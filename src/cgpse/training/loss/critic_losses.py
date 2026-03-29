from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------ helpers ------------------------

def _to_BLF(x: torch.Tensor) -> torch.Tensor:
    """
    Ensure logits are (B, L, F) with F=4 for heads.
    Accepts (B, L, 4) or (B, 4, L).
    """
    if x.dim() != 3:
        raise ValueError(f"Expected 3D tensor, got {tuple(x.shape)}")
    if x.shape[2] == 4:
        return x
    if x.shape[1] == 4:
        return x.permute(0, 2, 1).contiguous()
    raise ValueError(f"Unexpected logits shape {tuple(x.shape)}; expected (B,L,4) or (B,4,L)")


def _to_BLC(x: torch.Tensor, L: int) -> torch.Tensor:
    """
    Ensure features are (B, L, C).
    Accepts (B, L, C) or (B, C, L) where L matches.
    """
    if x.dim() != 3:
        raise ValueError(f"Expected 3D features, got {tuple(x.shape)}")
    if x.shape[1] == L:   # (B,L,C)
        return x
    if x.shape[2] == L:   # (B,C,L)
        return x.transpose(1, 2).contiguous()
    raise ValueError(f"Cannot reconcile feature shape {tuple(x.shape)} with length {L}")


# ------------------------ Unified bundle ------------------------

class CountScaledLMHeadLoss(nn.Module):
    """
    Count-scaled LM head loss (logit-space terms).

    Components:
      - KL(teacher || student) on masked positions.
      - Entropy matching on aligned positions.
      - REF-confidence matching term (Huber on REF CE gap).

    Count scaling mirrors fidelity's manifold alignment loss:
      - Compute N_total = sum_{t,l} gt_tracks[b, t, l] per example.
      - Apply count coefficient w(N_total) to each per-example loss.
    """

    def __init__(
        self,
        *,
        lambda_loss: float = 1.0,
        temp: float = 1.0,
        ref_ce_delta: float = 1.0,
        head_prefix: str = "gt_head",
        count_coeff_config=None,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.lambda_loss = float(lambda_loss)
        self.temp = float(temp) if temp > 0 else 1.0
        self.ref_ce_delta = float(ref_ce_delta)
        self.head_prefix = str(head_prefix)
        self.count_coeff_config = {} if count_coeff_config is None else dict(count_coeff_config)
        self.eps = float(eps)

    def _coeff(self, N: torch.Tensor) -> torch.Tensor:
        """
        Per-example coefficient w(N) with shape (B,).
        """
        typ = str(self.count_coeff_config.get("type", "none"))
        lower, upper = self.count_coeff_config.get("bounds", (None, None))

        N = N.clamp_min(0.0)
        if lower is not None:
            N = N.clamp_min(float(lower))

        if typ == "pow_softcap":
            alpha = float(self.count_coeff_config.get("alpha", 1.0))
            u = N ** alpha
            if upper is None:
                return u
            t = float(upper) ** alpha
            tt = u.new_tensor(t)
            return torch.where(u <= tt, u, 2.0 * torch.sqrt(u * tt) - tt)

        if upper is not None:
            N = N.clamp_max(float(upper))

        if typ == "total":
            return N
        if typ == "sqrt":
            return torch.sqrt(N)
        if typ == "pow":
            return N ** float(self.count_coeff_config.get("alpha", 1.0))
        if typ == "log1p":
            return torch.log1p(N)
        if typ == "none":
            return torch.ones_like(N)

        raise ValueError(f"Unknown count_coeff type: {typ}")

    @staticmethod
    def _per_example_masked_mean(
        x: torch.Tensor,   # (B,L)
        mask: torch.Tensor,  # (B,L) bool
        weights: Optional[torch.Tensor] = None,  # (B,L) or None
    ) -> torch.Tensor:
        mask_f = mask.float()
        if weights is not None:
            if weights.shape != mask.shape:
                raise ValueError(
                    f"weights shape {tuple(weights.shape)} != mask shape {tuple(mask.shape)}"
                )
            w = weights * mask_f
            num = (x * w).sum(dim=-1)                      # (B,)
            denom = w.sum(dim=-1).clamp_min(1.0)           # (B,)
        else:
            num = (x * mask_f).sum(dim=-1)                 # (B,)
            denom = mask_f.sum(dim=-1).clamp_min(1.0)      # (B,)
        return num / denom

    def forward(
        self,
        *,
        ref_onehot: torch.Tensor,           # (B,L,4)
        mask: torch.Tensor,                 # (B,L) bool (masked positions)
        teacher__logits: torch.Tensor,      # (B,L,4) or (B,4,L)
        student__logits: torch.Tensor,      # (B,L,4) or (B,4,L)
        gt_tracks: torch.Tensor,            # (B,T,L) counts for count-scaling
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if mask.dim() != 2:
            raise ValueError(f"mask must be (B,L), got {tuple(mask.shape)}")
        mask = mask.bool()

        if ref_onehot.dim() != 3 or ref_onehot.size(-1) != 4:
            raise ValueError(f"ref_onehot must be (B,L,4), got {tuple(ref_onehot.shape)}")
        B, L, _ = ref_onehot.shape
        if mask.shape != (B, L):
            raise ValueError(f"mask must be (B,L), got {tuple(mask.shape)}")

        if gt_tracks.dim() != 3:
            raise ValueError(f"gt_tracks must be (B,T,L), got {tuple(gt_tracks.shape)}")
        if gt_tracks.size(0) != B:
            raise ValueError(
                f"gt_tracks batch size {gt_tracks.size(0)} != ref_onehot batch size {B}"
            )

        # Per-example count scale from GT tracks.
        counts = gt_tracks.clamp_min(0.0)                      # (B,T,L)
        N_total = counts.sum(dim=(1, 2))                       # (B,)
        coeff = self._coeff(N_total)                           # (B,)

        # Convert logits to normalized distributions for KL/entropy terms.
        logits_t = _to_BLF(teacher__logits) / self.temp         # (B,L,4)
        logits_s = _to_BLF(student__logits) / self.temp         # (B,L,4)

        log_p = F.log_softmax(logits_t, dim=-1)                # (B,L,4)
        log_q = F.log_softmax(logits_s, dim=-1)                # (B,L,4)
        p = log_p.exp()
        q = log_q.exp()

        # KL(teacher || student) per position, reduced over masked positions.
        kl_pos = (p * (log_p - log_q)).sum(dim=-1)             # (B,L)
        H_p = -(p * log_p).sum(dim=-1)                         # (B,L)
        H_q = -(q * log_q).sum(dim=-1)                         # (B,L)

        kl_per_example = self._per_example_masked_mean(kl_pos, mask, weights=None)  # (B,)
        kl_unscaled = kl_per_example.mean()
        kl_scaled = (coeff * kl_per_example).mean()

        # Entropy match only where argmax aligns (still masked).
        argmax_t = logits_t.argmax(dim=-1)                     # (B,L)
        argmax_s = logits_s.argmax(dim=-1)                     # (B,L)
        align_mask = (argmax_t == argmax_s) & mask             # (B,L)

        ent_diff_sq = (H_q - H_p) ** 2                         # (B,L)
        ent_per_example = self._per_example_masked_mean(
            ent_diff_sq, align_mask, weights=None
        )                                                      # (B,)
        entropy_unscaled = ent_per_example.mean()
        entropy_scaled = (coeff * ent_per_example).mean()

        # REF-confidence one-sided Huber term on masked positions where teacher predicts REF.
        ref_idx = ref_onehot.argmax(dim=-1)                    # (B,L)
        ref_mask = (argmax_t == ref_idx) & mask # (B,L)

        log_p_ref = torch.gather(log_p, dim=-1, index=ref_idx.unsqueeze(-1)).squeeze(-1)
        log_q_ref = torch.gather(log_q, dim=-1, index=ref_idx.unsqueeze(-1)).squeeze(-1)

        ce_t_ref = -log_p_ref
        ce_s_ref = -log_q_ref

        gap = ce_t_ref - ce_s_ref                              # >0 => student more confident than teacher
        pos = F.relu(gap)                                      # one-sided

        delta = float(self.ref_ce_delta)
        quad = 0.5 * pos**2 / delta
        lin = pos - 0.5 * delta
        ref_over = torch.where(pos <= delta, quad, lin)        # one-sided Huber

        ref_ce_per_example = self._per_example_masked_mean(ref_over, ref_mask, weights=None)
        ref_ce_unscaled = ref_ce_per_example.mean()
        ref_ce_scaled = (coeff * ref_ce_per_example).mean()

        # Aggregate the three count-scaled terms under a single lambda.
        head_loss = kl_scaled + entropy_scaled + ref_ce_scaled
        total = self.lambda_loss * head_loss

        prefix = self.head_prefix
        stats: Dict[str, torch.Tensor] = {
            f"{prefix}/kl": kl_unscaled.detach(),
            f"{prefix}/count_scaled_kl": kl_scaled.detach(),
            f"{prefix}/entropy_loss": entropy_unscaled.detach(),
            f"{prefix}/count_scaled_entropy_loss": entropy_scaled.detach(),
            f"{prefix}/entropy_teacher": self._per_example_masked_mean(
                H_p, mask, weights=None
            ).mean().detach(),
            f"{prefix}/entropy_student": self._per_example_masked_mean(
                H_q, mask, weights=None
            ).mean().detach(),
            f"{prefix}/ref_ce_loss": ref_ce_unscaled.detach(),
            f"{prefix}/count_scaled_ref_ce_loss": ref_ce_scaled.detach(),
            f"{prefix}/loss": total.detach(),
        }

        return total, stats


class CountScaledTrackEmbeddingAlignmentHuberLoss(nn.Module):
    """
    Count-scaled Huber alignment loss between track embeddings.

    Expects:
      - emb_ref:  (B, L, C)
      - emb_pred: (B, L, C)

    Count scaling mirrors fidelity's manifold alignment loss:
      - Compute N_total = sum_{t,l} gt_tracks[b, t, l] per example.
      - Apply count coefficient w(N_total) to each per-example loss.
    """

    def __init__(
        self,
        *,
        count_coeff_config=None,
        delta: float = 1.0,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.count_coeff_config = {} if count_coeff_config is None else dict(count_coeff_config)
        self.delta = float(delta)
        self.eps = float(eps)

    def _coeff(self, N: torch.Tensor) -> torch.Tensor:
        """
        Per-example coefficient w(N) with shape (B,).
        """
        typ = str(self.count_coeff_config.get("type", "none"))
        lower, upper = self.count_coeff_config.get("bounds", (None, None))

        N = N.clamp_min(0.0)
        if lower is not None:
            N = N.clamp_min(float(lower))

        if typ == "pow_softcap":
            alpha = float(self.count_coeff_config.get("alpha", 1.0))
            u = N ** alpha
            if upper is None:
                return u
            t = float(upper) ** alpha
            tt = u.new_tensor(t)
            return torch.where(u <= tt, u, 2.0 * torch.sqrt(u * tt) - tt)

        if upper is not None:
            N = N.clamp_max(float(upper))

        if typ == "total":
            return N
        if typ == "sqrt":
            return torch.sqrt(N)
        if typ == "pow":
            return N ** float(self.count_coeff_config.get("alpha", 1.0))
        if typ == "log1p":
            return torch.log1p(N)
        if typ == "none":
            return torch.ones_like(N)

        raise ValueError(f"Unknown count_coeff type: {typ}")

    def forward(
        self,
        emb_ref: torch.Tensor,          # (B, L, C)
        emb_pred: torch.Tensor,         # (B, L, C)
        *,
        gt_tracks: torch.Tensor | None = None,  # (B, T, L) GT counts (optional)
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if emb_ref.shape != emb_pred.shape:
            raise ValueError(
                f"emb_ref and emb_pred must have same shape, got "
                f"{tuple(emb_ref.shape)} vs {tuple(emb_pred.shape)}"
            )
        if emb_ref.dim() != 3:
            raise ValueError(f"Expected (B,L,C) embeddings, got {tuple(emb_ref.shape)}")

        # Per-position L2 distance over channels, then Huber per position.
        diff = emb_pred - emb_ref                               # (B, L, C)
        dist = torch.sqrt((diff**2).sum(dim=-1) + self.eps)      # (B, L)

        abs_diff = dist
        quad = 0.5 * abs_diff**2 / self.delta
        lin = abs_diff - 0.5 * self.delta
        huber = torch.where(abs_diff <= self.delta, quad, lin)   # (B, L)

        huber_per = huber.mean(dim=-1)                           # (B,)

        if gt_tracks is not None:
            if gt_tracks.dim() != 3:
                raise ValueError(
                    f"gt_tracks must be (B,T,L), got {tuple(gt_tracks.shape)}"
                )
            counts = gt_tracks.clamp_min(0.0)                    # (B, T, L)
            N_total = counts.sum(dim=(1, 2))                     # (B,)
            w = self._coeff(N_total)                             # (B,)
            loss_per = w * huber_per                             # (B,)
            loss = loss_per.mean()
        else:
            loss = huber_per.mean()

        stats: Dict[str, torch.Tensor] = {
            "count_scaled_track_embed_huber": loss.detach(),
            "track_embed_huber": huber_per.mean().detach(),
            "track_embed_dist": dist.mean().detach(),
        }
        return loss, stats


class CriticLossBundle(nn.Module):
    """
    Critic loss bundle with count-scaled LM-head and track-embedding terms.

    Training objective:
        L_critic =
            lambda_lm_head * CountScaledLMHeadLoss(...)
          + lambda_track_embed * CountScaledTrackEmbeddingAlignmentHuberLoss(...)

    Notes
    -----
    - LM-head loss computes KL + entropy + REF-CE Huber terms on masked positions.
    - Track-embedding alignment is mask-agnostic.
    """

    def __init__(
        self,
        *,
        lambdas=None,
        count_coeffs=None,
        temp: float = 1.0,
        ref_ce_delta: float = 1.0,
        huber_delta: float = 1.0,
        head_prefix: str = "gt_head",
        eps: float = 1e-8,
    ):
        super().__init__()

        if lambdas is None:
            lambdas = {"lm_head": 1.0, "track_embed_align": 0.0}
        self.lambdas = dict(lambdas)

        if count_coeffs is None:
            count_coeffs = {
                "lm_head": {"type": "none", "bounds": (None, None)},
                "track_embed_align": {"type": "none", "bounds": (None, None)},
            }
        self.count_coeffs = dict(count_coeffs)

        self.lm_head_loss = CountScaledLMHeadLoss(
            lambda_loss=float(self.lambdas.get("lm_head", 0.0)),
            temp=temp,
            ref_ce_delta=ref_ce_delta,
            head_prefix=head_prefix,
            count_coeff_config=self.count_coeffs.get("lm_head", {}),
            eps=eps,
        )
        self.track_embed_align_loss = CountScaledTrackEmbeddingAlignmentHuberLoss(
            count_coeff_config=self.count_coeffs.get("track_embed_align", {}),
            delta=huber_delta,
            eps=eps,
        )

    # ---------------- main forward ---------------- #

    def forward(
        self,
        *,
        ref_onehot: torch.Tensor,           # (B,L,4)
        mask: torch.Tensor,                 # (B,L) bool (masked positions)
        teacher__logits: torch.Tensor,      # (B,L,4) or (B,4,L)
        student__logits: torch.Tensor,      # (B,L,4) or (B,4,L)
        gt_tracks: torch.Tensor,            # (B,T,L) counts for count-scaling
        teacher__track_embed: torch.Tensor,   # (B,L,C)
        student__track_embed: torch.Tensor,   # (B,L,C)
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        stats: Dict[str, torch.Tensor] = {}
        total = ref_onehot.new_tensor(0.0)

        # LM-head loss (always computed; lambda is internal).
        lm_loss, lm_stats = self.lm_head_loss(
            ref_onehot=ref_onehot,
            mask=mask,
            teacher__logits=teacher__logits,
            student__logits=student__logits,
            gt_tracks=gt_tracks,
        )
        total = total + lm_loss
        stats.update(lm_stats)

        # Track-embedding alignment (always computed; lambda applied here).
        lambda_track = float(self.lambdas.get("track_embed_align", 0.0))
        track_loss, track_stats = self.track_embed_align_loss(
            teacher__track_embed,
            student__track_embed,
            gt_tracks=gt_tracks,
        )
        total = total + lambda_track * track_loss
        stats.update(track_stats)
        stats["track_embed/loss"] = (lambda_track * track_loss).detach()

        # Global count stats (derived from LM-head coeffs).
        counts = gt_tracks.clamp_min(0.0)                      # (B,T,L)
        N_total = counts.sum(dim=(1, 2))                       # (B,)
        coeff = self.lm_head_loss._coeff(N_total)              # (B,)
        stats["critic/count_total_mean"] = N_total.mean().detach()
        stats["critic/count_coeff_mean"] = coeff.mean().detach()

        return total, stats



__all__ = [
    "CountScaledLMHeadLoss",
    "CountScaledTrackEmbeddingAlignmentHuberLoss",
    "CriticLossBundle",
]
