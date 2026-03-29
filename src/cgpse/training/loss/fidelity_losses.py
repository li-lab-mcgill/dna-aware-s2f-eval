from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn


class MultinomialNLL(nn.Module):
    """
    Full multinomial negative log-likelihood over the full window.

    Expects:
      - logps: (B, T, L) log-probabilities with logsumexp(logps, dim=-1) ≈ 0
      - counts: (B, T, L) non-negative counts

    Computes, per example:
        -log P(counts | p)
      including combinatorial terms:

        -log N! + sum_i log(count_i!) - sum_i count_i * log p_i

    Returns mean loss over batch and {"mnll": loss}.
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = float(eps)

    def forward(
        self,
        logps: torch.Tensor,      # (B, T, L) log-probs
        counts: torch.Tensor,     # (B, T, L) counts
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if logps.shape != counts.shape:
            raise ValueError(
                f"logps and counts must have same shape, got "
                f"{tuple(logps.shape)} vs {tuple(counts.shape)}"
            )

        # Clamp counts to be non-negative
        counts_clamped = counts.clamp_min(0.0)           # (B, T, L)

        # N = sum_i counts_i
        N = counts_clamped.sum(dim=-1)                   # (B, T)

        # log N!
        log_fact_sum = torch.lgamma(N + 1.0)             # (B, T)

        # sum_i log(count_i!)
        log_prod_fact = torch.lgamma(counts_clamped + 1.0).sum(dim=-1)  # (B, T)

        # sum_i counts_i * log p_i
        log_prod_exp = (counts_clamped * logps).sum(dim=-1)       # (B, T)

        # Full multinomial NLL per example
        loss_per_example = -log_fact_sum + log_prod_fact - log_prod_exp  # (B, T)
        loss = loss_per_example.mean()

        return loss, {"mnll": loss.detach()}


class CountScaledCE(nn.Module):
    """
    Count-weighted cross-entropy between reconstructed and GT.

    For each (B, T) example:
      - Construct p_gt = counts / sum_i counts_i.
      - CE = -sum_i p_gt(i) * log p_edit(i).
      - Let N = sum_i counts_i be the total GT count for that (B, T).
      - Apply a coefficient w(N) and return mean over (B, T):

            loss = mean_{B,T} [ w(N) * CE ].
    """
    def __init__(
        self,
        *,
        count_coeff_config=None,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.count_coeff_config = {} if count_coeff_config is None else dict(count_coeff_config)
        self.eps = float(eps)

    def _coeff(self, N: torch.Tensor) -> torch.Tensor:
        """
        Compute per-window per-track coefficients w(N) with shape (B,T).
        """
        # none is default
        typ = str(self.count_coeff_config.get("type", "none"))

        # lower and upperbounds are in form of total counts
        lower, upper = self.count_coeff_config.get("bounds", (None, None))

        # lowerbound hard clip
        N = N.clamp_min(0.0)
        if lower is not None:
            N = N.clamp_min(float(lower))

        # power transform with soft cap
        if typ == "pow_softcap":
            # power transform
            alpha = float(self.count_coeff_config.get("alpha", 1.0))
            u = N ** alpha
            if upper is None:
                return u
            # upperbound smooth cap with transition at upper**alpha
            t = float(upper) ** alpha
            tt = u.new_tensor(t)
            return torch.where(u <= tt, u, 2.0 * torch.sqrt(u * tt) - tt)

        # other transforms interpret the upper as hard clip
        if upper is not None:
            N = N.clamp_max(float(upper))

        # total count scaling
        if typ == "total":
            return N
        # square root scaling
        if typ == "sqrt":
            return torch.sqrt(N)
        # power scaling
        if typ == "pow":
            return N ** float(self.count_coeff_config.get("alpha", 1.0))
        # log1p scaling
        if typ == "log1p":
            return torch.log1p(N)
        # otherwise equal weighting
        if typ == "none":
            return torch.ones_like(N)
        
        raise ValueError(f"Unknown count_coeff type: {typ}")

    def forward(
        self,
        logps: torch.Tensor,      # (B, T, L) log-probs
        counts: torch.Tensor,     # (B, T, L) counts
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        
        if logps.shape != counts.shape:
            raise ValueError(
                f"logps and counts must have same shape, got "
                f"{tuple(logps.shape)} vs {tuple(counts.shape)}"
            )

        # Clamp counts and compute totals
        counts_clamped = counts.clamp_min(0.0)                          # (B, T, L)
        N = counts_clamped.sum(dim=-1)                                  # (B, T) per-track totals

        # Ground-truth probabilities per (B,T)
        denom = N.clamp_min(self.eps).unsqueeze(-1)                     # (B, T, 1)
        p_gt = counts_clamped / denom                                   # (B, T, L)

        # Cross-entropy per (B,T): CE = -sum_i p(i) * log p_edit(i)
        ce_per = -(p_gt * logps).sum(dim=-1)                            # (B, T)

        # Per-window count coefficient
        w = self._coeff(N)                                              # (B, T)

        # scale loss
        loss_per = w * ce_per                                           # (B, T)
        loss = loss_per.mean()

        stats: Dict[str, torch.Tensor] = {
            "count_scaled_ce": loss.detach(),
            "ce": ce_per.mean().detach(),
        }
        return loss, stats

class CountScaledManifoldAlignmentHuberLoss(nn.Module):
    """
    Count-scaled Huber alignment loss between two bottleneck representations.

    Expects:
      - z_gt:  (B, C_lat, L_lat)
      - z_recons: (B, C_lat, L_lat)

    Implementation:
      - Compute per-position L2 distance across channels:
            d_{b,l} = ||z_gt[b,:,l] - z_recons[b,:,l]||_2.
      - Apply Huber(d; delta) to get h_{b,l}.
      - Mean over L_lat to get a per-example loss.
      - Optionally apply a count-based coefficient c(N_b), where N_b is derived
        from `gt_tracks` (sum over tracks and positions).

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
        Compute per-window per-track coefficients w(N) with shape (B,T).
        """
        # none is default
        typ = str(self.count_coeff_config.get("type", "none"))

        # lower and upperbounds are in form of total counts
        lower, upper = self.count_coeff_config.get("bounds", (None, None))

        # lowerbound hard clip
        N = N.clamp_min(0.0)
        if lower is not None:
            N = N.clamp_min(float(lower))

        # power transform with soft cap
        if typ == "pow_softcap":
            # power transform
            alpha = float(self.count_coeff_config.get("alpha", 1.0))
            u = N ** alpha
            if upper is None:
                return u
            # upperbound smooth cap with transition at upper**alpha
            t = float(upper) ** alpha
            tt = u.new_tensor(t)
            return torch.where(u <= tt, u, 2.0 * torch.sqrt(u * tt) - tt)

        # other transforms interpret the upper as hard clip
        if upper is not None:
            N = N.clamp_max(float(upper))

        # total count scaling
        if typ == "total":
            return N
        # square root scaling
        if typ == "sqrt":
            return torch.sqrt(N)
        # power scaling
        if typ == "pow":
            return N ** float(self.count_coeff_config.get("alpha", 1.0))
        # log1p scaling
        if typ == "log1p":
            return torch.log1p(N)
        # otherwise equal weighting
        if typ == "none":
            return torch.ones_like(N)
        
        raise ValueError(f"Unknown count_coeff type: {typ}")

    def forward(
        self,
        z_gt: torch.Tensor,             # (B, C_lat, L_lat)
        z_recons: torch.Tensor,         # (B, C_lat, L_lat)
        *,
        gt_tracks: torch.Tensor | None = None,  # (B, T, L) GT counts (optional)
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        
        if z_gt.shape != z_recons.shape:
            raise ValueError(
                f"CountScaledManifoldAlignmentHuberLoss: shapes must match, got "
                f"{tuple(z_gt.shape)} vs {tuple(z_recons.shape)}"
            )

        # Per-position L2 distance across channels
        diff = z_recons - z_gt                                   # (B, C_lat, L_lat)
        dist = torch.sqrt((diff**2).sum(dim=1) + self.eps)       # (B, L_lat)

        # Huber(dist; delta)
        abs_diff = dist
        quad = 0.5 * abs_diff**2 / self.delta
        lin = abs_diff - 0.5 * self.delta
        huber = torch.where(abs_diff <= self.delta, quad, lin)   # (B, L_lat)

        # Mean Huber per example
        huber_per = huber.mean(dim=-1)                           # (B,)

        # Optional count scaling using GT tracks (reuse the existing GT counts tensor)
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
            "count_scaled_manifold_huber": loss.detach(),
            "manifold_huber": huber_per.mean().detach(),
            "manifold_dist": dist.mean().detach(),
        }
        return loss, stats


class FidelityLossBundle(nn.Module):
    """
    Combined fidelity loss on reconstructed profile shapes with count-weighted terms.

    Training objective:
        L_fid = sum_k lambdas[k] * L_k
        where k in {"ce", "manifold_align"} and each L_k includes its own
        count-based coefficient computed from gt_tracks.

    Monitoring-only:
      - Always computes MultinomialNLL(recons_logprobs, gt_tracks) as "mnll",
        but this term is never added to L_fid.

    Notes
    -----
    - `recons_logprobs` is assumed to be log-probs (e.g., output from log_softmax).
    - If a lambda is 0, the corresponding term is skipped.
    """

    def __init__(
        self,
        *,
        lambdas=None,
        count_coeffs_configs=None,
        eps: float = 1e-8,
        huber_delta: float = 1.0,
    ):
        super().__init__()

        # loss coeff
        if lambdas is None:
            # default to both losses
            lambdas = {"ce": 1.0, "manifold_align": 1.0}
        self.lambdas = dict(lambdas)

        # Monitor-only MNLL (never added to the training loss)
        self.mnll = MultinomialNLL(eps=eps)

        # per example count coeff
        if count_coeffs_configs is None:
            # default to no count reweighting
            count_coeffs_configs = {
                "ce": {"type": "none", "bounds": (None, None)},
                "manifold_align": {"type": "none", "bounds": (None, None)},
            }
        self.count_coeffs_configs = dict(count_coeffs_configs)

        self.eps = float(eps)
        self.huber_delta = float(huber_delta)

        # initialize losses
        self.count_ce = CountScaledCE(
            count_coeff_config=self.count_coeffs_configs.get("ce", {}),
            eps=self.eps,
        )
        self.count_manifold_align = CountScaledManifoldAlignmentHuberLoss(
            count_coeff_config=self.count_coeffs_configs.get("manifold_align", {}),
            delta=self.huber_delta,
            eps=self.eps,
        )

    def forward(
        self,
        *,
        recons_logprobs: torch.Tensor,  # (B, T, L) log-probs
        gt_tracks: torch.Tensor,        # (B, T, L)
        z_gt: torch.Tensor | None = None,      # (B, C_lat, L_lat)
        z_recons: torch.Tensor | None = None,  # (B, C_lat, L_lat)
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        
        if recons_logprobs.shape != gt_tracks.shape:
            raise ValueError(
                f"recons_logprobs and gt_tracks must have same shape, got "
                f"{tuple(recons_logprobs.shape)} vs {tuple(gt_tracks.shape)}"
            )

        stats: Dict[str, torch.Tensor] = {}
        # Use a scalar tensor so the return value is always a Tensor on the right device
        # (even if all lambdas are 0).
        total = recons_logprobs.new_tensor(0.0)

        # Always compute MNLL for monitoring (no gradient contribution).
        with torch.no_grad():
            _, mnll_stats = self.mnll(
                recons_logprobs.detach(),
                gt_tracks.detach(),
            )
        stats.update(mnll_stats)

        # Count-weighted CE term
        lambda_ce = float(self.lambdas.get("ce", 0.0))
        if lambda_ce > 0.0:
            ce_loss, ce_stats = self.count_ce(
                recons_logprobs,
                gt_tracks,
            )
            total = total + lambda_ce * ce_loss
            stats.update(ce_stats)

        # Count-weighted manifold alignment Huber term
        lambda_manifold_align = float(self.lambdas.get("manifold_align", 0.0))
        if lambda_manifold_align > 0.0:
            assert (z_gt is None) + (z_recons is None) == 0, (
                "When lambda_manifold_align > 0, z_gt and z_recons must both be provided"
            )
            align_loss, align_stats = self.count_manifold_align(
                z_gt,
                z_recons,
                gt_tracks=gt_tracks,
            )
            total = total + lambda_manifold_align * align_loss
            stats.update(align_stats)

        return total, stats

__all__ = [
    "MultinomialNLL",
    "CountScaledCE",
    "CountScaledManifoldAlignmentHuberLoss",
    "FidelityLossBundle",
]
