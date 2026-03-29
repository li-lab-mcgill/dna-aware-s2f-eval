from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..model import DNAAwareEditorAugmentedDAE
from .critic_adapter import CriticAdapter
from ..losses.critic_losses import CriticLossBundle
from ..losses import (
    FidelityLossBundle,
    GradientRatioController,
)


class TrainingOrchestrator(nn.Module):
    """
    Stage-4: DNA-aware editor training on top of a frozen DNA-free DAE.

    Wraps:
      * DNAAwareEditorAugmentedDAE (frozen DAE + trainable DNA-aware editor),
      * CriticAdapter (teacher=GT head, students=DAE reconstructions),
      * FidelityLossBundle (count-scaled CE vs GT, and optional manifold alignment).

    Expected batch keys
    -------------------
    dna         : (B, L_dna, 4) one-hot REF sequence
    exp_tracks  : (B, L, T)    GT profile-shape counts y_gt
    pred_tracks : (B, L, T)    S2F profile-shape counts y_s2f
    critic_input_mask  : (B, L) or (B, L, 4+T) — M_ci (DNA visibility to critic input)
    critic_output_mask : (B, L) or (B, L, 4+T) — M_co (audited/scored rows)

    Mask conventions:
      - 0 = visible, 1 = masked.
    """

    def __init__(
        self,
        *,
        dae: DNAAwareEditorAugmentedDAE,
        critic_adapter: CriticAdapter,
        critic_losses: CriticLossBundle,
        fidelity_losses: FidelityLossBundle,
        mode_probs: Tuple[float, float] = (0.1, 0.9),
        grad_ratio_controller: Optional[GradientRatioController] = None,
        grad_params: Optional[Iterable[nn.Parameter]] = None,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.dae = dae
        self.critic_adapter = critic_adapter
        self.critic_losses = critic_losses
        self.fidelity_losses = fidelity_losses

        mode_probs_tensor = torch.tensor(mode_probs, dtype=torch.float32)
        if mode_probs_tensor.numel() != 3:
            raise ValueError("mode_probs must have three entries (modes 1-3).")
        if float(mode_probs_tensor.sum()) <= 0.0:
            raise ValueError("mode_probs must sum to a positive value.")
        self.mode_probs = mode_probs_tensor / mode_probs_tensor.sum()

        self.grad_ratio_controller = grad_ratio_controller
        self.eps = float(eps)

        # By default, GradientRatioController uses y_sync (DAE outputs)
        # as the reference for gradient norms.
        if grad_params is None:
            self._grad_params = list(self.dae.parameters())
        else:
            self._grad_params = list(grad_params)

        if len(self._grad_params) == 0:
            raise ValueError("TrainingOrchestrator received an empty grad_params list.")

    @staticmethod
    def get_position_mask(mask: torch.Tensor) -> torch.Tensor:
        """
        Convert a mask in either geometry to a row-wise boolean mask (B, L).

        Accepts:
          - (B, L) mask (bool or 0/1 float/int)
          - (B, L, 4+T) mask over [DNA(4), profile-shape(T)] where DNA columns
            are reduced with an OR over the 4 bases.

        Returns
        -------
        (B, L) bool, True where masked.
        """
        if mask.dim() == 2:
            return mask.to(torch.bool) if mask.dtype == torch.bool else (mask > 0)
        if mask.dim() == 3:
            return (mask[:, :, :4] > 0).any(dim=-1)
        raise ValueError(
            f"Unsupported mask shape {tuple(mask.shape)}; expected (B,L) or (B,L,4+T)."
        )

    def train(self, mode: bool = True):
        """
        Custom train mode:
        - DAE follows the requested mode.
        - Critic (inside adapter) remains frozen and in eval mode.
        """
        super().train(mode)
        self.dae.train(mode)

        critic = getattr(self.critic_adapter, "critic", None)
        if critic is not None:
            critic.eval()
            for p in critic.parameters():
                p.requires_grad = False
        return self

    @staticmethod
    def _build_masked_dna_input(
        dna: torch.Tensor,      # (B, L, 4)
        dna_mask: torch.Tensor, # (B, L) 0 = visible, 1 = masked
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build fused DNA value+mask tensor given a (B,L) mask.

            dna_fused: (B, 2, L, 4)
              - channel 0: DNA values with masked positions zeroed
              - channel 1: DNA mask (1 = masked, 0 = visible), broadcast over 4 bases

        Returns
        -------
        dna_fused     : (B, 2, L, 4)
        dna_mask_bool : (B, L) bool mask (True where DNA is masked)
        """
        if dna_mask.dtype == torch.bool:
            dna_mask_bool = dna_mask
        else:
            dna_mask_bool = dna_mask > 0

        dna_mask_channel = dna_mask_bool.unsqueeze(-1).expand(-1, -1, 4).to(dna.dtype)  # (B,L,4)
        dna_values_masked = dna.masked_fill(dna_mask_channel > 0, 0.0)                  # (B,L,4)
        dna_fused = torch.stack([dna_values_masked, dna_mask_channel], dim=1)           # (B,2,L,4)
        return dna_fused, dna_mask_bool

    # ---------------- Main forward ----------------

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        *,
        mode: str = "train",
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Returns
        -------
        total_loss : scalar tensor
        stats      : dict[str, scalar tensor] for logging

        Notes
        -----
        - mode="train" uses stochastic mode sampling.
        - mode!="train" uses batch masks directly.
        """
        dna = batch["dna"]                    # (B, L_dna, 4)
        gt_tracks = batch["exp_tracks"]       # (B, L, T) GT counts
        s2f_tracks = batch["pred_tracks"]     # (B, L, T) S2F counts
        critic_input_mask = batch["critic_input_mask"]
        critic_output_mask = batch["critic_output_mask"]

        B, L, T = gt_tracks.shape
        L_dna = int(dna.shape[1])

        # Center-crop DNA to match track length if needed.
        if L_dna != L:
            offset = max(0, (L_dna - L) // 2)
            dna = dna[:, offset:offset + L, :]

        # Row-wise (B,L) masks are needed for:
        #  - CriticLossBundle supervision mask (expects a (B,L) mask).
        critic_input_row_base = self.get_position_mask(critic_input_mask).to(device=dna.device)     # (B,L) bool
        critic_output_row_base = self.get_position_mask(critic_output_mask).to(device=dna.device)   # (B,L) bool

        # CriticAdapter expects a full (B,L,4+T) mask over [DNA(4), track(T)].
        if critic_input_mask.dim() == 2:
            dna_mask_cols = (
                critic_input_row_base.to(device=dna.device, dtype=dna.dtype)
                .unsqueeze(-1)
                .expand(-1, -1, 4)
            )  # (B,L,4)
            track_mask_cols = torch.zeros((B, L, T), device=dna.device, dtype=dna.dtype)         # (B,L,T)
            critic_input_mask_full = torch.cat([dna_mask_cols, track_mask_cols], dim=-1)         # (B,L,4+T)
        elif critic_input_mask.dim() == 3:
            if critic_input_mask.shape != (B, L, 4 + T):
                raise ValueError(
                    f"critic_input_mask must be (B,L,4+T) with T={T}, got {tuple(critic_input_mask.shape)}"
                )
            critic_input_mask_full = critic_input_mask.to(device=dna.device, dtype=dna.dtype)
        else:
            raise ValueError(
                f"critic_input_mask must be (B,L) or (B,L,4+T), got {tuple(critic_input_mask.shape)}"
            )
        
        # ---------------- Mode selection for critic DNA visibility ---------------- #
        # Train: stochastic mode sampling.
        # Val/infer: use batch DNA mask and audit Co rows.
        if mode == "train":
            mode_probs = self.mode_probs.to(dna.device)
            mode_ids = torch.multinomial(mode_probs, num_samples=B, replacement=True)
            mode1 = mode_ids == 0
            mode2 = mode_ids == 1
            mode3 = mode_ids == 2

            critic_output_row = critic_output_row_base.clone()
            if mode1.any():
                critic_output_row[mode1] = True
                critic_input_mask_full = critic_input_mask_full.clone()
                critic_input_mask_full[mode1, :, :4] = 1.0
            if mode2.any():
                critic_output_row[mode2] = critic_output_row_base[mode2]
            if mode3.any():
                critic_output_row[mode3] = critic_output_row_base[mode3]
                critic_input_mask_full = critic_input_mask_full.clone()
                critic_input_mask_full[mode3, :, :4] = (
                    critic_output_row_base[mode3]
                    .to(device=dna.device, dtype=dna.dtype)
                    .unsqueeze(-1)
                    .expand(-1, -1, 4)
                )
        else:
            critic_output_row = critic_output_row_base

        # ---------------- DNA editor masking (coupled to critic output mask) ---------------- #
        # By default, S2F sees DNA where critic does NOT audit (invert Co).
        ds2f_mask = ~critic_output_row

        dna_fused_s2f, _ = self._build_masked_dna_input(dna, ds2f_mask)  # (B,2,L,4)

        # Ensure non-negative counts
        gt_nonneg = gt_tracks.clamp_min(0.0)      # (B, L, T)
        s2f_nonneg = s2f_tracks.clamp_min(0.0)    # (B, L, T)

        # ---------------- Tracks -> log-probs ---------------- #

        # GT
        denom_gt = gt_nonneg.sum(dim=1, keepdim=True).clamp_min(self.eps)   # (B, 1, T)
        p_gt = gt_nonneg / denom_gt                                         # (B, L, T)
        gt_logprobs_btl = torch.log(p_gt.clamp_min(self.eps)).permute(0, 2, 1).contiguous()  # (B, T, L)

        # S2F
        denom_s2f = s2f_nonneg.sum(dim=1, keepdim=True).clamp_min(self.eps)  # (B, 1, T)
        p_s2f = s2f_nonneg / denom_s2f                                       # (B, L, T)
        s2f_logprobs_btl = torch.log(p_s2f.clamp_min(self.eps)).permute(0, 2, 1).contiguous()  # (B, T, L)

        # ---------------- Editor forward ---------------- #
        # GT branch: latents only (for manifold alignment).
        z_gt = self.dae(gt_logprobs_btl, route="gt")                          # (B, C_lat, L_lat)
        # S2F branch: edited reconstruction + edited latents.
        recon_s2f_logits, z_s2f = self.dae(
            s2f_logprobs_btl,
            route="s2f",
            masked_dna=dna_fused_s2f,
        )                                                                     # (B,T,L), (B,C_lat,L_lat)
        recon_s2f_logprobs = F.log_softmax(recon_s2f_logits, dim=-1)          # (B,T,L)

        # ---------------- Fidelity losses (GT counts vs recon log-probs) ---------------- #
        fid_stats: Dict[str, torch.Tensor] = {}
        gt_tracks_btl = gt_nonneg.permute(0, 2, 1).contiguous()           # (B, T, L)

        fid_s2f, fid_stats_s2f = self.fidelity_losses(
            recons_logprobs=recon_s2f_logprobs,
            gt_tracks=gt_tracks_btl,
            z_gt=z_gt,
            z_recons=z_s2f,
        )

        fid_total = fid_s2f
        fid_stats["s2f/total"] = fid_s2f.detach()
        for k, v in fid_stats_s2f.items():
            if isinstance(v, torch.Tensor):
                fid_stats[f"s2f/{k}"] = v.detach()

        # ---------------- Critic: GT head alignment on audited positions ---------------- #
        critic_stats: Dict[str, torch.Tensor] = {}

        critic = getattr(self.critic_adapter, "critic", None)
        if critic is not None:
            critic_device = next(critic.parameters()).device
            if critic_device != dna.device:
                critic.to(dna.device)

        recon_s2f_probs_blt = recon_s2f_logprobs.exp().permute(0, 2, 1).contiguous()    # (B,L,T)

        critic_views = self.critic_adapter.compute_logits_and_embeddings(
            dna=dna,
            teacher_track=gt_nonneg,  # teacher is GT (counts)
            student_tracks={
                "s2f_recons": recon_s2f_probs_blt,
            },
            critic_input_mask=critic_input_mask_full,
        )

        gt_tracks_btl = gt_nonneg.permute(0, 2, 1).contiguous()                          # (B, T, L)                                             # (B, L) bool

        critic_loss_s2f, critic_stats_s2f = self.critic_losses(
            ref_onehot=dna,
            mask=critic_output_row,
            teacher__logits=critic_views["teacher__logits"],
            student__logits=critic_views["student__s2f_recons__logits"],
            gt_tracks=gt_tracks_btl,
            teacher__track_embed=critic_views["teacher__track_embed"],
            student__track_embed=critic_views["student__s2f_recons__track_embed"],
        )

        critic_total = critic_loss_s2f
        for k, v in critic_stats_s2f.items():
            if isinstance(v, torch.Tensor):
                critic_stats[f"s2f/{k}"] = v.detach()

        # ---------------- Combine losses ---------------- #

        lambda_t = 1.0
        g_fid_val: Optional[float] = None
        g_crit_val: Optional[float] = None
        if (
            self.grad_ratio_controller is not None
            and self.training
            and torch.is_grad_enabled()
        ):
            # Use z_s2f as the reference tensor for grad norms.
            g_fid_val = self.grad_ratio_controller.compute_grad_norm(
                fid_total, z_s2f
            )
            g_crit_val = self.grad_ratio_controller.compute_grad_norm(
                critic_total, z_s2f
            )
            lambda_t = self.grad_ratio_controller.update(g_fid_val, g_crit_val)

        total_loss = fid_total + float(lambda_t) * critic_total

        # ---------------- Stats ---------------- #

        stats: Dict[str, torch.Tensor] = {}
        stats["loss_total"] = total_loss.detach()

        stats["fid/total"] = fid_total.detach()
        for k, v in fid_stats.items():
            stats[f"fid/{k}"] = v

        stats["critic/total"] = critic_total.detach()
        for k, v in critic_stats.items():
            stats[f"critic/{k}"] = v

        if self.grad_ratio_controller is not None:
            stats["lambda_t"] = torch.tensor(float(lambda_t), device=total_loss.device)

        if g_fid_val is not None and g_crit_val is not None:
            stats["grad/g_fid"] = torch.tensor(float(g_fid_val), device=total_loss.device)
            stats["grad/g_critic"] = torch.tensor(float(g_crit_val), device=total_loss.device)

        return total_loss, stats


__all__ = ["TrainingOrchestrator"]
