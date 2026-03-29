from typing import Dict, Any, Optional

import torch
import torch.nn as nn

from .helpers import masked_ce_loss, masked_token_ce_from_labels


class MultiHeadedCriticTrainingOrchestrator(nn.Module):
    """
    Single-stem multi-head critic pretraining orchestrator.
    - Single stem (no regime routing)
    - Multiple heads (GT, S2F, DISC)
    - Masks come from BatchPreparer
    - All masked positions are supervised
    - Supports multi-track inputs (T >= 1)

    Responsibilities
    ----------------
    - Takes a batch from the batch preparer:
        dna         : (B,L,4)
        exp_tracks  : (B,L,T)  (GT / experimental, T tracks)
        pred_tracks : (B,L,T)  (S2F / predicted, T tracks, must match exp_tracks)
        mask        : (B,L,4+K) input mask from BatchPreparer
                      where the first 4 channels are DNA mask and the
                      remaining channels encode track masks. For
                      compatibility, we accept both 4+T and 4+2T layouts
                      and reduce the track mask blocks with a channel-wise
                      OR.
    - Globally equalizes S2F totals to GT totals (per-track scaling).
    - Constructs critic inputs for each modality (GT, S2F) using a shared
      track-channel geometry [dna(4), tracks(T)].
    - Routes through SingleStemSingleHeadRouterCritic and computes MLM losses.
    - Supervises ALL masked DNA positions.
    - Preserves input mask structure for visible modality, masks out all channels
      of the hidden modality.
    """

    def __init__(
        self,
        critic: nn.Module,
        *,
        lambda_GT: float = 1.0,   # loss weight for GT head
        lambda_S2F: float = 1.0,  # loss weight for S2F head
        lambda_DISC: float = 1.0, # loss weight for DISC head
    ):
        super().__init__()
        self.critic = critic

        self.loss_weights = {
            "GT": float(lambda_GT),
            "S2F": float(lambda_S2F),
            "DISC": float(lambda_DISC),
        }

    @staticmethod
    def _global_total_equalize(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """
        Scale `pred` so its total counts match `target` over the full window.

        For multi-channel tracks (e.g., +/- strands in TF ChIP), sums ALL
        channels together to compute a single global scale factor. This
        preserves relative strand ratios while matching total signal.

        pred/target: (B,L,T) where T can be any number of track channels
        Returns: (B,L,T) with scaled pred (single scale applied to all channels)
        """
        if pred is None or target is None:
            return pred  # type: ignore[return-value]
        if pred.shape != target.shape:
            raise ValueError(f"_global_total_equalize shape mismatch {pred.shape} vs {target.shape}")

        # Sum over positions AND channels for global equalization
        # This ensures multi-strand tracks (T>1) share a single scale factor
        pred_sum = pred.sum(dim=(1, 2), keepdim=True)  # (B,1,1)
        target_sum = target.sum(dim=(1, 2), keepdim=True)  # (B,1,1)
        scale = (target_sum + eps) / (pred_sum + eps)  # (B,1,1)
        return pred * scale  # (B,L,T) - broadcasts scale to all channels

    def _build_critic_input(
        self,
        dna: torch.Tensor,                     # (B,L,4)
        gt_track: Optional[torch.Tensor],      # (B,L,T) or None
        s2f_track: Optional[torch.Tensor],     # (B,L,T) or None
        input_mask: torch.Tensor,              # (B,L,4+K)
        modality: str,                         # "GT" or "S2F"
    ) -> torch.Tensor:
        """
        Build critic input tensor (B,2,L,4+T) for a given modality.

        For both GT and S2F modalities we use a shared track-channel
        geometry:
          - Values: [dna(4), tracks(T)]
          - Mask  : [dna_mask(4), track_mask(T)]

        `input_mask` can be:
          - (B,L,4+T)  : dna_mask(4) + track_mask(T)
          - (B,L,4+2T) : dna_mask(4) + two T-blocks (e.g. GT/S2F); in this
                         case we OR the two track-mask blocks into a single
                         T-channel mask.
        """
        B, L, _ = dna.shape
        device = dna.device

        # Determine number of track channels and validate
        if gt_track is not None and s2f_track is not None:
            T_gt = gt_track.size(2)
            T_s2f = s2f_track.size(2)
            if T_gt != T_s2f:
                raise ValueError(
                    f"GT tracks ({T_gt} channels) and S2F tracks ({T_s2f} channels) "
                    f"must have same number of channels"
                )
            num_track_channels = T_gt
        elif gt_track is not None:
            num_track_channels = gt_track.size(2)
        elif s2f_track is not None:
            num_track_channels = s2f_track.size(2)
        else:
            raise ValueError("At least one of gt_track or s2f_track must be provided")

        C_mask = input_mask.size(2)
        if C_mask == 4 + num_track_channels:
            dna_mask = input_mask[:, :, :4]            # (B,L,4)
            track_mask = input_mask[:, :, 4:]          # (B,L,T)
        elif C_mask == 4 + 2 * num_track_channels:
            dna_mask = input_mask[:, :, :4]            # (B,L,4)
            track_mask_1 = input_mask[:, :, 4:4+num_track_channels]                # (B,L,T)
            track_mask_2 = input_mask[:, :, 4+num_track_channels:4+2*num_track_channels]  # (B,L,T)
            track_mask = (track_mask_1.bool() | track_mask_2.bool()).float()
        else:
            raise ValueError(
                f"Mask channels {C_mask} != expected 4+T or 4+2T "
                f"(T={num_track_channels})"
            )

        if modality == "GT":
            if gt_track is None:
                raise ValueError("GT modality requested but gt_track is None")
            tracks = gt_track
        elif modality == "S2F":
            if s2f_track is None:
                raise ValueError("S2F modality requested but s2f_track is None")
            tracks = s2f_track
        else:
            raise ValueError(f"Unknown modality '{modality}' (expected 'GT' or 'S2F')")

        if tracks.shape != (B, L, num_track_channels):
            raise ValueError(
                f"tracks shape {tuple(tracks.shape)} != expected {(B, L, num_track_channels)}"
            )

        modality_mask = torch.cat([dna_mask, track_mask], dim=-1)  # (B,L,4+T)
        val = torch.cat([dna, tracks], dim=-1)                     # (B,L,4+T)
        if modality_mask.shape != val.shape:
            raise ValueError(
                f"modality_mask shape {tuple(modality_mask.shape)} != values {tuple(val.shape)}"
            )

        val_masked = val.masked_fill(modality_mask > 0, 0.0)
        x = torch.stack([val_masked, modality_mask], dim=1)  # (B,2,L,4+T)
        return x

    def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Compute single-stem multi-head critic MLM loss.

        Expects batch keys:
          - 'dna'        : (B,L,4)
          - 'exp_tracks' : (B,L,T) or None  (GT tracks, T can be any number >= 1)
          - 'pred_tracks': (B,L,T) or None  (S2F tracks, T must match exp_tracks if both present)
          - 'mask'       : (B,L,4+2T) input mask from BatchPreparer

        Supervises ALL masked DNA positions (no subset sampling).
        Supports multi-track inputs (T > 1).
        """
        dna: torch.Tensor = batch["dna"]
        gt: Optional[torch.Tensor] = batch.get("exp_tracks")  # (B,L,T) or None
        s2f: Optional[torch.Tensor] = batch.get("pred_tracks")  # (B,L,T) or None
        input_mask: torch.Tensor = batch["mask"]  # (B,L,4+K)

        L_dna = dna.shape[1]

        # Ensure DNA length matches track length via symmetric center crop if needed.
        L_track = None
        if isinstance(gt, torch.Tensor) and gt.dim() == 3 and gt.size(2) > 0:
            L_track = gt.size(1)
        if L_track is None and isinstance(s2f, torch.Tensor) and s2f.dim() == 3 and s2f.size(2) > 0:
            L_track = s2f.size(1)
        if L_track is not None and L_dna != L_track:
            # Symmetric crop DNA to track length
            offset = max(0, (L_dna - L_track) // 2)
            dna = dna[:, offset:offset + L_track, :]
            L = L_track
        else:
            L = L_dna

        # Ensure mask length matches
        if input_mask.size(1) != L:
            raise ValueError(f"Mask length {input_mask.size(1)} != DNA/track length {L}")

        # Validate track channel counts match
        if gt is not None and s2f is not None:
            if gt.size(2) != s2f.size(2):
                raise ValueError(
                    f"GT tracks ({gt.size(2)} channels) and S2F tracks ({s2f.size(2)} channels) "
                    f"must have same number of channels"
                )

        # Global total-count equalization: align S2F totals to GT totals across the window
        # This works for multi-track: sums over L, broadcasts scale over T
        if s2f is not None and gt is not None:
            s2f = self._global_total_equalize(s2f, gt)

        # Supervision mask: supervise ALL masked DNA positions
        # input_mask: (B,L,6) where first 4 channels are DNA mask
        # supervise_mask: (B,L) bool - True where ANY DNA channel is masked
        supervise_mask = (input_mask[:, :, :4] > 0).any(dim=-1)  # (B,L)

        # Ground-truth DNA one-hot for CE
        ref_genome_1hot = dna  # (B,L,4)

        losses: Dict[str, torch.Tensor] = {}
        disc_terms = []

        # Run GT head + DISC (if tracks available)
        if gt is not None:
            x = self._build_critic_input(dna, gt_track=gt, s2f_track=None, input_mask=input_mask, modality="GT")
            out = self.critic(x, heads=["GT", "DISC"])
            logits_gt = out["GT"]         # (B,L,4)
            logits_disc_gt = out["DISC"]  # (B,L,C_disc)
            losses["GT"] = masked_ce_loss(logits_gt, ref_genome_1hot, supervise_mask)

            labels_gt = torch.ones_like(supervise_mask, dtype=torch.long)
            disc_gt = masked_token_ce_from_labels(logits_disc_gt, labels_gt, supervise_mask)
            losses["DISC_GT"] = disc_gt
            disc_terms.append(disc_gt)

        # Run S2F head + DISC (if tracks available)
        if s2f is not None:
            x = self._build_critic_input(dna, gt_track=None, s2f_track=s2f, input_mask=input_mask, modality="S2F")
            out = self.critic(x, heads=["S2F", "DISC"])
            logits_s2f = out["S2F"]       # (B,L,4)
            logits_disc_s2f = out["DISC"] # (B,L,C_disc)
            losses["S2F"] = masked_ce_loss(logits_s2f, ref_genome_1hot, supervise_mask)

            labels_s2f = torch.zeros_like(supervise_mask, dtype=torch.long)
            disc_s2f = masked_token_ce_from_labels(logits_disc_s2f, labels_s2f, supervise_mask)
            losses["DISC_S2F"] = disc_s2f
            disc_terms.append(disc_s2f)

        if disc_terms:
            losses["DISC"] = torch.stack(disc_terms).mean()

        # Weighted total: use GT, S2F, DISC in optimization;
        # keep DISC_GT / DISC_S2F for logging only.
        total = None
        for name, loss in losses.items():
            if name in ("DISC_GT", "DISC_S2F"):
                continue
            w = self.loss_weights.get(name, 1.0)
            term = w * loss
            total = term if total is None else total + term
        if total is None:
            raise ValueError("No modality had available tracks; cannot compute critic loss.")

        losses["total"] = total
        return losses
