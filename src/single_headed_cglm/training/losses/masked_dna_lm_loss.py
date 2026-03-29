"""
multimodal_masked_loss.py
=========================

5-term loss for the multimodal-masked genome model
-------------------------------------------------------

Loss components
---------------
* DNA language modelling                    : L_dna           (Cross-entropy)
* Row-total magnitude (counts)              : L_mag_row       (Poisson NLL)
* Column-total magnitude                    : L_mag_col       (Poisson NLL with library size factor weighted aggregation)
* Distribution of row total to elements     : L_dist_row      (Library size weighted Cross-entropy)
* Distribution of column total to elements  : L_dist_col      (Cross-entropy with library size factor weighted aggregation)
* Track library size factor                 : coeff_track_library_size     (Library size factor)

Total loss
----------
    L = w_dna * L_dna
      + w_mag_row  * L_mag_row
      + w_mag_col  * L_mag_col
      + w_dist_row * L_dist_row
      + w_dist_col * L_dist_col

Default weights - DNA = 1, all others = 0.5.
Default library size factor is all ones

PCGrad
------
If `pcgrad=True`, the object exposes two gradient groups
    * magnitude_group   = [w_mag_row*L_mag_row, w_mag_col*L_mag_col]
    * distribution_group= [w_dist_row*L_dist_row, w_dist_col*L_dist_col]
and a helper :meth:`pcgrad_groups` that returns only *active* losses
(weights > 0) for use in an external PCGrad wrapper.

Author: Doruk Cakmakci & ChatGPT
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedDNALMLoss(nn.Module):
    # --------------------------------------------------------------------- #
    # constructor
    # --------------------------------------------------------------------- #
    def __init__(
        self,
        w_dna: float = 1.0,
        ignore_index=-100,
    ):
        """
        Parameters
        ----------
        w_*      : loss weights (set 0 to turn a term off)
        eps      : numerical stability constant
        pcgrad   : if True expose PCGrad groups via `pcgrad_groups`
        normalize_dual_loss: if True, normalize dual loss contributions
        ignore_index: value to fill unmasked positions of dna labels
        """
        super().__init__()

        self.ignore_index = ignore_index

        # store as buffers so they follow .to(device)
        self.register_buffer("w_dna", torch.tensor(float(w_dna)))


    # ------------------------------ API ----------------------------------- #
    def preprocess_data(self, target, mask):
        """
        Preprocess the labels for loss computation
        target: torch.Tensor
            (B, L, 4+T) observed data
        mask: torch.Tensor
            (B, L, 4+T)  float / {0,1}

        Returns:
            dna_labels: (B, L) - DNA class labels (0-3) with ignore_index for unmasked
            dna_mask: (B, L) - boolean mask indicating which positions have DNA masked
        """
        # prepare DNA labels
        dna_labels = torch.argmax(target[:, :, :4], dim=2)  # (B, L)
        dna_mask = mask[..., :4].any(-1)  # (B, L) - True where any DNA channel is masked
        # fill unmasked positions with ignore_index
        dna_labels = dna_labels.masked_fill(~dna_mask, self.ignore_index)

        return dna_labels, dna_mask

    def forward(
        self,
        pred: torch.Tensor,          # (B, L, 4) - DNA only predictions
        target: torch.Tensor,        # (B, L, T+4) - full targets
        mask: torch.Tensor,          # (B, L, T+4) - full mask
        return_undetached: bool = False
    ):
        """
        Parameters
        ----------
        pred: torch.Tensor
            (B, L, 4) DNA logits only
        target: torch.Tensor
            (B, L, T+4) observed data (full size for compatibility)
        mask: torch.Tensor
            (B, L, T+4) float / {0,1} (full size for compatibility)
        """
        # constants
        B, L, Tdna = pred.shape
        assert Tdna == 4, f"Expected DNA predictions to have 4 channels, got {Tdna}"

        # coefficients
        w_dna = self.w_dna.item()

        # preprocess observed data for loss computation
        dna_labels, dna_mask = self.preprocess_data(target, mask)

        # ------------------------------------------------------------------ #
        # DNA loss
        # ------------------------------------------------------------------ #
        if w_dna > 0 and dna_mask.any():
            # Get the logits for the DNA positions
            dna_logits = pred  # (B, L, 4) - already DNA only
            # Flatten the logits and true labels
            dna_logits = dna_logits.reshape(-1, 4)  # (N, 4)
            dna_true = dna_labels.reshape(-1)  # (N,)

            # Check for non-finite logits
            if torch.isfinite(dna_logits).all().item() is False:
                print("⚠️ Found non-finite logits!")
                print(f"  - NaN count: {torch.isnan(dna_logits).sum().item()}")
                print(f"  - Inf count: {torch.isinf(dna_logits).sum().item()}")
                print(f"  - Min/Max finite values: {dna_logits[torch.isfinite(dna_logits)].min().item():.4f} / {dna_logits[torch.isfinite(dna_logits)].max().item():.4f}")

            # Compute the cross-entropy loss
            L_dna = F.cross_entropy(dna_logits, dna_true, reduction="mean",
                ignore_index=self.ignore_index)

            # Check if L_dna is NaN - this is the smoking gun
            if torch.isnan(L_dna):
                print("🔥 FOUND IT! L_dna is NaN")
                print(f"  - Logits shape: {dna_logits.shape}")
                print(f"  - Logits finite: {torch.isfinite(dna_logits).all().item()}")
                print(f"  - Logits min/max: {dna_logits.min().item():.4f} / {dna_logits.max().item():.4f}")
                print(f"  - Labels unique: {torch.unique(dna_true)}")
                print(f"  - Valid labels count: {(dna_true != self.ignore_index).sum().item()}")
                print(f"  - Total labels: {len(dna_true)}")
                # Set to zero to prevent crash
                L_dna = torch.zeros((), device=pred.device)
        else:
            L_dna = torch.zeros((), device=pred.device)

        # ------------------------------------------------------------------ #
        # Total loss (DNA only)
        # ------------------------------------------------------------------ #

        # Final loss
        loss = w_dna * L_dna

        if return_undetached:
            return loss, {
                "loss": loss,  # Keep gradients
                "L_dna": L_dna,
            }
        else:
            return loss, {
                "loss": loss.detach(),
                "L_dna": L_dna.detach(),
            }
