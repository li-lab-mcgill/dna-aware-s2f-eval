from __future__ import annotations

from typing import Any, Dict, Tuple

import torch

class BatchPreparer:
    """
    Minimal device-aware batch preparer.

    Responsibilities:
      - Move tensors to device
      - Use dataset-provided mask for critic_input_mask/critic_output_mask.
    """

    def __init__(self) -> None:
        pass

    def __call__(
        self,
        raw_batch: Tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            list[str],
            list[str],
        ],
        *,
        device: torch.device,
        mask_dtype: torch.dtype = torch.float32,
    ) -> Dict[str, Any]:
        """
        raw_batch is the default-collated batch from the dataset:
          (dna, pred_tracks, exp_tracks, mask, region_types, identifiers)

        Notes
        -----
        - This module does *not* crop DNA to match track length. If DNA has a
          longer context window than the tracks, that is passed through as-is.
          Any cropping to align lengths is handled downstream (e.g., orchestrators).
        - Datasets may emit empty-channel tensors `(B, L, 0)` for missing tracks
          to support default PyTorch collation. Here we convert those to `None`
          so downstream modules can branch explicitly.
        - Masks are dataset-provided. If the mask is empty-channel `(B, L, 0)`,
          both `critic_input_mask` and `critic_output_mask` are `None`.
        """
        dna, pred_tracks, exp_tracks, mask, region_types, identifiers = raw_batch

        dna = dna.to(device=device, dtype=torch.float32)
        pred_tracks = pred_tracks.to(device=device, dtype=torch.float32)
        exp_tracks = exp_tracks.to(device=device, dtype=torch.float32)
        mask = mask.to(device=device, dtype=mask_dtype)

        B = int(dna.shape[0])

        pred_present = pred_tracks.size(-1) > 0
        exp_present = exp_tracks.size(-1) > 0
        if not (pred_present or exp_present):
            raise ValueError("Both pred_tracks and exp_tracks are empty (T=0); need at least one present.")


        # Set missing tracks to None (dataset uses empty-channel tensors for collation).
        pred_out = pred_tracks if pred_present else None
        exp_out = exp_tracks if exp_present else None

        # critic mask handling
        critic_input_mask = None
        critic_output_mask = None
        if mask.size(-1) > 0:
            critic_input_mask = mask
            critic_output_mask = mask

        return {
            "dna": dna,
            "pred_tracks": pred_out,
            "exp_tracks": exp_out,
            "critic_input_mask": critic_input_mask,
            "critic_output_mask": critic_output_mask,
            "region_types": region_types,
            "identifiers": identifiers,
            "pred_present": torch.tensor(bool(pred_present), device=device),
            "exp_present": torch.tensor(bool(exp_present), device=device),
        }


__all__ = ["BatchPreparer"]
