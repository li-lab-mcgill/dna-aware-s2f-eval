from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch

from ..masking.mask_sampler import MaskStrategySampler


class BatchPreparer:
    """
    Minimal device-aware batch preparer.

    Responsibilities:
      - Move tensors to device
      - Sample (critic_input_mask, critic_output_mask) per item, supporting both
        single-mask and nested-mask samplers.
    """

    def __init__(
        self,
        *,
        mask_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.mask_sampler = MaskStrategySampler(mask_config) if mask_config else None

    def __call__(
        self,
        raw_batch: Tuple[Any, ...],
        *,
        device: torch.device,
        mask_sampler: Optional[MaskStrategySampler] = None,
        mask_dtype: torch.dtype = torch.float32,
    ) -> Dict[str, Any]:
        """
        raw_batch is the default-collated batch from the dataset:
          - without controls: (dna, pred_tracks, exp_tracks, region_types, identifiers)
          - with controls:    (dna, pred_tracks, exp_tracks, control_tracks, region_types, identifiers)

        Notes
        -----
        - This module does *not* crop DNA to match track length. If DNA has a
          longer context window than the tracks, that is passed through as-is.
          Any cropping to align lengths is handled downstream (e.g., orchestrators).
        - Datasets may emit empty-channel tensors `(B, L, 0)` for missing tracks
          to support default PyTorch collation. Here we convert those to `None`
          so downstream modules can branch explicitly.
        - Mask sampling is optional: if no sampler is provided/configured, both
          `critic_input_mask` and `critic_output_mask` are returned as `None`.
        """
        if len(raw_batch) == 5:
            dna, pred_tracks, exp_tracks, region_types, identifiers = raw_batch
            control_tracks = None
        elif len(raw_batch) == 6:
            dna, pred_tracks, exp_tracks, control_tracks, region_types, identifiers = raw_batch
        else:
            raise ValueError(
                "raw_batch must have 5 or 6 elements: "
                "(dna, pred_tracks, exp_tracks[, control_tracks], region_types, identifiers)"
            )

        dna = dna.to(device=device, dtype=torch.float32)
        pred_tracks = pred_tracks.to(device=device, dtype=torch.float32)
        exp_tracks = exp_tracks.to(device=device, dtype=torch.float32)
        if control_tracks is not None:
            control_tracks = control_tracks.to(device=device, dtype=torch.float32)

        B = int(dna.shape[0])

        pred_present = pred_tracks.size(-1) > 0
        exp_present = exp_tracks.size(-1) > 0
        if not (pred_present or exp_present):
            raise ValueError("Both pred_tracks and exp_tracks are empty (T=0); need at least one present.")

        L_track = int(pred_tracks.shape[1] if pred_present else exp_tracks.shape[1])
        T = int(pred_tracks.shape[2] if pred_present else exp_tracks.shape[2])

        # Set missing tracks to None (dataset uses empty-channel tensors for collation).
        pred_out = pred_tracks if pred_present else None
        exp_out = exp_tracks if exp_present else None
        if control_tracks is None:
            control_present = False
            control_out = None
        else:
            control_present = control_tracks.size(-1) > 0
            control_out = control_tracks if control_present else None

        sampler = mask_sampler if mask_sampler is not None else self.mask_sampler
        critic_input_mask = None
        critic_output_mask = None
        mask = None
        if sampler is not None:
            in_masks = []
            out_masks = []
            for _ in range(B):
                # MaskStrategySampler returns a mask *strategy* instance.
                strategy = sampler()
                sample = strategy.sample_mask((L_track, 4 + T))

                # Single-mask sampler: input == output.
                if not isinstance(sample, (tuple, list)):
                    m_in = torch.as_tensor(sample, device=device, dtype=mask_dtype)
                    m_out = m_in
                else:
                    # Nested-mask sampler: first is a superset of second.
                    m_in, m_out = sample
                    m_in = torch.as_tensor(m_in, device=device, dtype=mask_dtype)
                    m_out = torch.as_tensor(m_out, device=device, dtype=mask_dtype)
                    if not torch.all(m_out <= m_in):
                        raise ValueError("Expected nested masks with critic_output_mask <= critic_input_mask.")

                in_masks.append(m_in)
                out_masks.append(m_out)

            critic_input_mask = torch.stack(in_masks, dim=0)
            critic_output_mask = torch.stack(out_masks, dim=0)
            
            # HACKY SOLUTION: 
            mask = critic_output_mask

        return {
            "dna": dna,
            "pred_tracks": pred_out,
            "exp_tracks": exp_out,
            "control_tracks": control_out,
            "mask": mask,
            "region_types": region_types,
            "identifiers": identifiers,
            "pred_present": torch.tensor(bool(pred_present), device=device),
            "exp_present": torch.tensor(bool(exp_present), device=device),
            "control_present": torch.tensor(bool(control_present), device=device),
        }


__all__ = ["BatchPreparer"]
