from __future__ import annotations

from typing import Tuple

import numpy as np

from .base import FidelityMetricBase, collapse_single_track


class CEMetric(FidelityMetricBase):
    """
    Cross-entropy metric between edited log-probs and GT histogram.

    Mirrors CountScaledCE in
    `utils/dna_aware_editor/losses/fidelity_losses.py` but without count weighting.
    """

    def compute_values(
        self,
        *,
        pred_logprobs: np.ndarray,  # (B, L)
        true_tracks: np.ndarray,    # (B, L)
    ) -> Tuple[np.ndarray, np.ndarray]:
        pred_logprobs = collapse_single_track(pred_logprobs, "pred_logprobs")
        true_tracks = collapse_single_track(true_tracks, "true_tracks")
        if pred_logprobs.shape != true_tracks.shape:
            raise ValueError(
                f"pred_logprobs and true_tracks must have same shape, got "
                f"{tuple(pred_logprobs.shape)} vs {tuple(true_tracks.shape)}"
            )

        # Clamp counts and compute totals per example (B,).
        counts_clamped = np.clip(true_tracks, a_min=0.0, a_max=None)
        N = counts_clamped.sum(axis=-1)                                 # (B,)

        # Ground-truth probabilities per example.
        denom = np.clip(N, a_min=self.eps, a_max=None)[..., None]       # (B, 1)
        p_gt = counts_clamped / denom                                   # (B, L)

        # Cross-entropy per example (B,).
        ce = -(p_gt * pred_logprobs).sum(axis=-1)
        return ce, N


__all__ = ["CEMetric"]
