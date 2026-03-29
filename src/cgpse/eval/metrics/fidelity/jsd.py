from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.spatial.distance import jensenshannon

from .base import FidelityMetricBase, collapse_single_track


class JSDMetric(FidelityMetricBase):
    """
    Jensen-Shannon divergence between two profiles.

    Mirrors the JSD evaluation logic used in legacy scripts, but computes
    per-example values and returns summary stats via FidelityMetricBase.
    """

    def compute_values(
        self,
        *,
        pred_logprobs: np.ndarray,  # (B, L)
        true_tracks: np.ndarray, # (B, L) counts (also the observed profile)
    ) -> Tuple[np.ndarray, np.ndarray]:
        pred_logprobs = collapse_single_track(pred_logprobs, "pred_logprobs")
        true_tracks = collapse_single_track(true_tracks, "true_tracks")
        if pred_logprobs.shape != true_tracks.shape:
            raise ValueError(
                f"pred_logprobs and true_tracks must have same shape, got "
                f"{tuple(pred_logprobs.shape)} vs {tuple(true_tracks.shape)}"
            )

        obs = np.clip(np.asarray(true_tracks), a_min=0.0, a_max=None)
        pred_probs = np.exp(np.asarray(pred_logprobs))

        if obs.ndim != 2:
            raise ValueError(f"Expected (B,L) for true_tracks, got {tuple(obs.shape)}")

        # Normalize to probabilities per example.
        obs_probs = obs / np.clip(obs.sum(axis=-1, keepdims=True), a_min=self.eps, a_max=None)
        pred_probs = pred_probs / np.clip(pred_probs.sum(axis=-1, keepdims=True), a_min=self.eps, a_max=None)

        B = obs_probs.shape[0]
        jsd_vals = np.zeros((B,), dtype=obs_probs.dtype)
        for b in range(B):
            # JSD per example using SciPy's reference implementation.
            jsd_vals[b] = float(jensenshannon(obs_probs[b], pred_probs[b]))

        # Total counts per example for the parallel log1p list.
        N = obs.sum(axis=-1)
        return jsd_vals, N


__all__ = ["JSDMetric"]
