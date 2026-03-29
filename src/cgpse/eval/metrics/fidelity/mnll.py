from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.special import gammaln

from .base import FidelityMetricBase, collapse_single_track


class MNLLMetric(FidelityMetricBase):
    """
    Multinomial negative log-likelihood metric.

    Mirrors MultinomialNLL in
    `utils/dna_aware_editor/losses/fidelity_losses.py`.
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
        N = counts_clamped.sum(axis=-1)                  # (B,)

        # Multinomial NLL per example (B,).
        log_fact_sum = gammaln(N + 1.0)                              # (B,)
        log_prod_fact = gammaln(counts_clamped + 1.0).sum(axis=-1)   # (B,)
        log_prod_exp = (counts_clamped * pred_logprobs).sum(axis=-1) # (B,)

        mnll = -log_fact_sum + log_prod_fact - log_prod_exp
        return mnll, N


__all__ = ["MNLLMetric"]
