from __future__ import annotations

import numpy as np

from .base import CriticMetricBase


class MaskedCEMetric(CriticMetricBase):
    """
    Masked cross-entropy: -sum_c P * log Q.
    """

    def compute_values(self, *, p_probs: np.ndarray, q_probs: np.ndarray) -> np.ndarray:
        p = np.asarray(p_probs)
        q = np.asarray(q_probs)

        q = np.clip(q, self.eps, 1.0)
        return -(p * np.log(q)).sum(axis=-1)


__all__ = ["MaskedCEMetric"]
