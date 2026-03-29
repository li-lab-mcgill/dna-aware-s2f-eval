from __future__ import annotations

import numpy as np

from .base import CriticMetricBase


class MaskedEntropyMetric(CriticMetricBase):
    """
    Masked entropy: -sum_c P * log P.
    """

    def compute_values(self, *, p_probs: np.ndarray, q_probs: np.ndarray) -> np.ndarray:
        x = np.asarray(p_probs)
        x = np.clip(x, self.eps, 1.0)
        return -(x * np.log(x)).sum(axis=-1)


__all__ = ["MaskedEntropyMetric"]
