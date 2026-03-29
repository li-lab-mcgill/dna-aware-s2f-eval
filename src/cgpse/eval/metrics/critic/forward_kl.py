from __future__ import annotations

import numpy as np

from .base import CriticMetricBase


class MaskedForwardKLMetric(CriticMetricBase):
    """
    Masked KL(P || Q) over nucleotide distributions.
    """

    def compute_values(self, *, p_probs: np.ndarray, q_probs: np.ndarray) -> np.ndarray:
        p = np.asarray(p_probs)
        q = np.asarray(q_probs)

        # Stabilize logs at zero-prob positions.
        p = np.clip(p, self.eps, 1.0)
        q = np.clip(q, self.eps, 1.0)

        log_p = np.log(p)
        log_q = np.log(q)

        # KL per position: sum_c p * (log p - log q)
        return (p * (log_p - log_q)).sum(axis=-1)


__all__ = ["MaskedForwardKLMetric"]
