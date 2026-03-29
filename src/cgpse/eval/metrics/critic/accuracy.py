from __future__ import annotations

from typing import Any, Dict

import numpy as np


class MaskedAccuracyMetric:
    """
    Masked argmax accuracy between P and Q.
    """

    def evaluate(
        self,
        *,
        p_probs: np.ndarray,
        q_probs: np.ndarray,
        mask: np.ndarray,
        prefix: str = "",
    ) -> Dict[str, Any]:
        p_probs = np.asarray(p_probs)
        q_probs = np.asarray(q_probs)
        mask = np.asarray(mask).astype(bool)

        if p_probs.shape != q_probs.shape:
            raise ValueError(
                f"p_probs shape {tuple(p_probs.shape)} != q_probs shape {tuple(q_probs.shape)}"
            )
        if p_probs.ndim != 3 or p_probs.shape[2] != 4:
            raise ValueError(f"p_probs must be (B, L, 4), got {tuple(p_probs.shape)}")
        if mask.shape != p_probs.shape[:2]:
            raise ValueError(
                f"mask shape {tuple(mask.shape)} != (B, L) {p_probs.shape[:2]}"
            )

        p_argmax = np.argmax(p_probs, axis=-1)
        q_argmax = np.argmax(q_probs, axis=-1)
        match_mask = (p_argmax == q_argmax)

        masked_matches = match_mask[mask]
        if masked_matches.size == 0:
            acc = float("nan")
        else:
            acc = float(np.mean(masked_matches))

        key_prefix = str(prefix) if prefix else ""
        return {
            f"{key_prefix}accuracy": acc,
            f"{key_prefix}match_mask": match_mask,
        }


__all__ = ["MaskedAccuracyMetric"]
