
import numpy as np

from typing import Any, Dict, Tuple

# ============================================================================
# Helpers 
# ============================================================================

def compute_summary_statistics(values: np.ndarray) -> dict:
    """
    Compute summary statistics for boxplot-style visualization.
    Returns dict with keys: min, q1, median, q3, max, mean, std, count.
    """
    arr = np.asarray(values)
    # Drop non-finite values before summary.
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            "min": float("nan"),
            "q1": float("nan"),
            "median": float("nan"),
            "q3": float("nan"),
            "max": float("nan"),
            "mean": float("nan"),
            "std": float("nan"),
            "count": 0,
        }
    return {
        "min": float(np.min(arr)),
        "q1": float(np.percentile(arr, 25.0)),
        "median": float(np.percentile(arr, 50.0)),
        "q3": float(np.percentile(arr, 75.0)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "count": int(arr.shape[0]),
    }


class CriticMetricBase:
    """
    Base class for critic metrics.

    Contract
    --------
    - Inputs: P_probs and Q_probs are (B, L, 4), mask is (B, L).
    - Subclass computes values in (B, L) before masking.
    - Masked values are flattened to K and summarized.
    """

    def __init__(self, *, eps: float = 1e-8) -> None:
        self.eps = float(eps)

    def compute_values(self, **kwargs: Any) -> np.ndarray:
        """
        Return values with shape (B, L). Subclasses must implement this.
        """
        raise NotImplementedError

    def evaluate(
        self,
        *,
        p_probs: np.ndarray,
        q_probs: np.ndarray,
        mask: np.ndarray,
        return_stats: bool = True,
        return_values: bool = True,
        return_mean: bool = False,
        prefix: str = "",
    ) -> Dict[str, Any]:
        p_probs = np.asarray(p_probs)
        q_probs = np.asarray(q_probs)
        mask = np.asarray(mask)

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

        values = self.compute_values(p_probs=p_probs, q_probs=q_probs)
        if values.shape != mask.shape:
            raise ValueError(
                f"values shape {tuple(values.shape)} != mask shape {tuple(mask.shape)}"
            )

        masked_values = values[mask.astype(bool)]

        key_prefix = str(prefix) if prefix else ""
        out: Dict[str, Any] = {}
        if return_stats:
            out[f"{key_prefix}stats"] = compute_summary_statistics(masked_values)
        if return_values:
            out[f"{key_prefix}values"] = masked_values.tolist()
        if return_mean:
            if masked_values.size == 0:
                out[f"{key_prefix}mean"] = float("nan")
            else:
                out[f"{key_prefix}mean"] = float(np.mean(masked_values))

        return out


__all__ = [
    "compute_summary_statistics",
    "CriticMetricBase",
]
