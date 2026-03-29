
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


def collapse_single_track(arr: np.ndarray, name: str) -> np.ndarray:
    """
    Accept (B, L), (B, 1, L), or (B, L, 1). If 3D, require T=1 and squeeze.
    """
    if arr.ndim == 2:
        return arr
    if arr.ndim != 3:
        raise ValueError(f"{name} must be 2D or 3D, got {tuple(arr.shape)}")
    if arr.shape[1] == 1:
        return arr[:, 0, :]
    if arr.shape[2] == 1:
        return arr[:, :, 0]
    raise ValueError(f"{name} must have T=1 in 3D input, got {tuple(arr.shape)}")


class FidelityMetricBase:
    """
    Base class for fidelity metrics.

    Design notes
    ------------
    - Metric values are computed per-example (or per-example-per-track).
    - We do NOT apply count-based weighting in the metric value itself.
    - We return parallel lists for raw metric values and total counts.
    - Summary statistics are computed on metric values only (count-agnostic).
    """

    def __init__(self, *, eps: float = 1e-8) -> None:
        self.eps = float(eps)

    def compute_values(self, **kwargs: Any) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return (values, counts_total) with matching shapes (no reduction).
        Subclasses must implement this.
        """
        raise NotImplementedError

    def evaluate(
        self,
        *,
        return_stats: bool = True,
        return_values: bool = True,
        return_counts_total: bool = True,
        prefix: str = "",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        values, counts_total = self.compute_values(**kwargs)
        if values.shape != counts_total.shape:
            raise ValueError(
                f"values shape {tuple(values.shape)} != counts_total shape {tuple(counts_total.shape)}"
            )

        values_flat = np.asarray(values).reshape(-1)
        counts_flat = np.asarray(counts_total).reshape(-1)

        values_np = values_flat
        counts_np = np.clip(counts_flat, a_min=0.0, a_max=None)

        # Filter non-finite values; track how many were dropped.
        finite_mask = np.isfinite(values_np)
        values_clean = values_np[finite_mask]
        counts_clean = counts_np[finite_mask]
        nonfinite_count = int((~finite_mask).sum())

        key_prefix = str(prefix) if prefix else ""
        out: Dict[str, Any] = {}
        if return_stats:
            out[f"{key_prefix}stats"] = compute_summary_statistics(values_clean)
            out[f"{key_prefix}nonfinite_count"] = nonfinite_count
            out[f"{key_prefix}total_count"] = int(values_np.shape[0])
        if return_values:
            out[f"{key_prefix}values"] = values_clean.tolist()
        if return_counts_total:
            out[f"{key_prefix}counts_total"] = counts_clean.tolist()

        return out


__all__ = [
    "compute_summary_statistics",
    "collapse_single_track",
    "FidelityMetricBase",
]
