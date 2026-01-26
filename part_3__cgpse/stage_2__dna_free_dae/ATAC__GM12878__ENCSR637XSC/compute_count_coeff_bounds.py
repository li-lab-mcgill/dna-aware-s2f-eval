#!/usr/bin/env python3
"""
Compute total observed counts for peaks and nonpeaks in the central 1kb window.

Outputs
-------
- PNG histograms for raw counts and log1p counts (peaks vs nonpeaks).
- JSON metadata with summary stats and configuration.
"""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import zarr

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Extend sys.path (do this before importing config).
sys.path.append(
    os.path.join(
        os.path.dirname(__file__),
        "../../../../../..",
    )
)
from config import *  # noqa


def _infer_predicted_track_name(script_dir: Path, cwd: Path) -> str:
    for probe in (script_dir.name, cwd.name):
        if probe == "alphagenome__predicted":
            return "alphagenome_predicted_track__summitcentered1000bp_symmetric500bpshift"
        if probe == "bpnet__predicted":
            return "bpnet_predicted_track__summitcentered1000bp_symmetric500bpshift"
    return "unknown"


def _navigate_to_dataset(store: zarr.Group, path_list: Iterable[str]) -> zarr.Array:
    current = store
    for key in path_list:
        if key not in current:
            raise KeyError(f"Key '{key}' not found in zarr group. Available keys: {list(current.keys())}")
        current = current[key]
    return current


def _load_split_indices(zarr_path: str, *, fold_name: str, split_name: str) -> np.ndarray:
    split_path = osp.join(zarr_path, "_auxiliary", "split_indices.npz")
    split_key = f"{fold_name}__{split_name}"
    splits = np.load(split_path)
    if split_key in splits:
        return splits[split_key]
    if split_name in splits:
        return splits[split_name]
    available = list(splits.keys())
    raise KeyError(f"Neither {split_key} nor {split_name} found in {split_path}. Available keys: {available}")


def _central_window_bounds(length: int, window_bp: int) -> tuple[int, int]:
    if window_bp <= 0:
        raise ValueError("window_bp must be > 0")
    start = (length // 2) - (window_bp // 2)
    end = start + window_bp
    if start < 0 or end > length:
        raise ValueError(
            f"window_bp={window_bp} does not fit within length={length} (start={start}, end={end})"
        )
    return start, end


def _sum_counts_for_indices(
    store: zarr.Group,
    dataset_paths: List[List[str]],
    indices: np.ndarray,
    window_bp: int,
    *,
    batch_size: int,
) -> np.ndarray:
    if not dataset_paths:
        raise ValueError("dataset_paths is empty; expected at least one experimental track dataset path.")

    indices = np.asarray(indices, dtype=np.int64)
    counts = np.zeros(indices.shape[0], dtype=np.float64)

    for path_list in dataset_paths:
        arr = _navigate_to_dataset(store, path_list)
        if arr.ndim < 2:
            raise ValueError(f"Expected track array with >=2 dims, got shape {arr.shape}")
        length = int(arr.shape[1])
        start, end = _central_window_bounds(length, window_bp)

        for offset in range(0, indices.shape[0], batch_size):
            batch_idx = indices[offset:offset + batch_size]
            if arr.ndim == 2:
                window = arr.oindex[batch_idx, start:end]
                batch_sum = window.sum(axis=1)
            else:
                window = arr.oindex[batch_idx, start:end, :]
                batch_sum = window.sum(axis=(1, 2))
            counts[offset:offset + batch_idx.shape[0]] += batch_sum

    return counts


def _summary_stats(values: np.ndarray) -> Dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {
            "count": 0,
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "std": 0.0,
            "p10": 0.0,
            "p25": 0.0,
            "p50": 0.0,
            "p75": 0.0,
            "p90": 0.0,
        }
    quantiles = np.percentile(values, [10, 25, 50, 75, 90])
    return {
        "count": int(values.size),
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "p10": float(quantiles[0]),
        "p25": float(quantiles[1]),
        "p50": float(quantiles[2]),
        "p75": float(quantiles[3]),
        "p90": float(quantiles[4]),
    }


def _plot_histogram(
    peaks: np.ndarray,
    nonpeaks: np.ndarray,
    *,
    title: str,
    xlabel: str,
    bins: int,
    output_path: Path,
) -> None:
    peaks = np.asarray(peaks, dtype=np.float64)
    nonpeaks = np.asarray(nonpeaks, dtype=np.float64)
    if peaks.size == 0 and nonpeaks.size == 0:
        raise ValueError("Cannot plot histogram: both peaks and nonpeaks arrays are empty.")
    peaks_min = float(peaks.min()) if peaks.size > 0 else 0.0
    nonpeaks_min = float(nonpeaks.min()) if nonpeaks.size > 0 else 0.0
    peaks_max = float(peaks.max()) if peaks.size > 0 else 0.0
    nonpeaks_max = float(nonpeaks.max()) if nonpeaks.size > 0 else 0.0
    combined_min = min(peaks_min, nonpeaks_min)
    combined_max = max(peaks_max, nonpeaks_max)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(
        peaks,
        bins=bins,
        range=(combined_min, combined_max),
        alpha=0.6,
        color="#1f77b4",
        label="peaks",
    )
    ax.hist(
        nonpeaks,
        bins=bins,
        range=(combined_min, combined_max),
        alpha=0.6,
        color="#ff7f0e",
        label="nonpeaks",
    )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-name", type=str, default="training")
    parser.add_argument("--fold-name", type=str, default="fold_0")
    parser.add_argument("--track-window-bp", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--bins", type=int, default=100)
    parser.add_argument("--metadata-dir", type=str, default="metadata")
    parser.add_argument(
        "--predicted-track-name",
        type=str,
        default=None,
        help="Optional label for the predicted track source (e.g., alphagenome or bpnet).",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    predicted_track_name = (
        args.predicted_track_name
        if args.predicted_track_name
        else _infer_predicted_track_name(script_dir, Path.cwd())
    )

    # Data configuration (mirrors run_train.py)
    peaks_zarr_path = osp.join(
        MANUSCRIPT_EXPERIMENTS_DIR,
        "section_1__ConditionalLM_as_diagnostic_probe",
        "workspace",
        "data",
        "ATAC",
        "GM12878__ENCSR637XSC",
        "preprocessed",
        "zarr_datasets",
        "peaks.all_folds.zarr",
    )
    nonpeaks_zarr_path = osp.join(
        MANUSCRIPT_EXPERIMENTS_DIR,
        "section_1__ConditionalLM_as_diagnostic_probe",
        "workspace",
        "data",
        "ATAC",
        "GM12878__ENCSR637XSC",
        "preprocessed",
        "zarr_datasets",
        f"nonpeaks.{args.fold_name}.zarr",
    )

    experimental_peaks_track_dataset_paths = [
        [
            "observed_tracks",
            "observed_track_summitcentered1000bp_symmetric500bpshift",
        ]
    ]
    experimental_nonpeaks_track_dataset_paths = [
        [
            "observed_tracks",
            "observed_track_summitcentered1000bp_symmetric500bpshift",
        ]
    ]

    peaks_store = zarr.open(peaks_zarr_path, mode="r")
    nonpeaks_store = zarr.open(nonpeaks_zarr_path, mode="r")

    peak_indices = _load_split_indices(
        peaks_zarr_path, fold_name=args.fold_name, split_name=args.split_name
    )
    nonpeak_indices = _load_split_indices(
        nonpeaks_zarr_path, fold_name=args.fold_name, split_name=args.split_name
    )

    print(f"Peaks: {len(peak_indices):,} indices ({args.split_name})")
    print(f"Nonpeaks: {len(nonpeak_indices):,} indices ({args.split_name})")

    peak_counts = _sum_counts_for_indices(
        peaks_store,
        experimental_peaks_track_dataset_paths,
        peak_indices,
        args.track_window_bp,
        batch_size=args.batch_size,
    )
    nonpeak_counts = _sum_counts_for_indices(
        nonpeaks_store,
        experimental_nonpeaks_track_dataset_paths,
        nonpeak_indices,
        args.track_window_bp,
        batch_size=args.batch_size,
    )

    peak_log1p = np.log1p(peak_counts)
    nonpeak_log1p = np.log1p(nonpeak_counts)

    metadata_dir = (script_dir / args.metadata_dir).resolve()
    metadata_dir.mkdir(parents=True, exist_ok=True)

    raw_hist_path = metadata_dir / f"total_counts_histogram__{args.split_name}.png"
    log_hist_path = metadata_dir / f"log1p_total_counts_histogram__{args.split_name}.png"

    _plot_histogram(
        peak_counts,
        nonpeak_counts,
        title="Total counts (central 1kb)",
        xlabel="Total counts",
        bins=args.bins,
        output_path=raw_hist_path,
    )
    _plot_histogram(
        peak_log1p,
        nonpeak_log1p,
        title="Log1p total counts (central 1kb)",
        xlabel="Log1p total counts",
        bins=args.bins,
        output_path=log_hist_path,
    )

    payload = {
        "created_at_unix": time.time(),
        "split_name": args.split_name,
        "fold_name": args.fold_name,
        "track_window_bp": args.track_window_bp,
        "predicted_track_name": predicted_track_name,
        "peaks_zarr_path": peaks_zarr_path,
        "nonpeaks_zarr_path": nonpeaks_zarr_path,
        "experimental_peaks_track_dataset_paths": experimental_peaks_track_dataset_paths,
        "experimental_nonpeaks_track_dataset_paths": experimental_nonpeaks_track_dataset_paths,
        "counts": {
            "peaks": _summary_stats(peak_counts),
            "nonpeaks": _summary_stats(nonpeak_counts),
            "peaks_log1p": _summary_stats(peak_log1p),
            "nonpeaks_log1p": _summary_stats(nonpeak_log1p),
        },
        "plots": {
            "total_counts_histogram_png": str(raw_hist_path),
            "log1p_total_counts_histogram_png": str(log_hist_path),
        },
    }

    json_path = metadata_dir / f"count_coeff_bounds__{args.split_name}.json"
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"Wrote metadata: {json_path}")
    print(f"Wrote plots: {raw_hist_path}, {log_hist_path}")


if __name__ == "__main__":
    main()
