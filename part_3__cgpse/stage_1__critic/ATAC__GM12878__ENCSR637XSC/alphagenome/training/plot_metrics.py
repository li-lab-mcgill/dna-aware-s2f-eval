#!/usr/bin/env python3
"""
Plotting script for validation metrics with multiple mask configurations.

Supports:
  - Critic pretraining metrics (GT/S2F/DISC heads).
  - DNA-conditional DAE metrics (loss_total, fid/total, critic/total) with
    optional S2F regimes (s2f_invco, s2f_no_dna).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# Ensure PDF text is editable in Illustrator
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42


# ===== HARDCODED METRICS PATH =====
DEFAULT_METRICS_PATH = "/home/mcb/users/dcakma3/multi_res_bench_v2/multimodal_masked_contrastive_main/manuscript_experiments/section_3_v2__Debiasing/workspace/runs/5__dna_conditional_dae/ATAC/GM12878__ENCSR637XSC/dna_conditional_dae/alphagenome__predicted/1000bp__mask_rco_10to60pct__rci_max_80pct/bottleneckC32/fidelity__ce+align/seed_42/lightning_logs/version_3/metrics.csv"


def _extract_mask_bins(df: pd.DataFrame) -> List[str]:
    mask_bins = set()
    for col in df.columns:
        if not col.startswith("val_"):
            continue
        rest = col[len("val_"):]
        mask = rest.split("/")[0]
        if mask:
            mask_bins.add(mask)

    def _mask_sort_key(mask: str) -> tuple[int, str]:
        match = re.search(r"(\\d+)", mask)
        if match:
            return (int(match.group(1)), mask)
        return (10**9, mask)

    return sorted(mask_bins, key=_mask_sort_key)


def _split_val_column(col: str) -> tuple[str, str, str]:
    rest = col[len("val_"):]
    parts = rest.split("/")
    mask = parts[0]
    if len(parts) >= 3 and parts[1] in {"s2f_invco", "s2f_no_dna"}:
        regime = parts[1]
        metric = "/".join(parts[2:])
    else:
        regime = "default"
        metric = "/".join(parts[1:])
    return mask, regime, metric


def sanitize_filename(name: str) -> str:
    """Convert metric group name to safe filename."""
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", name)


def build_validation_metric_groups(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Build validation metric groups organized by mask bins and metric types.

    Returns dict mapping group_name -> list of column names
    """
    groups: Dict[str, List[str]] = {}

    mask_bins = _extract_mask_bins(df)
    if not mask_bins:
        return groups

    metric_paths = ["loss_total", "fid/total", "critic/total"]
    regimes = ["default", "s2f_invco", "s2f_no_dna"]

    mask_order = {name: idx for idx, name in enumerate(mask_bins)}
    for metric_path in metric_paths:
        for regime in regimes:
            cols = []
            for mask_bin in mask_bins:
                if regime == "default":
                    col = f"val_{mask_bin}/{metric_path}"
                else:
                    col = f"val_{mask_bin}/{regime}/{metric_path}"
                if col in df.columns:
                    cols.append(col)
            if cols:
                cols_sorted = sorted(
                    cols,
                    key=lambda c: mask_order.get(_split_val_column(c)[0], 10**9),
                )
                group_name = f"val__{metric_path.replace('/', '_')}__{regime}"
                groups[group_name] = cols_sorted

    return groups


def build_training_metric_groups(df: pd.DataFrame) -> Dict[str, Tuple[List[str], str]]:
    """
    Build coarse training metric groups (step-based).
    """
    groups: Dict[str, Tuple[List[str], str]] = {}
    if "step" not in df.columns:
        return groups

    train_cols = {
        "train_totals_step": [
            "train/loss_total_step",
            "train/fid/total_step",
            "train/critic/total_step",
        ],
    }

    for group_name, cols in train_cols.items():
        found = [c for c in cols if c in df.columns]
        if found:
            groups[group_name] = (found, "step")

    return groups


def plot_group(
    df: pd.DataFrame,
    cols: List[str],
    title: str,
    out_file: Path,
    group_name: str | None = None,
    *,
    x_axis: str = "epoch",
    label_mode: str = "auto",
) -> None:
    """Plot a metric group across mask bins."""
    if not cols:
        return

    max_epoch = 300
    fig, ax = plt.subplots(figsize=(9, 4.5))
    plotted_any = False

    # Color map for mask bins (default)
    color_map = {
        "10to20": "C0",
        "20to40": "C1",
        "40to60": "C2",
        "60to80": "C3",
        "80to100": "C4",
        "100": "C5",
    }

    # Line style map for modalities (default)
    line_style_map = {
        "GT": "-",
        "S2F": "--",
    }

    # Special styling only for the combined GT+S2F validation plot
    is_combined_gt_s2f = group_name == "val_GT_and_S2F_all_masks"

    # Ordered mask bins (used for legend ordering)
    ordered_mask_bins = ["10to20", "20to40", "40to60", "60to80", "80to100", "100"]

    # Color maps for combined plot: GT (cool teal/blue-gray) and S2F (red)
    # Previous GT palette (browns) kept for reference:
    # gt_color_map = {
    #     "10to20": "#e3c6a8",
    #     "20to40": "#d3ac85",
    #     "40to60": "#c39262",
    #     "60to80": "#b3784a",
    #     "80to100": "#9f5f34",
    #     "100": "#8a461f",
    # }
    gt_color_map = {
        "10to20": "#c7d9df",
        "20to40": "#a9c2cc",
        "40to60": "#8cabba",
        "60to80": "#6f94a8",
        "80to100": "#527e96",
        "100": "#3b667c",
    }
    # Previous S2F palette (lilac) kept for reference:
    # s2f_color_map = {
    #     "10to20": "#e1d3f2",
    #     "20to40": "#ccb4ea",
    #     "40to60": "#b795e2",
    #     "60to80": "#a276da",
    #     "80to100": "#8b57d2",
    #     "100": "#7537ca",
    # }
    # AlphaGenome base tone from plot_script_v1.py: red (#e74c3c)
    s2f_color_map = {
        "10to20": "#f59a91",
        "20to40": "#f07266",
        "40to60": "#eb4d3f",
        "60to80": "#e53935",
        "80to100": "#d32f2f",
        "100": "#b71c1c",
    }

    # Keep track of lines for standalone legend in combined plot
    combined_lines = []

    for idx, col in enumerate(cols):
        if col not in df.columns or x_axis not in df.columns:
            continue

        # Drop NaN and ensure numeric
        sub = df[[x_axis, col]].dropna()
        sub = sub[
            pd.to_numeric(sub[x_axis], errors='coerce').notnull() &
            pd.to_numeric(sub[col], errors='coerce').notnull()
        ]
        if x_axis == "epoch":
            sub = sub[sub["epoch"] <= max_epoch]

        if sub.empty:
            continue

        if col.startswith("val_"):
            mask_bin, regime, metric_name = _split_val_column(col)
            if is_combined_gt_s2f:
                label = f"{mask_bin} ({metric_name})"
            elif label_mode == "mask":
                label = mask_bin
            elif label_mode == "metric":
                label = metric_name
            else:
                label = f"{mask_bin} ({regime}/{metric_name})"
        else:
            mask_bin = ""
            regime = "train"
            metric_name = col
            label = col

        # Styling
        if is_combined_gt_s2f and metric_name in ("GT", "S2F"):
            # Color encodes modality + mask bin; all solid lines
            if metric_name == "GT":
                color = gt_color_map.get(mask_bin, "#8a461f")
            else:
                color = s2f_color_map.get(mask_bin, "#e74c3c")
            linestyle = "-"
            linewidth = 2.0
        else:
            if col.startswith("val_"):
                # Default styling: color by mask bin, linestyle by modality
                color = color_map.get(mask_bin, "black")
                linestyle = line_style_map.get(metric_name, "-")
                linewidth = 2.0
            else:
                color = f"C{idx % 10}"
                linestyle = "-"
                linewidth = 2.0

        line, = ax.plot(
            sub[x_axis],
            sub[col],
            label=label,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            alpha=0.8,
        )
        if is_combined_gt_s2f and metric_name in ("GT", "S2F"):
            combined_lines.append((metric_name, mask_bin, line, label))
        plotted_any = True

    if not plotted_any:
        plt.close(fig)
        return

    if is_combined_gt_s2f:
        # No title; custom axis labels
        ax.set_xlabel("Epochs", fontsize=14)
        ax.set_ylabel("Masked Cross Entropy (Validation Set)", fontsize=14)

        # Build standalone legend with two columns:
        # left column: GT (browns), right column: S2F (lilacs)
        gt_entries = [
            (mask_bin, line, label)
            for metric_name, mask_bin, line, label in combined_lines
            if metric_name == "GT"
        ]
        s2f_entries = [
            (mask_bin, line, label)
            for metric_name, mask_bin, line, label in combined_lines
            if metric_name == "S2F"
        ]

        def _sort_key(entry):
            mask_bin, _, _ = entry
            try:
                return ordered_mask_bins.index(mask_bin)
            except ValueError:
                return len(ordered_mask_bins)

        gt_entries.sort(key=_sort_key)
        s2f_entries.sort(key=_sort_key)

        legend_handles = [line for _, line, _ in gt_entries + s2f_entries]
        legend_labels = [label for _, _, label in gt_entries + s2f_entries]

        # Standalone legend figure
        fig_leg, ax_leg = plt.subplots(figsize=(6, 4))
        ax_leg.axis("off")
        ax_leg.legend(
            legend_handles,
            legend_labels,
            loc="center",
            ncol=2,
            frameon=False,
            fontsize=10,
        )
        fig_leg.tight_layout()
        legend_path = out_file.with_name(out_file.stem + "_legend.pdf")
        fig_leg.savefig(legend_path)
        plt.close(fig_leg)
    else:
        # Default styling for all other plots
        ax.set_title(title, fontsize=16, fontweight='bold')
        if x_axis == "step":
            ax.set_xlabel("Step", fontsize=14)
        else:
            ax.set_xlabel("Epoch", fontsize=14)
        ax.set_ylabel("Loss", fontsize=14)
        ax.legend(loc="best", fontsize=10, framealpha=0.9, ncol=2)
    ax.grid(True, alpha=0.3, linestyle='--')
    fig.tight_layout()
    fig.savefig(out_file)
    plt.close(fig)
    print(f"  ✓ {out_file.name}")


def save_summary_table(df: pd.DataFrame, out_dir: Path) -> None:
    """Save summary statistics from the last epoch."""
    if "epoch" not in df.columns:
        return

    df_sorted = df.sort_values(by="epoch")
    last_epoch = int(df_sorted["epoch"].max())
    last_row = df_sorted[df_sorted["epoch"] == last_epoch].tail(1)

    if last_row.empty:
        return

    # Extract validation metrics only
    summary = {}
    for col in df.columns:
        if not col.startswith("val_"):
            continue
        try:
            val = last_row.iloc[0][col]
            if pd.notna(val) and isinstance(val, (int, float)):
                summary[col] = float(val)
        except Exception:
            continue

    if summary:
        summary_df = pd.Series(summary).to_frame(name="value")
        summary_path = out_dir / "validation_summary_last_epoch.csv"
        summary_df.to_csv(summary_path)
        print(f"\n  ✓ Summary: {summary_path.name}")

    # Epoch-compacted validation metrics CSV
    val_cols = [c for c in df.columns if c.startswith("val_")]
    if val_cols and "epoch" in df.columns:
        # Group by epoch and get last row per epoch
        compact_rows = []
        for epoch_num, epoch_group in df_sorted.groupby("epoch", sort=True):
            # Get validation metrics for this epoch
            last_row_epoch = epoch_group.tail(1).iloc[0]
            row_data = {"epoch": epoch_num}
            for val_col in val_cols:
                val_metric = epoch_group[val_col].dropna()
                if not val_metric.empty:
                    row_data[val_col] = val_metric.iloc[-1]
            compact_rows.append(row_data)

        compact = pd.DataFrame(compact_rows)
        compact_path = out_dir / "validation_metrics_epoch_compact.csv"
        compact.to_csv(compact_path, index=False)
        print(f"  ✓ Compacted: {compact_path.name} (validation metrics per epoch)")


def main() -> None:
    """Main plotting pipeline."""
    metrics_path = Path(DEFAULT_METRICS_PATH)

    if not metrics_path.exists():
        print(f"ERROR: Metrics file not found: {metrics_path}")
        return

    print(f"\n{'='*70}")
    print(f"Reading metrics: {metrics_path.name}")
    print(f"{'='*70}\n")

    df = pd.read_csv(metrics_path)
    print(f"  Shape: {df.shape[0]} rows × {df.shape[1]} columns")

    # Count validation columns
    val_cols = [c for c in df.columns if c.startswith("val_")]
    print(f"  Validation metrics: {len(val_cols)} columns")

    # Output directory (auto: sibling /plots dir)
    out_dir = metrics_path.parent / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Output: {out_dir}\n")

    # Build metric groups
    val_groups = build_validation_metric_groups(df)
    train_groups = build_training_metric_groups(df)
    print(f"Found {len(val_groups)} validation metric groups")
    print(f"Found {len(train_groups)} training/misc metric groups\n")

    # Generate plots
    print("Generating plots:")
    for group_name, group_cols in val_groups.items():
        out_file = out_dir / f"{sanitize_filename(group_name)}.png"
        plot_group(
            df,
            group_cols,
            group_name.replace("_", " ").title(),
            out_file,
            group_name=group_name,
            x_axis="epoch",
            label_mode="mask",
        )
    for group_name, (group_cols, axis) in train_groups.items():
        out_file = out_dir / f"{sanitize_filename(group_name)}.png"
        plot_group(
            df,
            group_cols,
            group_name.replace("_", " ").title(),
            out_file,
            group_name=group_name,
            x_axis=axis,
            label_mode="metric",
        )

    # Save summary tables
    print()
    save_summary_table(df, out_dir)

    print(f"\n{'='*70}")
    print(f"Done! All outputs in: {out_dir}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
