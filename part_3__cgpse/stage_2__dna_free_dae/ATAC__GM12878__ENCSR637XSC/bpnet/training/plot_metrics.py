#!/usr/bin/env python3
"""
Plotting script for final DNA-free DAE training metrics with validation mask bins.

Training plots focus on:
- Objective balance (loss_total, fid/critic totals, lambda_t, gradient norms)
- Fidelity decomposition (gt/s2f)
- Critic LM-head + track-embedding alignment (gt/s2f)
- Count-scaling diagnostics (gt/s2f)

Validation plots focus on mask-bin contrasts:
- loss_total, fid/total, critic/total across mask bins

Usage:
    python3 plot_metrics.py
    python3 plot_metrics.py --metrics /path/to/metrics.csv
    python3 plot_metrics.py --metrics /path/to/metrics_a.csv /path/to/metrics_b.csv
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# Ensure PDF text is editable in Illustrator
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42


# ===== HARDCODED METRICS PATHS =====
DEFAULT_METRICS_PATHS = [
    # "/home/mcb/users/dcakma3/multi_res_bench_v2/multimodal_masked_contrastive_main/manuscript_experiments/section_3_v2__Debiasing/workspace/runs/6__final_debiaser/ATAC/GM12878__ENCSR637XSC/dna_free_dae/bpnet__predicted/1000bp__mask_rco_10to60pct__rci_max_80pct/latentC64__sqrt__lb_nonpeak_133/fidelity__ce+align/seed_42/lightning_logs/version_0/metrics.csv",
    # "/home/mcb/users/dcakma3/multi_res_bench_v2/multimodal_masked_contrastive_main/manuscript_experiments/section_3_v2__Debiasing/workspace/runs/6__final_debiaser/ATAC/GM12878__ENCSR637XSC/dna_free_dae/bpnet__predicted/1000bp__mask_rco_10to60pct__rci_max_80pct/latentC64__sqrt__lb_nonpeak_133/fidelity__ce+align__critic_lm1__track0/seed_42/lightning_logs/version_0/metrics.csv"
    "/home/mcb/users/dcakma3/multi_res_bench_v2/multimodal_masked_contrastive_main/manuscript_experiments/section_3_v2__Debiasing/workspace/runs/6_1__final_debiaser/ATAC/GM12878__ENCSR637XSC/dna_free_dae/bpnet__predicted/1000bp__mask_rco_10to60pct__rci_max_80pct/latentC64__sqrt__lb_nonpeak_133/fidelity__ce+align__critic_lm1__track0/seed_42/lightning_logs/version_0/metrics.csv"
]

MASK_BIN_ORDER = ["10to20", "40to60", "100"]


def normalize_mask_bin(mask_bin: str) -> str:
    """Normalize mask-bin labels for consistent legends and colors."""
    if mask_bin.startswith("one_step_"):
        return mask_bin.replace("one_step_", "")
    return mask_bin


def mask_bin_sort_key(mask_bin: str) -> int:
    """Sort key to keep mask bins in a stable, publication-friendly order."""
    label = normalize_mask_bin(mask_bin)
    if label in MASK_BIN_ORDER:
        return MASK_BIN_ORDER.index(label)
    return len(MASK_BIN_ORDER)


def sanitize_filename(name: str) -> str:
    """Convert metric group name to safe filename."""
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", name)


def build_training_metric_groups(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Build training metric groups for monitoring training progress.

    Returns dict mapping group_name -> list of column names
    """
    groups: Dict[str, List[str]] = {}

    def add_group(name: str, cols: List[str]) -> None:
        existing = [col for col in cols if col in df.columns]
        if existing:
            groups[name] = existing

    # Objective balance
    add_group(
        "train_objective_balance",
        [
            "train/loss_total_epoch",
            "train/fid/total_epoch",
            "train/critic/total_epoch",
            "train/lambda_t_epoch",
            "train/grad/g_fid_epoch",
            "train/grad/g_critic_epoch",
        ],
    )

    # Fidelity decomposition (per route)
    add_group(
        "train_fid_gt_decomposition",
        [
            "train/fid/gt/total_epoch",
            "train/fid/gt/ce_epoch",
            "train/fid/gt/count_scaled_ce_epoch",
            "train/fid/gt/manifold_dist_epoch",
            "train/fid/gt/manifold_huber_epoch",
            "train/fid/gt/count_scaled_manifold_huber_epoch",
            "train/fid/gt/mnll_epoch",
        ],
    )
    add_group(
        "train_fid_s2f_decomposition",
        [
            "train/fid/s2f/total_epoch",
            "train/fid/s2f/ce_epoch",
            "train/fid/s2f/count_scaled_ce_epoch",
            "train/fid/s2f/manifold_dist_epoch",
            "train/fid/s2f/manifold_huber_epoch",
            "train/fid/s2f/count_scaled_manifold_huber_epoch",
            "train/fid/s2f/mnll_epoch",
        ],
    )

    # Critic LM-head behavior (per route)
    add_group(
        "train_critic_gt_lm_head",
        [
            "train/critic/gt/gt_head/loss_epoch",
            "train/critic/gt/gt_head/kl_epoch",
            "train/critic/gt/gt_head/count_scaled_kl_epoch",
            "train/critic/gt/gt_head/entropy_loss_epoch",
            "train/critic/gt/gt_head/count_scaled_entropy_loss_epoch",
            "train/critic/gt/gt_head/ref_ce_loss_epoch",
            "train/critic/gt/gt_head/count_scaled_ref_ce_loss_epoch",
            "train/critic/gt/gt_head/entropy_teacher_epoch",
            "train/critic/gt/gt_head/entropy_student_epoch",
        ],
    )
    add_group(
        "train_critic_s2f_lm_head",
        [
            "train/critic/s2f/gt_head/loss_epoch",
            "train/critic/s2f/gt_head/kl_epoch",
            "train/critic/s2f/gt_head/count_scaled_kl_epoch",
            "train/critic/s2f/gt_head/entropy_loss_epoch",
            "train/critic/s2f/gt_head/count_scaled_entropy_loss_epoch",
            "train/critic/s2f/gt_head/ref_ce_loss_epoch",
            "train/critic/s2f/gt_head/count_scaled_ref_ce_loss_epoch",
            "train/critic/s2f/gt_head/entropy_teacher_epoch",
            "train/critic/s2f/gt_head/entropy_student_epoch",
        ],
    )

    # Critic track-embedding alignment (per route)
    add_group(
        "train_critic_gt_track_embed",
        [
            "train/critic/gt/track_embed/loss_epoch",
            "train/critic/gt/track_embed_dist_epoch",
            "train/critic/gt/track_embed_huber_epoch",
            "train/critic/gt/count_scaled_track_embed_huber_epoch",
        ],
    )
    add_group(
        "train_critic_s2f_track_embed",
        [
            "train/critic/s2f/track_embed/loss_epoch",
            "train/critic/s2f/track_embed_dist_epoch",
            "train/critic/s2f/track_embed_huber_epoch",
            "train/critic/s2f/count_scaled_track_embed_huber_epoch",
        ],
    )

    # Count-scaling diagnostics (per route)
    add_group(
        "train_critic_gt_count_diagnostics",
        [
            "train/critic/gt/critic/count_coeff_mean_epoch",
            "train/critic/gt/critic/count_total_mean_epoch",
        ],
    )
    add_group(
        "train_critic_s2f_count_diagnostics",
        [
            "train/critic/s2f/critic/count_coeff_mean_epoch",
            "train/critic/s2f/critic/count_total_mean_epoch",
        ],
    )

    return groups


def build_validation_metric_groups(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Build validation metric groups organized by mask bins and metric types.

    Returns dict mapping group_name -> list of column names
    """
    groups: Dict[str, List[str]] = {}

    # Detect mask bins from available validation columns
    mask_bins = sorted(
        {
            col.split("/")[0].replace("val_", "")
            for col in df.columns
            if col.startswith("val_")
        },
        key=mask_bin_sort_key,
    )

    if not mask_bins:
        return groups

    def cols_for_metrics(metrics: List[str]) -> List[str]:
        cols: List[str] = []
        for mask_bin in mask_bins:
            for metric in metrics:
                col = f"val_{mask_bin}/{metric}"
                if col in df.columns:
                    cols.append(col)
        return cols

    # Validation mask-bin contrasts (per metric across all mask bins)
    loss_total_cols = cols_for_metrics(["loss_total"])
    if loss_total_cols:
        groups["val_loss_total_mask_contrast"] = loss_total_cols

    fid_total_cols = cols_for_metrics(["fid/total"])
    if fid_total_cols:
        groups["val_fid_total_mask_contrast"] = fid_total_cols

    critic_total_cols = cols_for_metrics(["critic/total"])
    if critic_total_cols:
        groups["val_critic_total_mask_contrast"] = critic_total_cols

    return groups


def plot_group(
    df: pd.DataFrame,
    cols: List[str],
    title: str,
    out_file: Path,
    group_name: str | None = None,
    is_training: bool = False,
    x_col: str = "epoch",
) -> None:
    """Plot a metric group (training or validation) across mask bins or components."""
    if not cols:
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    plotted_any = False

    # Color map for mask bins (3 bins)
    mask_bin_color_map = {
        "10to20": "#6baed6",  # Light blue
        "40to60": "#2171b5",  # Medium blue
        "100": "#08519c",     # Dark blue
    }

    # Line style map for different metric types
    line_style_map = {
        "total": "-",
        "mnll": "-",
        "ce": "--",
        "huber": ":",
        "logprob_huber": ":",
        "count_scaled": ":",
        "count_coeff": "--",
        "manifold": "-.",
        "ref_ce": "-.",
        "full": "-",
        "partial": "--",
        "kl": "-",
        "entropy": "--",
        "entropy_loss": "--",
        "dist": ":",
        "loss": "-",
        "g_critic": "-",
        "g_fid": "--",
    }

    # Special styling for primary plot (total loss across all masks)
    is_primary_plot = group_name is not None and group_name.startswith("val_loss_total")

    # Ordered mask bins for legend
    ordered_mask_bins = MASK_BIN_ORDER

    # Track lines for custom legend
    legend_entries = []

    for col in cols:
        if col not in df.columns or x_col not in df.columns:
            continue

        # Drop NaN and ensure numeric
        sub = df[[x_col, col]].dropna()
        sub = sub[
            pd.to_numeric(sub[x_col], errors='coerce').notnull() &
            pd.to_numeric(sub[col], errors='coerce').notnull()
        ]

        if sub.empty:
            continue

        # Parse column name to extract mask bin and metric type
        # Format: val_{mask_bin}/category/subcategory or train/category/subcategory
        col_parts = col.split("/")

        # Determine mask bin (if present)
        mask_bin = None
        if is_training:
            # Training metrics don't have mask bins
            metric_path = "/".join(col_parts[1:])  # Skip 'train'
        else:
            # Validation metrics: val_{mask_bin}/...
            if col_parts[0].startswith("val_"):
                mask_bin = col_parts[0].replace("val_", "")
                metric_path = "/".join(col_parts[1:])
            else:
                metric_path = "/".join(col_parts)

        # Build label
        mask_bin_label = normalize_mask_bin(mask_bin) if mask_bin else None
        if mask_bin_label:
            label = f"{mask_bin_label} | {metric_path}"
        else:
            label = metric_path

        # Determine color
        if mask_bin_label and mask_bin_label in mask_bin_color_map:
            color = mask_bin_color_map[mask_bin_label]
        else:
            # For training or unmapped, use default colors
            color = f"C{len(legend_entries) % 10}"

        # Determine line style based on metric type
        linestyle = "-"
        for key, style in line_style_map.items():
            if key in metric_path:
                linestyle = style
                break

        linewidth = 2.0 if is_primary_plot else 1.5

        line, = ax.plot(
            sub[x_col],
            sub[col],
            label=label,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            alpha=0.8,
        )
        legend_entries.append((mask_bin_label, line, label))
        plotted_any = True

    if not plotted_any:
        plt.close(fig)
        return

    if is_primary_plot:
        # Publication-ready styling: no title, clean labels
        ax.set_xlabel("Epochs", fontsize=14)
        ax.set_ylabel("Total Loss (Validation Set)", fontsize=14)

        # Sort legend entries by mask bin order
        def _sort_key(entry):
            mask_bin, _, _ = entry
            if mask_bin and mask_bin in ordered_mask_bins:
                return ordered_mask_bins.index(mask_bin)
            return len(ordered_mask_bins)

        legend_entries.sort(key=_sort_key)
        legend_handles = [line for _, line, _ in legend_entries]
        legend_labels = [label for _, _, label in legend_entries]

        # Standalone legend figure
        fig_leg, ax_leg = plt.subplots(figsize=(6, 4))
        ax_leg.axis("off")
        ax_leg.legend(
            legend_handles,
            legend_labels,
            loc="center",
            ncol=1,
            frameon=False,
            fontsize=10,
        )
        fig_leg.tight_layout()
        legend_path = out_file.with_name(out_file.stem + "_legend.pdf")
        fig_leg.savefig(legend_path)
        plt.close(fig_leg)
        print(f"  ✓ {legend_path.name}")
    else:
        # Standard styling for monitoring plots
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel(x_col.title(), fontsize=14)
        ax.set_ylabel("Loss", fontsize=14)

        # Smart legend placement based on number of items
        ncol = 2 if len(legend_entries) > 6 else 1
        ax.legend(loc="best", fontsize=9, framealpha=0.9, ncol=ncol)

    ax.grid(True, alpha=0.3, linestyle='--')
    fig.tight_layout()
    if out_file.suffix.lower() == ".png":
        fig.savefig(out_file, dpi=200)
    else:
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

    # Training metrics summary
    train_summary = {}
    for col in df.columns:
        if col.startswith("train/") and col.endswith("_epoch"):
            try:
                val = last_row.iloc[0][col]
                if pd.notna(val) and isinstance(val, (int, float)):
                    train_summary[col] = float(val)
            except Exception:
                continue

    if train_summary:
        train_summary_df = pd.Series(train_summary).to_frame(name="value")
        train_summary_path = out_dir / "training_summary_last_epoch.csv"
        train_summary_df.to_csv(train_summary_path)
        print(f"  ✓ Training Summary: {train_summary_path.name}")

    # Epoch-compacted training metrics CSV
    train_cols = [c for c in df.columns if c.startswith("train/") and c.endswith("_epoch")]
    if train_cols and "epoch" in df.columns:
        compact_rows = []
        for epoch_num, epoch_group in df_sorted.groupby("epoch", sort=True):
            last_row_epoch = epoch_group.tail(1).iloc[0]
            row_data = {"epoch": epoch_num}
            for train_col in train_cols:
                train_metric = epoch_group[train_col].dropna()
                if not train_metric.empty:
                    row_data[train_col] = train_metric.iloc[-1]
            compact_rows.append(row_data)

        compact = pd.DataFrame(compact_rows)
        compact_path = out_dir / "training_metrics_epoch_compact.csv"
        compact.to_csv(compact_path, index=False)
        print(f"  ✓ Compacted: {compact_path.name} (training metrics per epoch)")


def save_lambda_and_gradients(df: pd.DataFrame, out_dir: Path) -> None:
    """
    Save a CSV with the timeseries of lambda_t and gradient magnitudes that
    already appear in the metrics file.
    """
    if "epoch" not in df.columns:
        return

    lambda_grad_cols = []
    for col in [
        "train/lambda_t_epoch",
        "train/grad/g_critic_epoch",
        "train/grad/g_fid_epoch",
    ]:
        if col in df.columns:
            lambda_grad_cols.append(col)

    if not lambda_grad_cols:
        return

    subset = df[["epoch"] + lambda_grad_cols].copy()
    # Drop rows where all lambda/grad columns are NaN/empty
    subset = subset.dropna(axis=0, how="all", subset=lambda_grad_cols)
    if subset.empty:
        return

    out_path = out_dir / "lambda_and_gradients.csv"
    subset.to_csv(out_path, index=False)
    print(f"  ✓ Lambda/gradients: {out_path.name}")


def main() -> None:
    """Main plotting pipeline."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=DEFAULT_METRICS_PATHS,
        help="Path(s) to Lightning CSVLogger metrics.csv files.",
    )
    args = parser.parse_args()

    metrics_paths = [Path(p) for p in args.metrics]
    if not metrics_paths:
        print("ERROR: No metrics paths provided.")
        return

    for idx, metrics_path in enumerate(metrics_paths, start=1):
        if not metrics_path.exists():
            print(f"ERROR: Metrics file not found: {metrics_path}")
            continue

        print(f"\n{'='*70}")
        print(f"[{idx}/{len(metrics_paths)}] Reading metrics: {metrics_path}")
        print(f"{'='*70}\n")

        df = pd.read_csv(metrics_path)
        print(f"  Shape: {df.shape[0]} rows × {df.shape[1]} columns")

        # Count training and validation columns
        train_cols = [c for c in df.columns if c.startswith("train/")]
        val_cols = [c for c in df.columns if c.startswith("val_")]
        print(f"  Training metrics: {len(train_cols)} columns")
        print(f"  Validation metrics: {len(val_cols)} columns")

        # Output directory (auto: sibling /plots dir)
        out_dir = metrics_path.parent / "plots"
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"  Output: {out_dir}\n")

        # Build metric groups
        train_groups = build_training_metric_groups(df)
        val_groups = build_validation_metric_groups(df)
        print(f"Found {len(train_groups)} training metric groups")
        print(f"Found {len(val_groups)} validation metric groups\n")

        # Generate training plots
        if train_groups:
            print("Generating training plots:")
            for group_name, group_cols in train_groups.items():
                # MNLL plots are special: save epoch-scale as PDF, step-scale as PNG.
                if "mnll_epoch" in group_name:
                    out_file = out_dir / f"{sanitize_filename(group_name)}.pdf"
                    x_col = "epoch"
                elif "mnll_step" in group_name:
                    out_file = out_dir / f"{sanitize_filename(group_name)}.png"
                    x_col = "step"
                else:
                    out_file = out_dir / f"{sanitize_filename(group_name)}.png"
                    x_col = "epoch"
                plot_group(
                    df,
                    group_cols,
                    group_name.replace("_", " ").title(),
                    out_file,
                    group_name=group_name,
                    is_training=True,
                    x_col=x_col,
                )
            print()

        # Generate validation plots
        if val_groups:
            print("Generating validation plots:")
            for group_name, group_cols in val_groups.items():
                # Primary plot (total loss) is saved as PDF
                if group_name.startswith("val_loss_total") or "mnll" in group_name:
                    out_file = out_dir / f"{sanitize_filename(group_name)}.pdf"
                else:
                    out_file = out_dir / f"{sanitize_filename(group_name)}.png"
                plot_group(
                    df,
                    group_cols,
                    group_name.replace("_", " ").title(),
                    out_file,
                    group_name=group_name,
                    is_training=False,
                    x_col="epoch",
                )

        # Save summary tables
        print()
        save_summary_table(df, out_dir)
        save_lambda_and_gradients(df, out_dir)

        print(f"\n{'='*70}")
        print(f"Done! All outputs in: {out_dir}")
        print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
