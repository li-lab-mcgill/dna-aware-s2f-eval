from __future__ import annotations

from typing import Any, Dict

import numpy as np

from ..metrics.fidelity.components.ce import CEMetric
from ..metrics.fidelity.components.jsd import JSDMetric
from ..metrics.fidelity.components.mnll import MNLLMetric


def _counts_to_logprobs(counts: np.ndarray, eps: float) -> np.ndarray:
    counts = np.clip(counts, a_min=0.0, a_max=None)
    denom = np.clip(counts.sum(axis=-1, keepdims=True), a_min=eps, a_max=None)
    probs = counts / denom
    return np.log(np.clip(probs, a_min=eps, a_max=None))


def _split_stats_and_raw(
    metric_out: Dict[str, Any],
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    stats = dict(metric_out)
    raw: Dict[str, Any] = {}
    if "values" in metric_out:
        raw["values"] = metric_out["values"]
        stats.pop("values", None)
    if "counts_total" in metric_out:
        raw["counts_total"] = metric_out["counts_total"]
        stats.pop("counts_total", None)
    return stats, raw


def _compute_metrics_for_model(
    *,
    pred_logprobs: np.ndarray,
    true_tracks: np.ndarray,
    return_stats: bool,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    mnll_metric = MNLLMetric()
    ce_metric = CEMetric()
    jsd_metric = JSDMetric()

    outputs = {
        "mnll": mnll_metric.evaluate(
            pred_logprobs=pred_logprobs,
            true_tracks=true_tracks,
            return_stats=return_stats,
            return_values=True,
            return_counts_total=True,
        ),
        "ce": ce_metric.evaluate(
            pred_logprobs=pred_logprobs,
            true_tracks=true_tracks,
            return_stats=return_stats,
            return_values=True,
            return_counts_total=True,
        ),
        "jsd": jsd_metric.evaluate(
            pred_logprobs=pred_logprobs,
            true_tracks=true_tracks,
            return_stats=return_stats,
            return_values=True,
            return_counts_total=True,
        ),
    }

    stats_out: Dict[str, Any] = {}
    raw_out: Dict[str, Any] = {}
    for metric_name, metric_out in outputs.items():
        metric_stats, metric_raw = _split_stats_and_raw(metric_out)
        stats_out[metric_name] = metric_stats
        raw_out[metric_name] = metric_raw
    return stats_out, raw_out


def compute_fidelity_metrics(
    extracted: Dict[str, Any],
    *,
    eps: float = 1e-8,
    return_stats: bool = True,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Compute fidelity metrics for S2F and debiased outputs against GT counts.
    Returns (stats, raw_values).
    """
    gt_counts = np.asarray(extracted["gt_counts"])
    s2f_counts = np.asarray(extracted["s2f_counts"])
    dae_logprobs = extracted.get("dae_logprobs")
    if dae_logprobs is None:
        dae_logprobs = extracted.get("dae_debiased_logprobs")
    if dae_logprobs is None:
        raise KeyError("Missing dae_logprobs (or dae_debiased_logprobs) in extracted data.")
    dae_logprobs = np.asarray(dae_logprobs)
    editor_full_dna_logprobs = extracted.get("editor_full_dna_logprobs", None)
    editor_partial_dna_logprobs = extracted.get("editor_partial_dna_logprobs", None)

    s2f_logprobs = _counts_to_logprobs(s2f_counts, eps)

    stats_results: Dict[str, Any] = {}
    raw_results: Dict[str, Any] = {}

    model_logprobs = {
        "s2f": s2f_logprobs,
        "dae": dae_logprobs,
    }
    if editor_full_dna_logprobs is not None:
        model_logprobs["editor_full_dna"] = np.asarray(editor_full_dna_logprobs)
    if editor_partial_dna_logprobs is not None:
        model_logprobs["editor_partial_dna"] = np.asarray(editor_partial_dna_logprobs)

    for model_name, pred_logprobs in model_logprobs.items():
        stats_results[model_name], raw_results[model_name] = _compute_metrics_for_model(
            pred_logprobs=pred_logprobs,
            true_tracks=gt_counts,
            return_stats=return_stats,
        )

    return stats_results, raw_results


__all__ = ["compute_fidelity_metrics"]
