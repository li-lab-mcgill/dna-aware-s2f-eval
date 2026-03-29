import json
import os
import os.path as osp
from typing import Dict, List, Optional

import numpy as np
from scipy.spatial.distance import jensenshannon
from torch.utils.data import DataLoader
from tqdm import tqdm

import torch
import torch.nn.functional as F

from .eval_utils import convert_to_json_serializable, fprint
from ...pipeline.pipeline import bpnetlite_output_transform_fn
from ..data.base_zarr_dataset import create_dataset


def _resolve_dataset_paths(path_sets: Optional[List[List[str]]], fold_name: str) -> Optional[List[List[str]]]:
    if path_sets is None:
        return None

    resolved = []
    for path_set in path_sets:
        resolved.append([
            item.format(fold_name) if "{}" in item else item
            for item in path_set
        ])
    return resolved


def _build_dataset(config_dict):
    data_config = config_dict.data
    fold_name = config_dict.task.fold_name

    peaks_track_paths = _resolve_dataset_paths(
        getattr(data_config, "predicted_peaks_track_dataset_paths", None),
        fold_name,
    )
    nonpeaks_track_paths = _resolve_dataset_paths(
        getattr(data_config, "predicted_nonpeaks_track_dataset_paths", None),
        fold_name,
    )
    experimental_peaks_track_paths = _resolve_dataset_paths(
        getattr(data_config, "experimental_peaks_track_dataset_paths", None),
        fold_name,
    )
    experimental_nonpeaks_track_paths = _resolve_dataset_paths(
        getattr(data_config, "experimental_nonpeaks_track_dataset_paths", None),
        fold_name,
    )

    nonpeaks_zarr_path = data_config.nonpeaks_zarr_path.format(fold_name) \
        if "{}" in data_config.nonpeaks_zarr_path else data_config.nonpeaks_zarr_path

    return create_dataset(
        peaks_zarr_path=data_config.peaks_zarr_path,
        nonpeaks_zarr_path=nonpeaks_zarr_path,
        fold_name=fold_name,
        seq_dataset_path=data_config.seq_dataset_path,
        predicted_peaks_track_dataset_paths=peaks_track_paths,
        predicted_nonpeaks_track_dataset_paths=nonpeaks_track_paths,
        experimental_peaks_track_dataset_paths=experimental_peaks_track_paths,
        experimental_nonpeaks_track_dataset_paths=experimental_nonpeaks_track_paths,
        split_name=config_dict.task.split_name,
        peak_to_nonpeak_ratio=getattr(data_config, "peak_to_nonpeak_ratio", None),
        base_sampling_seed=getattr(data_config, "base_sampling_seed", 42),
        seq_width=data_config.seq_width,
        track_width=data_config.track_width,
        max_shift=getattr(data_config, "max_shift", 0),
        rc_aug=getattr(data_config, "rc_aug", False),
        shift_aug=getattr(data_config, "shift_aug", False),
        ddp_safe=getattr(data_config, "ddp_safe", "auto"),
        rc_strand_flip=getattr(data_config, "rc_strand_flip", False),
        strand_channel_pairs=getattr(data_config, "strand_channel_pairs", None),
    )


def _load_bpnet_model(model_path: str, device: str):
    from bpnetlite import BPNet

    if not osp.exists(model_path):
        raise FileNotFoundError(f"Missing BPNet-compatible checkpoint: {model_path}")

    model = BPNet.from_chrombpnet(model_path)
    model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model


def _load_chrombpnet_model(bias_model_path: str, accessibility_model_path: str, device: str):
    from bpnetlite import ChromBPNet

    if not osp.exists(bias_model_path):
        raise FileNotFoundError(f"Missing ChromBPNet bias checkpoint: {bias_model_path}")
    if not osp.exists(accessibility_model_path):
        raise FileNotFoundError(f"Missing ChromBPNet accessibility checkpoint: {accessibility_model_path}")

    model = ChromBPNet.from_chrombpnet(
        bias_model=bias_model_path,
        accessibility_model=accessibility_model_path,
        name="s2f_profile_fidelity",
    )
    model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model


def _load_torch_model(model_path: str, device: str):
    if not osp.exists(model_path):
        raise FileNotFoundError(f"Missing torch checkpoint: {model_path}")

    model = torch.load(model_path, map_location="cpu", weights_only=False)
    model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model


def _chrombpnet_signal_from_outputs(pred_profile_logits: torch.Tensor, pred_log1p_counts: torch.Tensor) -> torch.Tensor:
    if pred_profile_logits.dim() == 2:
        pred_profile_logits = pred_profile_logits.unsqueeze(1)
    if pred_log1p_counts.dim() == 1:
        pred_log1p_counts = pred_log1p_counts.unsqueeze(1)

    profile_probs = F.softmax(pred_profile_logits, dim=2)
    total_counts = torch.clamp(torch.exp(pred_log1p_counts) - 1.0, min=0.0)
    return profile_probs * total_counts.unsqueeze(2)


def _crop_to_central_width(signal: torch.Tensor, width: int) -> torch.Tensor:
    signal_width = signal.shape[1]
    if signal_width == width:
        return signal
    if signal_width < width:
        raise ValueError(f"Cannot crop width {signal_width} to larger width {width}")

    crop = (signal_width - width) // 2
    return signal[:, crop:crop + width, :]


def _crop_logits_to_central_width(logits: torch.Tensor, width: int) -> torch.Tensor:
    signal_width = logits.shape[2]
    if signal_width == width:
        return logits
    if signal_width < width:
        raise ValueError(f"Cannot crop logits width {signal_width} to larger width {width}")

    crop = (signal_width - width) // 2
    return logits[:, :, crop:crop + width]


def _load_s2f_model(config_dict):
    if not hasattr(config_dict, "s2f") or not hasattr(config_dict.s2f, "model_type"):
        return None

    model_type = config_dict.s2f.model_type
    device = config_dict.eval.device

    if model_type == "bpnet":
        return _load_bpnet_model(config_dict.s2f.checkpoint_path, device)
    if model_type == "chrombpnet_nobias":
        return _load_bpnet_model(config_dict.s2f.checkpoint_path, device)
    if model_type == "chrombpnet_bias":
        return _load_bpnet_model(config_dict.s2f.checkpoint_path, device)
    if model_type == "chrombpnet_full":
        return _load_chrombpnet_model(
            config_dict.s2f.bias_checkpoint_path,
            config_dict.s2f.accessibility_checkpoint_path,
            device,
        )
    if model_type == "bpnetlite_without_control":
        return _load_torch_model(config_dict.s2f.checkpoint_path, device)

    raise ValueError(f"Unsupported s2f.model_type={model_type}")


@torch.no_grad()
def _predict_tracks(seq_tensor: torch.Tensor, s2f_model, config_dict) -> torch.Tensor:
    if s2f_model is None:
        raise ValueError("S2F model is required for on-the-fly prediction")

    device = config_dict.eval.device
    model_type = config_dict.s2f.model_type

    sequence = seq_tensor.to(device).permute(0, 2, 1)

    if model_type in {"bpnet", "bpnetlite_without_control"}:
        outputs = s2f_model(sequence)
        return bpnetlite_output_transform_fn(outputs)

    if model_type == "chrombpnet_bias":
        pred_profile_logits, pred_log1p_counts = s2f_model(sequence)
        target_width = config_dict.data.track_width
        pred_profile_logits = _crop_logits_to_central_width(pred_profile_logits, target_width)
        pred_signal = _chrombpnet_signal_from_outputs(pred_profile_logits, pred_log1p_counts)
        return pred_signal.transpose(1, 2)

    if model_type in {"chrombpnet_nobias", "chrombpnet_full"}:
        pred_profile_logits, pred_log1p_counts = s2f_model(sequence)
        pred_signal = _chrombpnet_signal_from_outputs(pred_profile_logits, pred_log1p_counts)
        pred_signal = pred_signal.transpose(1, 2)
        target_width = config_dict.data.track_width
        pred_signal = _crop_to_central_width(pred_signal, target_width)
        return pred_signal

    raise ValueError(f"Unsupported s2f.model_type={model_type}")


def _tracks_to_probabilities(track_tensor: torch.Tensor):
    track_array = track_tensor.detach().cpu().numpy().astype(np.float64, copy=False)
    track_array = np.maximum(track_array, 0.0)
    flattened = track_array.reshape(track_array.shape[0], -1)
    totals = flattened.sum(axis=1)
    probabilities = np.zeros_like(flattened, dtype=np.float64)
    valid_totals = totals > 0
    probabilities[valid_totals] = flattened[valid_totals] / totals[valid_totals, None]
    return probabilities, totals


def _empty_split_result():
    return {
        "num_samples_total": 0,
        "num_valid_jsd": 0,
        "num_nan_jsd": 0,
        "num_zero_observed_total": 0,
        "num_zero_predicted_total": 0,
        "num_zero_both_total": 0,
        "jsd_mean": None,
        "jsd_median": None,
        "jsd_std": None,
        "jsd_min": None,
        "jsd_max": None,
        "jsd_p05": None,
        "jsd_p25": None,
        "jsd_p75": None,
        "jsd_p95": None,
    }


def _summarize_jsd(jsd_values: np.ndarray, observed_totals: np.ndarray, predicted_totals: np.ndarray):
    result = _empty_split_result()

    zero_observed = observed_totals == 0
    zero_predicted = predicted_totals == 0
    zero_both = zero_observed & zero_predicted
    nan_mask = np.isnan(jsd_values)
    valid_values = jsd_values[~nan_mask]

    result["num_samples_total"] = int(jsd_values.shape[0])
    result["num_valid_jsd"] = int(valid_values.shape[0])
    result["num_nan_jsd"] = int(nan_mask.sum())
    result["num_zero_observed_total"] = int(zero_observed.sum())
    result["num_zero_predicted_total"] = int(zero_predicted.sum())
    result["num_zero_both_total"] = int(zero_both.sum())

    if valid_values.size == 0:
        return result

    result["jsd_mean"] = float(np.mean(valid_values))
    result["jsd_median"] = float(np.median(valid_values))
    result["jsd_std"] = float(np.std(valid_values))
    result["jsd_min"] = float(np.min(valid_values))
    result["jsd_max"] = float(np.max(valid_values))
    result["jsd_p05"] = float(np.quantile(valid_values, 0.05))
    result["jsd_p25"] = float(np.quantile(valid_values, 0.25))
    result["jsd_p75"] = float(np.quantile(valid_values, 0.75))
    result["jsd_p95"] = float(np.quantile(valid_values, 0.95))
    return result


def _compute_batch_jsd(observed_tracks: torch.Tensor, predicted_tracks: torch.Tensor) -> Dict[str, np.ndarray]:
    observed_probs, observed_totals = _tracks_to_probabilities(observed_tracks)
    predicted_probs, predicted_totals = _tracks_to_probabilities(predicted_tracks)

    jsd_values = np.array([
        float(jensenshannon(observed_probs[index], predicted_probs[index]))
        for index in range(observed_probs.shape[0])
    ])

    return {
        "jsd": jsd_values,
        "observed_totals": observed_totals,
        "predicted_totals": predicted_totals,
    }


def _save_results(results: dict, save_path: str):
    os.makedirs(osp.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as handle:
        json.dump(convert_to_json_serializable(results), handle, indent=2)
    fprint(f"Saved results to: {save_path}")


def main(config_dict):
    fprint("=" * 60)
    fprint("S2F Profile-Shape Fidelity Evaluation")
    fprint("=" * 60)
    fprint(f"Assay: {config_dict.task.assay_type}")
    fprint(f"Experiment: {config_dict.task.assay_experiment_identifier}")
    fprint(f"Model: {config_dict.task.model_name}")
    fprint(f"Fold: {config_dict.task.fold_name}")
    fprint(f"Split: {config_dict.task.split_name}")

    dataset = _build_dataset(config_dict)
    fprint(f"Dataset length: {len(dataset)}")

    dataloader = DataLoader(
        dataset,
        batch_size=config_dict.eval.batch_size,
        shuffle=False,
        num_workers=config_dict.eval.num_workers,
        pin_memory=config_dict.eval.pin_memory,
        persistent_workers=(
            config_dict.eval.persistent_workers if config_dict.eval.num_workers > 0 else False
        ),
    )
    fprint(f"Dataloader batches: {len(dataloader)}")

    s2f_model = _load_s2f_model(config_dict)

    split_accumulators = {
        "peaks": {"jsd": [], "observed_totals": [], "predicted_totals": []},
        "nonpeaks": {"jsd": [], "observed_totals": [], "predicted_totals": []},
    }

    for batch in tqdm(dataloader, desc="Evaluating profile fidelity"):
        seq_tensor, pred_tensor, exp_tensor, region_types, identifiers = batch

        if exp_tensor.shape[-1] == 0:
            raise ValueError("Observed tracks are required for profile-fidelity evaluation")

        if s2f_model is None:
            if pred_tensor.shape[-1] == 0:
                raise ValueError("Predicted tracks are required when no S2F model is configured")
            predicted_tracks = pred_tensor
        else:
            predicted_tracks = _predict_tracks(seq_tensor, s2f_model, config_dict).cpu()

        if predicted_tracks.shape != exp_tensor.shape:
            raise ValueError(
                f"Predicted track shape {tuple(predicted_tracks.shape)} does not match "
                f"observed track shape {tuple(exp_tensor.shape)}"
            )

        batch_metrics = _compute_batch_jsd(exp_tensor, predicted_tracks)
        region_types = np.asarray(region_types)

        for split_label, region_name in (("peaks", "peak"), ("nonpeaks", "nonpeak")):
            split_mask = region_types == region_name
            if not np.any(split_mask):
                continue
            split_accumulators[split_label]["jsd"].append(batch_metrics["jsd"][split_mask])
            split_accumulators[split_label]["observed_totals"].append(
                batch_metrics["observed_totals"][split_mask]
            )
            split_accumulators[split_label]["predicted_totals"].append(
                batch_metrics["predicted_totals"][split_mask]
            )

    results = {
        "metadata": {
            "assay_type": config_dict.task.assay_type,
            "assay_experiment_identifier": config_dict.task.assay_experiment_identifier,
            "model_name": config_dict.task.model_name,
            "fold_name": config_dict.task.fold_name,
            "split_name": config_dict.task.split_name,
            "evaluation_mode": "profile_shape_fidelity_jsd",
        }
    }

    for split_label in ("peaks", "nonpeaks"):
        if split_accumulators[split_label]["jsd"]:
            jsd_values = np.concatenate(split_accumulators[split_label]["jsd"])
            observed_totals = np.concatenate(split_accumulators[split_label]["observed_totals"])
            predicted_totals = np.concatenate(split_accumulators[split_label]["predicted_totals"])
            results[split_label] = _summarize_jsd(jsd_values, observed_totals, predicted_totals)
        else:
            results[split_label] = _empty_split_result()

    for split_label in ("peaks", "nonpeaks"):
        split_result = results[split_label]
        fprint(
            f"{split_label}: total={split_result['num_samples_total']}, "
            f"valid={split_result['num_valid_jsd']}, "
            f"nan={split_result['num_nan_jsd']}, "
            f"median={split_result['jsd_median']}"
        )

    _save_results(results, config_dict.task.save_path)

    if s2f_model is not None:
        del s2f_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return results
