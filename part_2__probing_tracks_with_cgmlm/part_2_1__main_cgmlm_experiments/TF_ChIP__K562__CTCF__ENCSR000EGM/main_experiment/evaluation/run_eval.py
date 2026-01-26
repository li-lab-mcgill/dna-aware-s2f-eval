import pdb
import os
import sys
import time
import torch
import torch.nn.functional as F
import numpy as np
import json
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pathlib import Path
from ml_collections import ConfigDict
from copy import deepcopy
from tqdm import tqdm
import zarr
import os.path as osp
import pickle
import gzip

# Add project root to path
print(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../../../../.."))
from config import * 

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from eval_utils.mask_replicate_dataset import create_dataset
from eval_utils.eval_utils import (
    compute_metrics, 
    convert_to_json_serializable,
    compute_entropy_statistics,
)

def load_model(checkpoint_path, config_dict, device):
    """Load a multimodal masked model from a checkpoint."""
    from local_utils.models.masked_unet.multimodal_input_dna_output import MaskedGenomeUNet
    # init model 
    model = MaskedGenomeUNet(**config_dict.to_dict())
    # load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    # retain key with model. prefix then consume it
    checkpoint["state_dict"] = {k.replace("model.", ""): v for k, v in checkpoint['state_dict'].items() if k.startswith("model.")}    
    # load state dict
    model.load_state_dict(checkpoint["state_dict"])
    # set to eval mode
    model.eval()
    # move to device
    model.to(device)
    # return
    return model


def evaluate_model_on_replicate(model, dataloader, config_dict):
    """
    Evaluate model on a single replicate with peak/nonpeak separation
    
    Returns:
        metrics: {
            'peaks': {'cross_entropy': float, 'accuracy': float, 'matched_entropy': list, 'unmatched_entropy': list},
            'nonpeaks': {'cross_entropy': float, 'accuracy': float, 'matched_entropy': list, 'unmatched_entropy': list}
        }
    """
    all_logits = []
    all_targets = []
    all_masks = []
    all_region_types = []
    all_identifiers = []
    
    model.eval()
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Processing batches", leave=False, disable=True):
            # Shapes: (B, 2, L, 4+T), (B, L, 4+T), (B, L, 4+T)
            # NEW FORMAT: includes region_type and identifier
            masked_data_tensor, data_matrix, mask_matrix, region_type, identifier = batch
            
            # Move to device
            masked_data_tensor = masked_data_tensor.to(config_dict.eval.device)
            data_matrix = data_matrix.to(config_dict.eval.device)
            mask_matrix = mask_matrix.to(config_dict.eval.device)
            
            # Forward pass - model predicts (B, L, 4) DNA only
            preds = model(masked_data_tensor)  # (B, L, 4)
            
            # Extract DNA targets and masks
            targets = data_matrix[..., :4]  # (B, L, 4)
            masks = mask_matrix[..., :4]    # (B, L, 4)
            
            # Store results including metadata
            all_logits.append(preds.float().cpu())
            all_targets.append(targets.float().cpu())
            all_masks.append(masks.float().cpu())
            all_region_types.extend(region_type)  # List of strings
            all_identifiers.extend(identifier)    # List of strings
    
    # Concatenate tensors
    logits = torch.cat(all_logits, dim=0)  # (N, L, 4)
    targets = torch.cat(all_targets, dim=0)  # (N, L, 4)
    masks = torch.cat(all_masks, dim=0)      # (N, L, 4)
    
    # Convert region types to tensor for indexing
    region_types_array = np.array(all_region_types)
    peak_indices = np.where(region_types_array == 'peak')[0]
    nonpeak_indices = np.where(region_types_array == 'nonpeak')[0]
    
    # Compute peak-specific metrics
    peak_metrics = {}
    if len(peak_indices) > 0:
        peak_logits = logits[peak_indices]
        peak_targets = targets[peak_indices]
        peak_masks = masks[peak_indices]
        peak_metrics = compute_metrics(peak_logits, peak_targets, peak_masks, config_dict.eval.ignore_index)
    
    # Compute nonpeak-specific metrics
    nonpeak_metrics = {}
    if len(nonpeak_indices) > 0:
        nonpeak_logits = logits[nonpeak_indices]
        nonpeak_targets = targets[nonpeak_indices]
        nonpeak_masks = masks[nonpeak_indices]
        nonpeak_metrics = compute_metrics(nonpeak_logits, nonpeak_targets, nonpeak_masks, config_dict.eval.ignore_index)
    
    return {
        'peaks': peak_metrics,
        'nonpeaks': nonpeak_metrics,
        'num_peaks': len(peak_indices),
        'num_nonpeaks': len(nonpeak_indices)
    }


def save_replicate_results(results, fold_name, split_name, replicate_name, save_dir):
    """Save results for a single replicate after all mask types are completed"""
    # Create save directory
    replicate_save_dir = save_dir.format(fold_name, split_name, replicate_name)
    os.makedirs(replicate_save_dir, exist_ok=True)
    
    # Process results to convert entropy lists to statistics
    processed_results = deepcopy(results)
    for model_name, model_data in processed_results.items():
        metrics = model_data["metrics"]
        
        # Process peaks
        if "peaks" in metrics and metrics["peaks"]:
            peaks_matched_stats = compute_entropy_statistics(metrics["peaks"]["matched_entropy"])
            peaks_unmatched_stats = compute_entropy_statistics(metrics["peaks"]["unmatched_entropy"])
            metrics["peaks"]["matched_entropy_stats"] = peaks_matched_stats
            metrics["peaks"]["unmatched_entropy_stats"] = peaks_unmatched_stats
            del metrics["peaks"]["matched_entropy"]  # Remove raw lists
            del metrics["peaks"]["unmatched_entropy"]
        
        # Process nonpeaks
        if "nonpeaks" in metrics and metrics["nonpeaks"]:
            nonpeaks_matched_stats = compute_entropy_statistics(metrics["nonpeaks"]["matched_entropy"])
            nonpeaks_unmatched_stats = compute_entropy_statistics(metrics["nonpeaks"]["unmatched_entropy"])
            metrics["nonpeaks"]["matched_entropy_stats"] = nonpeaks_matched_stats
            metrics["nonpeaks"]["unmatched_entropy_stats"] = nonpeaks_unmatched_stats
            del metrics["nonpeaks"]["matched_entropy"]  # Remove raw lists
            del metrics["nonpeaks"]["unmatched_entropy"]
    
    # Save processed results
    results_file = osp.join(replicate_save_dir, "evaluation_results.json")
    with open(results_file, 'w') as f:
        json.dump(convert_to_json_serializable(processed_results), f, indent=2)
    
    print(f"💾 Saved replicate results to: {results_file}")
    return results_file


def main(config_dict):
    """Main evaluation function with peak/nonpeak separation"""
    
    print(f"🚀 Starting evaluation of {len(config_dict.task.model_names_and_checkpoints)} models")
    print(f"🔄 Evaluating on {len(config_dict.eval.replicate_ids)} replicate(s)")
    
    # Loop over folds
    for fold_name in config_dict.task.fold_names:
        print(f"\n=== Evaluating Fold {fold_name} ===")
        
        # Loop over splits
        for split_name in config_dict.task.split_names:
            print(f"\n--- Split {split_name} ---")
            
            # Loop over replicates
            for replicate_id in config_dict.eval.replicate_ids:
                replicate_name = f"replicate_{replicate_id}"
                print(f"\n--- Replicate {replicate_name} ---")
                
                # Results for this specific replicate across all models
                replicate_results = {}
                
                # Loop over each model and its MATCHING mask type
                for model_name, checkpoint_path in config_dict.task.model_names_and_checkpoints.items():
                    mask_type = model_name  # Model name IS the mask type
                    print(f"\n-- Model {model_name} on matching mask type {mask_type} --")

                    if checkpoint_path is None:
                        print(f"Skipping {model_name} because checkpoint path is None")
                        continue
                    
                    # Load model for this mask type
                    print(f"Loading {model_name} from {checkpoint_path}")
                    model = load_model(
                        checkpoint_path, 
                        config_dict.model, 
                        config_dict.eval.device
                    )

                    # Format fold related variables
                    nonpeaks_zarr_path = config_dict.data.nonpeaks_zarr_path.format(fold_name)
                    
                    # Create dataset with MATCHING mask type
                    dataset = create_dataset(
                        peaks_zarr_path=config_dict.data.peaks_zarr_path,  # ← NO formatting
                        nonpeaks_zarr_path=nonpeaks_zarr_path,  # ← Format this
                        fold_name=fold_name,
                        seq_dataset_path=config_dict.data.seq_dataset_path,
                        peaks_track_dataset_paths=config_dict.data.peaks_track_dataset_paths,  # ← Format this
                        nonpeaks_track_dataset_paths=config_dict.data.nonpeaks_track_dataset_paths,  # ← Format this
                        split_name=split_name,
                        mask_type_str=mask_type,  # Use matching mask type
                        replicate_name=replicate_name,
                        peak_to_nonpeak_ratio=config_dict.data.peak_to_nonpeak_ratio,
                        base_sampling_seed=config_dict.data.base_sampling_seed,
                        seq_width=config_dict.data.seq_width,
                        track_width=config_dict.data.track_width,
                        max_shift=config_dict.data.max_shift,
                        rc_aug=config_dict.data.rc_aug,
                        shift_aug=config_dict.data.shift_aug,
                        ddp_safe=config_dict.data.ddp_safe
                    )
                    
                    dataloader = DataLoader(
                        dataset,
                        batch_size=config_dict.eval.batch_size,
                        shuffle=False,
                        num_workers=config_dict.eval.num_workers,
                        pin_memory=config_dict.eval.pin_memory,
                        persistent_workers=config_dict.eval.persistent_workers
                    )
                    
                    # Evaluate model on this replicate
                    metrics = evaluate_model_on_replicate(model, dataloader, config_dict)
                    
                    # Store metrics for this model
                    replicate_results[model_name] = {
                        "checkpoint_path": checkpoint_path,
                        "mask_type": mask_type,
                        "metrics": metrics
                    }
                    
                    # Print summary for peaks and nonpeaks only
                    print(f"  Total: {metrics['num_peaks'] + metrics['num_nonpeaks']} samples")

                    if metrics['num_peaks'] > 0:
                        print(f"  Peaks: {metrics['num_peaks']} samples")
                        print(f"    Cross-entropy: {metrics['peaks']['cross_entropy']:.4f}")
                        print(f"    Accuracy: {metrics['peaks']['accuracy']:.4f}")

                    if metrics['num_nonpeaks'] > 0:
                        print(f"  Nonpeaks: {metrics['num_nonpeaks']} samples")
                        print(f"    Cross-entropy: {metrics['nonpeaks']['cross_entropy']:.4f}")
                        print(f"    Accuracy: {metrics['nonpeaks']['accuracy']:.4f}")
                    
                    # Clean up model to save memory
                    del model
                    torch.cuda.empty_cache()
                
                # Save results for this replicate after all models are done
                save_replicate_results(
                    replicate_results, 
                    fold_name, 
                    split_name, 
                    replicate_name, 
                    config_dict.task.save_dir
                )
    
    print(f"\n🎉 Evaluation complete! All models evaluated with peak/nonpeak separation.")


if __name__ == "__main__":
    # Create configuration
    config_dict = ConfigDict()

    # Task configuration
    config_dict.task = ConfigDict()
    config_dict.task.fold_names = ["fold_0"]
    config_dict.task.split_names = ["test"]
    config_dict.task.model_names_and_checkpoints = {
        "10_to_20pct": "/home/mcb/users/dcakma3/multi_res_bench_v2/multimodal_masked_contrastive_main/manuscript_experiments/section_1__ConditionalLM_as_diagnostic_probe/workspace/runs/TF_ChIP/K562__CTCF__ENCSR000EGM/ConditionalLM/observed__main_experiment/fold_0/tiny__variable_rate_partial_row_mask__from_0.1_to_0.2_pct__p_dna_1.0__p_track_0.0__/DNA_1.0/lightning_logs/version_0/checkpoints/best_val_loss-epoch=122-step=17589.ckpt",
        "20_to_40pct": "/home/mcb/users/dcakma3/multi_res_bench_v2/multimodal_masked_contrastive_main/manuscript_experiments/section_1__ConditionalLM_as_diagnostic_probe/workspace/runs/TF_ChIP/K562__CTCF__ENCSR000EGM/ConditionalLM/observed__main_experiment/fold_0/tiny__variable_rate_partial_row_mask__from_0.2_to_0.4_pct__p_dna_1.0__p_track_0.0__/DNA_1.0/lightning_logs/version_0/checkpoints/best_val_loss-epoch=47-step=6864.ckpt",
        "40_to_60pct": "/home/mcb/users/dcakma3/multi_res_bench_v2/multimodal_masked_contrastive_main/manuscript_experiments/section_1__ConditionalLM_as_diagnostic_probe/workspace/runs/TF_ChIP/K562__CTCF__ENCSR000EGM/ConditionalLM/observed__main_experiment/fold_0/tiny__variable_rate_partial_row_mask__from_0.4_to_0.6_pct__p_dna_1.0__p_track_0.0__/DNA_1.0/lightning_logs/version_0/checkpoints/best_val_loss-epoch=53-step=7722.ckpt",
        "60_to_80pct": "/home/mcb/users/dcakma3/multi_res_bench_v2/multimodal_masked_contrastive_main/manuscript_experiments/section_1__ConditionalLM_as_diagnostic_probe/workspace/runs/TF_ChIP/K562__CTCF__ENCSR000EGM/ConditionalLM/observed__main_experiment/fold_0/tiny__variable_rate_partial_row_mask__from_0.6_to_0.8_pct__p_dna_1.0__p_track_0.0__/DNA_1.0/lightning_logs/version_0/checkpoints/best_val_loss-epoch=34-step=5005.ckpt",
        "80_to_100pct": "/home/mcb/users/dcakma3/multi_res_bench_v2/multimodal_masked_contrastive_main/manuscript_experiments/section_1__ConditionalLM_as_diagnostic_probe/workspace/runs/TF_ChIP/K562__CTCF__ENCSR000EGM/ConditionalLM/observed__main_experiment/fold_0/tiny__variable_rate_partial_row_mask__from_0.8_to_1.0_pct__p_dna_1.0__p_track_0.0__/DNA_1.0/lightning_logs/version_0/checkpoints/best_val_loss-epoch=25-step=3718.ckpt",
        "100pct": "/home/mcb/users/dcakma3/multi_res_bench_v2/multimodal_masked_contrastive_main/manuscript_experiments/section_1__ConditionalLM_as_diagnostic_probe/workspace/runs/TF_ChIP/K562__CTCF__ENCSR000EGM/ConditionalLM/observed__main_experiment/fold_0/tiny__variable_rate_partial_row_mask__from_1.0_to_1.0_pct__p_dna_1.0__p_track_0.0__/DNA_1.0/lightning_logs/version_0/checkpoints/best_val_loss-epoch=27-step=4004.ckpt",
    }
    config_dict.task.save_dir = os.path.join(
        "./results",
        "{}", # fold name 
        "{}", # split name
        "{}", # replicate name
    )

    # Evaluation configuration
    config_dict.eval = ConfigDict()
    config_dict.eval.device = "cuda:1"
    config_dict.eval.batch_size = 8192
    config_dict.eval.num_workers = 0
    config_dict.eval.pin_memory = True
    config_dict.eval.persistent_workers = False
    config_dict.eval.ignore_index = -100
    config_dict.eval.replicate_ids = [0]
    
    # Model configuration
    config_dict.model = ConfigDict()
    config_dict.model.T = 2 + 4  # 2 functional tracks (+/-) + 4 DNA one-hot
    config_dict.model.depths = [128, 192, 256]  # Tiny model depths
    config_dict.model.input_kernel_size = 23
    config_dict.model.input_conv_channels = 64
    config_dict.model.num_groups = 8
    config_dict.model.conv_kernel_size = 3
    config_dict.model.dropout = 0.1
    config_dict.model.padding_mode = 'same'
    
    # Bottleneck configuration
    config_dict.model.bottleneck_config = {
        'num_layers': 2,
        'heads': 4,
        'ff_mult': 2,
        'dropout': 0.1,
        'use_gqa': True,
        'use_flash': False,
        'rotary_emb_fraction': 0.5,
        'rotary_emb_base': 20000.0
    }

    # Data configuration - USE TEMPLATES with {} placeholders
    config_dict.data = ConfigDict()
    config_dict.data.peaks_zarr_path = osp.join(
        MANUSCRIPT_EXPERIMENTS_DIR,
        "section_1__ConditionalLM_as_diagnostic_probe",
        "workspace",
        "data",
        "TF_ChIP",
        "K562__CTCF__ENCSR000EGM",
        "preprocessed",
        "zarr_datasets",
        "peaks.all_folds.zarr"
    )
    config_dict.data.nonpeaks_zarr_path = osp.join(
        MANUSCRIPT_EXPERIMENTS_DIR,
        "section_1__ConditionalLM_as_diagnostic_probe",
        "workspace",
        "data",
        "TF_ChIP",
        "K562__CTCF__ENCSR000EGM",
        "preprocessed",
        "zarr_datasets",
        "nonpeaks.{}.zarr"  # ← Template for fold formatting
    )
    config_dict.data.seq_dataset_path = [
        "reference_dna", 
        "dna_summitcentered1000bp_symmetric500bpshift"
    ]
    config_dict.data.peaks_track_dataset_paths = [
        [
            "observed_tracks", 
            "observed_main_track_summitcentered1000bp_symmetric500bpshift"  # Main experiment track
        ]
    ]
    config_dict.data.nonpeaks_track_dataset_paths = [
        [
            "observed_tracks", 
            "observed_main_track_summitcentered1000bp_symmetric500bpshift"  # Main experiment track
        ]
    ]
    # peak/nonpeak sampling parameters
    config_dict.data.peak_to_nonpeak_ratio = None  # Test set: use all data
    config_dict.data.base_sampling_seed = 42
    # sequence and track parameters
    config_dict.data.seq_width = 1000
    config_dict.data.track_width = 1000
    config_dict.data.max_shift = 0
    config_dict.data.rc_aug = False
    config_dict.data.shift_aug = False
    config_dict.data.ddp_safe = "auto"
    
    # Run main function
    main(config_dict)