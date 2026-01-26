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

sys.path.append(os.path.join(
    os.path.dirname(__file__), 
    "../../../../../../../.."
))
from config import MANUSCRIPT_SECTION_2_DIR, MANUSCRIPT_EXPERIMENTS_DIR  # noqa 
sys.path.append(MANUSCRIPT_SECTION_2_DIR)

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from eval_utils.base_zarr_dataset import create_dataset
from eval_utils.eval_utils import (
    load_clm_model,
    compute_metrics, 
    convert_to_json_serializable,
    construct_tclm_input
)
from eval_utils.prepare_batch import BatchPreparer




def evaluate_model_on_replicate(model, dataloader, batch_preparer, config_dict):
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
    
    model.eval()
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Processing batches", leave=False, disable=True):

            # 1) Prepare on device
            prepared = batch_preparer(
                batch,
                device=config_dict.eval.device,
            )

            # 2) Extract tensors from prepared batch
            tracks = prepared[config_dict.track_key_in_batch]  # (B, L, T)
            dna = prepared['dna']                                   # (B, L, 4)
            mask_matrix = prepared['mask']                          # (B, L, 4+T)
            region_types = prepared['region_types']                 # list[str]

            # 3) Build masked input for the model
            masked_data_tensor = construct_tclm_input(tracks, dna, mask_matrix)  # (B, 2, L, 4+T)
            
            # Forward pass - model predicts (B, L, 4) DNA only
            preds = model(masked_data_tensor)  # (B, L, 4)
            
            # Extract DNA targets and masks
            targets = dna[..., :4]  # (B, L, 4)
            masks = mask_matrix[..., :4]    # (B, L, 4)
            
            # Store results including metadata
            all_logits.append(preds.float().cpu())
            all_targets.append(targets.float().cpu())
            all_masks.append(masks.float().cpu())
            all_region_types.extend(region_types)  # List of strings
    
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
      
    # Save processed results
    results_file = osp.join(replicate_save_dir, "evaluation_results.json")
    with open(results_file, 'w') as f:
        json.dump(convert_to_json_serializable(results), f, indent=2)
    
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
                
                # Results for this specific replicate across all mask types and resolutions
                replicate_results = {}

                # Outer loop: iterate over mask types
                for mask_type, resolution_dict in config_dict.task.model_names_and_checkpoints.items():
                    print(f"\n-- Mask Type: {mask_type} --")

                    # Initialize nested dictionary for this mask type
                    if mask_type not in replicate_results:
                        replicate_results[mask_type] = {}

                    # Inner loop: iterate over resolutions for this mask type
                    for resolution_name, checkpoint_path in resolution_dict.items():
                        bin_width = int(resolution_name)  # Resolution name IS the bin_width
                        print(f"\n-- Resolution (bin_width): {bin_width} (mask type: {mask_type}) --")

                        if checkpoint_path is None:
                            print(f"Skipping resolution {resolution_name} for mask {mask_type} because checkpoint path is None")
                            continue

                        # Load model for this resolution
                        print(f"Loading model with bin_width {bin_width} from {checkpoint_path}")
                        model = load_clm_model(
                            checkpoint_path,
                            config_dict.model,
                            config_dict.eval.device
                        )

                        # Create batch preparer for this specific resolution
                        # For resolution ablation: region_lengths is FIXED at (-1,) = full 1000bp
                        # Only bin_width varies
                        batch_preparer = BatchPreparer(
                            region_lengths=(-1,),  # FIXED: Always use full 1000bp sequence
                            bin_width=bin_width,   # VARIABLE: Changes with resolution
                            equalize_counts=config_dict.batch_preparer.equalize_counts,
                            use_logits=config_dict.batch_preparer.use_logits
                        )

                        # Format nonpeaks related variables
                        nonpeaks_zarr_path = config_dict.data.nonpeaks_zarr_path.format(fold_name)

                        # Create dataset with the current mask type
                        dataset = create_dataset(
                            peaks_zarr_path=config_dict.data.peaks_zarr_path,  # ← NO formatting
                            nonpeaks_zarr_path=nonpeaks_zarr_path,  # ← Format this
                            fold_name=fold_name,
                            seq_dataset_path=config_dict.data.seq_dataset_path,
                            predicted_peaks_track_dataset_paths=config_dict.data.predicted_peaks_track_dataset_paths,
                            predicted_nonpeaks_track_dataset_paths=config_dict.data.predicted_nonpeaks_track_dataset_paths,
                            experimental_peaks_track_dataset_paths=config_dict.data.experimental_peaks_track_dataset_paths,
                            experimental_nonpeaks_track_dataset_paths=config_dict.data.experimental_nonpeaks_track_dataset_paths,
                            split_name=split_name,
                            mask_type_str=mask_type,  # Use current mask type
                            replicate_name=replicate_name,
                            subsample_frac=config_dict.data.subsample_frac,
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
                        metrics = evaluate_model_on_replicate(model, dataloader, batch_preparer, config_dict)

                        # Store metrics for this resolution under this mask type
                        replicate_results[mask_type][resolution_name] = {
                            "checkpoint_path": checkpoint_path,
                            "bin_width": bin_width,
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
        "10_to_20pct": {
            "1": "/home/mcb/users/dcakma3/multi_res_bench_v2/multimodal_masked_contrastive_main/manuscript_experiments/section_2__ConditionalLM_controlled_experiments/workspace/runs/ATAC/GM12878__ENCSR637XSC/ablation_1__count_equalization/observed/fold_0/mask_10to20pct/seed_42/lightning_logs/version_0/checkpoints/best_val_loss-epoch=106-step=51039.ckpt",
            "2": "/home/mcb/users/dcakma3/multi_res_bench_v2/multimodal_masked_contrastive_main/manuscript_experiments/section_2__ConditionalLM_controlled_experiments/workspace/runs/ATAC/GM12878__ENCSR637XSC/ablation_2__resolution__shift_aug/observed/fold_0/mask_10to20pct/seed_42__resolution_2/lightning_logs/version_0/checkpoints/best_val_loss-epoch=92-step=44361.ckpt",
            "4": "/home/mcb/users/dcakma3/multi_res_bench_v2/multimodal_masked_contrastive_main/manuscript_experiments/section_2__ConditionalLM_controlled_experiments/workspace/runs/ATAC/GM12878__ENCSR637XSC/ablation_2__resolution__shift_aug/observed/fold_0/mask_10to20pct/seed_42__resolution_4/lightning_logs/version_0/checkpoints/best_val_loss-epoch=37-step=18126.ckpt",
            "8": "/home/mcb/users/dcakma3/multi_res_bench_v2/multimodal_masked_contrastive_main/manuscript_experiments/section_2__ConditionalLM_controlled_experiments/workspace/runs/ATAC/GM12878__ENCSR637XSC/ablation_2__resolution__shift_aug/observed/fold_0/mask_10to20pct/seed_42__resolution_8/lightning_logs/version_0/checkpoints/best_val_loss-epoch=69-step=33390.ckpt",
        },
        "100pct": {
            "1": "/home/mcb/users/dcakma3/multi_res_bench_v2/multimodal_masked_contrastive_main/manuscript_experiments/section_2__ConditionalLM_controlled_experiments/workspace/runs/ATAC/GM12878__ENCSR637XSC/ablation_1__count_equalization/observed/fold_0/mask_100pct/seed_42/lightning_logs/version_0/checkpoints/best_val_loss-epoch=22-step=10971.ckpt",
            "2": "/home/mcb/users/dcakma3/multi_res_bench_v2/multimodal_masked_contrastive_main/manuscript_experiments/section_2__ConditionalLM_controlled_experiments/workspace/runs/ATAC/GM12878__ENCSR637XSC/ablation_2__resolution__shift_aug/observed/fold_0/mask_100pct/seed_42__resolution_2/lightning_logs/version_0/checkpoints/best_val_loss-epoch=16-step=8109.ckpt",
            "4": "/home/mcb/users/dcakma3/multi_res_bench_v2/multimodal_masked_contrastive_main/manuscript_experiments/section_2__ConditionalLM_controlled_experiments/workspace/runs/ATAC/GM12878__ENCSR637XSC/ablation_2__resolution__shift_aug/observed/fold_0/mask_100pct/seed_42__resolution_4/lightning_logs/version_0/checkpoints/best_val_loss-epoch=33-step=16218.ckpt",
            "8": "/home/mcb/users/dcakma3/multi_res_bench_v2/multimodal_masked_contrastive_main/manuscript_experiments/section_2__ConditionalLM_controlled_experiments/workspace/runs/ATAC/GM12878__ENCSR637XSC/ablation_2__resolution__shift_aug/observed/fold_0/mask_100pct/seed_42__resolution_8/lightning_logs/version_0/checkpoints/best_val_loss-epoch=25-step=12402.ckpt",
        }
    }
    config_dict.task.save_dir = os.path.join(
        "./results",
        "{}", # fold name 
        "{}", # split name
        "{}", # replicate name
    )

    # Evaluation configuration
    config_dict.eval = ConfigDict()
    config_dict.eval.device = "cuda:3"
    config_dict.eval.batch_size = 8192 - 1024
    config_dict.eval.num_workers = 0
    config_dict.eval.pin_memory = True
    config_dict.eval.persistent_workers = False
    config_dict.eval.ignore_index = -100
    config_dict.eval.replicate_ids = [2]
    
    # Model configuration
    config_dict.model = ConfigDict()
    config_dict.model.T = 1 + 4  # 1 functional track + 4 DNA one-hot
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
        "ATAC",
        "GM12878__ENCSR637XSC",
        "preprocessed",
        "zarr_datasets",
        "peaks.all_folds.zarr"
    )
    config_dict.data.nonpeaks_zarr_path = osp.join(
        MANUSCRIPT_EXPERIMENTS_DIR,
        "section_1__ConditionalLM_as_diagnostic_probe",
        "workspace",
        "data",
        "ATAC",
        "GM12878__ENCSR637XSC",
        "preprocessed",
        "zarr_datasets",
        "nonpeaks.{}.zarr"  # ← Template for fold formatting
    )
    config_dict.data.seq_dataset_path = [
        "reference_dna", 
        "dna_summitcentered1000bp_symmetric500bpshift"
    ]
    # Predicted tracks are NOT provided for observed runs
    config_dict.data.predicted_peaks_track_dataset_paths = None
    config_dict.data.predicted_nonpeaks_track_dataset_paths = None
    # Experimental tracks (fold-agnostic)
    config_dict.data.experimental_peaks_track_dataset_paths = [
        [
            "observed_tracks",
            "observed_track_summitcentered1000bp_symmetric500bpshift"
        ]
    ]
    config_dict.data.experimental_nonpeaks_track_dataset_paths = [
        [
            "observed_tracks",
            "observed_track_summitcentered1000bp_symmetric500bpshift"
        ]
    ]
    # peak/nonpeak sampling parameters
    config_dict.data.peak_to_nonpeak_ratio = 10
    config_dict.data.base_sampling_seed = 42
    # optional subsampling for eval speed
    config_dict.data.subsample_frac = None
    # sequence and track parameters
    config_dict.data.seq_width = 1000
    config_dict.data.track_width = 1000
    config_dict.data.max_shift = 0
    config_dict.data.rc_aug = False
    config_dict.data.shift_aug = False
    config_dict.data.ddp_safe = "auto"
    
    # Batch preparer configuration template (instantiated per resolution in the loop)
    # For resolution ablation: region_lengths is FIXED at (-1,) = full 1000bp
    # Each model evaluation will create its own BatchPreparer with matching bin_width
    config_dict.batch_preparer = ConfigDict()
    # bin_width will be set per-model in the loop (1, 2, 4, or 8)
    # Using observed tracks as input: no equalization needed
    config_dict.batch_preparer.equalize_counts = False
    config_dict.batch_preparer.use_logits = False     # use tracks path

    # dynamic track key used by batch preparer when constructing model input
    config_dict.track_key_in_batch = "exp_tracks"
    
    # Run main function
    main(config_dict)
