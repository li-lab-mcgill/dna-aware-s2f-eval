import pdb
import os
import sys
import numpy as np
import pandas as pd
import zarr
import pyBigWig
from tqdm import tqdm
from ml_collections import ConfigDict
import matplotlib.pyplot as plt
import logging

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from bpnetlite import BPNet, ChromBPNet

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../../../.."))
from config import * 

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../.."))
from local_utils.datasets.S2F.tile_dataset import create_dataset

def load_models_for_fold(config_dict, fold_name):
    """Load ChromBPNet models for a specific fold"""
    
    print(f"Loading ChromBPNet models for {fold_name}...")
    
    # Dictionary to store loaded models
    models = {}
    
    # Fill fold-specific filepaths
    fold_model_filepaths = {}
    for model_name, filepath_template in config_dict.model.model_filepaths.items():
        fold_model_filepaths[model_name] = filepath_template.format(fold_name, fold_name)
    
    # 1. Load scaled bias model as BPNet
    print("Loading scaled bias model...")
    models['scaled_bias'] = BPNet.from_chrombpnet(fold_model_filepaths['bias_scaled'])
    
    print("✓ All models loaded successfully")
    return models

def process_chrombpnet_output(pred_log1p_counts, pred_profile_logits):
    """Convert ChromBPNet raw outputs to predicted signal"""
    
    # Handle different output formats
    if pred_profile_logits.dim() == 2:
        # Shape is (B, L) - add track dimension
        pred_profile_logits = pred_profile_logits.unsqueeze(1)  # (B, 1, L)
    
    if pred_log1p_counts.dim() == 1:
        # Shape is (B,) - add track dimension  
        pred_log1p_counts = pred_log1p_counts.unsqueeze(1)  # (B, 1)
    
    # Apply softmax to profile logits along sequence dimension  
    profile_probs = F.softmax(pred_profile_logits, dim=2)  # (B, T, L)
    
    # Apply inverse log1p transform: exp(x) - 1, then clamp at 0
    total_counts = torch.clamp(torch.exp(pred_log1p_counts) - 1, min=0.0)  # (B, T)
    
    # Multiply profile probabilities by total counts for each batch entry
    # total_counts needs to be expanded to match profile_probs shape
    total_counts_expanded = total_counts.unsqueeze(2)  # (B, T, 1)
    predicted_signal = profile_probs * total_counts_expanded  # (B, T, L)
    
    return predicted_signal

def crop_to_1000bp(signal, model_name):
    """
    Crop signal to 1000bp if needed
    scaled_bias outputs 1960bp, others output 1000bp
    """
    signal_length = signal.shape[-1]
    
    if model_name == 'scaled_bias' and signal_length == 1960:
        # Crop from 1960bp to 1000bp (remove 480bp from each side)
        crop_width = (signal_length - 1000) // 2  # 480
        return signal[:, :, crop_width:-crop_width]  # (B, T, 1000)
    else:
        # Already 1000bp or different model
        return signal

def create_predicted_track_dataset(fold_name: str, config_dict: ConfigDict):
    """Add predicted track datasets to existing zarr file"""
    
    zarr_path = os.path.join(config_dict.output_dir, f"peaks.all_folds.zarr")
    
    # Setup logging
    print(f"=== Adding observed track datasets ===")
    
    # Load regions dataframe
    regions_df = pd.read_csv(os.path.join(zarr_path, "_auxiliary", "regions_df.tsv"), sep='\t')
    N = len(regions_df)
    
    print(f"Zarr path: {zarr_path}")
    print(f"Regions: {N}")
    
    # Load models
    models = load_models_for_fold(config_dict, fold_name)

    # create dataset
    seq_dataset_path = ["reference_dna", "dna_summitcentered2114bp_symmetric750bpshift"]
    dataset = create_dataset(
        zarr_path=zarr_path,
        fold_name=fold_name,
        seq_dataset_path=seq_dataset_path,
        model_input_len=config_dict.input_region_len,
        tile_size=config_dict.tile_size,
    )

    # create dataloader
    batch_size = 1024*7
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False, # Critical: keeps tiles grouped by region
        num_workers=0,
    )

    print(f"Dataset size: {len(dataset)} tiles ({len(dataset)//4} regions)")
    print(f"Batch size: {batch_size}, Number of batches: {len(dataloader)}")

    # device setup
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Open zarr in read/write mode to add new datasets
    root = zarr.open(zarr_path, mode="r+")

    # Create new datasets with same chunking as DNA output dataset
    existing_dna_output = root["reference_dna"]["dna_summitcentered1000bp_symmetric500bpshift"]
    chunk_len = existing_dna_output.chunks[0]  # Get chunk size from existing dataset
    
    compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=zarr.Blosc.BITSHUFFLE)
    
    # Dataset names
    group_name = f"{fold_name}__model_based_tracks"
    target_zarr_group = root[group_name]

    # iterate over all models
    for model_name, model in models.items():
        print(f"Processing ({fold_name}) model: {model_name}")

        # Dataset name
        dataset_name = f"{model_name}_predicted_track_summitcentered1000bp_symmetric500bpshift"

        # Create zarr dataset
        if dataset_name in target_zarr_group:
            print(f"Dataset {dataset_name} already exists, will overwrite. If allowed, press c to continue")
            pdb.set_trace()
            del target_zarr_group[dataset_name]
        
        predicted_dataset = target_zarr_group.create_dataset(
            dataset_name,
            shape=(N, config_dict.shift_extended_output_region_len, 1),  # (N, 2000, 1)
            chunks=(chunk_len, config_dict.shift_extended_output_region_len, 1),
            dtype="float32",
            compressor=compressor,
        )
        
        print(f"Created dataset: {group_name}/{dataset_name}")
        print(f"Shape: {predicted_dataset.shape}")
        
        # Move model to device and set to eval mode
        model = model.to(device)
        model.eval()
        
        # Reset dataloader iterator for this model
        dataloader_iter = iter(dataloader)
        
        # Storage for all predictions in order
        all_predicted_signals = []
        all_region_indices = []
        all_tile_indices = []
        
        # Process batches
        batch_count = 0
        with torch.no_grad():
            for batch_sequences, batch_metadata in tqdm(dataloader_iter, desc=f"{model_name}"):
                batch_count += 1
                
                # Move sequences to device
                batch_sequences = batch_sequences.to(device)  # (batch_size, 4, 2114)
                
                # Forward pass
                outputs = model(batch_sequences)
                
                # Process outputs
                if isinstance(outputs, tuple) and len(outputs) == 2:
                    pred_profile_logits, pred_log1p_counts = outputs
                    predicted_signal = process_chrombpnet_output(pred_log1p_counts, pred_profile_logits)
                else:
                    print(f"Unexpected output format: {type(outputs)}")
                    continue
                
                # Crop to 1000bp if needed (scaled_bias model outputs 1960bp)
                predicted_signal = crop_to_1000bp(predicted_signal, model_name)
                
                # Store predicted signal (move to CPU to save GPU memory)
                all_predicted_signals.append(predicted_signal.cpu())
                
                # Store metadata (extract from collated dict format)
                batch_region_indices = batch_metadata['region_idx'].tolist()  # Convert tensor to list
                batch_tile_indices = batch_metadata['tile_idx'].tolist()      # Convert tensor to list
                
                all_region_indices.extend(batch_region_indices)
                all_tile_indices.extend(batch_tile_indices)
                
                # Log progress
                if batch_count % 100 == 0:
                    print(f"Processed {batch_count} batches")
                
                # Clean up batch outputs
                del outputs, predicted_signal

        print(f"Completed inference for {model_name}: processed {batch_count} batches")

        # Now reconstruct 2000bp signals from all collected outputs
        print("Reconstructing 2000bp signals from tiles...")

        # Lists to store processed signals and indices in order
        processed_region_signals = []  # List of 2000bp signals
        processed_region_ids = []      # List of region indices
        processed_tile_ids = []        # List of tile indices
        
        sample_idx = 0  # Global sample index across all batches
        
        # Process all collected outputs in order
        for batch_idx, batch_signals in enumerate(tqdm(all_predicted_signals, desc="Reconstructing")):
            # batch_signals: (batch_size, T, L) where L should be 1000
            batch_size = batch_signals.shape[0]
            
            for i in range(batch_size):
                region_idx = all_region_indices[sample_idx]
                tile_idx = all_tile_indices[sample_idx]
                
                # Extract central 500bp from each 1000bp prediction for this sample
                signal_length = batch_signals.shape[-1]  # Should be 1000
                central_start = (signal_length - 500) // 2  # Should be 250
                central_end = central_start + 500
                
                central_signal = batch_signals[i, :, central_start:central_end]  # (T, 500)
                
                # Store processed results
                processed_region_signals.append(central_signal)
                processed_region_ids.append(region_idx)
                processed_tile_ids.append(tile_idx)
                
                sample_idx += 1

        # Assert ordering: region IDs should be sorted with 4 tiles each
        for i in range(0, len(processed_region_ids), 4):
            # Check that we have 4 consecutive tiles for same region
            region_group = processed_region_ids[i:i+4]
            tile_group = processed_tile_ids[i:i+4]
            
            assert len(region_group) == 4, f"Expected 4 tiles per region, got {len(region_group)}"
            assert all(r == region_group[0] for r in region_group), f"Region IDs not grouped: {region_group}"
            assert tile_group == [0, 1, 2, 3], f"Tile indices not 0,1,2,3: {tile_group}"
            
            # Check region ordering (should be ascending)
            if i > 0:
                prev_region = processed_region_ids[i-4]
                curr_region = region_group[0]
                assert curr_region > prev_region, f"Regions not in ascending order: {prev_region} -> {curr_region}"
        
        print("✓ Tile ordering validated")
        
        # Reconstruct 2000bp signals by combining 4 tiles per region
        print("Combining tiles into 2000bp signals...")
        num_regions = len(processed_region_ids) // 4
        final_predictions = np.zeros((N, config_dict.shift_extended_output_region_len, 1), dtype=np.float32)
        
        for region_i in range(num_regions):
            base_idx = region_i * 4
            region_idx = processed_region_ids[base_idx]
            
            # Combine 4 tiles of 500bp each into 2000bp signal
            region_signal_2000bp = torch.zeros(2000)
            
            for tile_i in range(4):
                tile_signal = processed_region_signals[base_idx + tile_i][0]  # Take first track: (500,)
                start_pos = tile_i * 500
                end_pos = start_pos + 500
                region_signal_2000bp[start_pos:end_pos] = tile_signal
            
            # Store in final array
            final_predictions[region_idx, :, 0] = region_signal_2000bp.numpy()
        
        print(f"Reconstructed {num_regions} regions into 2000bp signals")
        
        # Write to zarr dataset
        print(f"Writing {model_name} predictions to zarr...")
        predicted_dataset[:] = final_predictions
        
        # Update attributes
        root.attrs[f'{model_name}_predicted_track_dataset'] = dataset_name
        
        # Clean up
        del model, predicted_dataset, all_predicted_signals, all_region_indices, all_tile_indices
        
        print(f"✓ Successfully added {model_name} predicted track dataset")
    
    # Close zarr
    del root

    # Clear models
    del models
    torch.cuda.empty_cache()

    print(f"✓ Successfully added all predicted track datasets to {fold_name}")


def main():
    # Configuration
    config_dict = ConfigDict()
    config_dict.input_region_len = 2114
    config_dict.output_region_len = 1000
    config_dict.shift_bp = 500
    config_dict.shift_bp_for_tile = 750
    config_dict.shift_extended_input_region_len = config_dict.input_region_len + 2 * config_dict.shift_bp  # 3114
    config_dict.shift_extended_input_region_for_tile_len = \
        config_dict.input_region_len + 2 * config_dict.shift_bp_for_tile  # 3614
    config_dict.shift_extended_output_region_len = config_dict.output_region_len + 2 * config_dict.shift_bp  # 2000
    config_dict.tile_size = 500

    config_dict.output_dir = os.path.join(
        MANUSCRIPT_EXPERIMENTS_DIR,
        "section_1__ConditionalLM_as_diagnostic_probe",
        "workspace",
        "data",
        "ATAC",
        "GM12878__ENCSR637XSC",
        "preprocessed",
        "zarr_datasets",
    )

    config_dict.data = ConfigDict()
    config_dict.data.fold_names = [f"fold_{i}" for i in range(1)]

    config_dict.model = ConfigDict()
    config_dict.model.model_dir = os.path.join(
        MANUSCRIPT_EXPERIMENTS_DIR,
        "section_1__ConditionalLM_as_diagnostic_probe",
        "workspace",
        "data",
        "ATAC",
        "GM12878__ENCSR637XSC",
        "chrombpnet_annotation",
    )

    config_dict.model.model_filepaths = {
        "bias_scaled": os.path.join(config_dict.model.model_dir, "ENCFF142IOR", "{}", "model.bias_scaled.{}.ENCSR637XSC.h5"),
    }
    
    # Process all folds
    print("Adding predicted track dataset...")
    
    for fold_name in config_dict.data.fold_names:
        create_predicted_track_dataset(fold_name, config_dict)
    
    print(f"✓ Completed adding predicted track dataset")

if __name__ == "__main__":
    main()
