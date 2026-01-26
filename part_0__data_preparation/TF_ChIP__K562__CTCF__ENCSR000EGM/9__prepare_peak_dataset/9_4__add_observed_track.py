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

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../../.."))
from config import * 


def create_observed_track_dataset(config_dict: ConfigDict):
    """Add observed track datasets to existing zarr file"""
    
    zarr_path = os.path.join(config_dict.output_dir, f"peaks.all_folds.zarr")
    
    # Setup logging
    print(f"=== Adding observed track datasets ===")
    
    # Load regions dataframe
    regions_df = pd.read_csv(os.path.join(zarr_path, "_auxiliary", "regions_df.tsv"), sep='\t')
    N = len(regions_df)
    
    print(f"Zarr path: {zarr_path}")
    print(f"Regions: {N}")
    print(f"Main (+) BigWig file: {config_dict.data.main_plus_bigwig_file}")
    print(f"Main (-) BigWig file: {config_dict.data.main_minus_bigwig_file}")
    print(f"Control (+) BigWig file: {config_dict.data.control_plus_bigwig_file}")
    print(f"Control (-) BigWig file: {config_dict.data.control_minus_bigwig_file}")

    
    # Check if region_idx matches pandas index
    index_matches_region_idx = (regions_df.index == regions_df['region_idx']).all()
    print(f"DataFrame index matches region_idx column: {index_matches_region_idx}")
    
    # Open BigWig files
    if not os.path.exists(config_dict.data.main_plus_bigwig_file):
        raise FileNotFoundError(f"BigWig file not found: {config_dict.data.main_plus_bigwig_file}")
    if not os.path.exists(config_dict.data.main_minus_bigwig_file):
        raise FileNotFoundError(f"BigWig file not found: {config_dict.data.main_minus_bigwig_file}")
    if not os.path.exists(config_dict.data.control_plus_bigwig_file):
        raise FileNotFoundError(f"BigWig file not found: {config_dict.data.control_plus_bigwig_file}")
    if not os.path.exists(config_dict.data.control_minus_bigwig_file):
        raise FileNotFoundError(f"BigWig file not found: {config_dict.data.control_minus_bigwig_file}")
    
    bw_main_plus = pyBigWig.open(config_dict.data.main_plus_bigwig_file)
    bw_main_minus = pyBigWig.open(config_dict.data.main_minus_bigwig_file)
    bw_control_plus = pyBigWig.open(config_dict.data.control_plus_bigwig_file)
    bw_control_minus = pyBigWig.open(config_dict.data.control_minus_bigwig_file)
    print("Successfully opened BigWig files")
    
    # Open zarr in read/write mode to add new datasets
    root = zarr.open(zarr_path, mode="r+")
    
    # Dataset names
    main_dataset_name = "observed_main_track_summitcentered1000bp_symmetric500bpshift"
    control_dataset_name = "observed_control_track_summitcentered1000bp_symmetric500bpshift"
    main_input_dataset_name = "observed_main_track_summitcentered2114bp_symmetric750bpshift"
    control_input_dataset_name = "observed_control_track_summitcentered2114bp_symmetric750bpshift"
    
    # Check if datasets already exist
    if main_dataset_name in root["observed_tracks"]:
        print(f"Dataset {main_dataset_name} already exists, will overwrite")
        del root["observed_tracks"][main_dataset_name]
    if control_dataset_name in root["observed_tracks"]:
        print(f"Dataset {control_dataset_name} already exists, will overwrite")
        del root["observed_tracks"][control_dataset_name]
    if main_input_dataset_name in root["observed_tracks"]:
        print(f"Dataset {main_input_dataset_name} already exists, will overwrite")
        del root["observed_tracks"][main_input_dataset_name]
    if control_input_dataset_name in root["observed_tracks"]:
        print(f"Dataset {control_input_dataset_name} already exists, will overwrite")
        del root["observed_tracks"][control_input_dataset_name]
    
    # Create new datasets with same chunking as DNA output dataset
    existing_dna_output = root["reference_dna"]["dna_summitcentered1000bp_symmetric500bpshift"]
    chunk_len = existing_dna_output.chunks[0]  # Get chunk size from existing dataset
    
    compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=zarr.Blosc.BITSHUFFLE)
    
    main_dataset = root["observed_tracks"].create_dataset(
        main_dataset_name,
        shape=(N, config_dict.shift_extended_output_region_len, 2),  # (N, 2000, 2) +/- track
        chunks=(chunk_len, config_dict.shift_extended_output_region_len, 2),
        dtype="float32",
        compressor=compressor,
    )
    control_dataset = root["observed_tracks"].create_dataset(
        control_dataset_name,
        shape=(N, config_dict.shift_extended_output_region_len, 2),  # (N, 2000, 2) +/- track
        chunks=(chunk_len, config_dict.shift_extended_output_region_len, 2),
        dtype="float32",
        compressor=compressor,
    )
    
    print(f"Created datasets: {main_dataset_name} and {control_dataset_name}")
    print(f"Shape: {main_dataset.shape} and {control_dataset.shape}")
    print(f"Chunks: {main_dataset.chunks} and {control_dataset.chunks}")
    
    # Create input length datasets
    main_input_dataset = root["observed_tracks"].create_dataset(
        main_input_dataset_name,
        shape=(N, config_dict.shift_extended_input_region_for_tile_len, 2),  # (N, 3614, 2) +/- track
        chunks=(chunk_len, config_dict.shift_extended_input_region_for_tile_len, 2),
        dtype="float32",
        compressor=compressor,
    )
    control_input_dataset = root["observed_tracks"].create_dataset(
        control_input_dataset_name,
        shape=(N, config_dict.shift_extended_input_region_for_tile_len, 2),  # (N, 3614, 2) +/- track
        chunks=(chunk_len, config_dict.shift_extended_input_region_for_tile_len, 2),
        dtype="float32",
        compressor=compressor,
    )
    
    print(f"Created input datasets: {main_input_dataset_name} and {control_input_dataset_name}")
    print(f"Input shape: {main_input_dataset.shape} and {control_input_dataset.shape}")
    
    # Pre-allocate track tensors (corrected shapes with 2 channels)
    main_track_values = np.empty((N, config_dict.shift_extended_output_region_len, 2), dtype=np.float32)
    control_track_values = np.empty((N, config_dict.shift_extended_output_region_len, 2), dtype=np.float32)
    main_input_track_values = np.empty((N, config_dict.shift_extended_input_region_for_tile_len, 2), dtype=np.float32)
    control_input_track_values = np.empty((N, config_dict.shift_extended_input_region_for_tile_len, 2), dtype=np.float32)
    
    # NaN analysis storage for both datasets
    main_nan_counts_per_region = []
    main_regions_with_length_mismatch = 0
    main_regions_with_no_data = 0
    control_nan_counts_per_region = []
    control_regions_with_length_mismatch = 0
    control_regions_with_no_data = 0
    
    # Extract track values by iterating over regions_df in order
    print("Extracting track values by iterating over regions_df...")
    for i, row in tqdm(regions_df.iterrows(), total=N, desc="regions"):
        region_idx = row['region_idx']
        chrom = row['Chromosome']
        
        # Use output extended coordinates (2000bp centered on summit)
        start = int(row['output_extended_start'])
        end = int(row['output_extended_end'])
        
        # Process main BigWigs
        try:
            main_plus_values = bw_main_plus.values(chrom, start, end)
            main_minus_values = bw_main_minus.values(chrom, start, end)
            
            # Handle main track
            if main_plus_values is None:
                main_plus_values = np.zeros(config_dict.shift_extended_output_region_len, dtype=np.float32)
                main_regions_with_no_data += 1
            else:
                main_plus_values = np.array(main_plus_values, dtype=np.float32)
                if len(main_plus_values) != config_dict.shift_extended_output_region_len:
                    main_regions_with_length_mismatch += 1
                    raise ValueError(f"Main plus length mismatch for {chrom}:{start}-{end}, expected {config_dict.shift_extended_output_region_len}, got {len(main_plus_values)}")
                main_plus_values = np.nan_to_num(main_plus_values, nan=0.0)
                
            if main_minus_values is None:
                main_minus_values = np.zeros(config_dict.shift_extended_output_region_len, dtype=np.float32)
            else:
                main_minus_values = np.array(main_minus_values, dtype=np.float32)
                if len(main_minus_values) != config_dict.shift_extended_output_region_len:
                    main_regions_with_length_mismatch += 1
                    raise ValueError(f"Main minus length mismatch for {chrom}:{start}-{end}, expected {config_dict.shift_extended_output_region_len}, got {len(main_minus_values)}")
                main_minus_values = np.nan_to_num(main_minus_values, nan=0.0)
            
            # Stack + and - channels for main track
            main_track_values[region_idx, :, 0] = main_plus_values  # + channel
            main_track_values[region_idx, :, 1] = main_minus_values  # - channel
            
            # Count NaNs for main track
            main_nan_count = np.sum(np.isnan(main_plus_values)) + np.sum(np.isnan(main_minus_values))
            main_nan_counts_per_region.append(main_nan_count)
            
        except Exception as e:
            print(f"Error processing main track for region {chrom}:{start}-{end}: {e}")
            main_track_values[region_idx, :, :] = 0.0
            main_nan_counts_per_region.append(config_dict.shift_extended_output_region_len * 2)
        
        # Process control BigWigs
        try:
            control_plus_values = bw_control_plus.values(chrom, start, end)
            control_minus_values = bw_control_minus.values(chrom, start, end)
            
            # Handle control track
            if control_plus_values is None:
                control_plus_values = np.zeros(config_dict.shift_extended_output_region_len, dtype=np.float32)
                control_regions_with_no_data += 1
            else:
                control_plus_values = np.array(control_plus_values, dtype=np.float32)
                if len(control_plus_values) != config_dict.shift_extended_output_region_len:
                    control_regions_with_length_mismatch += 1
                    raise ValueError(f"Control plus length mismatch for {chrom}:{start}-{end}, expected {config_dict.shift_extended_output_region_len}, got {len(control_plus_values)}")
                control_plus_values = np.nan_to_num(control_plus_values, nan=0.0)
                
            if control_minus_values is None:
                control_minus_values = np.zeros(config_dict.shift_extended_output_region_len, dtype=np.float32)
            else:
                control_minus_values = np.array(control_minus_values, dtype=np.float32)
                if len(control_minus_values) != config_dict.shift_extended_output_region_len:
                    control_regions_with_length_mismatch += 1
                    raise ValueError(f"Control minus length mismatch for {chrom}:{start}-{end}, expected {config_dict.shift_extended_output_region_len}, got {len(control_minus_values)}")
                control_minus_values = np.nan_to_num(control_minus_values, nan=0.0)
            
            # Stack + and - channels for control track
            control_track_values[region_idx, :, 0] = control_plus_values  # + channel
            control_track_values[region_idx, :, 1] = control_minus_values  # - channel
            
            # Count NaNs for control track
            control_nan_count = np.sum(np.isnan(control_plus_values)) + np.sum(np.isnan(control_minus_values))
            control_nan_counts_per_region.append(control_nan_count)
            
        except Exception as e:
            print(f"Error processing control track for region {chrom}:{start}-{end}: {e}")
            control_track_values[region_idx, :, :] = 0.0
            control_nan_counts_per_region.append(config_dict.shift_extended_output_region_len * 2)
        
        # Process input length tracks (3614bp)
        input_start = int(row['input_extended_start_for_tile'])
        input_end = int(row['input_extended_end_for_tile'])
        
        # Main input track
        try:
            main_plus_input = bw_main_plus.values(chrom, input_start, input_end)
            main_minus_input = bw_main_minus.values(chrom, input_start, input_end)
            
            if main_plus_input is None:
                main_plus_input = np.zeros(config_dict.shift_extended_input_region_for_tile_len, dtype=np.float32)
            else:
                main_plus_input = np.nan_to_num(np.array(main_plus_input, dtype=np.float32), nan=0.0)
                
            if main_minus_input is None:
                main_minus_input = np.zeros(config_dict.shift_extended_input_region_for_tile_len, dtype=np.float32)
            else:
                main_minus_input = np.nan_to_num(np.array(main_minus_input, dtype=np.float32), nan=0.0)
            
            main_input_track_values[region_idx, :, 0] = main_plus_input
            main_input_track_values[region_idx, :, 1] = main_minus_input
        except:
            main_input_track_values[region_idx, :, :] = 0.0
        
        # Control input track  
        try:
            control_plus_input = bw_control_plus.values(chrom, input_start, input_end)
            control_minus_input = bw_control_minus.values(chrom, input_start, input_end)
            
            if control_plus_input is None:
                control_plus_input = np.zeros(config_dict.shift_extended_input_region_for_tile_len, dtype=np.float32)
            else:
                control_plus_input = np.nan_to_num(np.array(control_plus_input, dtype=np.float32), nan=0.0)
                
            if control_minus_input is None:
                control_minus_input = np.zeros(config_dict.shift_extended_input_region_for_tile_len, dtype=np.float32)
            else:
                control_minus_input = np.nan_to_num(np.array(control_minus_input, dtype=np.float32), nan=0.0)
            
            control_input_track_values[region_idx, :, 0] = control_plus_input
            control_input_track_values[region_idx, :, 1] = control_minus_input
        except:
            control_input_track_values[region_idx, :, :] = 0.0
    
    # Write all data to zarr datasets at once
    print("Writing track values to zarr datasets...")
    main_dataset[:] = main_track_values
    control_dataset[:] = control_track_values
    main_input_dataset[:] = main_input_track_values
    control_input_dataset[:] = control_input_track_values
    
    # Update zarr attributes
    root.attrs['observed_main_track_dataset'] = main_dataset_name
    root.attrs['observed_control_track_dataset'] = control_dataset_name
    root.attrs['main_plus_bigwig_source'] = config_dict.data.main_plus_bigwig_file
    root.attrs['main_minus_bigwig_source'] = config_dict.data.main_minus_bigwig_file
    root.attrs['control_plus_bigwig_source'] = config_dict.data.control_plus_bigwig_file
    root.attrs['control_minus_bigwig_source'] = config_dict.data.control_minus_bigwig_file
    root.attrs['track_dtype'] = 'float32'
    
    # Close handles
    bw_main_plus.close()
    bw_main_minus.close()
    bw_control_plus.close()
    bw_control_minus.close()
    del root
    
    # Convert to numpy arrays for analysis
    main_nan_counts = np.array(main_nan_counts_per_region)
    control_nan_counts = np.array(control_nan_counts_per_region)
    
    # Analysis for main track
    print("=== Main Track Analysis ===")
    print(f"  Regions with no data: {main_regions_with_no_data}")
    print(f"  Regions with length mismatch: {main_regions_with_length_mismatch}")
    main_regions_all_nan = np.sum(main_nan_counts == config_dict.shift_extended_output_region_len * 2)
    print(f"  Regions with ALL NaN values: {main_regions_all_nan}")
    print(f"  Total NaN positions: {np.sum(main_nan_counts):,}")
    print(f"  Mean NaN count per region: {np.mean(main_nan_counts):.2f}")
    print(f"  Median NaN count per region: {np.median(main_nan_counts):.2f}")
    
    # Analysis for control track
    print("=== Control Track Analysis ===")
    print(f"  Regions with no data: {control_regions_with_no_data}")
    print(f"  Regions with length mismatch: {control_regions_with_length_mismatch}")
    control_regions_all_nan = np.sum(control_nan_counts == config_dict.shift_extended_output_region_len * 2)
    print(f"  Regions with ALL NaN values: {control_regions_all_nan}")
    print(f"  Total NaN positions: {np.sum(control_nan_counts):,}")
    print(f"  Mean NaN count per region: {np.mean(control_nan_counts):.2f}")
    print(f"  Median NaN count per region: {np.median(control_nan_counts):.2f}")

    print(f"✓ Successfully added observed track datasets")
    
    # Validate the new datasets
    print("Validating new datasets...")
    validation_root = zarr.open(zarr_path, mode="r")
    
    main_track_data = validation_root["observed_tracks"][main_dataset_name]
    control_track_data = validation_root["observed_tracks"][control_dataset_name]
    
    print(f"Main dataset shape: {main_track_data.shape}")
    print(f"Main data type: {main_track_data.dtype}")
    print(f"Main min value: {np.min(main_track_data[:]):.6f}")
    print(f"Main max value: {np.max(main_track_data[:]):.6f}")
    print(f"Main mean value: {np.mean(main_track_data[:]):.6f}")
    print(f"Main non-zero positions: {np.count_nonzero(main_track_data[:]):,}")
    
    print(f"Control dataset shape: {control_track_data.shape}")
    print(f"Control data type: {control_track_data.dtype}")
    print(f"Control min value: {np.min(control_track_data[:]):.6f}")
    print(f"Control max value: {np.max(control_track_data[:]):.6f}")
    print(f"Control mean value: {np.mean(control_track_data[:]):.6f}")
    print(f"Control non-zero positions: {np.count_nonzero(control_track_data[:]):,}")
    
    main_input_track_data = validation_root["observed_tracks"][main_input_dataset_name]
    control_input_track_data = validation_root["observed_tracks"][control_input_dataset_name]
    
    print(f"Main input dataset shape: {main_input_track_data.shape}")
    print(f"Main input data type: {main_input_track_data.dtype}")
    print(f"Main input min value: {np.min(main_input_track_data[:]):.6f}")
    print(f"Main input max value: {np.max(main_input_track_data[:]):.6f}")
    print(f"Main input mean value: {np.mean(main_input_track_data[:]):.6f}")
    print(f"Main input non-zero positions: {np.count_nonzero(main_input_track_data[:]):,}")
    
    print(f"Control input dataset shape: {control_input_track_data.shape}")
    print(f"Control input data type: {control_input_track_data.dtype}")
    print(f"Control input min value: {np.min(control_input_track_data[:]):.6f}")
    print(f"Control input max value: {np.max(control_input_track_data[:]):.6f}")
    print(f"Control input mean value: {np.mean(control_input_track_data[:]):.6f}")
    print(f"Control input non-zero positions: {np.count_nonzero(control_input_track_data[:]):,}")
    
    del validation_root
    
    return True


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

    config_dict.output_dir = os.path.join(
        MANUSCRIPT_EXPERIMENTS_DIR,
        "section_1__ConditionalLM_as_diagnostic_probe",
        "workspace",
        "data",
        "TF_ChIP",
        "K562__CTCF__ENCSR000EGM",
        "preprocessed",
        "zarr_datasets",
    )

    config_dict.data = ConfigDict()
    # BigWig file paths (fixed missing commas)
    config_dict.data.main_plus_bigwig_file = os.path.join(
        MANUSCRIPT_EXPERIMENTS_DIR,
        "section_1__ConditionalLM_as_diagnostic_probe",
        "workspace",
        "data",
        "TF_ChIP",
        "K562__CTCF__ENCSR000EGM",
        "preprocessed",
        "main_experiment.plus.bw"
    )
    config_dict.data.main_minus_bigwig_file = os.path.join(
        MANUSCRIPT_EXPERIMENTS_DIR,
        "section_1__ConditionalLM_as_diagnostic_probe",
        "workspace",
        "data",
        "TF_ChIP",
        "K562__CTCF__ENCSR000EGM",
        "preprocessed",
        "main_experiment.minus.bw"
    )
    config_dict.data.control_plus_bigwig_file = os.path.join(
        MANUSCRIPT_EXPERIMENTS_DIR,
        "section_1__ConditionalLM_as_diagnostic_probe",
        "workspace",
        "data",
        "TF_ChIP",
        "K562__CTCF__ENCSR000EGM",
        "preprocessed",
        "control_experiment.plus.bw"
    )
    config_dict.data.control_minus_bigwig_file = os.path.join(
        MANUSCRIPT_EXPERIMENTS_DIR,
        "section_1__ConditionalLM_as_diagnostic_probe",
        "workspace",
        "data",
        "TF_ChIP",
        "K562__CTCF__ENCSR000EGM",
        "preprocessed",
        "control_experiment.minus.bw"
    )
    
    # Process all folds
    print("Adding observed track dataset...")
    
    create_observed_track_dataset(config_dict)
    
    print(f"✓ Completed adding observed track dataset")

if __name__ == "__main__":
    main()
