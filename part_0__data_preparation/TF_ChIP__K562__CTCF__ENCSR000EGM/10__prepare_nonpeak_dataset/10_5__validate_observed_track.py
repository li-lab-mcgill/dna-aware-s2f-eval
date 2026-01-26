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


def validate_observed_track_dataset(config_dict: ConfigDict, fold_name: str):
    """Validate observed track datasets by re-extracting from BigWig and comparing with zarr storage"""
    
    zarr_path = os.path.join(config_dict.output_dir, f"nonpeaks.{fold_name}.zarr")
    
    # Setup logging
    print(f"=== Validating observed track datasets for {fold_name} ===")
    
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
    
    # Open zarr in read-only mode to load existing datasets
    root = zarr.open(zarr_path, mode="r")
    
    # Dataset names
    main_dataset_name = "observed_main_track_summitcentered1000bp_symmetric500bpshift"
    control_dataset_name = "observed_control_track_summitcentered1000bp_symmetric500bpshift"
    
    # Check if datasets exist
    if main_dataset_name not in root["observed_tracks"]:
        raise ValueError(f"Dataset {main_dataset_name} does not exist in zarr file. Run 11_4 first to create it.")
    if control_dataset_name not in root["observed_tracks"]:
        raise ValueError(f"Dataset {control_dataset_name} does not exist in zarr file. Run 11_4 first to create it.")
    
    # Load existing datasets
    stored_main_dataset = root["observed_tracks"][main_dataset_name]
    stored_control_dataset = root["observed_tracks"][control_dataset_name]
    print(f"Loaded stored datasets: {main_dataset_name} and {control_dataset_name}")
    print(f"Main stored shape: {stored_main_dataset.shape}")
    print(f"Main stored dtype: {stored_main_dataset.dtype}")
    print(f"Control stored shape: {stored_control_dataset.shape}")
    print(f"Control stored dtype: {stored_control_dataset.dtype}")
    
    # Pre-allocate track tensors for re-extraction
    reextracted_main_values = np.empty((N, config_dict.shift_extended_output_region_len, 2), dtype=np.float32)
    reextracted_control_values = np.empty((N, config_dict.shift_extended_output_region_len, 2), dtype=np.float32)
    
    # Validation statistics for main track
    main_nan_counts_per_region = []
    main_regions_with_length_mismatch = 0
    main_regions_with_no_data = 0
    main_regions_with_mismatches = 0
    main_total_mismatched_positions = 0
    main_max_absolute_diff = 0.0
    
    # Validation statistics for control track
    control_nan_counts_per_region = []
    control_regions_with_length_mismatch = 0
    control_regions_with_no_data = 0
    control_regions_with_mismatches = 0
    control_total_mismatched_positions = 0
    control_max_absolute_diff = 0.0
    
    # Re-extract track values by iterating over regions_df in order
    print("Re-extracting track values and comparing with stored data...")
    for i, row in tqdm(regions_df.iterrows(), total=N, desc="validating regions"):
        region_idx = row['region_idx']
        chrom = row['Chromosome']
        
        # Use output extended coordinates (2000bp centered on summit)
        start = int(row['output_extended_start'])
        end = int(row['output_extended_end'])
        
        # Re-extract main track from BigWig using same logic as 11_4
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
            reextracted_main_values[region_idx, :, 0] = main_plus_values  # + channel
            reextracted_main_values[region_idx, :, 1] = main_minus_values  # - channel
            
            # Count NaNs for main track
            main_nan_count = np.sum(np.isnan(main_plus_values)) + np.sum(np.isnan(main_minus_values))
            main_nan_counts_per_region.append(main_nan_count)
            
        except Exception as e:
            print(f"Error processing main track for region {chrom}:{start}-{end}: {e}")
            reextracted_main_values[region_idx, :, :] = 0.0
            main_nan_counts_per_region.append(config_dict.shift_extended_output_region_len * 2)
        
        # Re-extract control track from BigWig using same logic as 11_4
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
            reextracted_control_values[region_idx, :, 0] = control_plus_values  # + channel
            reextracted_control_values[region_idx, :, 1] = control_minus_values  # - channel
            
            # Count NaNs for control track
            control_nan_count = np.sum(np.isnan(control_plus_values)) + np.sum(np.isnan(control_minus_values))
            control_nan_counts_per_region.append(control_nan_count)
            
        except Exception as e:
            print(f"Error processing control track for region {chrom}:{start}-{end}: {e}")
            reextracted_control_values[region_idx, :, :] = 0.0
            control_nan_counts_per_region.append(config_dict.shift_extended_output_region_len * 2)
    
    # Compare re-extracted values with stored values
    print("Comparing re-extracted values with stored values...")
    
    # Load stored data for this region in chunks to avoid memory issues
    chunk_size = 1000  # Process 1000 regions at a time
    main_all_match = True
    control_all_match = True
    main_non_integer_values_count = 0
    control_non_integer_values_count = 0
    
    print("=== Validating Main Track ===")
    for chunk_start in tqdm(range(0, N, chunk_size), desc="comparing main track chunks"):
        chunk_end = min(chunk_start + chunk_size, N)
        
        # Get stored and reextracted data for this chunk
        stored_main_chunk = stored_main_dataset[chunk_start:chunk_end]
        reextracted_main_chunk = reextracted_main_values[chunk_start:chunk_end]
        
        # Check if values are integers (should be whole numbers)
        stored_main_is_int = np.isclose(stored_main_chunk, np.round(stored_main_chunk))
        reextracted_main_is_int = np.isclose(reextracted_main_chunk, np.round(reextracted_main_chunk))
        
        # Count non-integer values
        non_integer_stored_main = np.sum(~stored_main_is_int)
        non_integer_reextracted_main = np.sum(~reextracted_main_is_int)
        main_non_integer_values_count += non_integer_stored_main + non_integer_reextracted_main
        
        if non_integer_stored_main > 0 or non_integer_reextracted_main > 0:
            print(f"Warning: Found non-integer values in main track chunk {chunk_start}-{chunk_end}")
            print(f"  Non-integer stored values: {non_integer_stored_main}")
            print(f"  Non-integer reextracted values: {non_integer_reextracted_main}")
        
        # Convert to integers for exact comparison
        stored_main_chunk_int = np.round(stored_main_chunk).astype(np.int32)
        reextracted_main_chunk_int = np.round(reextracted_main_chunk).astype(np.int32)
        
        # Compare integer versions for EXACT equality
        exact_main_match = np.array_equal(stored_main_chunk_int, reextracted_main_chunk_int)
        
        if not exact_main_match:
            main_all_match = False
            
            # Find specific mismatches in this chunk
            main_mismatch_mask = (stored_main_chunk_int != reextracted_main_chunk_int)
            
            chunk_main_mismatched_positions = np.sum(main_mismatch_mask)
            main_total_mismatched_positions += chunk_main_mismatched_positions
            
            # Calculate differences on original float values
            main_diff = np.abs(stored_main_chunk - reextracted_main_chunk)
            chunk_main_max_diff = np.max(main_diff)
            main_max_absolute_diff = max(main_max_absolute_diff, chunk_main_max_diff)
            
            # Count regions with mismatches in this chunk
            main_regions_with_mismatches_in_chunk = np.sum(np.any(main_mismatch_mask, axis=(1, 2)))
            main_regions_with_mismatches += main_regions_with_mismatches_in_chunk
            
            # Log details for first few mismatches
            if chunk_start < 5000:  # Only log for first few chunks
                main_mismatch_regions = np.where(np.any(main_mismatch_mask, axis=(1, 2)))[0]
                for region_in_chunk in main_mismatch_regions[:2]:  # Show first 2 mismatches
                    global_region_idx = chunk_start + region_in_chunk
                    region_main_diff = main_diff[region_in_chunk]
                    max_main_diff_in_region = np.max(region_main_diff)
                    exact_main_mismatches_in_region = np.sum(main_mismatch_mask[region_in_chunk])
                    
                    region_row = regions_df.iloc[global_region_idx]
                    print(f"  Main mismatch in region {global_region_idx}: {region_row['Chromosome']}:{region_row['output_extended_start']}-{region_row['output_extended_end']}")
                    print(f"    Max diff: {max_main_diff_in_region:.8f}, Exact integer mismatches: {exact_main_mismatches_in_region}")
    
    print("=== Validating Control Track ===")
    for chunk_start in tqdm(range(0, N, chunk_size), desc="comparing control track chunks"):
        chunk_end = min(chunk_start + chunk_size, N)
        
        # Get stored and reextracted data for this chunk
        stored_control_chunk = stored_control_dataset[chunk_start:chunk_end]
        reextracted_control_chunk = reextracted_control_values[chunk_start:chunk_end]
        
        # Check if values are integers (should be whole numbers)
        stored_control_is_int = np.isclose(stored_control_chunk, np.round(stored_control_chunk))
        reextracted_control_is_int = np.isclose(reextracted_control_chunk, np.round(reextracted_control_chunk))
        
        # Count non-integer values
        non_integer_stored_control = np.sum(~stored_control_is_int)
        non_integer_reextracted_control = np.sum(~reextracted_control_is_int)
        control_non_integer_values_count += non_integer_stored_control + non_integer_reextracted_control
        
        if non_integer_stored_control > 0 or non_integer_reextracted_control > 0:
            print(f"Warning: Found non-integer values in control track chunk {chunk_start}-{chunk_end}")
            print(f"  Non-integer stored values: {non_integer_stored_control}")
            print(f"  Non-integer reextracted values: {non_integer_reextracted_control}")
        
        # Convert to integers for exact comparison
        stored_control_chunk_int = np.round(stored_control_chunk).astype(np.int32)
        reextracted_control_chunk_int = np.round(reextracted_control_chunk).astype(np.int32)
        
        # Compare integer versions for EXACT equality
        exact_control_match = np.array_equal(stored_control_chunk_int, reextracted_control_chunk_int)
        
        if not exact_control_match:
            control_all_match = False
            
            # Find specific mismatches in this chunk
            control_mismatch_mask = (stored_control_chunk_int != reextracted_control_chunk_int)
            
            chunk_control_mismatched_positions = np.sum(control_mismatch_mask)
            control_total_mismatched_positions += chunk_control_mismatched_positions
            
            # Calculate differences on original float values
            control_diff = np.abs(stored_control_chunk - reextracted_control_chunk)
            chunk_control_max_diff = np.max(control_diff)
            control_max_absolute_diff = max(control_max_absolute_diff, chunk_control_max_diff)
            
            # Count regions with mismatches in this chunk
            control_regions_with_mismatches_in_chunk = np.sum(np.any(control_mismatch_mask, axis=(1, 2)))
            control_regions_with_mismatches += control_regions_with_mismatches_in_chunk
            
            # Log details for first few mismatches
            if chunk_start < 5000:  # Only log for first few chunks
                control_mismatch_regions = np.where(np.any(control_mismatch_mask, axis=(1, 2)))[0]
                for region_in_chunk in control_mismatch_regions[:2]:  # Show first 2 mismatches
                    global_region_idx = chunk_start + region_in_chunk
                    region_control_diff = control_diff[region_in_chunk]
                    max_control_diff_in_region = np.max(region_control_diff)
                    exact_control_mismatches_in_region = np.sum(control_mismatch_mask[region_in_chunk])
                    
                    region_row = regions_df.iloc[global_region_idx]
                    print(f"  Control mismatch in region {global_region_idx}: {region_row['Chromosome']}:{region_row['output_extended_start']}-{region_row['output_extended_end']}")
                    print(f"    Max diff: {max_control_diff_in_region:.8f}, Exact integer mismatches: {exact_control_mismatches_in_region}")
    
    # Close handles
    bw_main_plus.close()
    bw_main_minus.close()
    bw_control_plus.close()
    bw_control_minus.close()
    del root
    
    # Print validation results
    print(f"\n=== VALIDATION RESULTS for {fold_name} ===")
    print(f"Total regions processed: {N:,}")
    
    # Main track results
    print(f"\n=== Main Track Results ===")
    print(f"Regions with no data: {main_regions_with_no_data}")
    print(f"Regions with length mismatch: {main_regions_with_length_mismatch}")
    
    if main_all_match:
        print(f"✅ MAIN TRACK VALIDATION PASSED: All re-extracted values match stored values perfectly!")
    else:
        print(f"❌ MAIN TRACK VALIDATION FAILED: Found mismatches between re-extracted and stored values")
        print(f"  Regions with mismatches: {main_regions_with_mismatches:,}")
        print(f"  Total mismatched positions: {main_total_mismatched_positions:,}")
        print(f"  Maximum absolute difference: {main_max_absolute_diff:.8f}")
        
        # Calculate percentage of mismatches
        total_main_positions = N * config_dict.shift_extended_output_region_len * 2  # 2 channels
        main_mismatch_percentage = (main_total_mismatched_positions / total_main_positions) * 100
        print(f"  Percentage of positions with mismatches: {main_mismatch_percentage:.6f}%")
    
    # Control track results
    print(f"\n=== Control Track Results ===")
    print(f"Regions with no data: {control_regions_with_no_data}")
    print(f"Regions with length mismatch: {control_regions_with_length_mismatch}")
    
    if control_all_match:
        print(f"✅ CONTROL TRACK VALIDATION PASSED: All re-extracted values match stored values perfectly!")
    else:
        print(f"❌ CONTROL TRACK VALIDATION FAILED: Found mismatches between re-extracted and stored values")
        print(f"  Regions with mismatches: {control_regions_with_mismatches:,}")
        print(f"  Total mismatched positions: {control_total_mismatched_positions:,}")
        print(f"  Maximum absolute difference: {control_max_absolute_diff:.8f}")
        
        # Calculate percentage of mismatches
        total_control_positions = N * config_dict.shift_extended_output_region_len * 2  # 2 channels
        control_mismatch_percentage = (control_total_mismatched_positions / total_control_positions) * 100
        print(f"  Percentage of positions with mismatches: {control_mismatch_percentage:.6f}%")
    
    # Additional statistics
    main_nan_counts = np.array(main_nan_counts_per_region)
    control_nan_counts = np.array(control_nan_counts_per_region)
    
    main_regions_all_nan = np.sum(main_nan_counts == config_dict.shift_extended_output_region_len * 2)
    control_regions_all_nan = np.sum(control_nan_counts == config_dict.shift_extended_output_region_len * 2)
    
    print(f"\nMain track re-extraction statistics:")
    print(f"  Regions with ALL NaN values: {main_regions_all_nan}")
    print(f"  Total NaN positions: {np.sum(main_nan_counts):,}")
    print(f"  Mean NaN count per region: {np.mean(main_nan_counts):.2f}")
    print(f"  Median NaN count per region: {np.median(main_nan_counts):.2f}")
    
    print(f"\nControl track re-extraction statistics:")
    print(f"  Regions with ALL NaN values: {control_regions_all_nan}")
    print(f"  Total NaN positions: {np.sum(control_nan_counts):,}")
    print(f"  Mean NaN count per region: {np.mean(control_nan_counts):.2f}")
    print(f"  Median NaN count per region: {np.median(control_nan_counts):.2f}")
    
    # Summary statistics of re-extracted data
    print(f"\nMain track re-extracted data statistics:")
    print(f"  Min value: {np.min(reextracted_main_values):.6f}")
    print(f"  Max value: {np.max(reextracted_main_values):.6f}")
    print(f"  Mean value: {np.mean(reextracted_main_values):.6f}")
    print(f"  Non-zero positions: {np.count_nonzero(reextracted_main_values):,}")
    
    print(f"\nControl track re-extracted data statistics:")
    print(f"  Min value: {np.min(reextracted_control_values):.6f}")
    print(f"  Max value: {np.max(reextracted_control_values):.6f}")
    print(f"  Mean value: {np.mean(reextracted_control_values):.6f}")
    print(f"  Non-zero positions: {np.count_nonzero(reextracted_control_values):,}")
    
    return main_all_match and control_all_match


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
    # BigWig file paths (same as 11_4)
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
    
    # Process each fold
    fold_names = [f"fold_{i}" for i in range(5)]
    all_validation_passed = True
    
    for fold_name in fold_names:
        print(f"\n=== Validating {fold_name} ===")
        
        # Check if zarr file exists
        zarr_path = os.path.join(config_dict.output_dir, f"nonpeaks.{fold_name}.zarr")
        if not os.path.exists(zarr_path):
            print(f"⚠️  Zarr file not found: {zarr_path}, skipping...")
            continue
            
        validation_passed = validate_observed_track_dataset(config_dict, fold_name)
        if not validation_passed:
            all_validation_passed = False
            
        print(f"✓ Completed validation for {fold_name}")
    
    if all_validation_passed:
        print(f"✅ All fold validations completed successfully - all data matches!")
    else:
        print(f"❌ Some fold validations failed - mismatches found!")
        sys.exit(1)

if __name__ == "__main__":
    main()
