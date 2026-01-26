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


def validate_observed_track_dataset(fold_name: str, config_dict: ConfigDict):
    """Validate observed track datasets by re-extracting from BigWig and comparing with zarr storage"""
    
    zarr_path = os.path.join(config_dict.output_dir, f"nonpeaks.{fold_name}.zarr")
    
    # Setup logging
    print(f"=== Validating observed track datasets for {fold_name} ===")
    
    # Load regions dataframe
    regions_df = pd.read_csv(os.path.join(zarr_path, "_auxiliary", "regions_df.tsv"), sep='\t')
    N = len(regions_df)
    
    print(f"Zarr path: {zarr_path}")
    print(f"Regions: {N}")
    print(f"BigWig file: {config_dict.data.bigwig_file}")
    
    # Check if region_idx matches pandas index
    index_matches_region_idx = (regions_df.index == regions_df['region_idx']).all()
    print(f"DataFrame index matches region_idx column: {index_matches_region_idx}")
    
    # Open BigWig files
    if not os.path.exists(config_dict.data.bigwig_file):
        raise FileNotFoundError(f"BigWig file not found: {config_dict.data.bigwig_file}")
    
    bw = pyBigWig.open(config_dict.data.bigwig_file)
    print("Successfully opened BigWig file")
    
    # Open zarr in read-only mode to load existing datasets
    root = zarr.open(zarr_path, mode="r")
    
    # Dataset names
    dataset_name = "observed_track_summitcentered1000bp_symmetric500bpshift"
    
    # Check if dataset exists
    if dataset_name not in root["observed_tracks"]:
        raise ValueError(f"Dataset {dataset_name} does not exist in zarr file. Run 3_4 first to create it.")
    
    # Load existing dataset
    stored_dataset = root["observed_tracks"][dataset_name]
    print(f"Loaded stored dataset: {dataset_name}")
    print(f"Stored shape: {stored_dataset.shape}")
    print(f"Stored dtype: {stored_dataset.dtype}")
    
    # Pre-allocate track tensors for re-extraction
    reextracted_values = np.empty((N, config_dict.shift_extended_output_region_len, 1), dtype=np.float32)
    
    # Validation statistics
    nan_counts_per_region = []
    regions_with_length_mismatch = 0
    regions_with_no_data = 0
    regions_with_mismatches = 0
    total_mismatched_positions = 0
    max_absolute_diff = 0.0
    
    # Re-extract track values by iterating over regions_df in order
    print("Re-extracting track values and comparing with stored data...")
    for i, row in tqdm(regions_df.iterrows(), total=N, desc="validating regions"):
        region_idx = row['region_idx']
        chrom = row['Chromosome']
        
        # Use output extended coordinates (2000bp centered on summit)
        start = int(row['output_extended_start'])
        end = int(row['output_extended_end'])
        
        # Re-extract from BigWig using same logic as 3_4
        try:
            values = bw.values(chrom, start, end)
            
            if values is None:
                values = np.zeros(config_dict.shift_extended_output_region_len, dtype=np.float32)
                nan_counts_per_region.append(config_dict.shift_extended_output_region_len)
                regions_with_no_data += 1
            else:
                values = np.array(values, dtype=np.float32)
                nan_count = np.sum(np.isnan(values))
                nan_counts_per_region.append(nan_count)
                
                if len(values) != config_dict.shift_extended_output_region_len:
                    regions_with_length_mismatch += 1
                    raise ValueError(f"Length mismatch for {chrom}:{start}-{end}, expected {config_dict.shift_extended_output_region_len}, got {len(values)}")
                
                values = np.nan_to_num(values, nan=0.0)
            
            reextracted_values[region_idx] = values.reshape(-1, 1)
            
        except Exception as e:
            print(f"Error processing region {chrom}:{start}-{end}: {e}")
            reextracted_values[region_idx] = np.zeros((config_dict.shift_extended_output_region_len, 1), dtype=np.float32)
            nan_counts_per_region.append(config_dict.shift_extended_output_region_len)
    
    # Compare re-extracted values with stored values
    print("Comparing re-extracted values with stored values...")
    
    # Load stored data for this region in chunks to avoid memory issues
    chunk_size = 1000  # Process 1000 regions at a time
    all_match = True
    non_integer_values_count = 0
    
    for chunk_start in tqdm(range(0, N, chunk_size), desc="comparing chunks"):
        chunk_end = min(chunk_start + chunk_size, N)
        
        # Get stored and reextracted data for this chunk
        stored_chunk = stored_dataset[chunk_start:chunk_end]
        reextracted_chunk = reextracted_values[chunk_start:chunk_end]
        
        # Check if values are integers (should be whole numbers)
        stored_is_int = np.isclose(stored_chunk, np.round(stored_chunk))
        reextracted_is_int = np.isclose(reextracted_chunk, np.round(reextracted_chunk))
        
        # Count non-integer values
        non_integer_stored = np.sum(~stored_is_int)
        non_integer_reextracted = np.sum(~reextracted_is_int)
        non_integer_values_count += non_integer_stored + non_integer_reextracted
        
        if non_integer_stored > 0 or non_integer_reextracted > 0:
            print(f"Warning: Found non-integer values in chunk {chunk_start}-{chunk_end}")
            print(f"  Non-integer stored values: {non_integer_stored}")
            print(f"  Non-integer reextracted values: {non_integer_reextracted}")
        
        # Convert to integers for exact comparison
        stored_chunk_int = np.round(stored_chunk).astype(np.int32)
        reextracted_chunk_int = np.round(reextracted_chunk).astype(np.int32)
        
        # Compare integer versions for EXACT equality
        exact_match = np.array_equal(stored_chunk_int, reextracted_chunk_int)
        
        if not exact_match:
            all_match = False
            
            # Find specific mismatches in this chunk
            mismatch_mask = (stored_chunk_int != reextracted_chunk_int)
            
            chunk_mismatched_positions = np.sum(mismatch_mask)
            total_mismatched_positions += chunk_mismatched_positions
            
            # Calculate differences on original float values
            diff = np.abs(stored_chunk - reextracted_chunk)
            chunk_max_diff = np.max(diff)
            max_absolute_diff = max(max_absolute_diff, chunk_max_diff)
            
            # Count regions with mismatches in this chunk
            regions_with_mismatches_in_chunk = np.sum(np.any(mismatch_mask, axis=(1, 2)))
            regions_with_mismatches += regions_with_mismatches_in_chunk
            
            # Log details for first few mismatches
            if chunk_start < 5000:  # Only log for first few chunks
                mismatch_regions = np.where(np.any(mismatch_mask, axis=(1, 2)))[0]
                for region_in_chunk in mismatch_regions[:3]:  # Show first 3 mismatches
                    global_region_idx = chunk_start + region_in_chunk
                    region_diff = diff[region_in_chunk]
                    max_diff_in_region = np.max(region_diff)
                    exact_mismatches_in_region = np.sum(mismatch_mask[region_in_chunk])
                    
                    region_row = regions_df.iloc[global_region_idx]
                    print(f"  Mismatch in region {global_region_idx}: {region_row['Chromosome']}:{region_row['output_extended_start']}-{region_row['output_extended_end']}")
                    print(f"    Max diff: {max_diff_in_region:.8f}, Exact integer mismatches: {exact_mismatches_in_region}")
                    
                    # Show a few actual value pairs for debugging
                    mismatch_positions = np.where(mismatch_mask[region_in_chunk])
                    if len(mismatch_positions[0]) > 0:
                        for pos_idx in range(min(3, len(mismatch_positions[0]))):  # Show first 3 mismatched positions
                            pos = mismatch_positions[0][pos_idx]
                            stored_val = stored_chunk[region_in_chunk, pos, 0]
                            reextracted_val = reextracted_chunk[region_in_chunk, pos, 0]
                            stored_int = stored_chunk_int[region_in_chunk, pos, 0]
                            reextracted_int = reextracted_chunk_int[region_in_chunk, pos, 0]
                            print(f"      Position {pos}: stored={stored_val:.6f}({stored_int}), reextracted={reextracted_val:.6f}({reextracted_int})")
    
    # Close handles
    bw.close()
    del root
    
    # Print validation results
    print(f"\n=== VALIDATION RESULTS for {fold_name} ===")
    print(f"Total regions processed: {N:,}")
    print(f"Regions with no data: {regions_with_no_data}")
    print(f"Regions with length mismatch: {regions_with_length_mismatch}")
    print(f"Non-integer values found: {non_integer_values_count}")
    
    if all_match:
        print(f"✅ VALIDATION PASSED: All re-extracted values match stored values perfectly!")
    else:
        print(f"❌ VALIDATION FAILED: Found mismatches between re-extracted and stored values")
        print(f"  Regions with mismatches: {regions_with_mismatches:,}")
        print(f"  Total mismatched positions: {total_mismatched_positions:,}")
        print(f"  Maximum absolute difference: {max_absolute_diff:.8f}")
        
        # Calculate percentage of mismatches
        total_positions = N * config_dict.shift_extended_output_region_len
        mismatch_percentage = (total_mismatched_positions / total_positions) * 100
        print(f"  Percentage of positions with mismatches: {mismatch_percentage:.6f}%")
    
    # Additional statistics
    nan_counts = np.array(nan_counts_per_region)
    regions_all_nan = np.sum(nan_counts == config_dict.shift_extended_output_region_len)
    print(f"\nRe-extraction statistics:")
    print(f"  Regions with ALL NaN values: {regions_all_nan}")
    print(f"  Total NaN positions: {np.sum(nan_counts):,}")
    print(f"  Mean NaN count per region: {np.mean(nan_counts):.2f}")
    print(f"  Median NaN count per region: {np.median(nan_counts):.2f}")
    
    # Summary statistics of re-extracted data
    print(f"\nRe-extracted data statistics:")
    print(f"  Min value: {np.min(reextracted_values):.6f}")
    print(f"  Max value: {np.max(reextracted_values):.6f}")
    print(f"  Mean value: {np.mean(reextracted_values):.6f}")
    print(f"  Non-zero positions: {np.count_nonzero(reextracted_values):,}")
    
    return all_match


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
        "ATAC",
        "K562__ENCSR868FGK",
        "preprocessed",
        "zarr_datasets",
    )

    config_dict.data = ConfigDict()
    config_dict.data.fold_names = [f"fold_{i}" for i in range(5)]
    # BigWig file paths
    config_dict.data.bigwig_file = os.path.join(
        MANUSCRIPT_EXPERIMENTS_DIR,
        "section_1__ConditionalLM_as_diagnostic_probe",
        "workspace",
        "data",
        "ATAC",
        "K562__ENCSR868FGK",
        "chrombpnet_annotation",
        "ENCFF874FUM.bigWig"
    )
    
    # Validate observed track dataset for all folds
    print("Validating observed track datasets...")
    
    all_folds_passed = True
    for fold_name in config_dict.data.fold_names:
        validation_passed = validate_observed_track_dataset(fold_name, config_dict)
        if not validation_passed:
            all_folds_passed = False
            print(f"❌ Validation failed for {fold_name}!")
        else:
            print(f"✅ Validation passed for {fold_name}!")
        print("-" * 50)
    
    if all_folds_passed:
        print(f"✅ All folds validation completed successfully - all data matches!")
    else:
        print(f"❌ Some folds validation failed - mismatches found!")
        sys.exit(1)

if __name__ == "__main__":
    main()
