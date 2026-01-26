import os
import sys
import numpy as np
import pandas as pd
import zarr
import pyBigWig
import pyfaidx
from tqdm import tqdm
from ml_collections import ConfigDict

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../../.."))
from config import * 
from utils.data.zarr_preparation.extract_windows import dna_one_hot

def validate_fold_data(fold_name: str, config_dict: ConfigDict):
    """Validate both DNA sequences and observed tracks for one fold by comparing zarr data with fresh extraction"""
    print(f"\n=== Validating {fold_name} ===")
    
    zarr_path = os.path.join(config_dict.output_dir, f"nonpeaks.{fold_name}.zarr")
    
    # Load regions dataframe
    regions_df = pd.read_csv(os.path.join(zarr_path, "_auxiliary", "regions_df.tsv"), sep='\t')
    N = len(regions_df)
    
    print(f"Zarr path: {zarr_path}")
    print(f"Regions: {N}")
    print(f"BigWig file: {config_dict.data.bigwig_file}")
    print(f"FASTA file: {config_dict.data.fasta_file}")
    
    # Display the regions_df structure for verification
    print(f"Regions DataFrame columns: {list(regions_df.columns)}")
    print(f"First few rows of regions_df:")
    print(regions_df.head(3).to_string())
    
    # Check if region_idx matches pandas index
    index_matches_region_idx = (regions_df.index == regions_df['region_idx']).all()
    print(f"DataFrame index matches region_idx column: {index_matches_region_idx}")
    
    # Load zarr data (stored data)
    zarr_store = zarr.open(zarr_path, mode="r")
    
    # Load stored datasets from the grouped structure
    stored_dna_input = zarr_store["reference_dna"]["dna_summitcentered2114bp_symmetric500bpshift"][:]
    stored_dna_input_750 = zarr_store["reference_dna"]["dna_summitcentered2114bp_symmetric750bpshift"][:]
    stored_dna_output = zarr_store["reference_dna"]["dna_summitcentered1000bp_symmetric500bpshift"][:]
    stored_observed_track = zarr_store["observed_tracks"]["observed_track_summitcentered1000bp_symmetric500bpshift"][:]
    
    print(f"Stored DNA input (500bp shift) shape: {stored_dna_input.shape}")
    print(f"Stored DNA input (750bp shift) shape: {stored_dna_input_750.shape}")
    print(f"Stored DNA output shape: {stored_dna_output.shape}")
    print(f"Stored observed track shape: {stored_observed_track.shape}")
    
    # Open genome and cache all chromosomes
    genome = pyfaidx.Fasta(config_dict.data.fasta_file, as_raw=True, sequence_always_upper=True)
    
    print("Caching chromosome sequences...")
    chrom_cache = {}
    unique_chroms = regions_df['Chromosome'].unique()
    for chrom in tqdm(unique_chroms, desc="chromosomes"):
        chrom_cache[chrom] = str(genome[chrom])
    
    # Open BigWig file
    if not os.path.exists(config_dict.data.bigwig_file):
        raise FileNotFoundError(f"BigWig file not found: {config_dict.data.bigwig_file}")
    
    bw = pyBigWig.open(config_dict.data.bigwig_file)
    print("Successfully opened BigWig file")
    
    # Pre-allocate validation tensors
    validation_dna_input = np.empty((N, config_dict.shift_extended_input_region_len, 4), dtype=np.uint8)
    validation_dna_input_750 = np.empty((N, config_dict.shift_extended_input_region_for_tile_len, 4), dtype=np.uint8)
    validation_dna_output = np.empty((N, config_dict.shift_extended_output_region_len, 4), dtype=np.uint8)
    validation_observed_track = np.empty((N, config_dict.shift_extended_output_region_len, 1), dtype=np.float32)
    
    # Validation statistics
    dna_input_mismatches = 0
    dna_input_750_mismatches = 0
    dna_output_mismatches = 0
    observed_track_mismatches = 0
    
    track_nan_counts = []
    
    # Extract sequences and tracks by iterating over regions_df in order
    print("Extracting validation data by iterating over regions_df...")
    for i, row in tqdm(regions_df.iterrows(), total=N, desc="regions"):
        region_idx = row['region_idx']
        chrom = row['Chromosome']
        
        # Get cached chromosome sequence
        chrom_dna = chrom_cache[chrom]
        
        # ===== DNA INPUT VALIDATION (500bp shift) =====
        # Use the pre-calculated coordinates from regions_df
        input_start = int(row['input_extended_start'])
        input_end = int(row['input_extended_end'])
        input_seq = chrom_dna[input_start:input_end]
        
        # Validate length
        expected_len = config_dict.shift_extended_input_region_len
        assert len(input_seq) == expected_len, \
            f"Input sequence length {len(input_seq)} != {expected_len} for region {region_idx}"
        
        # One-hot encode and store
        validation_dna_input[region_idx] = dna_one_hot(input_seq)
        
        # ===== DNA INPUT VALIDATION (750bp shift) =====
        # Use the pre-calculated coordinates from regions_df
        input_start_750 = int(row['input_extended_start_for_tile'])
        input_end_750 = int(row['input_extended_end_for_tile'])
        input_seq_750 = chrom_dna[input_start_750:input_end_750]
        
        # Validate length
        expected_len_750 = config_dict.shift_extended_input_region_for_tile_len
        assert len(input_seq_750) == expected_len_750, \
            f"Input sequence (750bp) length {len(input_seq_750)} != {expected_len_750} for region {region_idx}"
        
        # One-hot encode and store
        validation_dna_input_750[region_idx] = dna_one_hot(input_seq_750)
        
        # ===== DNA OUTPUT VALIDATION =====
        # Use the pre-calculated coordinates from regions_df
        output_start = int(row['output_extended_start'])
        output_end = int(row['output_extended_end'])
        output_seq = chrom_dna[output_start:output_end]
        
        # Validate length
        expected_output_len = config_dict.shift_extended_output_region_len
        assert len(output_seq) == expected_output_len, \
            f"Output sequence length {len(output_seq)} != {expected_output_len} for region {region_idx}"
        
        # One-hot encode and store
        validation_dna_output[region_idx] = dna_one_hot(output_seq)
        
        # ===== OBSERVED TRACK VALIDATION =====
        try:
            track_values = bw.values(chrom, output_start, output_end)
            
            if track_values is None:
                track_values = np.zeros(expected_output_len, dtype=np.float32)
                track_nan_counts.append(expected_output_len)
            else:
                track_values = np.array(track_values, dtype=np.float32)
                track_nan_count = np.sum(np.isnan(track_values))
                track_nan_counts.append(track_nan_count)
                
                if len(track_values) != expected_output_len:
                    raise ValueError(f"Track length mismatch for {chrom}:{output_start}-{output_end}")
                
                track_values = np.nan_to_num(track_values, nan=0.0)
            
            validation_observed_track[region_idx] = track_values.reshape(-1, 1)
            
        except Exception as e:
            print(f"Error processing track for region {chrom}:{output_start}-{output_end}: {e}")
            validation_observed_track[region_idx] = np.zeros((expected_output_len, 1), dtype=np.float32)
            track_nan_counts.append(expected_output_len)
    
    # Close file handles
    genome.close()
    bw.close()
    
    # ===== COMPARISON PHASE =====
    print("Comparing stored vs freshly extracted data...")
    
    # Compare DNA sequences
    dna_input_matches = np.all(np.equal(stored_dna_input, validation_dna_input), axis=(1, 2))
    dna_input_750_matches = np.all(np.equal(stored_dna_input_750, validation_dna_input_750), axis=(1, 2))
    dna_output_matches = np.all(np.equal(stored_dna_output, validation_dna_output), axis=(1, 2))
    
    dna_input_mismatches = np.sum(~dna_input_matches)
    dna_input_750_mismatches = np.sum(~dna_input_750_matches)
    dna_output_mismatches = np.sum(~dna_output_matches)
    
    # Compare tracks (use chunked comparison for memory efficiency) - ENHANCED VERSION
    chunk_size = 1000

    # Track validation statistics - enhanced
    track_float_exact = True
    track_int_match = True
    track_tolerance_match = True

    track_float_mismatches = 0
    track_int_mismatches = 0
    track_max_diff = 0.0

    # Tolerance settings for floating-point comparison
    rtol = 1e-6  # Relative tolerance
    atol = 1e-8  # Absolute tolerance

    for chunk_start in tqdm(range(0, N, chunk_size), desc="comparing track chunks"):
        chunk_end = min(chunk_start + chunk_size, N)
        
        # ===== OBSERVED TRACK COMPARISON =====
        stored_track_chunk = stored_observed_track[chunk_start:chunk_end]
        validation_track_chunk = validation_observed_track[chunk_start:chunk_end]
        
        # 1. EXACT FLOAT COMPARISON
        track_float_chunk_match = np.array_equal(stored_track_chunk, validation_track_chunk)
        if not track_float_chunk_match:
            track_float_exact = False
            float_mismatch_mask = (stored_track_chunk != validation_track_chunk)
            track_float_mismatches += np.sum(float_mismatch_mask)
        
        # 2. INTEGER COMPARISON (rounded)
        stored_track_int = np.round(stored_track_chunk).astype(np.int32)
        validation_track_int = np.round(validation_track_chunk).astype(np.int32)
        track_int_chunk_match = np.array_equal(stored_track_int, validation_track_int)
        if not track_int_chunk_match:
            track_int_match = False
            int_mismatch_mask = (stored_track_int != validation_track_int)
            track_int_mismatches += np.sum(int_mismatch_mask)
        
        # 3. TOLERANCE COMPARISON (for floating-point precision issues)
        track_tolerance_chunk_match = np.allclose(stored_track_chunk, validation_track_chunk, 
                                                 rtol=rtol, atol=atol)
        if not track_tolerance_chunk_match:
            track_tolerance_match = False
        
        # 4. Calculate maximum absolute difference
        track_diff = np.abs(stored_track_chunk - validation_track_chunk)
        track_max_diff_chunk = np.max(track_diff)
        track_max_diff = max(track_max_diff, track_max_diff_chunk)

    # ===== ENHANCED RESULTS REPORTING =====
    print(f"\n=== VALIDATION RESULTS FOR {fold_name} ===")
    print(f"Total regions: {N}")

    # DNA Results
    print(f"\nDNA INPUT (500bp shift):")
    if dna_input_mismatches == 0:
        print(f"  ✅ PASSED: {N - dna_input_mismatches}/{N} matched (0 mismatches)")
    else:
        print(f"  ❌ FAILED: {N - dna_input_mismatches}/{N} matched ({dna_input_mismatches} mismatches)")

    print(f"\nDNA INPUT (750bp shift):")
    if dna_input_750_mismatches == 0:
        print(f"  ✅ PASSED: {N - dna_input_750_mismatches}/{N} matched (0 mismatches)")
    else:
        print(f"  ❌ FAILED: {N - dna_input_750_mismatches}/{N} matched ({dna_input_750_mismatches} mismatches)")

    print(f"\nDNA OUTPUT:")
    if dna_output_mismatches == 0:
        print(f"  ✅ PASSED: {N - dna_output_mismatches}/{N} matched (0 mismatches)")
    else:
        print(f"  ❌ FAILED: {N - dna_output_mismatches}/{N} matched ({dna_output_mismatches} mismatches)")

    # Enhanced Track Results
    print(f"\nOBSERVED TRACK:")
    print(f"  Float exact match: {'✅ YES' if track_float_exact else f'❌ NO ({track_float_mismatches:,} mismatches)'}")
    print(f"  Integer match (rounded): {'✅ YES' if track_int_match else f'❌ NO ({track_int_mismatches:,} mismatches)'}")
    print(f"  Tolerance match (rtol={rtol}, atol={atol}): {'✅ YES' if track_tolerance_match else '❌ NO'}")
    print(f"  Maximum absolute difference: {track_max_diff:.12f}")

    # Overall validation result - enhanced
    all_passed = (dna_input_mismatches == 0 and dna_input_750_mismatches == 0 and 
                  dna_output_mismatches == 0 and track_float_exact)

    # Additional diagnostic info
    if not track_float_exact:
        print(f"\n🔍 DIAGNOSTIC INFO:")
        print(f"  Track: {track_float_mismatches:,} float mismatches out of {N * config_dict.shift_extended_output_region_len:,} total positions")
        print(f"  Track mismatch rate: {(track_float_mismatches / (N * config_dict.shift_extended_output_region_len)) * 100:.8f}%")
        
        # Show significance of differences
        if track_max_diff > 1e-6:
            print(f"  ⚠️  Found significant differences (> 1e-6) - this could affect training!")
        elif track_max_diff > 1e-12:
            print(f"  ⚠️  Found minor floating-point differences - likely due to precision")
        else:
            print(f"  ✅ Differences are negligible (< 1e-12)")

    # Show coordinate validation for first few regions
    print(f"\n🔍 COORDINATE VALIDATION (first 3 regions):")
    for i in range(min(3, N)):
        row = regions_df.iloc[i]
        summit_abs = row['summit_abs']
        
        # Validate summit calculation: summit_abs should equal Start + Summit
        calculated_summit = row['Start'] + row['Summit']
        summit_match = (summit_abs == calculated_summit)
        
        # Validate coordinate calculations
        input_500_center = (row['input_extended_start'] + row['input_extended_end']) // 2
        input_750_center = (row['input_extended_start_for_tile'] + row['input_extended_end_for_tile']) // 2
        output_center = (row['output_extended_start'] + row['output_extended_end']) // 2
        
        print(f"  Region {i} ({row['identifier']}):")
        print(f"    Summit abs: {summit_abs}, calculated: {calculated_summit}, match: {summit_match}")
        print(f"    Input (500bp) center: {input_500_center}, diff from summit: {input_500_center - summit_abs}")
        print(f"    Input (750bp) center: {input_750_center}, diff from summit: {input_750_center - summit_abs}")
        print(f"    Output center: {output_center}, diff from summit: {output_center - summit_abs}")

    if all_passed:
        print(f"\n✅ {fold_name} VALIDATION PASSED - All data matches perfectly!")
    else:
        if track_int_match:
            print(f"\n⚠️  {fold_name} VALIDATION: Integer values match but float precision differs")
            print(f"   This is likely due to floating-point precision and may not affect training")
        else:
            print(f"\n❌ {fold_name} VALIDATION FAILED - Found significant mismatches in stored data!")
    
    # Track extraction statistics
    track_nan_counts_array = np.array(track_nan_counts)
    print(f"\nTrack extraction statistics:")
    print(f"  Total NaNs: {np.sum(track_nan_counts_array):,}, mean per region: {np.mean(track_nan_counts_array):.2f}")
    
    del zarr_store
    return all_passed

def main():
    # Configuration - Updated for manuscript experiments structure
    config_dict = ConfigDict()
    config_dict.input_region_len = 2114
    config_dict.output_region_len = 1000
    config_dict.shift_bp = 500
    config_dict.shift_bp_for_tile = 750
    config_dict.shift_extended_input_region_len = config_dict.input_region_len + 2 * config_dict.shift_bp  # 3114
    config_dict.shift_extended_input_region_for_tile_len = config_dict.input_region_len + 2 * config_dict.shift_bp_for_tile  # 3614
    config_dict.shift_extended_output_region_len = config_dict.output_region_len + 2 * config_dict.shift_bp  # 2000

    # MANUSCRIPT experiment paths (new dataset)
    config_dict.output_dir = os.path.join(
        MANUSCRIPT_EXPERIMENTS_DIR,
        "section_1__ConditionalLM_as_diagnostic_probe",
        "workspace", "data", "ATAC", "K562__ENCSR868FGK", "preprocessed", "zarr_datasets"
    )

    config_dict.data = ConfigDict()
    config_dict.data.fold_names = [f"fold_{i}" for i in range(5)]
    # config_dict.data.fold_names = [f"fold_{i}" for i in range(1)]  # Uncomment for single fold testing
    
    config_dict.data.fasta_file = "/home/mcb/users/dcakma3/multi_res_bench_v2/data/genomes/hg38/hg38.fa"
    
    # BigWig file path - Updated for manuscript experiment structure
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
    
    # Validate all folds
    print("Starting comprehensive DNA + Track validation for manuscript experiment format...")
    print(f"Will validate {len(config_dict.data.fold_names)} folds")
    
    all_folds_passed = True
    
    for fold_name in config_dict.data.fold_names:
        fold_passed = validate_fold_data(fold_name, config_dict)
        if not fold_passed:
            all_folds_passed = False
    
    print(f"\n{'='*60}")
    if all_folds_passed:
        print(f"🎉 ALL FOLDS VALIDATION PASSED!")
        print(f"✅ Zarr storage is consistent with source data")
        print(f"✅ Training data integrity confirmed")
    else:
        print(f"❌ VALIDATION FAILED!")
        print(f"⚠️  Found inconsistencies in zarr storage")
        print(f"⚠️  Training data may be corrupted")
        sys.exit(1)

if __name__ == "__main__":
    main()
