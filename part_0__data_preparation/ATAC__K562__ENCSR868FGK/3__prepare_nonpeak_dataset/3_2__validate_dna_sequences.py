import os
import sys
import numpy as np
import pandas as pd
import zarr
import pyfaidx
from tqdm import tqdm
from ml_collections import ConfigDict

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../../.."))
from config import * 
from utils.data.zarr_preparation.extract_windows import dna_one_hot

def validate_fold_zarr_dataset(fold_name: str, config_dict: ConfigDict):
    """Validate a single fold zarr dataset by comparing with fresh extraction"""
    print(f"\n=== Validating nonpeaks.{fold_name}.zarr ===")
    
    zarr_path = os.path.join(config_dict.output_dir, f"nonpeaks.{fold_name}.zarr")
    
    # Load zarr data (actual data)
    zarr_store = zarr.open(zarr_path, mode="r")
    actual_input = zarr_store["dna_summitcentered2114bp_symmetric500bpshift"][:]
    actual_input_750 = zarr_store["dna_summitcentered2114bp_symmetric750bpshift"][:]
    actual_output = zarr_store["dna_summitcentered1000bp_symmetric500bpshift"][:]
    
    # Load regions dataframe
    regions_df = pd.read_csv(os.path.join(zarr_path, "_auxiliary", "regions_df.tsv"), sep='\t')
    N = len(regions_df)
    
    print(f"Zarr: {N} regions")
    print(f"Actual input shape (500bp shift): {actual_input.shape}")
    print(f"Actual input shape (750bp shift): {actual_input_750.shape}")
    print(f"Actual output shape: {actual_output.shape}")
    
    # Check if region_idx matches pandas index
    index_matches_region_idx = (regions_df.index == regions_df['region_idx']).all()
    print(f"DataFrame index matches region_idx column: {index_matches_region_idx}")
    
    # Load genome and cache all chromosomes
    genome = pyfaidx.Fasta(config_dict.data.fasta_file, as_raw=True, sequence_always_upper=True)
    
    print("Caching chromosome sequences...")
    chrom_cache = {}
    unique_chroms = regions_df['Chromosome'].unique()
    for chrom in tqdm(unique_chroms, desc="chromosomes"):
        chrom_cache[chrom] = str(genome[chrom])
    
    # Pre-allocate validation tensors
    validation_input = np.empty((N, config_dict.shift_extended_input_region_len, 4), dtype=np.uint8)
    validation_input_750 = np.empty((N, config_dict.shift_extended_input_region_for_tile_len, 4), dtype=np.uint8)
    validation_output = np.empty((N, config_dict.shift_extended_output_region_len, 4), dtype=np.uint8)
    
    # Extract sequences by iterating over regions_df in order
    print("Extracting validation sequences by iterating over regions_df...")
    for i, row in tqdm(regions_df.iterrows(), total=N, desc="regions"):
        region_idx = row['region_idx']
        chrom = row['Chromosome']
        
        # Get cached chromosome sequence
        chrom_dna = chrom_cache[chrom]
        
        # Extract input sequence (500bp shift)
        input_start = row['input_extended_start']
        input_end = row['input_extended_end']
        input_seq = chrom_dna[input_start:input_end]
        
        # Validate length
        assert len(input_seq) == config_dict.shift_extended_input_region_len, \
            f"Input sequence length {len(input_seq)} != {config_dict.shift_extended_input_region_len} for region {region_idx}"
        
        # One-hot encode and store
        validation_input[region_idx] = dna_one_hot(input_seq)
        
        # Extract input sequence (750bp shift)
        input_start_750 = row['input_extended_start_for_tile']
        input_end_750 = row['input_extended_end_for_tile']
        input_seq_750 = chrom_dna[input_start_750:input_end_750]
        
        # Validate length
        assert len(input_seq_750) == config_dict.shift_extended_input_region_for_tile_len, \
            f"Input sequence length (750bp shift) {len(input_seq_750)} != {config_dict.shift_extended_input_region_for_tile_len} for region {region_idx}"
        
        # One-hot encode and store
        validation_input_750[region_idx] = dna_one_hot(input_seq_750)
        
        # Extract output sequence  
        output_start = row['output_extended_start']
        output_end = row['output_extended_end']
        output_seq = chrom_dna[output_start:output_end]
        
        # Validate length
        assert len(output_seq) == config_dict.shift_extended_output_region_len, \
            f"Output sequence length {len(output_seq)} != {config_dict.shift_extended_output_region_len} for region {region_idx}"
        
        # One-hot encode and store
        validation_output[region_idx] = dna_one_hot(output_seq)
    
    genome.close()
    
    # Compare actual vs validation data
    print("Comparing sequences...")
    input_matches = np.all(np.equal(actual_input, validation_input), axis=(1, 2))
    input_750_matches = np.all(np.equal(actual_input_750, validation_input_750), axis=(1, 2))
    output_matches = np.all(np.equal(actual_output, validation_output), axis=(1, 2))
    
    input_mismatches = np.sum(~input_matches)
    input_750_mismatches = np.sum(~input_750_matches)
    output_mismatches = np.sum(~output_matches)
    
    # Results
    print(f"Input (500bp shift):  {N - input_mismatches}/{N} matched ({input_mismatches} mismatches)")
    print(f"Input (750bp shift):  {N - input_750_mismatches}/{N} matched ({input_750_mismatches} mismatches)")
    print(f"Output: {N - output_mismatches}/{N} matched ({output_mismatches} mismatches)")
    
    if input_mismatches > 0:
        mismatched_idx = np.where(~input_matches)[0]
        print(f"Input (500bp shift) mismatches at regions: {mismatched_idx[:10].tolist()}")
        
    if input_750_mismatches > 0:
        mismatched_idx = np.where(~input_750_matches)[0]
        print(f"Input (750bp shift) mismatches at regions: {mismatched_idx[:10].tolist()}")
        
    if output_mismatches > 0:
        mismatched_idx = np.where(~output_matches)[0]
        print(f"Output mismatches at regions: {mismatched_idx[:10].tolist()}")
    
    # Fail if any mismatches
    assert input_mismatches == 0, f"Input sequence (500bp shift) mismatches found: {input_mismatches}"
    assert input_750_mismatches == 0, f"Input sequence (750bp shift) mismatches found: {input_750_mismatches}"
    assert output_mismatches == 0, f"Output sequence mismatches found: {output_mismatches}"
    
    print(f"✓ nonpeaks.{fold_name}.zarr PASSED")
    return True

def main():
    # Configuration - match 3_1 exactly
    config_dict = ConfigDict()
    config_dict.input_region_len = 2114
    config_dict.output_region_len = 1000
    config_dict.shift_bp = 500
    config_dict.shift_bp_for_tile = 750
    config_dict.shift_extended_input_region_len = \
        config_dict.input_region_len + 2 * config_dict.shift_bp  # 3114
    config_dict.shift_extended_input_region_for_tile_len = \
        config_dict.input_region_len + 2 * config_dict.shift_bp_for_tile  # 3614
    config_dict.shift_extended_output_region_len = \
        config_dict.output_region_len + 2 * config_dict.shift_bp  # 2000

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
    config_dict.data.fasta_file = "/home/mcb/users/dcakma3/multi_res_bench_v2/data/genomes/hg38/hg38.fa"
    
    # Validate each fold's zarr dataset
    print("Starting DNA sequence validation for nonpeak datasets...")
    
    for fold_name in config_dict.data.fold_names:
        validate_fold_zarr_dataset(fold_name, config_dict)
    
    print(f"\n🎉 ALL NONPEAK VALIDATIONS PASSED!")

if __name__ == "__main__":
    main()
