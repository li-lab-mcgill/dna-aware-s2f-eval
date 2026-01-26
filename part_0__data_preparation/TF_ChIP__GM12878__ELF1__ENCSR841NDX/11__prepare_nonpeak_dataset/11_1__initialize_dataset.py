import json
import pdb
import os
import sys
import logging
import multiprocessing
from datetime import datetime
import time # Added for optional pause

import numpy as np
import pandas as pd
import zarr
from tqdm import tqdm
from ml_collections import ConfigDict

# Add project root to path
print(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../../.."))
from config import * 
from utils.data.zarr_preparation.extract_windows import dna_one_hot

def prepare_regions_df(regions_file, config_dict):
    """
    Prepare and augment regions dataframe with shift-extended coordinates
    """
    # Load regions (same format as peaks - no header)
    regions_df = pd.read_csv(regions_file, sep='\t', header=None)

    # add columns for MACS2 peak columns (same as 10_1)
    regions_df.columns = \
        ['Chromosome', 'Start', 'End'] + \
        [f"col_{i}" for i in range(4, 10)] + \
        ['Summit']
    
    # FIX: Ensure all coordinate columns are integers when loading from TSV
    regions_df['Start'] = regions_df['Start'].astype(int)
    regions_df['End'] = regions_df['End'].astype(int)
    regions_df['Summit'] = regions_df['Summit'].astype(int)
    
    # 2.1: Sort by chromosome for optimal processing (following extract_windows pattern)
    regions_df = regions_df.sort_values(['Chromosome', 'Start']).reset_index(drop=True)
    
    # 2.2: Calculate shift-augmented coordinates
    # Original summit position (absolute genomic coordinate)
    regions_df['summit_abs'] = regions_df['Start'] + regions_df['Summit']
    
    # Shift-extended input region (2114 + 2*500 = 3114bp centered on summit)
    half_input_extended = config_dict.shift_extended_input_region_len // 2  # 1557
    regions_df['input_extended_start'] = regions_df['summit_abs'] - half_input_extended
    regions_df['input_extended_end'] = regions_df['summit_abs'] + half_input_extended
    
    # Shift-extended input region with 750bp shift (2114 + 2*750 = 3614bp centered on summit)
    half_input_extended_for_tile = config_dict.shift_extended_input_region_for_tile_len // 2  # 1807
    regions_df['input_extended_start_for_tile'] = regions_df['summit_abs'] - half_input_extended_for_tile
    regions_df['input_extended_end_for_tile'] = regions_df['summit_abs'] + half_input_extended_for_tile
    
    # Shift-extended output region (1000 + 2*500 = 2000bp centered on summit) 
    half_output_extended = config_dict.shift_extended_output_region_len // 2  # 1000
    regions_df['output_extended_start'] = regions_df['summit_abs'] - half_output_extended
    regions_df['output_extended_end'] = regions_df['summit_abs'] + half_output_extended
    
    # CRITICAL: Validate all extended regions have exact same length
    input_lengths = regions_df['input_extended_end'] - regions_df['input_extended_start']
    input_lengths_for_tile = regions_df['input_extended_end_for_tile'] - regions_df['input_extended_start_for_tile']
    output_lengths = regions_df['output_extended_end'] - regions_df['output_extended_start']
    
    assert all(input_lengths == config_dict.shift_extended_input_region_len), \
        f"Input extended regions must all be {config_dict.shift_extended_input_region_len}bp"
    assert all(input_lengths_for_tile == config_dict.shift_extended_input_region_for_tile_len), \
        f"Input extended regions (750bp shift) must all be {config_dict.shift_extended_input_region_for_tile_len}bp"
    assert all(output_lengths == config_dict.shift_extended_output_region_len), \
        f"Output extended regions must all be {config_dict.shift_extended_output_region_len}bp"
    
    # 2.3: CRITICAL - Make pandas index equal to region_idx for perfect alignment
    # This ensures zarr row index = region_idx = pandas index
    regions_df['region_idx'] = range(len(regions_df))
    regions_df.index = regions_df['region_idx']  # Make pandas index = region_idx
    
    return regions_df

def create_split_indices(regions_df, config_dict, fold_name):
    """
    Extract train/val/test split indices for the current fold based on chromosome assignments
    """
    all_split_indices = {}
    
    # Load the splits file for this specific fold only
    splits_file = os.path.join(config_dict.data.splits_dir, f"{fold_name}.json")
    with open(splits_file, 'r') as f:
        splits = json.load(f)
    
    # Extract indices for each split based on chromosome assignments for this fold only
    train_indices = regions_df[regions_df['Chromosome'].isin(splits['train'])]['region_idx'].tolist()
    val_indices = regions_df[regions_df['Chromosome'].isin(splits['valid'])]['region_idx'].tolist()
    test_indices = regions_df[regions_df['Chromosome'].isin(splits['test'])]['region_idx'].tolist()
    
    # Store with the naming convention for this fold only
    all_split_indices[f"{fold_name}__training"] = train_indices
    all_split_indices[f"{fold_name}__validation"] = val_indices
    all_split_indices[f"{fold_name}__test"] = test_indices
    
    return all_split_indices

def initialize_zarr_dataset(regions_df, config_dict, zarr_name):
    """Initialize zarr dataset with optimized chunking (excluding chrY)"""
    
    zarr_path = os.path.join(config_dict.output_dir, zarr_name)
    N = len(regions_df)
    
    # Calculate chunk size excluding problematic small chromosomes
    chrom_groups = regions_df.groupby('Chromosome')
    main_chrom_sizes = []
    
    for chrom, group in chrom_groups:
        if chrom not in ['chrY']:  # Exclude tiny chromosomes
            main_chrom_sizes.append(len(group))
    
    # Use largest power of 2 <= smallest main chromosome
    min_main_chrom = min(main_chrom_sizes)
    chunk_len = 1
    while chunk_len * 2 <= min_main_chrom:
        chunk_len *= 2
    
    chunk_len = max(chunk_len, 1024)  # Minimum reasonable size
    print(f"Chunk length: {chunk_len}")
    
    # Create zarr store
    root = zarr.open(zarr_path, mode="w")
    compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=zarr.Blosc.BITSHUFFLE)
    
    # Create reference_dna group
    reference_dna_group = root.create_group("reference_dna")
    
    # DNA sequences - three versions with different augmentation windows
    reference_dna_group.create_dataset(
        "dna_summitcentered2114bp_symmetric500bpshift",
        shape=(N, config_dict.shift_extended_input_region_len, 4),  # (N, 3114, 4)
        chunks=(chunk_len, config_dict.shift_extended_input_region_len, 4),
        dtype="uint8",
        compressor=compressor,
    )
    
    reference_dna_group.create_dataset(
        "dna_summitcentered2114bp_symmetric750bpshift",
        shape=(N, config_dict.shift_extended_input_region_for_tile_len, 4),  # (N, 3614, 4)
        chunks=(chunk_len, config_dict.shift_extended_input_region_for_tile_len, 4),
        dtype="uint8",
        compressor=compressor,
    )
    
    reference_dna_group.create_dataset(
        "dna_summitcentered1000bp_symmetric500bpshift", 
        shape=(N, config_dict.shift_extended_output_region_len, 4),  # (N, 2000, 4)
        chunks=(chunk_len, config_dict.shift_extended_output_region_len, 4),
        dtype="uint8",
        compressor=compressor,
    )
    
    # Create observed_tracks group for future use
    observed_tracks_group = root.create_group("observed_tracks")
    
    # Store attributes only (no metadata group)
    root.attrs.update(
        n_regions=N,
        input_region_len=config_dict.input_region_len,
        output_region_len=config_dict.output_region_len,
        shift_bp=config_dict.shift_bp,
        shift_bp_for_tile=config_dict.shift_bp_for_tile,
        shift_extended_input_region_len=config_dict.shift_extended_input_region_len,
        shift_extended_input_region_for_tile_len=config_dict.shift_extended_input_region_for_tile_len,
        shift_extended_output_region_len=config_dict.shift_extended_output_region_len,
        created_at=str(datetime.now()),
        regions_df_path="_auxiliary/regions_df.tsv",  # Reference to auxiliary data
        split_indices_path="_auxiliary/split_indices.npz"
    )
    
    # CRITICAL: Delete zarr handle (following extract_windows.py pattern)
    del root
    
    return zarr_path

def save_auxiliary_files(zarr_path, regions_df, all_split_indices):
    """Save auxiliary files to {zarr_path}/_auxiliary/"""
    
    aux_dir = os.path.join(zarr_path, "_auxiliary")
    os.makedirs(aux_dir, exist_ok=True)
    
    # Save regions dataframe (this is the sorted, augmented version)
    regions_df.to_csv(os.path.join(aux_dir, "regions_df.tsv"), sep='\t', index=False)
    
    # Save split indices - each key becomes an array in the npz file
    np.savez_compressed(
        os.path.join(aux_dir, "split_indices.npz"),
        **all_split_indices  # Unpacks the dict as keyword arguments
    )
    
    # Save metadata
    metadata = {
        'n_regions': len(regions_df),
        'split_keys': list(all_split_indices.keys()),
        'created_at': str(datetime.now())
    }
    
    # Add counts for each split
    for key, indices in all_split_indices.items():
        metadata[f'n_{key}'] = len(indices)
    
    with open(os.path.join(aux_dir, "metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)

def chromosome_dna_worker(
    chrom: str,
    chrom_indices: np.ndarray,
    regions_df: pd.DataFrame,
    fasta_path: str,
    zarr_path: str,
    config_dict: ConfigDict
) -> None:
    """Process DNA extraction for one chromosome (parallel worker)"""
    
    import pyfaidx
    
    print(f"Worker processing {chrom}: {len(chrom_indices)} regions")
    
    # Load FASTA chromosome into memory (following extract_windows pattern)
    genome = pyfaidx.Fasta(fasta_path, as_raw=True, sequence_always_upper=True)
    chrom_dna = str(genome[chrom])
    chrom_len = len(chrom_dna)
    
    # Get regions for this chromosome
    chrom_regions = regions_df.iloc[chrom_indices]
    n_regions = len(chrom_regions)
    
    # Pre-allocate buffers
    input_buf = np.empty(
        (n_regions, config_dict.shift_extended_input_region_len, 4), 
        dtype=np.uint8
    )
    input_buf_for_tile = np.empty(
        (n_regions, config_dict.shift_extended_input_region_for_tile_len, 4), 
        dtype=np.uint8
    )
    output_buf = np.empty(
        (n_regions, config_dict.shift_extended_output_region_len, 4), 
        dtype=np.uint8
    )
    
    # Process regions
    for i, (_, row) in enumerate(chrom_regions.iterrows()):
        # Input sequence (3114bp) with 500bp shift
        input_start = row['input_extended_start']
        input_end = row['input_extended_end']
        
        # Check boundary conditions - RAISE ERROR if out of bounds
        if input_start < 0:
            raise ValueError(f"Input start coordinate {input_start} < 0 for region {row['region_idx']} on {chrom}")
        if input_end > chrom_len:
            raise ValueError(f"Input end coordinate {input_end} > chromosome length {chrom_len} for region {row['region_idx']} on {chrom}")
        
        # Extract sequence (no boundary handling needed - already validated)
        input_seq = chrom_dna[input_start:input_end]
        
        # Validate length
        assert len(input_seq) == config_dict.shift_extended_input_region_len, \
            f"Input sequence length {len(input_seq)} != {config_dict.shift_extended_input_region_len} for region {row['region_idx']}"
        input_buf[i] = dna_one_hot(input_seq)
        
        # Input sequence (3614bp) with 750bp shift
        input_start_for_tile = row['input_extended_start_for_tile']
        input_end_for_tile = row['input_extended_end_for_tile']
        
        # Check boundary conditions - RAISE ERROR if out of bounds
        if input_start_for_tile < 0:
            raise ValueError(f"Input start coordinate (750bp shift) {input_start_for_tile} < 0 for region {row['region_idx']} on {chrom}")
        if input_end_for_tile > chrom_len:
            raise ValueError(f"Input end coordinate (750bp shift) {input_end_for_tile} > chromosome length {chrom_len} for region {row['region_idx']} on {chrom}")
        
        # Extract sequence (no boundary handling needed - already validated)
        input_seq_for_tile = chrom_dna[input_start_for_tile:input_end_for_tile]
        
        # Validate length
        assert len(input_seq_for_tile) == config_dict.shift_extended_input_region_for_tile_len, \
            f"Input sequence length (750bp shift) {len(input_seq_for_tile)} != {config_dict.shift_extended_input_region_for_tile_len} for region {row['region_idx']}"
        input_buf_for_tile[i] = dna_one_hot(input_seq_for_tile)
        
        # Output sequence (2000bp)
        output_start = row['output_extended_start']
        output_end = row['output_extended_end']
        
        # Check boundary conditions - RAISE ERROR if out of bounds
        if output_start < 0:
            raise ValueError(f"Output start coordinate {output_start} < 0 for region {row['region_idx']} on {chrom}")
        if output_end > chrom_len:
            raise ValueError(f"Output end coordinate {output_end} > chromosome length {chrom_len} for region {row['region_idx']} on {chrom}")
        
        # Extract sequence (no boundary handling needed - already validated)
        output_seq = chrom_dna[output_start:output_end]
        
        # Validate length
        assert len(output_seq) == config_dict.shift_extended_output_region_len, \
            f"Output sequence length {len(output_seq)} != {config_dict.shift_extended_output_region_len} for region {row['region_idx']}"
        output_buf[i] = dna_one_hot(output_seq)
    
    # Write to zarr (NO SYNCHRONIZER - following extract_windows pattern)
    root = None
    try:
        root = zarr.open(zarr_path, mode="r+")
        root["reference_dna"]["dna_summitcentered2114bp_symmetric500bpshift"][chrom_indices] = input_buf
        root["reference_dna"]["dna_summitcentered2114bp_symmetric750bpshift"][chrom_indices] = input_buf_for_tile
        root["reference_dna"]["dna_summitcentered1000bp_symmetric500bpshift"][chrom_indices] = output_buf
    except Exception as e:
        print(f"Failed to write chromosome {chrom}: {e}")
        raise
    finally:
        # CRITICAL: Delete zarr handle (following extract_windows.py pattern)
        if root is not None:
            del root
    
    # Close genome file
    genome.close()
    print(f"✓ Completed {chrom}: {n_regions} regions")

def extract_and_store_dna_sequences_parallel(zarr_path, regions_df, fasta_file, config_dict):
    """Extract DNA sequences - configurable parallel/sequential processing"""
    
    # Get configuration
    num_parallel_rounds = getattr(config_dict, 'num_parallel_rounds', 3)  # Default to 3 rounds
    special_chromosomes = getattr(config_dict, 'special_chromosomes', ['chrY'])  # Default chrY only
    
    chrom_groups = regions_df.groupby('Chromosome')
    main_tasks = []
    special_tasks = []
    
    # If num_parallel_rounds = -1, process everything sequentially
    if num_parallel_rounds == -1:
        for chrom, group in chrom_groups:
            indices = group.index.values
            special_tasks.append((chrom, indices))
    else:
        # Normal parallel/round-robin processing
        for chrom, group in chrom_groups:
            indices = group.index.values
            if chrom in special_chromosomes:
                special_tasks.append((chrom, indices))
            else:
                main_tasks.append((chrom, indices))
    
    print(f"Main task Chromosomes: {[task[0] for task in main_tasks]}")
    print(f"Special task Chromosomes: {[task[0] for task in special_tasks]}")
    
    # Process main chromosomes using round-robin assignment (if any)
    if main_tasks:
        max_workers = min(8, multiprocessing.cpu_count() - 2)
        
        # Assign chromosomes to rounds using round-robin
        rounds = [[] for _ in range(num_parallel_rounds)]
        
        for i, task in enumerate(main_tasks):
            round_idx = i % num_parallel_rounds
            rounds[round_idx].append(task)
        
        # Log the round assignments
        for round_idx, round_tasks in enumerate(rounds):
            if round_tasks:
                chrom_names = [task[0] for task in round_tasks]
        
        # Process each round sequentially, but chromosomes within each round in parallel
        for round_idx, round_tasks in enumerate(rounds):
            if not round_tasks:
                continue
                
            workers = min(max_workers, len(round_tasks))
            
            with multiprocessing.Pool(workers) as pool:
                results = []
                for chrom, chrom_indices in round_tasks:
                    result = pool.apply_async(
                        chromosome_dna_worker,
                        (chrom, chrom_indices, regions_df, fasta_file, zarr_path, config_dict)
                    )
                    results.append(result)
                
                # Wait for completion of this round
                for result in tqdm(results, desc=f"round {round_idx + 1}"):
                    result.get()
            
    
    # Process special chromosomes sequentially
    if special_tasks:
        for chrom, chrom_indices in special_tasks:
            chromosome_dna_worker(chrom, chrom_indices, regions_df, fasta_file, zarr_path, config_dict)
    
    processing_mode = "sequential" if num_parallel_rounds == -1 else f"round-robin ({num_parallel_rounds} rounds)"


def main(config_dict):
    """Process all folds for nonpeaks dataset"""
    
    # Process each fold separately
    for fold_name in config_dict.data.fold_names:
        print(f"\n=== Processing {fold_name} ===")
        
        # 1. Load and prepare fold-specific region data
        regions_file = os.path.join(config_dict.data.data_dir, f"nonpeaks.{fold_name}.ext_col.bed")
        if not os.path.exists(regions_file):
            print(f"Warning: File {regions_file} does not exist, skipping {fold_name}")
            continue
            
        regions_df = prepare_regions_df(regions_file, config_dict)
        print(f"Loaded {len(regions_df)} regions from {regions_file}")
        
        # 2. Extract split indices for this fold
        all_split_indices = create_split_indices(regions_df, config_dict, fold_name)
        
        # 3. Initialize empty zarr dataset for this fold
        zarr_name = f"nonpeaks.{fold_name}.zarr"
        zarr_path = initialize_zarr_dataset(regions_df, config_dict, zarr_name)
        print(f"Created zarr dataset: {zarr_path}")
        
        # 4. Save auxiliary files (regions_df and split indices)
        save_auxiliary_files(zarr_path, regions_df, all_split_indices)
        
        # 5. Extract and store DNA sequences using parallel workers
        extract_and_store_dna_sequences_parallel(zarr_path, regions_df, config_dict.data.fasta_file, config_dict)
        
        print(f"✓ Completed {fold_name}")


if __name__ == "__main__":
    # Create configuration
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

    # Processing configuration
    config_dict.num_parallel_rounds = -1  # Set to -1 for sequential, or 3 for round-robin
    config_dict.special_chromosomes = ['chrY']  # Chromosomes to always process sequentially

    config_dict.output_dir = os.path.join(
        MANUSCRIPT_EXPERIMENTS_DIR,
        "section_1__ConditionalLM_as_diagnostic_probe",
        "workspace",
        "data",
        "TF_ChIP",
        "GM12878__ELF1__ENCSR841NDX",
        "preprocessed",
        "zarr_datasets",
    )
    os.makedirs(config_dict.output_dir, exist_ok=True)

    config_dict.data = ConfigDict()
    config_dict.data.fold_names = [f"fold_{i}" for i in range(5)]
    config_dict.data.data_dir = os.path.join(
        MANUSCRIPT_EXPERIMENTS_DIR,
        "section_1__ConditionalLM_as_diagnostic_probe",
        "workspace",
        "data",
        "TF_ChIP",
        "GM12878__ELF1__ENCSR841NDX",
        "preprocessed",
    )
    
    # Add fasta file path
    config_dict.data.fasta_file = "/home/mcb/users/dcakma3/multi_res_bench_v2/data/genomes/hg38/hg38.fa"
    config_dict.data.splits_dir = "/home/mcb/users/dcakma3/multi_res_bench_v2/data/genomes/hg38/splits"
    
    # Run main function
    main(config_dict)