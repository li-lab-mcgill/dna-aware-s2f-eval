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


def create_observed_track_dataset(fold_name: str, config_dict: ConfigDict):
    """Add both public and inhouse observed track datasets to existing zarr file"""
    
    zarr_path = os.path.join(config_dict.output_dir, f"nonpeaks.{fold_name}.zarr")
    
    # Setup logging
    print(f"=== Adding observed track datasets to {fold_name} ===")
    
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
    
    # Open zarr in read/write mode to add new datasets
    root = zarr.open(zarr_path, mode="r+")
    
    # Dataset names
    dataset_name = "observed_track_summitcentered1000bp_symmetric500bpshift"
    
    # Check if datasets already exist
    if dataset_name in root["observed_tracks"]:
        print(f"Dataset {dataset_name} already exists, will overwrite")
        del root["observed_tracks"][dataset_name]
    
    # Create new datasets with same chunking as DNA output dataset
    existing_dna_output = root["reference_dna"]["dna_summitcentered1000bp_symmetric500bpshift"]
    chunk_len = existing_dna_output.chunks[0]  # Get chunk size from existing dataset
    
    compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=zarr.Blosc.BITSHUFFLE)
    
    dataset = root["observed_tracks"].create_dataset(
        dataset_name,
        shape=(N, config_dict.shift_extended_output_region_len, 1),  # (N, 2000, 1)
        chunks=(chunk_len, config_dict.shift_extended_output_region_len, 1),
        dtype="float32",
        compressor=compressor,
    )
    
    print(f"Created datasets: {dataset_name}")
    print(f"Shape: {dataset.shape}")
    print(f"Chunks: {dataset.chunks}")
    
    # Extract track values by iterating over regions_df in order
    print("Extracting BigWig values for output regions...")
    
    # Pre-allocate track tensors
    track_values = np.empty((N, config_dict.shift_extended_output_region_len, 1), dtype=np.float32)
    
    # NaN analysis storage for both datasets
    nan_counts_per_region = []
    regions_with_length_mismatch = 0
    regions_with_no_data = 0
    
    # Extract track values by iterating over regions_df in order
    print("Extracting track values by iterating over regions_df...")
    for i, row in tqdm(regions_df.iterrows(), total=N, desc="regions"):
        region_idx = row['region_idx']
        chrom = row['Chromosome']
        
        # Use output extended coordinates (2000bp centered on summit)
        start = int(row['output_extended_start'])
        end = int(row['output_extended_end'])
        
        # Process public BigWig
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
                    raise ValueError(f"Public length mismatch for {chrom}:{start}-{end}, expected {config_dict.shift_extended_output_region_len}, got {len(values)}")
                
                values = np.nan_to_num(values, nan=0.0)
            
            track_values[region_idx] = values.reshape(-1, 1)
            
        except Exception as e:
            print(f"Error processing region {chrom}:{start}-{end}: {e}")
            track_values[region_idx] = np.zeros((config_dict.shift_extended_output_region_len, 1), dtype=np.float32)
            nan_counts_per_region.append(config_dict.shift_extended_output_region_len)       
            
            # Log every 50,000 regions (only 10 log entries for 500K regions)
            if i % 50000 == 0:
                print(f"  Region {i}: {chrom}:{start}-{end}, NaNs: {nan_count}")
    
    # Write all data to zarr datasets at once
    print("Writing track values to zarr datasets...")
    dataset[:] = track_values
    
    # Update zarr attributes
    root.attrs['observed_track_dataset'] = dataset_name
    root.attrs['bigwig_source_file'] = config_dict.data.bigwig_file
    root.attrs['track_dtype'] = 'float32'
    
    # Close handles
    bw.close()
    del root
    
    # Convert to numpy arrays for analysis
    nan_counts = np.array(nan_counts_per_region)
    
    # REMOVED: Peak/nonpeak separation - now only overall analysis
    
    # Simplified comparison analysis
    print(f"  Regions with no data: {regions_with_no_data}")
    print(f"  Regions with length mismatch: {regions_with_length_mismatch}")
    regions_all_nan = np.sum(nan_counts == config_dict.shift_extended_output_region_len)
    print(f"  Regions with ALL NaN values: {regions_all_nan}")
    print(f"  Total NaN positions: {np.sum(nan_counts):,}")
    print(f"  Mean NaN count per region: {np.mean(nan_counts):.2f}")
    print(f"  Median NaN count per region: {np.median(nan_counts):.2f}")
    
    

    print(f"✓ Successfully added observed track dataset to {fold_name}")
    
    # Validate the new datasets
    print("Validating new datasets...")
    validation_root = zarr.open(zarr_path, mode="r")
    
    track_data = validation_root["observed_tracks"][dataset_name]
    print(f"dataset shape: {track_data.shape}")
    print(f"data type: {track_data.dtype}")
    print(f"min value: {np.min(track_data[:]):.6f}")
    print(f"max value: {np.max(track_data[:]):.6f}")
    print(f"mean value: {np.mean(track_data[:]):.6f}")
    print(f"non-zero positions: {np.count_nonzero(track_data[:]):,}")
    
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
    
    # Process all folds
    print("Adding observed track dataset...")
    
    for fold_name in config_dict.data.fold_names:
        create_observed_track_dataset(fold_name, config_dict)
    
    print(f"✓ Completed adding observed track dataset")

if __name__ == "__main__":
    main()
