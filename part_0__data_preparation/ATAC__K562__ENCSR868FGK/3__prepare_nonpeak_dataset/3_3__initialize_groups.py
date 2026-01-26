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
        "ATAC",
        "K562__ENCSR868FGK",
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
        "ATAC",
        "K562__ENCSR868FGK",
        "preprocessed",
        "regions",
    )

    # Process each fold's nonpeak zarr file separately
    for fold_name in config_dict.data.fold_names:
        print(f"\n=== Processing {fold_name} ===")
        
        # open zarr store for this fold
        zarr_path = os.path.join(config_dict.output_dir, f"nonpeaks.{fold_name}.zarr")
        zarr_store = zarr.open(zarr_path, mode="r+")

        # initialize_groups - only current fold's model_based_tracks + shared groups
        group_names = [
            f"{fold_name}__model_based_tracks",  # Only current fold
            "fold_agnostic__model_based_tracks",
            "reference_dna", 
            "observed_tracks"
        ]
        
        # Create groups
        for group_name in group_names:
            zarr_store.require_group(group_name)
            print(f"Created/ensured group: {group_name}")
        
        # List existing top-level items
        print("Current top-level items:", list(zarr_store.keys()))
        
        # Move DNA datasets to reference_dna group if they exist at top level
        dna_datasets = [
            "dna_summitcentered2114bp_symmetric500bpshift",
            "dna_summitcentered2114bp_symmetric750bpshift", 
            "dna_summitcentered1000bp_symmetric500bpshift"
        ]
        
        reference_dna_group = zarr_store["reference_dna"]
        
        for dna_dataset in dna_datasets:
            if dna_dataset in zarr_store:
                print(f"Moving {dna_dataset} to reference_dna group...")
                
                # Copy dataset to the new group
                zarr.copy(zarr_store[dna_dataset], reference_dna_group, name=dna_dataset, shallow=False, if_exists="replace")
                
                # Verify copy worked by checking shapes match
                original_shape = zarr_store[dna_dataset].shape
                copied_shape = reference_dna_group[dna_dataset].shape
                
                if original_shape == copied_shape:
                    print(f"  ✓ Copy verified: {original_shape}")
                    # Delete the old dataset at top level
                    del zarr_store[dna_dataset]
                    print(f"  ✓ Deleted original {dna_dataset} from top level")
                else:
                    print(f"  ✗ Copy verification failed! Original: {original_shape}, Copied: {copied_shape}")
                    raise ValueError(f"Copy verification failed for {dna_dataset}")
            else:
                print(f"Dataset {dna_dataset} not found at top level, skipping")
        
        # Move alphagenome dataset to fold_agnostic__model_based_tracks if it exists
        alphagenome_dataset = "alphagenome_predicted_track__summitcentered1000bp_symmetric500bpshift"
        
        if alphagenome_dataset in zarr_store:
            print(f"Moving {alphagenome_dataset} to fold_agnostic__model_based_tracks group...")
            
            fold_agnostic_group = zarr_store["fold_agnostic__model_based_tracks"]
            
            # Copy dataset to the new group
            zarr.copy(zarr_store[alphagenome_dataset], fold_agnostic_group, name=alphagenome_dataset, shallow=False, if_exists="replace")
            
            # Verify copy worked by checking shapes match
            original_shape = zarr_store[alphagenome_dataset].shape
            copied_shape = fold_agnostic_group[alphagenome_dataset].shape
            
            if original_shape == copied_shape:
                print(f"  ✓ Copy verified: {original_shape}")
                # Delete the old dataset at top level
                del zarr_store[alphagenome_dataset]
                print(f"  ✓ Deleted original {alphagenome_dataset} from top level")
            else:
                print(f"  ✗ Copy verification failed! Original: {original_shape}, Copied: {copied_shape}")
                raise ValueError(f"Copy verification failed for {alphagenome_dataset}")
        else:
            print(f"Dataset {alphagenome_dataset} not found at top level, skipping")
        
        print("Final top-level items:", list(zarr_store.keys()))
        print("Groups created:", list(zarr_store.groups()))
        
        # Show what's in each group
        for group_name in group_names:
            if group_name in zarr_store:
                group_items = list(zarr_store[group_name].keys())
                if group_items:
                    print(f"  {group_name}: {group_items}")
        
        print(f"✓ Completed {fold_name}")
