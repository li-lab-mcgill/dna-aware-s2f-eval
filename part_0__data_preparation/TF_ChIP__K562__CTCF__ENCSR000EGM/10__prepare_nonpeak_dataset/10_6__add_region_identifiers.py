import os
import sys
import pandas as pd
import zarr
from ml_collections import ConfigDict

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../../.."))
from config import *

def add_identifiers_to_regions_df(zarr_path):
    """Add identifier column to regions_df in existing zarr dataset"""
    
    print(f"=== Adding identifiers to regions_df ===")
    print(f"Zarr path: {zarr_path}")
    
    # Load existing regions_df
    regions_df_path = os.path.join(zarr_path, "_auxiliary", "regions_df.tsv")
    if not os.path.exists(regions_df_path):
        raise FileNotFoundError(f"regions_df.tsv not found: {regions_df_path}")
    
    regions_df = pd.read_csv(regions_df_path, sep='\t')
    print(f"Loaded regions_df with {len(regions_df)} regions")
    
    # Check if identifier column already exists
    if 'identifier' in regions_df.columns:
        print("Identifier column already exists, skipping...")
        return
    
    # Create identifier: chr:start:summit (relative summit position)
    regions_df['identifier'] = (
        regions_df['Chromosome'].astype(str) + ':' + 
        regions_df['Start'].astype(str) + ':' + 
        regions_df['Summit'].astype(str)
    )
    
    print(f"Created {len(regions_df)} identifiers")
    print(f"Example identifiers: {regions_df['identifier'].head(3).tolist()}")
    
    # Save updated regions_df
    regions_df.to_csv(regions_df_path, sep='\t', index=False)
    print(f"✓ Updated regions_df saved to {regions_df_path}")

def main():
    """Main function to add identifiers to all nonpeaks datasets"""
    
    # Configuration
    config_dict = ConfigDict()
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
    config_dict.data.fold_names = [f"fold_{i}" for i in range(5)]
    
    # Add identifiers to each fold-specific nonpeaks dataset
    for fold_name in config_dict.data.fold_names:
        print(f"\n=== Processing {fold_name} ===")
        
        zarr_path = os.path.join(config_dict.output_dir, f"nonpeaks.{fold_name}.zarr")
        if not os.path.exists(zarr_path):
            print(f"Warning: Zarr dataset not found: {zarr_path}, skipping...")
            continue
        
        add_identifiers_to_regions_df(zarr_path)
        print(f"✓ Completed {fold_name}")
    
    print("✓ Completed adding identifiers to all nonpeaks datasets")

if __name__ == "__main__":
    main()
