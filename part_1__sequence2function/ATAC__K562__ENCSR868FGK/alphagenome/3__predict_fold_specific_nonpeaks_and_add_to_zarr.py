import json
import pdb
import os
import sys
import logging
from time import sleep

import numpy as np
import pandas as pd
import zarr
from tqdm import tqdm

from ml_collections import ConfigDict

from alphagenome.models import dna_client
from alphagenome.data import genome

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../../../.."))
from config import * 



def crop_to_central_width(signal, width):
    """
    Crop signal to central width
    """
    signal_length = signal.shape[-1]
    crop_width = (signal_length - width) // 2
    return signal[:, crop_width:-crop_width]  # (N, width)

def create_predicted_track_dataset(fold_name: str, config_dict: ConfigDict):
    """Add predicted track datasets to existing zarr file"""
    
    zarr_path = os.path.join(config_dict.output_dir, f"nonpeaks.{fold_name}.zarr")
    
    # Load regions dataframe
    regions_df = pd.read_csv(os.path.join(zarr_path, "_auxiliary", "regions_df.tsv"), sep='\t')
    N = len(regions_df)
    
    print(f"Zarr path: {zarr_path}")
    print(f"Regions: {N}")
    
    # Open zarr in read/write mode
    root = zarr.open(zarr_path, mode="r+")
    
    # Get chunking from existing dataset
    existing_dna_output = root["reference_dna"]["dna_summitcentered1000bp_symmetric500bpshift"]
    chunk_len = existing_dna_output.chunks[0]
    
    compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=zarr.Blosc.BITSHUFFLE)
    
    # Get the target group and dataset name
    group_name = f"{fold_name}__model_based_tracks"
    target_group = root[group_name]
    dataset_name = f"alphagenome_predicted_track__summitcentered1000bp_symmetric500bpshift"
        
    # Create zarr dataset in the correct group
    if dataset_name in target_group:
        print(f"Dataset {dataset_name} already exists in {group_name}, will overwrite")
        pdb.set_trace()
        del target_group[dataset_name]
        
    predicted_dataset = target_group.create_dataset(
        dataset_name,
        shape=(N, config_dict.shift_extended_output_region_len, 1),  # (N, 2000, 1)
        chunks=(chunk_len, config_dict.shift_extended_output_region_len, 1),
        dtype="float32",
        compressor=compressor,
    )
        
    print(f"Created dataset: {group_name}/{dataset_name}")
    print(f"Shape: {predicted_dataset.shape}")
        
    # prepare region interval at once
    target_length = 2**11
    regions_df["Start_alphagenome"] = regions_df["summit_abs"] - target_length//2
    regions_df["End_alphagenome"] = regions_df["summit_abs"] + target_length//2
    assert all((regions_df["End_alphagenome"] - regions_df["Start_alphagenome"]) == target_length)

    # regions_df = regions_df.iloc[:1000]

    # # create a list of intervals
    # intervals = [
    #     genome.Interval(
    #         chromosome=row["Chromosome"],
    #         start=row["Start_alphagenome"],
    #         end=row["End_alphagenome"]
    #     ) for _, row in regions_df.iterrows()
    # ]

    # # predict all intervals at once
    # outputs = config_dict.alphagenome_model.predict_intervals(
    #     intervals=intervals,
    #     requested_outputs=[
    #         dna_client.OutputType.ATAC,
    #     ],
    #     ontology_terms=[
    #         'EFO:0002067', # K562.
    #     ],
    #     max_workers=32,
    # )

    # Process region df one row at a time
    predicted_signals = []
    for region_idx, row in tqdm(regions_df.iterrows(), desc=f"alphagenome_predicted_track__summitcentered1000bp_symmetric500bpshift"):

        # prepare region interval
        output = config_dict.alphagenome_model.predict_interval(
            interval=genome.Interval(
                chromosome=row["Chromosome"],
                start=row["Start_alphagenome"],
                end=row["End_alphagenome"]
            ),
            requested_outputs=[
                dna_client.OutputType.ATAC,
            ],
            ontology_terms=[
                'EFO:0002067', # K562
            ],
        )

        predicted_signals.append(output.atac.values.T)

        if region_idx != 0 and region_idx % 200 == 0:
            sleep(10)

    # predicted signals is a list of (1, 2**11)
    predicted_signals = np.concatenate(predicted_signals, axis=0) # (N, 2**11)
                
    # Crop to 2000bp
    predicted_signals = crop_to_central_width(predicted_signals, config_dict.shift_extended_output_region_len)
    # (N, 2000)

    # add new dummy dimension to predicted signals at -1
    predicted_signals = predicted_signals[..., np.newaxis]
    # (N, 2000, 1)
    
    # Write to zarr dataset
    print(f"Writing {dataset_name} predictions to zarr...")
    predicted_dataset[:] = predicted_signals
    
    # Update attributes
    root.attrs[f'{dataset_name}_predicted_track_dataset'] = f"{group_name}/{dataset_name}"
    
    # Clear memory
    del predicted_signals
    
    # Close zarr
    del root
    
    print(f"✓ Successfully added predicted track dataset to {group_name}")
    
    return True

def main():
    # Configuration
    config_dict = ConfigDict()
    config_dict.input_region_len = 2114
    config_dict.output_region_len = 1000
    config_dict.shift_bp = 500
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
    config_dict.data.fold_names = [f"fold_{i}" for i in range(1)]

    # alphagenome model 
    API_KEY = "fill"
    config_dict.alphagenome_model = dna_client.create(API_KEY)
    
    # Process all folds
    print("Adding predicted track datasets...")
    for fold_name in config_dict.data.fold_names:
        create_predicted_track_dataset(fold_name, config_dict)
    
    print(f"✓ Completed adding predicted track datasets")

if __name__ == "__main__":
    main()