import json
import pdb
import os
import sys
import pandas as pd
from ml_collections import ConfigDict
import numpy as np
import logging
import pyBigWig
import pyranges as pr

from data_prep_utils import setup_logging, load_fold_data, load_bed_gz_file

# Add project root to path
print(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../.."))
from config import * 


def pool_peaks_and_nonpeaks(fold_data):
    """Pool train/valid/test splits and add identifiers"""
    
    # Pool peaks across splits
    all_peaks = pd.concat([
        fold_data['peak__training'],
        fold_data['peak__validation'], 
        fold_data['peak__test']
    ], ignore_index=True)
    
    # Pool nonpeaks across splits
    all_nonpeaks = pd.concat([
        fold_data['nonpeak__training'],
        fold_data['nonpeak__validation'],
        fold_data['nonpeak__test']
    ], ignore_index=True)
    
    return {"peaks": all_peaks, "nonpeaks": all_nonpeaks}


def main(config_dict):
    """Main function to process all folds"""
    
    os.makedirs(config_dict.data.data_output_dir, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(config_dict.data.data_output_dir, "prepare_regions")
    logger.info("Starting region preprocessing for all folds")
    logger.info("Set random seed to 42 for reproducible subsampling")
    np.random.seed(42)

    # load fold data 
    fold_data = load_fold_data(config_dict, logger)
    logger.info(f"Loaded regions for all folds:")
    for fold_name, fold_dct in fold_data.items():
        logger.info(f"  {fold_name}:")
        for key, df in fold_dct.items():
            logger.info(f"    {key}: {len(df)} regions")

    # Add identifiers
    for fold_name, fold_dct in fold_data.items():
        for key, df in fold_dct.items():
            df['identifier'] = \
                df['Chromosome'].astype(str) + ':' + \
                df['Start'].astype(str) + ':' + \
                df['Summit'].astype(str)
            logger.info(f"Added identifiers to {key} for {fold_name}")

    # create mapping from identifier to split
    fold_splits_to_identifier = {}
    for fold_name, fold_dct in fold_data.items():
        fold_splits_to_identifier[fold_name] = {}
        for key, df in fold_dct.items():
            region_type, split_name = key.split('__')
            if region_type not in fold_splits_to_identifier[fold_name]:
                fold_splits_to_identifier[fold_name][region_type] = {}
            fold_splits_to_identifier[fold_name][region_type][split_name] = df['identifier'].tolist()

            logger.info(f"Created mapping from identifier to split for {key} for {fold_name}")

    # Remove the pdb.set_trace()
    # pdb.set_trace()
    
    # Pool splits per peak/nonpeak per fold 
    pooled_fold_data = {}
    for fold_name, fold_dct in fold_data.items():
        pooled_fold_data[fold_name] = pool_peaks_and_nonpeaks(fold_dct)
          
        logger.info(f"Pooled data for {fold_name}:")
        logger.info(f" Total Peaks: {len(pooled_fold_data[fold_name]['peaks'])} regions")
        logger.info(f" Unique Peaks: {len(pooled_fold_data[fold_name]['peaks']['identifier'].unique())}")
        logger.info(f" Total Nonpeaks: {len(pooled_fold_data[fold_name]['nonpeaks'])} regions")
        logger.info(f" Unique Nonpeaks: {len(pooled_fold_data[fold_name]['nonpeaks']['identifier'].unique())}")

    # deduplicate nonpeaks per fold 
    for fold_name, fold_dct in pooled_fold_data.items():
        fold_dct['nonpeaks'] = fold_dct['nonpeaks'].drop_duplicates(subset=['identifier'])
        logger.info(f"Deduplicated nonpeaks for {fold_name}: {len(fold_dct['nonpeaks'])} regions")

    # Combine peaks and deduplicate
    combined_peaks = pd.concat([pooled_fold_data[fold_name]['peaks'] for fold_name in config_dict.data.fold_names], ignore_index=True)
    combined_peaks = combined_peaks.drop_duplicates(subset=['identifier'])
    logger.info(f"Combined peaks: {len(combined_peaks)} regions")

    # Add fold-specific split columns to combined peaks
    logger.info("Adding fold-specific split columns to combined peaks:")
    for fold_name in config_dict.data.fold_names:
        combined_peaks[f"{fold_name}__splits"] = None
        
        # Use the mapping to assign splits
        for split_name, identifiers in fold_splits_to_identifier[fold_name]['peak'].items():
            mask = combined_peaks['identifier'].isin(identifiers)
            combined_peaks.loc[mask, f"{fold_name}__splits"] = split_name
            
        null_count = combined_peaks[f"{fold_name}__splits"].isna().sum()
        logger.info(f"  {fold_name}: {null_count} peaks have no split assignments")

    # save combined peaks and per fold nonpeaks as tsv files
    # peaks 
    filepath = os.path.join(config_dict.data.data_output_dir, "peaks.all_folds.tsv")
    combined_peaks.to_csv(filepath, sep='\t', index=False)
    logger.info(f"Saved combined peaks to: {filepath}")
    
    # nonpeaks - add splits column and save
    logger.info("Adding splits column to nonpeaks:")
    for fold_name, fold_dct in pooled_fold_data.items():
        # Add splits column to this fold's nonpeaks
        nonpeaks_df = fold_dct['nonpeaks']
        nonpeaks_df['splits'] = None
        
        # Use the mapping to assign splits
        for split_name, identifiers in fold_splits_to_identifier[fold_name]['nonpeak'].items():
            mask = nonpeaks_df['identifier'].isin(identifiers)
            nonpeaks_df.loc[mask, 'splits'] = split_name

        null_count = nonpeaks_df['splits'].isna().sum()
        logger.info(f"  {fold_name}: {null_count} nonpeaks have no split assignments")
        
        logger.info(f"Added splits column to {fold_name} nonpeaks")
        
        filepath = os.path.join(config_dict.data.data_output_dir, f"nonpeaks.{fold_name}.tsv")
        fold_dct['nonpeaks'].to_csv(filepath, sep='\t', index=False)
        logger.info(f"Saved nonpeaks for {fold_name} to: {filepath}")
        

if __name__ == "__main__":
    # Create configuration
    config_dict = ConfigDict()
    
    config_dict.data = ConfigDict()
    config_dict.data.fold_names = [f"fold_{i}" for i in range(5)]
    
    base_dir = os.path.join(
        MANUSCRIPT_EXPERIMENTS_DIR,
        "section_1__ConditionalLM_as_diagnostic_probe",
        "workspace",
        "data",
        "ATAC",
        "K562__ENCSR868FGK",
    )
    config_dict.data.data_dir = os.path.join(
        base_dir,
        "chrombpnet_annotation",
    )
    
    config_dict.data.fold_specific_peak_region_template = os.path.join(
        config_dict.data.data_dir,
        "ENCFF991RUK",
        "{}", # fold name
        "peaks.{}set.{}.ENCSR868FGK.bed.gz") # NOTE: first {} is fold name, second is split name, third is fold name
    config_dict.data.fold_specific_nonpeak_region_template = os.path.join(
        config_dict.data.data_dir,
        "ENCFF991RUK",
        "{}", # fold name
        "nonpeaks.{}set.{}.ENCSR868FGK.bed.gz") # NOTE: first {} is fold name, second is split name, third is fold name
    
    # Add public observed BigWig file path
    config_dict.data.public_observed_bigwig = os.path.join(
        config_dict.data.data_dir,
        "ENCFF874FUM.bigWig"  
    )
    
    config_dict.data.data_output_dir = os.path.join(
        base_dir,
        "preprocessed",
        "regions",
    )
        
    # Run main function
    main(config_dict)