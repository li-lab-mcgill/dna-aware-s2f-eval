import json
import pdb
import os
import sys
import pandas as pd
from ml_collections import ConfigDict
import pyranges as pr
import matplotlib.pyplot as plt
import numpy as np
import logging

def setup_logging(output_dir, logger_name):
    """Setup logging to save all output to files without timestamps"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Create file handler
    log_file = os.path.join(output_dir, f'{logger_name}.log')
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.INFO)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create formatter without timestamp
    formatter = logging.Formatter('%(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def load_bed_gz_file(filepath, logger):
    """Load a compressed BED file (.bed.gz)"""
    if not os.path.exists(filepath):
        logger.warning(f"File not found: {filepath}, exiting...")
        raise FileNotFoundError(f"File not found: {filepath}")
    
    df = pd.read_csv(filepath, sep='\t', compression='gzip', header=None)
    
    # Assign standard BED column names (pyranges format)
    bed_columns = ['Chromosome', 'Start', 'End'] + \
          [f"col_{i}" for i in range(4, 10)] + ["Summit"]
    if len(df.columns) == len(bed_columns): # MACS2 narrowpeak format
        df.columns = bed_columns
    elif len(df.columns) == 3: # Chrom, Start, End
        df.columns = bed_columns[:3]
    else:
        raise ValueError(f"File {filepath} has {len(df.columns)} columns, which is not supported. Please check the file format.")
    
    # Convert numeric columns
    df['Start'] = pd.to_numeric(df['Start'])
    df['End'] = pd.to_numeric(df['End'])
    if "Summit" in df.columns:
        df['Summit'] = pd.to_numeric(df['Summit'])
    else:
        df['Summit'] = (df['End'] - df['Start']) // 2
    
    logger.info(f"Successfully loaded BED file: {filepath}")
    logger.info(f"Shape: {df.shape}")
    return df


def load_fold_data(config_dict, logger):
    """Load peak and nonpeak regions for train/validation/test sets"""

    datasets = {}
    sets = ['training', 'validation', 'test']

    for fold_name in config_dict.data.fold_names:
        logger.info(f"\n=== Loading fold data for {fold_name} ===")
        datasets[f'{fold_name}'] = {}
        
        for set_name in sets:
            # Load peak regions
            peak_filepath = config_dict.data.fold_specific_peak_region_template.format(fold_name, set_name, fold_name)
            peak_df = load_bed_gz_file(peak_filepath, logger)
            
            # Load nonpeak regions  
            nonpeak_filepath = config_dict.data.fold_specific_nonpeak_region_template.format(fold_name, set_name, fold_name)
            nonpeak_df = load_bed_gz_file(nonpeak_filepath, logger)
            
            datasets[f'{fold_name}'][f'peak__{set_name}'] = peak_df
            datasets[f'{fold_name}'][f'nonpeak__{set_name}'] = nonpeak_df
            
            logger.info(f"{set_name.capitalize()} set:")
            logger.info(f"  Peak regions: {len(peak_df):,}")
            logger.info(f"  Nonpeak regions: {len(nonpeak_df):,}")
    
    return datasets