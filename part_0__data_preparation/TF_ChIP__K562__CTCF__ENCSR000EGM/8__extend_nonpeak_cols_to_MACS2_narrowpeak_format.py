#!/usr/bin/env python3

import pandas as pd
import os

# Define paths
base_dir = "/home/mcb/users/dcakma3/multi_res_bench_v2/multimodal_masked_contrastive_main/manuscript_experiments/section_1__ConditionalLM_as_diagnostic_probe/workspace/data/TF_ChIP/K562__CTCF__ENCSR000EGM"
nonpeaks_dir = os.path.join(base_dir, "nonpeaks")
preprocessed_dir = os.path.join(base_dir, "preprocessed")

# Create preprocessed directory if it doesn't exist
os.makedirs(preprocessed_dir, exist_ok=True)

# Iterate over 5 cross-validation folds
for fold_id in range(5):
    fold_name = f"fold_{fold_id}"
    
    print(f"Processing {fold_name}...")
    
    # Load both replicates for this fold
    replicate_files = [
        os.path.join(nonpeaks_dir, f"nonpeaks.{fold_name}.replicate_1.bed"),
        os.path.join(nonpeaks_dir, f"nonpeaks.{fold_name}.replicate_2.bed")
    ]
    
    # Read and concatenate the replicate files
    dfs = []
    for file_path in replicate_files:
        df = pd.read_csv(file_path, sep='\t', header=None, names=['chr', 'start', 'end'])
        dfs.append(df)
    
    # Concatenate tables
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Remove exact duplicates
    combined_df = combined_df.drop_duplicates().reset_index(drop=True)
    
    print(f"  {fold_name}: {len(combined_df)} unique nonpeak regions")
    
    # Save basic format
    basic_output_path = os.path.join(preprocessed_dir, f"nonpeaks.{fold_name}.bed")
    combined_df.to_csv(basic_output_path, sep='\t', header=False, index=False)
    
    # Calculate length and assert it's exactly 2114
    combined_df['length'] = combined_df['end'] - combined_df['start']
    assert all(combined_df['length'] == 2114), f"Not all regions in {fold_name} are exactly 2114 bp long!"
    
    # Calculate relative summit (length/2)
    relative_summit = 2114 // 2  # 1057
    
    # Create MACS2 format (10 columns)
    macs2_df = pd.DataFrame({
        'chr': combined_df['chr'],
        'start': combined_df['start'],
        'end': combined_df['end'],
        'name': '.',
        'score': ".",
        'strand': '.',
        'signalValue': '.',
        'pValue': ".",
        'qValue': ".",
        'peak': relative_summit
    })
    
    # Save extended format
    ext_output_path = os.path.join(preprocessed_dir, f"nonpeaks.{fold_name}.ext_col.bed")
    macs2_df.to_csv(ext_output_path, sep='\t', header=False, index=False)
    
    print(f"  Saved {fold_name} basic format: {basic_output_path}")
    print(f"  Saved {fold_name} MACS2 format: {ext_output_path}")

print("All folds processed successfully!")
