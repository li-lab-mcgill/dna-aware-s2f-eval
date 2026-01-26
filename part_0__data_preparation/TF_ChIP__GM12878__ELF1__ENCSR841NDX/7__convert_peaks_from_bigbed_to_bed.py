#!/usr/bin/env python3

import os
import subprocess
import sys

def convert_bigbed_to_bed(bigbed_file, output_bed_file):
    """
    Convert BigBed file to BED format using bigBedToBed tool.
    
    Args:
        bigbed_file (str): Path to input BigBed file
        output_bed_file (str): Path to output BED file
    """
    
    # Check if input file exists
    if not os.path.exists(bigbed_file):
        raise FileNotFoundError(f"BigBed file not found: {bigbed_file}")
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_bed_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Use the specific bigBedToBed tool path
    bigbed_tool = "/home/mcb/users/dcakma3/multi_res_bench_v2/multimodal_masked_contrastive_main/utils/ucsctools/bigBedToBed"
    cmd = [bigbed_tool, bigbed_file, output_bed_file]
    
    try:
        print(f"Converting {bigbed_file} to {output_bed_file}...")
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"Successfully converted to {output_bed_file}")
        
        # Print some stats
        with open(output_bed_file, 'r') as f:
            line_count = sum(1 for line in f)
        print(f"Output BED file contains {line_count} regions")
        
    except subprocess.CalledProcessError as e:
        print(f"Error converting BigBed to BED: {e}")
        print(f"Command: {' '.join(cmd)}")
        print(f"Error output: {e.stderr}")
        sys.exit(1)

def main():
    # Hardcoded paths
    bigbed_file = "/home/mcb/users/dcakma3/multi_res_bench_v2/multimodal_masked_contrastive_main/manuscript_experiments/section_1__ConditionalLM_as_diagnostic_probe/workspace/data/TF_ChIP/GM12878__ELF1__ENCSR841NDX/peaks/ENCFF692SMY.bigBed"
    output_bed_file = "/home/mcb/users/dcakma3/multi_res_bench_v2/multimodal_masked_contrastive_main/manuscript_experiments/section_1__ConditionalLM_as_diagnostic_probe/workspace/data/TF_ChIP/GM12878__ELF1__ENCSR841NDX/preprocessed/peaks.bed"
    
    convert_bigbed_to_bed(bigbed_file, output_bed_file)

if __name__ == "__main__":
    main()