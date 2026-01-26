#!/bin/bash

# Set up environment
export TMPDIR="/home/mcb/users/dcakma3/multi_res_bench/workspace/temp"


# Create output directory if it doesn't exist
output_path="/home/mcb/users/dcakma3/multi_res_bench_v2/multimodal_masked_contrastive_main/manuscript_experiments/section_1__ConditionalLM_as_diagnostic_probe/workspace/data/TF_ChIP/K562__CTCF__ENCSR000EGM/filtered_bams/main_experiment"
mkdir -p "${output_path}"

# Change to output directory before downloading
cd "${output_path}"

# files.txt already in the output directory
xargs -n 1 curl -O -L < files.txt