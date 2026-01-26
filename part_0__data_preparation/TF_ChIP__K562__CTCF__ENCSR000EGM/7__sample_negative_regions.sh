#!/bin/bash

# Set up environment
export TMPDIR="/home/mcb/users/dcakma3/multi_res_bench/workspace/temp"

# Create output directory if it doesn't exist
output_path="/home/mcb/users/dcakma3/multi_res_bench_v2/multimodal_masked_contrastive_main/experiments/section__ConditionalLM_as_diagnostic_probe/workspace/data/K562__CTCF__ENCSR000EGM/nonpeaks"
mkdir -p "${output_path}"

# Change to output directory before downloading
cd "${output_path}"

# run bpnetlite commandline tool to sample negative regions
# NOTE: used plus strand main experiment bigwig file
BIGWIG_FILEPATH="/home/mcb/users/dcakma3/multi_res_bench_v2/multimodal_masked_contrastive_main/manuscript_experiments/section_1__ConditionalLM_as_diagnostic_probe/workspace/data/TF_ChIP/K562__CTCF__ENCSR000EGM/preprocessed/main_experiment.plus.bw"
PEAKS_FILEPATH="/home/mcb/users/dcakma3/multi_res_bench_v2/multimodal_masked_contrastive_main/manuscript_experiments/section_1__ConditionalLM_as_diagnostic_probe/workspace/data/TF_ChIP/K562__CTCF__ENCSR000EGM/preprocessed/peaks.bed"
REFERENCE_GENOME_FILEPATH="/home/mcb/users/dcakma3/multi_res_bench_v2/multimodal_masked_contrastive_main/manuscript_experiments/section_1__ConditionalLM_as_diagnostic_probe/workspace/data/genomes/hg38/hg38.fa"
OUTPUT_DIR="/home/mcb/users/dcakma3/multi_res_bench_v2/multimodal_masked_contrastive_main/manuscript_experiments/section_1__ConditionalLM_as_diagnostic_probe/workspace/data/TF_ChIP/K562__CTCF__ENCSR000EGM/nonpeaks"
LOG_DIR="${OUTPUT_DIR}/logs"
mkdir -p "${LOG_DIR}"

# NOTE: run with hydra conda env
# Iterate over 5 cross-validation folds and generate 2 replicates each
for fold_id in {0..4}; do
    fold_name="fold_${fold_id}"
    
    for replicate_id in {1..2}; do
        replicate_name="replicate_${replicate_id}"
        output_filename="nonpeaks.${fold_name}.${replicate_name}.bed"
        log_filename="${fold_name}.${replicate_name}.nonpeaks.log"
        
        bpnet negatives \
          --peaks "${PEAKS_FILEPATH}" \
          --fasta "${REFERENCE_GENOME_FILEPATH}" \
          --bigwig "${BIGWIG_FILEPATH}" \
          --beta 0.5 \
          --max_n_perc 0.05 \
          --output "${OUTPUT_DIR}/${output_filename}" \
          --verbose > "${LOG_DIR}/${log_filename}" 2>&1
    done
done
