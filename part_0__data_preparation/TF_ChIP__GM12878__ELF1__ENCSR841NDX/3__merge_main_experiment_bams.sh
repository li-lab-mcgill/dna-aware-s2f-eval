#!/bin/bash

# Set up environment
export TMPDIR="/home/mcb/users/dcakma3/multi_res_bench/workspace/temp"

# things will be saved here
preprocessed_data_dir=/home/mcb/users/dcakma3/multi_res_bench_v2/multimodal_masked_contrastive_main/manuscript_experiments/section_1__ConditionalLM_as_diagnostic_probe/workspace/data/TF_ChIP/GM12878__ELF1__ENCSR841NDX/preprocessed
mkdir -p "${preprocessed_data_dir}"
cd "${preprocessed_data_dir}"

# Create output directory if it doesn't exist
filtered_bam_dir=/home/mcb/users/dcakma3/multi_res_bench_v2/multimodal_masked_contrastive_main/manuscript_experiments/section_1__ConditionalLM_as_diagnostic_probe/workspace/data/TF_ChIP/GM12878__ELF1__ENCSR841NDX/filtered_bams
main_experiment_bam1_path="${filtered_bam_dir}/main_experiment/ENCFF041QIN.bam"
main_experiment_bam2_path="${filtered_bam_dir}/main_experiment/ENCFF290PIQ.bam"
merged_main_experiment_bam_path="${preprocessed_data_dir}/main_experiment_merged.bam"

# merge
samtools merge -f ${merged_main_experiment_bam_path} ${main_experiment_bam1_path} ${main_experiment_bam2_path}

# index the merged BAM file
samtools index ${merged_main_experiment_bam_path}
