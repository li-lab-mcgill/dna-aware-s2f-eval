#!/bin/bash

# Set up environment
export TMPDIR="/home/mcb/users/dcakma3/multi_res_bench/workspace/temp"

# things will be saved here
preprocessed_data_dir=/home/mcb/users/dcakma3/multi_res_bench_v2/multimodal_masked_contrastive_main/manuscript_experiments/section_1__ConditionalLM_as_diagnostic_probe/workspace/data/TF_ChIP/GM12878__ELF1__ENCSR841NDX/preprocessed
mkdir -p "${preprocessed_data_dir}"
cd "${preprocessed_data_dir}"

# Merged BAM files from scripts 3 and 4
merged_main_experiment_bam_path="${preprocessed_data_dir}/main_experiment_merged.bam"

# chrom sizes file needed below
chrom_sizes_path="/home/mcb/users/dcakma3/multi_res_bench_v2/data/genomes/hg38/hg38.chrom.sizes"


####### Main experiment processing #######

# get coverage of 5' positions of the plus strand
main_plus_bedgraph_path="${preprocessed_data_dir}/main_experiment.plus.bedGraph"
bedtools genomecov -5 -bg -strand + \
        -g ${chrom_sizes_path} -ibam ${merged_main_experiment_bam_path} \
        | sort -k1,1 -k2,2n > ${main_plus_bedgraph_path}

# get coverage of 5' positions of the minus strand
main_minus_bedgraph_path="${preprocessed_data_dir}/main_experiment.minus.bedGraph"
bedtools genomecov -5 -bg -strand - \
        -g ${chrom_sizes_path} -ibam ${merged_main_experiment_bam_path} \
        | sort -k1,1 -k2,2n > ${main_minus_bedgraph_path}

# Convert bedGraph files to bigWig files for plus 
main_plus_bigwig_path="${preprocessed_data_dir}/main_experiment.plus.bw"
bedGraphToBigWig ${main_plus_bedgraph_path} ${chrom_sizes_path} ${main_plus_bigwig_path}

# Convert bedGraph files to bigWig files for minus
main_minus_bigwig_path="${preprocessed_data_dir}/main_experiment.minus.bw"
bedGraphToBigWig ${main_minus_bedgraph_path} ${chrom_sizes_path} ${main_minus_bigwig_path}

