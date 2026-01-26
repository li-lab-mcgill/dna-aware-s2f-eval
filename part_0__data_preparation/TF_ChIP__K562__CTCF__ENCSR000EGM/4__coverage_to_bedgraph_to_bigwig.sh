#!/bin/bash

# Set up environment
export TMPDIR="/home/mcb/users/dcakma3/multi_res_bench/workspace/temp"

# things will be saved here
preprocessed_data_dir=/home/mcb/users/dcakma3/multi_res_bench_v2/multimodal_masked_contrastive_main/manuscript_experiments/section_1__ConditionalLM_as_diagnostic_probe/workspace/data/TF_ChIP/K562__CTCF__ENCSR000EGM/preprocessed
mkdir -p "${preprocessed_data_dir}"
cd "${preprocessed_data_dir}"

# Create output directory if it doesn't exist
filtered_bam_dir=/home/mcb/users/dcakma3/multi_res_bench_v2/multimodal_masked_contrastive_main/manuscript_experiments/section_1__ConditionalLM_as_diagnostic_probe/workspace/data/TF_ChIP/K562__CTCF__ENCSR000EGM/filtered_bams
control_experiment_bam_path="${filtered_bam_dir}/control_experiment/ENCFF396DTD.bam"
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


####### Control experiment processing #######

# get coverage of 5' positions of the plus strand
control_plus_bedgraph_path="${preprocessed_data_dir}/control_experiment.plus.bedGraph"
bedtools genomecov -5 -bg -strand + \
        -g ${chrom_sizes_path} -ibam ${control_experiment_bam_path} \
        | sort -k1,1 -k2,2n > ${control_plus_bedgraph_path}

# get coverage of 5' positions of the minus strand
control_minus_bedgraph_path="${preprocessed_data_dir}/control_experiment.minus.bedGraph"
bedtools genomecov -5 -bg -strand - \
        -g ${chrom_sizes_path} -ibam ${control_experiment_bam_path} \
        | sort -k1,1 -k2,2n > ${control_minus_bedgraph_path}

# Convert bedGraph files to bigWig files for plus 
control_plus_bigwig_path="${preprocessed_data_dir}/control_experiment.plus.bw"
bedGraphToBigWig ${control_plus_bedgraph_path} ${chrom_sizes_path} ${control_plus_bigwig_path}

# Convert bedGraph files to bigWig files for minus
control_minus_bigwig_path="${preprocessed_data_dir}/control_experiment.minus.bw"
bedGraphToBigWig ${control_minus_bedgraph_path} ${chrom_sizes_path} ${control_minus_bigwig_path}
