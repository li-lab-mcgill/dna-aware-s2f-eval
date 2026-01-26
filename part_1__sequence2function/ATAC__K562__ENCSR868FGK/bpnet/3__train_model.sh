#!/bin/bash


# Set up environment
export TMPDIR="/home/mcb/users/dcakma3/multi_res_bench/workspace/temp"
export CUDA_VISIBLE_DEVICES="1"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

FOLD_NAME="fold_0"
BASE_DIR="/home/mcb/users/dcakma3/multi_res_bench_v2/multimodal_masked_contrastive_main/manuscript_experiments/section_1__ConditionalLM_as_diagnostic_probe/workspace/data"
ASSAY_DATA_DIR="$BASE_DIR/ATAC/K562__ENCSR868FGK"
OUTPUT_DIR="/home/mcb/users/dcakma3/multi_res_bench_v2/multimodal_masked_contrastive_main/manuscript_experiments/section_1__ConditionalLM_as_diagnostic_probe/workspace/runs/ATAC/K562__ENCSR868FGK/S2F/bpnet/$FOLD_NAME"
mkdir -p "$OUTPUT_DIR"

# # ========= 7. Train ChromBPNet Model ========= #
echo "Running ChromBPNet single stage model training..."
chrombpnet single_stage_model__train \
    -ibw "$ASSAY_DATA_DIR/chrombpnet_annotation/ENCFF874FUM.bigWig" \
    -d "ATAC" \
    -g "$BASE_DIR/genomes/hg38/hg38.fa" \
    -c "$BASE_DIR/genomes/hg38/hg38.chrom.subset.sizes" \
    -p "$ASSAY_DATA_DIR/preprocessed/chrombpnet_annotation__regions/peaks.bed" \
    -n "$ASSAY_DATA_DIR/preprocessed/chrombpnet_annotation__regions/nonpeaks.$FOLD_NAME.bed" \
    -fl "$BASE_DIR/genomes/hg38/splits/$FOLD_NAME.json" \
    -o "$OUTPUT_DIR"

