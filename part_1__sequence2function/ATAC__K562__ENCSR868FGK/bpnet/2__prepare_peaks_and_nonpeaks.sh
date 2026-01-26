#!/bin/bash

# Set up environment
export TMPDIR="/home/mcb/users/dcakma3/multi_res_bench/workspace/temp"

FOLD_NAMES=("fold_0" "fold_1" "fold_2" "fold_3" "fold_4")
BASE_DIR="/home/mcb/users/dcakma3/multi_res_bench_v2/multimodal_masked_contrastive_main/manuscript_experiments/section_1__ConditionalLM_as_diagnostic_probe/workspace/data"
INPUT_DIR="$BASE_DIR/ATAC/K562__ENCSR868FGK/chrombpnet_annotation/ENCFF991RUK"
OUTPUT_DIR="$BASE_DIR/ATAC/K562__ENCSR868FGK/preprocessed/chrombpnet_annotation__regions"
mkdir -p "$OUTPUT_DIR"

# peaks are .bed.gz files unzip them to a target directory
BED_FILE="$INPUT_DIR/peaks.all_input_regions.ENCSR868FGK.bed.gz"
TARGET_FILE="$OUTPUT_DIR/peaks.bed"
gunzip -c "$BED_FILE" > "$TARGET_FILE"

# nonpeaks are fold specific and also .bed.gz files
for FOLD_NAME in "${FOLD_NAMES[@]}"; do
    echo "Processing fold: $FOLD_NAME"
    BED_FILE="$INPUT_DIR/$FOLD_NAME/nonpeaks.all_input_regions.$FOLD_NAME.ENCSR868FGK.bed.gz"
    TARGET_FILE="$OUTPUT_DIR/nonpeaks.$FOLD_NAME.bed"
    gunzip -c "$BED_FILE" > "$TARGET_FILE"
done

echo "Done"


