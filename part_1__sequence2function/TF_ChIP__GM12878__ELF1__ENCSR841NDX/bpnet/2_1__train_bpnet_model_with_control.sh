#!/bin/bash

# Set up enrironment
export CUDA_VISIBLE_DEVICES="3"

FOLD_NAME="fold_0"
BASE_DIR="/home/mcb/users/dcakma3/multi_res_bench_v2/multimodal_masked_contrastive_main/manuscript_experiments/section_1__ConditionalLM_as_diagnostic_probe"

cd $BASE_DIR/workspace/runs/TF_ChIP/GM12878__ELF1__ENCSR841NDX/S2F/bpnetlite__with_control_track_input/$FOLD_NAME

bpnet fit -p fit_config.json
