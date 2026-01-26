#!/bin/bash

# Set up enrironment
export CUDA_VISIBLE_DEVICES="3"

FOLD_NAME="fold_0"
BASE_DIR="/home/mcb/users/dcakma3/multi_res_bench_v2/multimodal_masked_contrastive_main/manuscript_experiments/section_1__ConditionalLM_as_diagnostic_probe"

cd $BASE_DIR/workspace/runs/TF_ChIP/K562__CTCF__ENCSR000EGM/S2F/bpnetlite__without_control_track_input/$FOLD_NAME

bpnet fit -p fit_config.json
