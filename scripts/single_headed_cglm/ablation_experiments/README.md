# Single-headed cgLM ablation experiments

This directory is intended for the single-headed cgLM ablation analyses used in Figure S6.

## Dataset
- `ATAC__GM12878__ENCSR637XSC`

## Figure Mapping

### S2F-predicted cgLM conditioning track total count rescaling (Figure S6a)
- `ablation_1__count_equalization`
  - `observed`
  - `bpnet__predicted`
  - `chrombpnet__nobias_model__predicted`
  - `alphagenome__predicted`

### cgLM conditioning track resolution (Figure S6b)
- `ablation_2__resolution__shift_aug`
  - `observed`
  - `bpnet__predicted`
  - `chrombpnet__nobias_model__predicted`
  - `alphagenome__predicted`

### cgLM input width (Figure S6c)
- `ablation_3__locality`
  - `observed`
  - `bpnet__predicted`
  - `chrombpnet__nobias_model__predicted`
  - `alphagenome__predicted`

## Inputs
This directory expects the matched GM12878 ATAC dataset under `workspace/datasets/ATAC__GM12878__ENCSR637XSC/` and the released ablation checkpoints under `workspace/models/ATAC__GM12878__ENCSR637XSC___checkpoints/single_headed_cglm/ablation_experiments/`.

## Notes
Exact reproduction scripts and commands for this directory are in preparation.
