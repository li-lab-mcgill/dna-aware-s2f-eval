# Multi-headed cgLM critic evaluation

This directory is intended for the evaluation of multi-headed cgLM critics used in the manuscript's ATAC GM12878 analyses.

## Dataset
- `ATAC__GM12878__ENCSR637XSC`

## Figure Mapping

### Critics trained on S2F-predicted tracks before debiasing (Figure 3b)
- `pre_debiasing`
  - `bpnet__predicted`
  - `alphagenome__predicted`
  - `chrombpnet__nobias_model__predicted`

### Critics trained on S2F-predicted tracks after debiasing (Figure 5c and Figure S7)
- `post_debiasing`
  - `bpnet__predicted`
  - `alphagenome__predicted`

## Inputs
This directory expects the matched GM12878 ATAC dataset under `workspace/datasets/ATAC__GM12878__ENCSR637XSC/` and the released critic checkpoints under `workspace/models/ATAC__GM12878__ENCSR637XSC___checkpoints/multi_headed_cglm/`.

## Notes
Exact reproduction scripts and commands for this directory are in preparation.
