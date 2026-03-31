# Single-headed cgLM ablation experiments

This directory covers the single-headed cgLM ablation analyses used in Figure S6.

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


## Notes
- `figS6a__total_count_equalization` reproduces the total-count equalization control.
- `figS6b__cgLM_input_resolution` reproduces the resolution ablation.
- `figS6c__cgLM_input_length` reproduces the locality/input-length ablation.
- Raw evaluation outputs are written under `workspace/logs/single_headed_cglm/ablation_experiments/`, and rendered figures are written under `workspace/plots/single_headed_cglm/ablation_experiments/`.
