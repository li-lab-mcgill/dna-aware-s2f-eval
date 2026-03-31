# Single-headed cgLM diagnostic experiments

This directory contains the single-headed cgLM scripts used for the manuscript's main diagnostic analyses.

## Figure Mapping
- Figures 1 and 2
- Figures S4 and S5

## Datasets
- `ATAC__GM12878__ENCSR637XSC`
- `ATAC__K562__ENCSR868FGK`
- `TF_ChIP__GM12878__ELF1__ENCSR841NDX`
- `TF_ChIP__K562__CTCF__ENCSR000EGM`

## Checkpoint targets

### ATAC-seq
For both `ATAC__GM12878__ENCSR637XSC` and `ATAC__K562__ENCSR868FGK`, the released checkpoint targets are:

- `observed`
- `dna_only`
- `bpnet__predicted`
- `chrombpnet__nobias_model__predicted`
- `alphagenome__predicted`

### TF ChIP-seq
For both `TF_ChIP__GM12878__ELF1__ENCSR841NDX` and `TF_ChIP__K562__CTCF__ENCSR000EGM`, the released checkpoint targets are:

- `dna_only`
- `observed__main_experiment`
- `observed__control_experiment`
- `predicted__bpnetlite_with_control`
- `predicted__bpnetlite_without_control`

## Inputs
This directory expects:

- matched dataset bundles under `workspace/datasets/`
- released single-headed cgLM discovery checkpoints under `workspace/models/*___checkpoints/single_headed_cglm/diagnostic_experiments/`
- checked-in checkpoint-bundle configs under `configs/single_headed_cglm/diagnostic_experiments/`

## Notes
- `fig1__GM12878_bars` and `figS4__K562_bars` correspond to the held-out peak evaluations shown in Figures 1 and S4.
- `fig2__GM12878_table` and `figS5__K562_table` correspond to the peak, nonpeak, shuffled, and JSD summaries used in Figures 2 and S5.
- Raw evaluation outputs are written under `workspace/logs/single_headed_cglm/diagnostic_experiments/`, and rendered figures are written under `workspace/plots/single_headed_cglm/diagnostic_experiments/`.
