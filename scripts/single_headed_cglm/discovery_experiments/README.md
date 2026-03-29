# Single-headed cgLM discovery experiments

This directory is intended for the single-headed cgLM discovery experiments used for the manuscript's main diagnostic analyses.

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
- `chrombpnet__bias_model__predicted`
- `chrombpnet__full__predicted`
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
This directory expects the matched dataset bundles under `workspace/datasets/` and the released single-headed cgLM discovery checkpoints under `workspace/models/*___checkpoints/single_headed_cglm/`.

## Notes
Exact reproduction scripts and commands for this directory are in preparation. For bundle structure and checkpoint naming details, refer to the top-level README and the Zenodo archive notes.
