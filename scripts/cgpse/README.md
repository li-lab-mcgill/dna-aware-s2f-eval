# Critic-Guided Profile-Shape Editing (CGPSE)

This directory is intended for the manuscript's preliminary CGPSE analyses on GM12878 ATAC-seq.

## Dataset
- `ATAC__GM12878__ENCSR637XSC`

## Figure Mapping

### Evaluation of debiased profile shapes in terms of profile shape fidelity (Figure 5a)
- `dna_free_dae`
  - `bpnet__predicted`
  - `alphagenome__predicted`
- `dna_aware_editor`
  - `bpnet__predicted`
  - `alphagenome__predicted`

### Evaluation of debiased profile shapes with training objective defining critic model (Figure 5b)
- `dna_free_dae`
  - `bpnet__predicted`
  - `alphagenome__predicted`
- `dna_aware_editor`
  - `bpnet__predicted`
  - `alphagenome__predicted`

## Inputs
This directory expects the matched GM12878 ATAC dataset under `workspace/datasets/ATAC__GM12878__ENCSR637XSC/` and the released CGPSE checkpoints under `workspace/models/ATAC__GM12878__ENCSR637XSC___checkpoints/`.

## Notes
Exact reproduction scripts and commands for this directory are in preparation.
