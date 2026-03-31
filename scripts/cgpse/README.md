# Critic-Guided Profile-Shape Editing (CGPSE)

This directory covers the manuscript's preliminary CGPSE analyses on GM12878 ATAC-seq.

## Dataset
- `ATAC__GM12878__ENCSR637XSC`

## Figure Mapping

### Critic-based evaluation of original and edited profile shapes (Figure 5a)
- `dna_free_dae`
  - `bpnet__predicted`
  - `alphagenome__predicted`
- `dna_aware_editor`
  - `bpnet__predicted`
  - `alphagenome__predicted`

### Profile-shape fidelity evaluation (Figure 5b)
- `dna_free_dae`
  - `bpnet__predicted`
  - `alphagenome__predicted`
- `dna_aware_editor`
  - `bpnet__predicted`
  - `alphagenome__predicted`

## Notes
- `fig5a__pre_debiasing_critic_based_eval/` contains critic-based evaluation scripts for original versus edited outputs.
- `fig5b__profile_shape_fidelity_eval/` contains the profile-shape fidelity summaries used in Figure 5b.
- Raw evaluation outputs are written under `workspace/logs/cgpse/`, and rendered figures are written under `workspace/plots/cgpse/`.
