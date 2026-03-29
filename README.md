# DNA-aware evaluation and debiasing of sequence-to-function models

[![DOI](https://zenodo.org/badge/541773560.svg)](https://doi.org/10.5281/zenodo.19210955)

> This repository accompanies our manuscript on DNA-aware evaluation and debiasing of sequence-to-function (S2F) models. We first use track-conditional genome language models (cgLMs) as DNA-aware probes of experimental and S2F-predicted functional genomics tracks. We then present a preliminary critic-guided debiasing framework that uses a frozen cgLM critic to define an objective function for reducing the discrepancy it detects between experimental and S2F-predicted tracks. Here, “bias” is defined operationally as the discrepancy between experimental and S2F-predicted tracks under the cgLM probe.
>
> Cakmakci and Li. "DNA-aware evaluation and debiasing of sequence-to-function models" (manuscript in preparation)
>
> The trained model checkpoints and preprocessed datasets required to reproduce our results are available [here](https://doi.org/10.5281/zenodo.19210955).
>
> License: [MIT](LICENSE)

## Authors

Doruk Cakmakci, Yue Li

## Questions & comments

[firstauthorname].[firstauthorsurname]@mail.mcgill.ca  
[lastauthorfirstname][lastauthorsurname]@cs.mcgill.ca

## Table of Contents
- [Environment Setup](#environment-setup)
- [Workspace Preparation](#workspace-preparation)
- [Dataset Contents](#dataset-contents)
- [Model Checkpoint Contents](#model-checkpoint-contents)
- [Single-headed cgLM Experiments](#single-headed-cglm-experiments)
- [Multi-headed cgLM Critic Experiments](#multi-headed-cglm-critic-experiments)
- [Critic-Guided Profile-Shape Editing (CGPSE) Experiments](#critic-guided-profile-shape-editing-cgpse-experiments)
- [Citation](#citation)
- [License](#license)

## Environment Setup
This repository uses a conda-based environment. Preparation of the public environment YAML file, compatible with Linux, is in preparation.

## Workspace Preparation
The repository expects preprocessed datasets and trained checkpoints under `workspace/`. Start by creating the workspace skeleton:

```bash
./init_workspace.sh
```

This creates `models` and `datasets` directories under `workspace`.

Our study analyzes two assay types across two cell lines, with a particular focus on GM12878 ATAC-seq. Accordingly, the Zenodo archive contains four dataset-checkpoint pairs:

- `ATAC__GM12878__ENCSR637XSC`
- `ATAC__K562__ENCSR868FGK`
- `TF_ChIP__GM12878__ELF1__ENCSR841NDX`
- `TF_ChIP__K562__CTCF__ENCSR000EGM`

You may reproduce the analyses for any pair of interest by downloading and unpacking the corresponding archives under `models` and `datasets`. For example, if you would like to analyze GM12878 ATAC-seq and GM12878 ELF1 ChIP-seq, your workspace should look like this:

```text
workspace/
|-- datasets/
|   |-- ATAC__GM12878__ENCSR637XSC/
|   |   |-- peaks.all_folds.zarr/
|   |   `-- nonpeaks.fold_0.zarr/
|   |-- TF_ChIP__GM12878__ELF1__ENCSR841NDX/
|   |   |-- peaks.all_folds.zarr/
|   |   `-- nonpeaks.fold_0.zarr/
`-- models/
    |-- ATAC__GM12878__ENCSR637XSC___checkpoints/
    `-- TF_ChIP__GM12878__ELF1__ENCSR841NDX___checkpoints/
```

## Dataset Contents
Prepared datasets are provided for two assays, ATAC-seq and TF ChIP-seq, across two cell lines, GM12878 and K562. Each dataset bundle contains the peak and nonpeak zarr stores used in the manuscript, together with reference DNA, observed tracks, model-based tracks, and precomputed masks used for training and evaluation.

Each zarr store contains the following components:

- `_auxiliary/`
  - includes `regions_df.tsv` and `split_indices.npz`
- `reference_dna/`
  - contains the genomic sequences used by the models
- `observed_tracks/`
  - contains the experimental tracks
- `fold_0__model_based_tracks/` and, where applicable, `fold_agnostic__model_based_tracks/`
  - contain the S2F-predicted tracks
- `precomputed_masks/`
  - contains the DNA masks used for training and evaluation

For ATAC-seq, the released zarr stores contain:

- observed tracks
  - `observed_track_summitcentered1000bp_symmetric500bpshift`
- S2F-predicted tracks
  - `bpnet_predicted_track_summitcentered1000bp_symmetric500bpshift`
  - `chrombpnet_nobias_predicted_track_summitcentered1000bp_symmetric500bpshift`
  - `alphagenome_predicted_track__summitcentered1000bp_symmetric500bpshift`

For TF ChIP-seq, the released zarr stores contain:

- observed tracks
  - `observed_main_track_*`
  - `observed_control_track_*`
- S2F-predicted tracks
  - `bpnet_with_control_predicted_track_*`
  - `bpnet_without_control_predicted_track_*`

Rows within each zarr store are matched to `_auxiliary/regions_df.tsv`, and dataset splits are provided in `_auxiliary/split_indices.npz`. The precomputed masks include the replicate sets used for the main manuscript evaluations. For exact file trees, naming conventions, and assay-specific details, please refer to `datasets_readme.txt` in the Zenodo archive.

## Model Checkpoint Contents
Model checkpoint bundles are also provided for ATAC-seq and TF ChIP-seq in GM12878 and K562. The GM12878 ATAC-seq bundle contains the checkpoint families used for the manuscript's broader GM12878 ATAC-seq analyses, including the single-headed cgLMs, the single-headed cgLM ablation experiments, the multi-headed cgLM critics, and the preliminary CGPSE models (`dna_free_dae` and `dna_aware_editor`).

The remaining three bundles, GM12878 TF ChIP-seq, K562 ATAC-seq, and K562 TF ChIP-seq, follow a reduced structure. They contain the single-headed cgLM discovery models used for the main diagnostic evaluations, together with the base S2F models used in those comparisons.

These model bundles are distributed as PyTorch Lightning checkpoints rather than drop-in `nn.Module` objects. Please use the checkpoint loading utilities provided in this repository for the relevant model families.

For exact checkpoint trees, fold organization, naming conventions, and additional checkpoint loading notes, please refer to `checkpoints_readme.txt` in the Zenodo archive.

## Single-headed cgLM Experiments

### Discovery models (Figures 1, 2, S4, and S5)
Single-headed cgLM discovery models reproduce the manuscript's main diagnostic analyses for Figures 1, 2, S4, and S5. See [scripts/single_headed_cglm/discovery_experiments/README.md](scripts/single_headed_cglm/discovery_experiments/README.md).

### Ablation models (Figure S6)
Ablation models reproduce Figure S6. See [scripts/single_headed_cglm/ablation_experiments/README.md](scripts/single_headed_cglm/ablation_experiments/README.md).

Exact reproduction instructions are in preparation.

## Multi-headed cgLM Critic Experiments
Multi-headed cgLM critic experiments were performed for `ATAC__GM12878__ENCSR637XSC`.

### Critics trained on S2F-predicted tracks before debiasing (Figure 3b)
Critics trained on S2F-predicted tracks before debiasing reproduce Figure 3b.

### Critics trained on S2F-predicted tracks after debiasing (Figure 5c and Figure S7)
Critics trained on S2F-predicted tracks after debiasing reproduce Figure 5c and Figure S7.

Detailed figure-to-checkpoint mapping is provided in [scripts/multi_headed_cglm/eval/README.md](scripts/multi_headed_cglm/eval/README.md).

Exact reproduction instructions are in preparation.


## Critic-Guided Profile-Shape Editing (CGPSE) Experiments
CGPSE experiments were performed for `ATAC__GM12878__ENCSR637XSC`.

### Evaluation of debiased profile shapes in terms of profile shape fidelity (Figure 5a)
Evaluation of debiased profile shapes in terms of profile shape fidelity corresponds to Figure 5a.

### Evaluation of debiased profile shapes with training objective defining critic model (Figure 5b)
Evaluation of debiased profile shapes with the training objective defining critic model corresponds to Figure 5b.

Detailed figure-to-checkpoint mapping is provided in [scripts/cgpse/README.md](scripts/cgpse/README.md).

Exact reproduction instructions are in preparation.

## Citation
Manuscript in preparation.

## License
This repository is released under the [MIT License](LICENSE).
