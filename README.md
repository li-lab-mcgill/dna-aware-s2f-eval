# DNA-aware evaluation and debiasing of sequence-to-function models

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19297875.svg)](https://doi.org/10.5281/zenodo.19297875)

> This repository accompanies our manuscript on DNA-aware evaluation and debiasing of sequence-to-function (S2F) models. We first use track-conditional genome language models (cgLMs) as DNA-aware probes of experimental and S2F-predicted functional genomics tracks. We then present a preliminary critic-guided debiasing framework that uses a frozen cgLM critic to define a differentiable objective function for reducing the discrepancy it detects between experimental and S2F-predicted tracks. Here, “bias” is defined operationally as the discrepancy between experimental and S2F-predicted tracks under the cgLM probe.
>
> Cakmakci and Li. "DNA-aware evaluation and debiasing of sequence-to-function models" (manuscript in preparation)
>
> The trained model checkpoints and preprocessed datasets required to reproduce our results are available [here](https://doi.org/10.5281/zenodo.19297875).
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
- [Critic-Guided Profile-Shape Editing (CGPSE) Experiments](#critic-guided-profile-shape-editing-cgpse-experiments)
- [Citation](#citation)
- [License](#license)

## Environment Setup
This repository uses a conda-managed Python 3.10 environment named `cglm`, with Python packages pinned in [`requirements.txt`](requirements.txt).

To prepare the environment once, run:

```bash
./init_env.sh
```

This script creates the `cglm` conda environment if it does not already exist, upgrades `pip`, and installs the pinned Python dependencies from [`requirements.txt`](requirements.txt).

After that, activate the environment when you want to work in the repo:

```bash
conda activate cglm
```

## Workspace Preparation
The repository expects preprocessed datasets and trained model checkpoints under `workspace/`. Start by creating the workspace skeleton:

```bash
./init_workspace.sh
```

This creates `datasets`, `models`, `logs`, and `plots` directories under `workspace/`.

Our study analyzes two assay types across two cell lines. Accordingly, the Zenodo archive contains four dataset-checkpoint pairs:

- `ATAC__GM12878__ENCSR637XSC`
- `ATAC__K562__ENCSR868FGK`
- `TF_ChIP__GM12878__ELF1__ENCSR841NDX`
- `TF_ChIP__K562__CTCF__ENCSR000EGM`

Prepared dataset and checkpoint bundles can be downloaded together from Zenodo with:

```bash
python download_zenodo.py
```

To download only a subset of the four supported assay/cell-line pairs, pass the dataset IDs explicitly:

```bash
python download_zenodo.py --datasets ATAC__GM12878__ENCSR637XSC TF_ChIP__GM12878__ELF1__ENCSR841NDX
```

These scripts download and unpack the prepared datasets and checkpoints from Zenodo into the expected `workspace/datasets/` and `workspace/models/` directories. They also download `datasets_readme.txt` into `workspace/datasets/` and `checkpoints_readme.txt` into `workspace/models/`. Please refer to those downloaded readmes for the detailed Zenodo bundle structure and naming conventions.

For example, if you would like to analyze GM12878 ATAC-seq and GM12878 ELF1 ChIP-seq, your workspace should look like this after running the above command:

```text
workspace/
|-- datasets/
|   |-- datasets_readme.txt
|   |-- ATAC__GM12878__ENCSR637XSC/
|   |   |-- peaks.all_folds.zarr/
|   |   `-- nonpeaks.fold_0.zarr/
|   |-- TF_ChIP__GM12878__ELF1__ENCSR841NDX/
|   |   |-- peaks.all_folds.zarr/
|   |   `-- nonpeaks.fold_0.zarr/
|-- logs/
|   `-- ...
|-- plots/
|   `-- ...
`-- models/
    |-- checkpoints_readme.txt
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

Rows within each zarr store are matched to `_auxiliary/regions_df.tsv`, and dataset splits are provided in `_auxiliary/split_indices.npz`. The precomputed masks include the replicate sets used for the main manuscript evaluations. For file trees, naming conventions, and assay-specific details, please refer to `workspace/datasets/datasets_readme.txt` in the Zenodo archive.

## Model Checkpoint Contents
Model checkpoint bundles are also provided for ATAC-seq and TF ChIP-seq in GM12878 and K562. The GM12878 ATAC-seq bundle contains the checkpoint families used for the manuscript's broader GM12878 ATAC-seq analyses, including the single-headed cgLMs, the single-headed cgLM ablation experiments, the multi-headed cgLM critics, and the preliminary CGPSE models (`dna_free_dae` and `dna_aware_editor`).

The remaining three bundles, GM12878 TF ChIP-seq, K562 ATAC-seq, and K562 TF ChIP-seq, follow a reduced structure. They contain the single-headed cgLM discovery models used for the main diagnostic evaluations, together with the base S2F models used in those comparisons.

These model bundles are distributed as PyTorch Lightning checkpoints rather than drop-in `nn.Module` objects. Please use the checkpoint loading utilities provided in this repository for the relevant model families.

For exact checkpoint trees, fold organization, naming conventions, and additional checkpoint loading notes, please refer to `checkpoints_readme.txt` in the Zenodo archive.

## Single-headed cgLM Experiments

### Diagnostic experiments (Figures 1, 2, S4, and S5)
Evaluation of single-headed cgLM models on 1-Kbp regions from held-out chromosomes. Please see [scripts/single_headed_cglm/diagnostic_experiments/README.md](scripts/single_headed_cglm/diagnostic_experiments/README.md).

### Control experiments (Figure S6)
Evaluation of different single-headed cgLM design choices on held-out chromosomes.  Please see [scripts/single_headed_cglm/ablation_experiments/README.md](scripts/single_headed_cglm/ablation_experiments/README.md).

## Multi-headed cgLM Critic Experiments
Multi-headed cgLM critic experiments were performed for `ATAC__GM12878__ENCSR637XSC`.
Please see [scripts/multi_headed_cglm/README.md](scripts/multi_headed_cglm/README.md).


## Critic-Guided Profile-Shape Editing (CGPSE) Experiments
CGPSE experiments were performed for `ATAC__GM12878__ENCSR637XSC`.



Please see [scripts/cgpse/README.md](scripts/cgpse/README.md).

## Citation
Manuscript in preparation.

## License
This repository is released under the [MIT License](LICENSE).
