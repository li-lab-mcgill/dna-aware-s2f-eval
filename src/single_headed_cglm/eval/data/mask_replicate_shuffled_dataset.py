"""
Dataset for evaluating cgLM on optionally shuffled sequences.

Extends local BasePeakNonpeakZarrDataset to add precomputed mask loading
and optional sequence shuffling for debiasing evaluation.
"""
import torch
import numpy as np
from typing import Optional

from .base_peak_nonpeak_dataset import BasePeakNonpeakZarrDataset
from tangermeme.ersatz import dinucleotide_shuffle


class MaskedPeakNonpeakReplicateZarrDataset(BasePeakNonpeakZarrDataset):
    """
    Dataset for evaluating cgLM on optionally shuffled sequences.

    Loads sequences and precomputed masks from zarr stores, optionally applying
    sequence shuffling for debiasing evaluation.

    Key features:
    - Loads DNA sequences only (tracks computed by S2F at eval time)
    - Loads precomputed position masks from zarr stores
    - Optional dinucleotide or mononucleotide shuffling with reproducible seeds
    - No augmentation (deterministic evaluation)
    """

    def __init__(
        self,
        # Zarr dataset parameters
        peaks_zarr_path: str,
        nonpeaks_zarr_path: str,
        fold_name: str,
        seq_dataset_path: list,
        split_name: str = "validation",
        # Masking parameters
        mask_type_str: str = "10_to_20pct",
        replicate_name: str = "replicate_0",
        # Sequence parameters
        seq_width: int = 2114,
        track_width: int = 1000,
        # Shuffling parameters
        shuffle_seed: Optional[int] = None,
        enable_shuffling: bool = False,
        shuffle_mode: Optional[str] = None,
    ):
        """
        Simplified dataset for evaluating on optionally shuffled sequences.

        Args:
            Dataset parameters:
                peaks_zarr_path: Path to peaks zarr store (contains all folds)
                nonpeaks_zarr_path: Path to nonpeaks zarr store (fold-specific)
                fold_name: Fold identifier (e.g., "fold_0", "fold_1")
                seq_dataset_path: List of nested keys to sequence dataset
                split_name: Name of the split ('training', 'validation', 'test')
            Masking parameters:
                mask_type_str: Type of mask (e.g., "10_to_20pct", "20_to_40pct")
                replicate_name: Replicate identifier (e.g., "replicate_0", "replicate_1")
            Sequence parameters:
                seq_width: Width of sequence to extract (e.g., 2114bp)
                track_width: Width for mask alignment (e.g., 1000bp)
            Shuffling parameters:
                shuffle_seed: Base seed for shuffling (None = no shuffling)
                enable_shuffling: Whether to apply shuffling to sequences
                shuffle_mode: One of {"none", "dinucleotide", "mononucleotide"}.
                    If omitted, existing behavior is preserved:
                    enable_shuffling=False -> "none"
                    enable_shuffling=True -> "dinucleotide"

        NOTE: Tracks are NOT loaded from zarr - they must be computed on shuffled
              sequences by S2F model at evaluation time.
        """
        # Store mask parameters before calling parent constructor
        self.mask_type_str = mask_type_str
        self.replicate_name = replicate_name

        # Store shuffling parameters
        self.shuffle_seed = shuffle_seed
        self.enable_shuffling = enable_shuffling
        self.shuffle_mode = self._resolve_shuffle_mode(enable_shuffling, shuffle_mode)

        # Initialize parent class
        super().__init__(
            peaks_zarr_path=peaks_zarr_path,
            nonpeaks_zarr_path=nonpeaks_zarr_path,
            fold_name=fold_name,
            seq_dataset_path=seq_dataset_path,
            split_name=split_name,
            seq_width=seq_width,
            track_width=track_width,
        )

        # Filter out sequences with N bases when using dinucleotide shuffling.
        if self.shuffle_seed is not None and self.shuffle_mode == "dinucleotide":
            self._filter_sequences_with_n_bases()

        # Load precomputed masks from both zarr stores
        self._load_precomputed_masks()

    def _filter_sequences_with_n_bases(self):
        """
        Remove sequences containing N bases from the dataset.

        N bases are encoded as all-zero rows in one-hot encoding.
        tangermeme's dinucleotide_shuffle requires strictly one-hot encoded
        sequences, so we must filter out any sequences with N bases.
        """
        print("Filtering sequences with N bases...")
        original_count = len(self.indices)

        # Track which indices to keep
        valid_indices = []

        for region_type, zarr_idx, identifier in self.indices:
            # Get the full sequence
            if region_type == 'peak':
                seq_data = self.peak_seq_data[zarr_idx]
            else:
                seq_data = self.nonpeak_seq_data[zarr_idx]

            # Extract the window we'll actually use
            if region_type == 'peak':
                center = self.peak_seq_zarr_summit
            else:
                center = self.nonpeak_seq_zarr_summit
            start = center - self.seq_width // 2
            end = start + self.seq_width
            seq_window = seq_data[start:end, :]

            # Check for N bases (all-zero rows)
            row_sums = seq_window.sum(axis=1)
            has_n_bases = (row_sums == 0).any()

            if not has_n_bases:
                valid_indices.append((region_type, zarr_idx, identifier))

        self.indices = valid_indices
        filtered_count = original_count - len(self.indices)

        print(f"  Removed {filtered_count:,} sequences with N bases")
        print(f"  Remaining: {len(self.indices):,} sequences")

    def _load_precomputed_masks(self):
        """Load precomputed masks from zarr stores"""
        print(f"Loading precomputed masks: {self.mask_type_str}/{self.replicate_name}")

        # Define mask path within zarr structure
        mask_path = ["precomputed_masks", "eval_on_reference_nucleotides", self.mask_type_str, self.replicate_name]

        # Load peak masks (all sequences, keep zarr ordering)
        try:
            peak_mask_dataset = self._navigate_to_dataset(self.peak_store, mask_path)
            self.peak_precomputed_masks = torch.from_numpy(np.asarray(peak_mask_dataset)).float()
            print(f"  Loaded peak masks: shape {self.peak_precomputed_masks.shape}")
        except Exception as e:
            raise ValueError(f"Failed to load peak masks from {self.peaks_zarr_path} at {'/'.join(mask_path)}: {e}")

        # Load nonpeak masks (all sequences, keep zarr ordering)
        try:
            nonpeak_mask_dataset = self._navigate_to_dataset(self.nonpeak_store, mask_path)
            self.nonpeak_precomputed_masks = torch.from_numpy(np.asarray(nonpeak_mask_dataset)).float()
            print(f"  Loaded nonpeak masks: shape {self.nonpeak_precomputed_masks.shape}")
        except Exception as e:
            raise ValueError(f"Failed to load nonpeak masks from {self.nonpeaks_zarr_path} at {'/'.join(mask_path)}: {e}")

    def _resolve_shuffle_mode(self, enable_shuffling: bool, shuffle_mode: Optional[str]) -> str:
        """Normalize shuffle mode while preserving backward compatibility."""
        if shuffle_mode is None:
            return "dinucleotide" if enable_shuffling else "none"

        valid_modes = {"none", "dinucleotide", "mononucleotide"}
        if shuffle_mode not in valid_modes:
            raise ValueError(f"Invalid shuffle_mode={shuffle_mode!r}. Expected one of {sorted(valid_modes)}")

        if not enable_shuffling and shuffle_mode != "none":
            raise ValueError(
                f"Conflicting shuffling configuration: enable_shuffling={enable_shuffling} with "
                f"shuffle_mode={shuffle_mode!r}"
            )

        if enable_shuffling and shuffle_mode == "none":
            raise ValueError(
                f"Conflicting shuffling configuration: enable_shuffling={enable_shuffling} with "
                f"shuffle_mode={shuffle_mode!r}"
            )

        return shuffle_mode

    def _shuffle_sequence(self, seq_tensor: torch.Tensor, random_state: int) -> torch.Tensor:
        """
        Apply configured shuffle preserving reproducibility.

        Dinucleotide mode preserves dinucleotide frequencies via tangermeme.
        Mononucleotide mode permutes positions, preserving mononucleotide counts.

        Args:
            seq_tensor: (L, 4) one-hot encoded DNA sequence
            random_state: Seed for reproducibility

        Returns:
            shuffled: (L, 4) shuffled sequence
        """
        if self.shuffle_mode == "none":
            return seq_tensor

        if self.shuffle_mode == "mononucleotide":
            generator = torch.Generator()
            generator.manual_seed(int(random_state))
            permutation = torch.randperm(seq_tensor.shape[0], generator=generator)
            return seq_tensor[permutation].contiguous()

        if self.shuffle_mode != "dinucleotide":
            raise ValueError(f"Unsupported shuffle_mode={self.shuffle_mode}")

        # Convert (L, 4) -> (1, 4, L) for tangermeme format
        seq_tangermeme = seq_tensor.T.unsqueeze(0).float()

        # Shuffle with n=1 to get single shuffle
        shuffled = dinucleotide_shuffle(seq_tangermeme, n=1, random_state=random_state)
        # Output shape: (1, 1, 4, L)

        # Convert back to (L, 4) format
        return shuffled[0, 0].T.contiguous()

    def _get_shuffle_seed(self, zarr_idx: int) -> int:
        """
        Get deterministic seed for a specific sequence.

        Combines base shuffle_seed with zarr_idx to ensure:
        - Same sequence gets same shuffle across runs (reproducibility)
        - Different sequences get different shuffles (diversity)

        Args:
            zarr_idx: Index of the sequence in the zarr store

        Returns:
            int: Seed for this specific sequence
        """
        return self.shuffle_seed + zarr_idx

    def __getitem__(self, idx):
        """
        Returns:
            seq_tensor: (seq_width, 4) - DNA sequence (shuffled if enabled)
            position_mask: (track_width,) - precomputed position mask (1=masked, 0=visible)
            region_type: str - "peak" or "nonpeak"
            identifier: str - region identifier

        NOTE: Tracks are NOT returned - they must be computed by S2F on the
              (potentially shuffled) sequence at evaluation time.
        """
        # Get the base data from parent class
        # Parent returns: (seq_tensor, region_type, identifier, zarr_idx)
        seq_tensor, region_type, identifier, zarr_idx = super().__getitem__(idx)

        # Apply shuffling if enabled
        if self.shuffle_mode != "none" and self.shuffle_seed is not None:
            shuffle_seed = self._get_shuffle_seed(zarr_idx)
            seq_tensor = self._shuffle_sequence(seq_tensor, shuffle_seed)

        # Get precomputed position mask (track_width length, aligned with track center)
        if region_type == 'peak':
            position_mask = self.peak_precomputed_masks[zarr_idx]
        else:  # nonpeak
            position_mask = self.nonpeak_precomputed_masks[zarr_idx]

        return seq_tensor, position_mask, region_type, identifier


def create_dataset(
    peaks_zarr_path,
    nonpeaks_zarr_path,
    fold_name,
    seq_dataset_path,
    split_name="validation",
    mask_type_str="10_to_20pct",
    replicate_name="replicate_0",
    seq_width=2114,
    track_width=1000,
    shuffle_seed=None,
    enable_shuffling=False,
    shuffle_mode=None,
):
    """
    Dataset factory for shuffled sequence evaluation.

    Creates a dataset that loads sequences and precomputed masks from zarr stores,
    optionally applying configurable sequence shuffling.

    Args:
        peaks_zarr_path: Path to peaks zarr store (contains all folds)
        nonpeaks_zarr_path: Path to nonpeaks zarr store (fold-specific)
        fold_name: Fold identifier (e.g., "fold_0")
        seq_dataset_path: List of nested keys to sequence dataset
        split_name: Name of the split ('training', 'validation', 'test')
        mask_type_str: Type of mask (e.g., "10_to_20pct")
        replicate_name: Replicate name (e.g., "replicate_0")
        seq_width: Width of sequence to extract (e.g., 2114bp)
        track_width: Width for mask alignment (e.g., 1000bp)
        shuffle_seed: Base seed for sequence shuffling (None = no shuffling)
        enable_shuffling: Whether to apply shuffling
        shuffle_mode: Optional explicit mode in {"none", "dinucleotide", "mononucleotide"}

    Returns:
        Dataset yielding (seq_tensor, position_mask, region_type, identifier)
    """
    return MaskedPeakNonpeakReplicateZarrDataset(
        peaks_zarr_path=peaks_zarr_path,
        nonpeaks_zarr_path=nonpeaks_zarr_path,
        fold_name=fold_name,
        seq_dataset_path=seq_dataset_path,
        split_name=split_name,
        mask_type_str=mask_type_str,
        replicate_name=replicate_name,
        seq_width=seq_width,
        track_width=track_width,
        shuffle_seed=shuffle_seed,
        enable_shuffling=enable_shuffling,
        shuffle_mode=shuffle_mode,
    )
