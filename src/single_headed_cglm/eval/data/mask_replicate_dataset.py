import pdb
import zarr
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import json
import os
import time
import gc
from filelock import FileLock
from typing import List, Optional

import sys
import os.path as osp

from .base_zarr_dataset import BasePeakNonpeakZarrDataset

# Global flag to prevent multiple set_start_method calls
_FORK_CONTEXT_SET = False


class MaskedPeakNonpeakReplicateZarrDataset(BasePeakNonpeakZarrDataset):
    """
    PyTorch Dataset for pre-extracted multi-track genomics data with precomputed replicate masking.
    Extends the v3 BasePeakNonpeakZarrDataset (which supports different seq_width and track_width)
    and adds precomputed mask loading capabilities.

    Key features:
    - Supports different seq_width (e.g., 2114bp) and track_width (e.g., 1000bp)
    - Loads precomputed position masks from zarr stores
    - Returns separate seq/track tensors plus position mask for orchestrator compatibility
    """

    # Global shared cache - keyed by dataset combination + mask parameters
    _global_data_cache = {}

    def __init__(
        self,
        # Dual zarr dataset parameters
        peaks_zarr_path: str,
        nonpeaks_zarr_path: str,
        fold_name: str,
        seq_dataset_path: list,
        # Track paths - now using predicted/experimental naming like v3 base
        predicted_peaks_track_dataset_paths: Optional[List[List[str]]] = None,
        predicted_nonpeaks_track_dataset_paths: Optional[List[List[str]]] = None,
        experimental_peaks_track_dataset_paths: Optional[List[List[str]]] = None,
        experimental_nonpeaks_track_dataset_paths: Optional[List[List[str]]] = None,
        split_name: str = "validation",
        # Masking parameters
        mask_type_str: str = "10_to_20pct",  # e.g., "10_to_20pct"
        replicate_name: str = "replicate_0",  # e.g., "replicate_0"
        # Peak/nonpeak sampling parameters
        peak_to_nonpeak_ratio: int = 10,
        base_sampling_seed: int = 42,
        # Sequence and track parameters (can be different!)
        seq_width: int = 2114,
        track_width: int = 1000,
        max_shift: int = 0,
        rc_aug: bool = False,
        shift_aug: bool = False,
        # Memory, ddp and caching parameters
        ddp_safe: str = "single",
        cache_dir: str = None,
    ):
        """
        Args:
            Dataset parameters:
                peaks_zarr_path: Path to peaks zarr store (contains all folds)
                nonpeaks_zarr_path: Path to nonpeaks zarr store (fold-specific)
                fold_name: Fold identifier (e.g., "fold_0", "fold_1")
                seq_dataset_path: List of nested keys to sequence dataset
                predicted_peaks_track_dataset_paths: List of lists for predicted peak tracks
                predicted_nonpeaks_track_dataset_paths: List of lists for predicted nonpeak tracks
                experimental_peaks_track_dataset_paths: List of lists for experimental peak tracks
                experimental_nonpeaks_track_dataset_paths: List of lists for experimental nonpeak tracks
                split_name: Name of the split ('training', 'validation', 'test')
            Masking parameters:
                mask_type_str: Type of mask (e.g., "10_to_20pct", "20_to_40pct")
                replicate_name: Replicate identifier (e.g., "replicate_0", "replicate_1")
            Nonpeak sampling parameters:
                peak_to_nonpeak_ratio: Ratio of peaks to nonpeaks (default: 10)
                base_sampling_seed: Base seed for reproducible nonpeak sampling
            Sequence and track parameters:
                seq_width: Width of sequence to extract (can differ from track_width)
                track_width: Width of tracks to extract (can differ from seq_width)
                max_shift: Maximum shift in bp for shift augmentation
                rc_aug: Enable reverse complement augmentation
                shift_aug: Enable shift augmentation
            Memory, ddp and caching parameters:
                ddp_safe: "auto" (detect), "single" (in-memory), "multi" (memory-mapped)
                cache_dir: Directory to store .npy cache files for memory-mapped mode

        NOTE: seq_width and track_width CAN be different (e.g., 2114bp seq, 1000bp track)
        """

        # Store mask parameters before calling parent constructor
        self.mask_type_str = mask_type_str
        self.replicate_name = replicate_name

        # Initialize parent class (handles all the dual zarr functionality)
        super().__init__(
            peaks_zarr_path=peaks_zarr_path,
            nonpeaks_zarr_path=nonpeaks_zarr_path,
            fold_name=fold_name,
            seq_dataset_path=seq_dataset_path,
            predicted_peaks_track_dataset_paths=predicted_peaks_track_dataset_paths,
            predicted_nonpeaks_track_dataset_paths=predicted_nonpeaks_track_dataset_paths,
            experimental_peaks_track_dataset_paths=experimental_peaks_track_dataset_paths,
            experimental_nonpeaks_track_dataset_paths=experimental_nonpeaks_track_dataset_paths,
            split_name=split_name,
            peak_to_nonpeak_ratio=peak_to_nonpeak_ratio,
            base_sampling_seed=base_sampling_seed,
            seq_width=seq_width,
            track_width=track_width,
            max_shift=max_shift,
            rc_aug=rc_aug,
            shift_aug=shift_aug,
            ddp_safe=ddp_safe,
            cache_dir=cache_dir,
        )

        # Load precomputed masks from both zarr stores
        self._load_precomputed_masks()

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

    def __getitem__(self, idx):
        """
        Returns:
            seq_tensor: (seq_width, 4) - DNA sequence
            pred_tensor: (track_width, T) or empty tensor - predicted tracks
            exp_tensor: (track_width, T) or empty tensor - experimental tracks
            position_mask: (track_width,) - precomputed position mask (1=masked, 0=visible)
            region_type: str - "peak" or "nonpeak"
            identifier: str - region identifier
        """
        # Get the base data from parent class (returns separate seq/track tensors)
        # Parent returns: (seq_tensor, pred_tensor, exp_tensor, region_type, identifier)
        seq_tensor, pred_tensor, exp_tensor, region_type, identifier = super().__getitem__(idx)

        # Get the zarr index for this sample from the mixed indices
        region_type_from_indices, zarr_idx, identifier_from_indices = self.indices[idx]
        assert region_type == region_type_from_indices, f"Region type mismatch"

        # Get precomputed position mask (track_width length, aligned with track center)
        if region_type == 'peak':
            position_mask = self.peak_precomputed_masks[zarr_idx]
        else:  # nonpeak
            position_mask = self.nonpeak_precomputed_masks[zarr_idx]

        # Return separate tensors plus position mask
        return seq_tensor, pred_tensor, exp_tensor, position_mask, region_type, identifier

    def subsample_nonpeaks_for_epoch(self, epoch_id, verbose=False):
        """
        Called by Lightning module to resample nonpeaks for new epoch.
        Delegates to parent class implementation.
        """
        return super().subsample_nonpeaks_for_epoch(epoch_id, verbose)


def create_dataset(
    peaks_zarr_path,
    nonpeaks_zarr_path,
    fold_name,
    seq_dataset_path,
    # Track paths - support both old single-list format and new predicted/experimental format
    predicted_peaks_track_dataset_paths=None,
    predicted_nonpeaks_track_dataset_paths=None,
    experimental_peaks_track_dataset_paths=None,
    experimental_nonpeaks_track_dataset_paths=None,
    split_name="validation",
    mask_type_str="10_to_20pct",
    replicate_name="replicate_0",
    **kwargs,
):
    """
    Dataset factory for dual zarr (peaks + nonpeaks) with precomputed replicate masking.
    Supports different seq_width and track_width (e.g., 2114bp seq, 1000bp track).

    Args:
        peaks_zarr_path: Path to peaks zarr store (contains all folds)
        nonpeaks_zarr_path: Path to nonpeaks zarr store (fold-specific)
        fold_name: Fold identifier (e.g., "fold_0")
        seq_dataset_path: List of nested keys to sequence dataset
        predicted_peaks_track_dataset_paths: List of lists for predicted peak tracks
        predicted_nonpeaks_track_dataset_paths: List of lists for predicted nonpeak tracks
        experimental_peaks_track_dataset_paths: List of lists for experimental peak tracks
        experimental_nonpeaks_track_dataset_paths: List of lists for experimental nonpeak tracks
        split_name: Name of the split ('training', 'validation', 'test')
        mask_type_str: Type of mask (e.g., "10_to_20pct")
        replicate_name: Replicate name (e.g., "replicate_0")
        **kwargs: Additional arguments (seq_width, track_width, etc.)

    Returns:
        Dataset that yields:
        (seq_tensor, pred_tensor, exp_tensor, position_mask, region_type, identifier)
    """
    return MaskedPeakNonpeakReplicateZarrDataset(
        peaks_zarr_path=peaks_zarr_path,
        nonpeaks_zarr_path=nonpeaks_zarr_path,
        fold_name=fold_name,
        seq_dataset_path=seq_dataset_path,
        predicted_peaks_track_dataset_paths=predicted_peaks_track_dataset_paths,
        predicted_nonpeaks_track_dataset_paths=predicted_nonpeaks_track_dataset_paths,
        experimental_peaks_track_dataset_paths=experimental_peaks_track_dataset_paths,
        experimental_nonpeaks_track_dataset_paths=experimental_nonpeaks_track_dataset_paths,
        split_name=split_name,
        mask_type_str=mask_type_str,
        replicate_name=replicate_name,
        **kwargs,
    )


# Testing code demonstrating precomputed replicate masking functionality with dual zarr
if __name__ == "__main__":
    # Dual zarr paths (using 2114bp seq, 1000bp tracks like bias model training)
    peaks_zarr_path = "/home/mcb/users/dcakma3/multi_res_bench_v2/multimodal_masked_contrastive_main/manuscript_experiments/section_1__ConditionalLM_as_diagnostic_probe/workspace/data/ATAC/GM12878__ENCSR637XSC/preprocessed/zarr_datasets/peaks.all_folds.zarr"
    nonpeaks_zarr_path = "/home/mcb/users/dcakma3/multi_res_bench_v2/multimodal_masked_contrastive_main/manuscript_experiments/section_1__ConditionalLM_as_diagnostic_probe/workspace/data/ATAC/GM12878__ENCSR637XSC/preprocessed/zarr_datasets/nonpeaks.fold_0.zarr"
    fold_name = "fold_0"
    seq_dataset_path = ["reference_dna", "dna_summitcentered2114bp_symmetric500bpshift"]

    # Use predicted/experimental track path format (like v3 base dataset)
    predicted_peaks_track_dataset_paths = [
        ["fold_agnostic__model_based_tracks", "alphagenome_predicted_track__summitcentered1000bp_symmetric500bpshift"]
    ]
    predicted_nonpeaks_track_dataset_paths = [
        ["fold_0__model_based_tracks", "alphagenome_predicted_track__summitcentered1000bp_symmetric500bpshift"]
    ]
    experimental_peaks_track_dataset_paths = [
        ["observed_tracks", "observed_track_summitcentered1000bp_symmetric500bpshift"]
    ]
    experimental_nonpeaks_track_dataset_paths = [
        ["observed_tracks", "observed_track_summitcentered1000bp_symmetric500bpshift"]
    ]

    print("=== Testing Precomputed Replicate Masked Peak-Nonpeak Dataset (v3 compatible) ===")

    # Test with different mask types and replicates
    mask_type_str = "10_to_20pct"
    replicate_name = "replicate_0"

    # Load validation split
    print("\n--- Creating Validation Dataset ---")

    valid_dataset = create_dataset(
        peaks_zarr_path=peaks_zarr_path,
        nonpeaks_zarr_path=nonpeaks_zarr_path,
        fold_name=fold_name,
        seq_dataset_path=seq_dataset_path,
        predicted_peaks_track_dataset_paths=predicted_peaks_track_dataset_paths,
        predicted_nonpeaks_track_dataset_paths=predicted_nonpeaks_track_dataset_paths,
        experimental_peaks_track_dataset_paths=experimental_peaks_track_dataset_paths,
        experimental_nonpeaks_track_dataset_paths=experimental_nonpeaks_track_dataset_paths,
        split_name='validation',
        mask_type_str=mask_type_str,
        replicate_name=replicate_name,
        peak_to_nonpeak_ratio=10,
        base_sampling_seed=42,
        seq_width=2114,  # Different from track_width!
        track_width=1000,
        ddp_safe="single"
    )

    print(f"\n📊 Dataset size: {len(valid_dataset):,} samples")
    print(f"🎭 Mask type: {mask_type_str}")
    print(f"🔄 Replicate: {replicate_name}")

    # Test sample output format
    print(f"\n🎭 Testing Output Format:")

    seq_tensor, pred_tensor, exp_tensor, position_mask, region_type, identifier = valid_dataset[0]
    print(f"  seq_tensor shape: {seq_tensor.shape}")  # (2114, 4)
    print(f"  pred_tensor shape: {pred_tensor.shape}")  # (1000, T)
    print(f"  exp_tensor shape: {exp_tensor.shape}")  # (1000, T)
    print(f"  position_mask shape: {position_mask.shape}")  # (1000,) - track_width
    print(f"  region_type: {region_type}")
    print(f"  identifier: {identifier}")
    print(f"  mask percentage: {position_mask.float().mean().item():.3f}")

    # Verify precomputed masks are consistent
    sample1 = valid_dataset[0]
    sample2 = valid_dataset[0]
    masks_identical = torch.equal(sample1[3], sample2[3])
    print(f"  Precomputed masks consistent: {masks_identical}")

    print("\n✅ Dataset ready for bias model evaluation!")
    print("✅ Supports different seq_width (2114) and track_width (1000)!")
    print("✅ Precomputed masks loaded from zarr!")
    print("✅ Returns separate tensors: (seq, pred, exp, mask, region_type, identifier)")
