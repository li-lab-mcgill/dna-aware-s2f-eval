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

from typing import Optional

from .base_zarr_dataset import BasePeakNonpeakZarrDataset
from ..masking.mask_sampler import MaskSampler

# Global flag to prevent multiple set_start_method calls
_FORK_CONTEXT_SET = False

class MaskedPeakNonpeakZarrDataset(BasePeakNonpeakZarrDataset):
    """
    Ultra-fast in-memory PyTorch Dataset for pre-extracted multi-track genomics data with masking support.
    Supports dual zarr sources (peaks + nonpeaks) with epoch-based nonpeak resampling and masking capabilities.
    Inherits all functionality from BasePeakNonpeakZarrDataset and adds masking capabilities.
    """

    # Global shared cache - keyed by (peaks_zarr_path, nonpeaks_zarr_path, fold_name, seq_dataset_path, tuple(peaks_track_dataset_paths), tuple(nonpeaks_track_dataset_paths))
    _global_data_cache = {}
    
    def __init__(
        self,
        # NEW: dual zarr dataset parameters
        peaks_zarr_path: str,
        nonpeaks_zarr_path: str,
        fold_name: str,
        seq_dataset_path: list,
        predicted_peaks_track_dataset_paths: Optional[list],
        predicted_nonpeaks_track_dataset_paths: Optional[list],
        experimental_peaks_track_dataset_paths: Optional[list],
        experimental_nonpeaks_track_dataset_paths: Optional[list],
        split_name: str,
        # masking parameters
        mask_config: dict,
        # peak/nonpeak sampling parameters
        peak_to_nonpeak_ratio: int = 10,
        base_sampling_seed: int = 42,
        # sequence and track parameters
        seq_width: int = 1000,
        track_width: int = 1000,
        max_shift: int = 0,
        rc_aug: bool = False,
        shift_aug: bool = False,
        # memory, ddp and caching parameters
        ddp_safe: str = "single",  # "auto", "single", "multi"
        cache_dir: str = None,
    ):
        """
        Args:
            Dataset parameters:
                peaks_zarr_path: Path to peaks zarr store (contains all folds)
                nonpeaks_zarr_path: Path to nonpeaks zarr store (fold-specific)
                fold_name: Fold identifier (e.g., "fold_0", "fold_1")
                seq_dataset_path: List of nested keys to sequence dataset
                peaks_track_dataset_paths: List of lists, each containing nested keys to peak track datasets
                nonpeaks_track_dataset_paths: List of lists, each containing nested keys to nonpeak track datasets
                split_name: Name of the split ('training', 'validation', 'test')
            Masking parameters:
                mask_config: Configuration for the mask sampler
            Nonpeak sampling parameters:
                peak_to_nonpeak_ratio: Ratio of peaks to nonpeaks (default: 10)
                base_sampling_seed: Base seed for reproducible nonpeak sampling
            Sequence and track parameters:
                seq_width: Width of sequence to extract 
                track_width: Width of tracks to extract 
                max_shift: Maximum shift in bp for shift augmentation (only applied to 'training')
                rc_aug: Enable reverse complement augmentation (only applied to 'training')
                shift_aug: Enable shift augmentation (only applied to 'training')
            Memory, ddp and caching parameters:
                ddp_safe: "auto" (detect), "single" (in-memory), "multi" (memory-mapped)
                cache_dir: Directory to store .npy cache files for memory-mapped mode

        """
        
        # Store mask config before calling parent constructor
        self.mask_config = mask_config
        
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

        # Initialize mask sampler (this is the only new functionality)
        self.mask_generator_sampler = MaskSampler(mask_config)

    def __getitem__(self, idx):
        """Returns:
        - seq_data_tensor: (seq_width, 4)
        - track_data_tensor: (track_width, n_tracks)
        - mask_tensor: (track_width, 4 + n_tracks) - the mask
        - region_type: str - "peak" or "nonpeak"
        - identifier: str - region identifier
        """
        # Get the base data from parent class (already augmented, returns separate seq/track tensors)
        seq_data_tensor, pred_track_tensor, exp_track_tensor, region_type, identifier = super().__getitem__(idx)
        
        # Sample a mask generator instance from mask sampler
        mask_generator = self.mask_generator_sampler()
        
        # Sample mask for S2F output region using the generator
        # mask for both dna and tracks
        # returns mask (track_width, 4+n_tracks) 
        if pred_track_tensor is not None:
            L, T = pred_track_tensor.shape
        elif exp_track_tensor is not None:
            L, T = exp_track_tensor.shape
        else:
            raise ValueError("No track tensor available to determine mask shape.")
        mask_matrix = mask_generator.sample_mask((L, 4 + T))
        
        # Return all components including metadata from parent
        return seq_data_tensor, pred_track_tensor, exp_track_tensor, mask_matrix, region_type, identifier

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
    predicted_peaks_track_dataset_paths,
    predicted_nonpeaks_track_dataset_paths,
    experimental_peaks_track_dataset_paths,
    experimental_nonpeaks_track_dataset_paths,
    split_name,
    mask_config,
    **kwargs,
):
    """
    Smart dataset factory for dual zarr (peaks + nonpeaks) with masking that loads data once and shares across splits.
    
    Args:
        peaks_zarr_path: Path to peaks zarr store (contains all folds)
        nonpeaks_zarr_path: Path to nonpeaks zarr store (fold-specific)
        fold_name: Fold identifier (e.g., "fold_0")
        seq_dataset_path: List of nested keys to sequence dataset
        peaks_track_dataset_paths: List of lists, each containing nested keys to peak track datasets
        nonpeaks_track_dataset_paths: List of lists, each containing nested keys to nonpeak track datasets
        split_name: Name of the split ('training', 'validation', 'test')
        mask_config: Configuration for the mask sampler
        **kwargs: Additional arguments passed to dataset constructor
    
    Usage:
        train_ds = create_dataset(
            peaks_zarr_path="/path/to/peaks.zarr",
            nonpeaks_zarr_path="/path/to/nonpeaks_fold_0.zarr", 
            fold_name="fold_0",
            seq_dataset_path=["reference_dna", "dna_summitcentered1000bp_symmetric500bpshift"],
            peaks_track_dataset_paths=[["fold_0__observed_tracks", "observed_track_summitcentered1000bp_symmetric500bpshift"]],
            nonpeaks_track_dataset_paths=[["observed_tracks", "observed_track_summitcentered1000bp_symmetric500bpshift"]],
            split_name="training",
            mask_config=mask_config
        )
    """
    return MaskedPeakNonpeakZarrDataset(
        peaks_zarr_path=peaks_zarr_path,
        nonpeaks_zarr_path=nonpeaks_zarr_path,
        fold_name=fold_name,
        seq_dataset_path=seq_dataset_path,
        predicted_peaks_track_dataset_paths=predicted_peaks_track_dataset_paths,
        predicted_nonpeaks_track_dataset_paths=predicted_nonpeaks_track_dataset_paths,
        experimental_peaks_track_dataset_paths=experimental_peaks_track_dataset_paths,
        experimental_nonpeaks_track_dataset_paths=experimental_nonpeaks_track_dataset_paths,
        split_name=split_name,
        mask_config=mask_config,
        **kwargs
    )
