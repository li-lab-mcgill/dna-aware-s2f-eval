"""
Simplified base dataset for loading sequences from peak/nonpeak zarr stores.

Stripped-down version for evaluation - no tracks, no augmentations.
"""
import zarr
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from typing import List
import os


class BasePeakNonpeakZarrDataset(Dataset):
    """
    Simple dataset for loading DNA sequences from peak/nonpeak zarr stores.

    Key features:
    - Loads sequences only (no tracks)
    - No augmentation (deterministic evaluation)
    - Returns all peaks and nonpeaks (no ratio-based sampling)
    - Simple in-memory loading
    """

    def __init__(
        self,
        peaks_zarr_path: str,
        nonpeaks_zarr_path: str,
        fold_name: str,
        seq_dataset_path: List[str],
        split_name: str = "validation",
        seq_width: int = 2114,
        track_width: int = 1000,
    ):
        """
        Args:
            peaks_zarr_path: Path to peaks zarr store (contains all folds)
            nonpeaks_zarr_path: Path to nonpeaks zarr store (fold-specific)
            fold_name: Fold identifier (e.g., "fold_0", "fold_1")
            seq_dataset_path: List of nested keys to sequence dataset
            split_name: Name of the split ('training', 'validation', 'test')
            seq_width: Width of sequence to extract (e.g., 2114bp)
            track_width: Width for mask alignment (e.g., 1000bp)
        """
        self.fold_name = fold_name
        self.split_name = split_name
        self.peaks_zarr_path = peaks_zarr_path
        self.nonpeaks_zarr_path = nonpeaks_zarr_path
        self.seq_dataset_path = seq_dataset_path
        self.seq_width = seq_width
        self.track_width = track_width

        print(f"=== Initializing Sequence-Only Dataset ({split_name}) ===")
        print(f"Peaks zarr: {peaks_zarr_path}")
        print(f"Nonpeaks zarr: {nonpeaks_zarr_path}")
        print(f"Fold: {fold_name}")

        # Open both zarr stores
        self.peak_store = zarr.open(peaks_zarr_path, mode='r')
        self.nonpeak_store = zarr.open(nonpeaks_zarr_path, mode='r')

        # Get zarr widths
        temp = self._navigate_to_dataset(self.peak_store, self.seq_dataset_path)
        self.peak_seq_zarr_width = temp.shape[1]
        self.peak_seq_zarr_summit = self.peak_seq_zarr_width // 2

        temp = self._navigate_to_dataset(self.nonpeak_store, self.seq_dataset_path)
        self.nonpeak_seq_zarr_width = temp.shape[1]
        self.nonpeak_seq_zarr_summit = self.nonpeak_seq_zarr_width // 2

        # Validate dimensions
        assert seq_width <= self.peak_seq_zarr_width, \
            f"seq_width ({seq_width}) must be <= peak zarr width ({self.peak_seq_zarr_width})"
        assert seq_width <= self.nonpeak_seq_zarr_width, \
            f"seq_width ({seq_width}) must be <= nonpeak zarr width ({self.nonpeak_seq_zarr_width})"

        # Load regions_df from _auxiliary
        temp = os.path.join(peaks_zarr_path, "_auxiliary", "regions_df.tsv")
        self.peak_regions_df = pd.read_csv(temp, sep='\t')

        temp = os.path.join(nonpeaks_zarr_path, "_auxiliary", "regions_df.tsv")
        self.nonpeak_regions_df = pd.read_csv(temp, sep='\t')

        # Load split indices
        temp = os.path.join(peaks_zarr_path, "_auxiliary", "split_indices.npz")
        split_key = f"{fold_name}__{split_name}"
        self.peak_split_indices = np.load(temp)[split_key]

        temp = os.path.join(nonpeaks_zarr_path, "_auxiliary", "split_indices.npz")
        nonpeak_splits = np.load(temp)
        if split_key in nonpeak_splits:
            self.nonpeak_split_indices = nonpeak_splits[split_key]
        elif split_name in nonpeak_splits:
            self.nonpeak_split_indices = nonpeak_splits[split_name]
        else:
            available_keys = list(nonpeak_splits.keys())
            raise KeyError(f"Neither {split_key} nor {split_name} found. Available: {available_keys}")

        # Generate indices (all peaks + all nonpeaks, no sampling)
        self.indices = self._generate_indices()

        print(f"Loaded {len(self.indices):,} samples for {split_name} split")
        print(f"  Peaks: {len(self.peak_split_indices):,}")
        print(f"  Nonpeaks: {len(self.nonpeak_split_indices):,}")
        print(f"Extraction window: seq_width={seq_width}")

        # Load sequence data into memory
        self._load_sequences()

    def _navigate_to_dataset(self, store, path_list):
        """Navigate through nested zarr groups using path list"""
        current = store
        for key in path_list:
            if key not in current:
                raise ValueError(f"Key '{key}' not found. Available: {list(current.keys())}")
            current = current[key]
        return current

    def _generate_indices(self):
        """Generate indices for all peaks and nonpeaks (no sampling)"""
        indices = []

        # Add all peak indices
        for peak_zarr_idx in self.peak_split_indices:
            peak_identifier = self.peak_regions_df.iloc[peak_zarr_idx]['identifier']
            indices.append(('peak', peak_zarr_idx, peak_identifier))

        # Add all nonpeak indices
        for nonpeak_zarr_idx in self.nonpeak_split_indices:
            nonpeak_identifier = self.nonpeak_regions_df.iloc[nonpeak_zarr_idx]['identifier']
            indices.append(('nonpeak', nonpeak_zarr_idx, nonpeak_identifier))

        return indices

    def _load_sequences(self):
        """Load all sequence data into memory"""
        print("Loading sequences into memory...")

        # Load peak sequences
        peak_seq_data = self._navigate_to_dataset(self.peak_store, self.seq_dataset_path)
        self.peak_seq_data = np.asarray(peak_seq_data)
        peak_gb = self.peak_seq_data.nbytes / (1024**3)
        print(f"  Peak sequences: {self.peak_seq_data.shape} = {peak_gb:.2f} GB")

        # Load nonpeak sequences
        nonpeak_seq_data = self._navigate_to_dataset(self.nonpeak_store, self.seq_dataset_path)
        self.nonpeak_seq_data = np.asarray(nonpeak_seq_data)
        nonpeak_gb = self.nonpeak_seq_data.nbytes / (1024**3)
        print(f"  Nonpeak sequences: {self.nonpeak_seq_data.shape} = {nonpeak_gb:.2f} GB")

        total_gb = peak_gb + nonpeak_gb
        print(f"  Total: {total_gb:.2f} GB")

    def _extract_window(self, seq_data, region_type):
        """Extract centered window from sequence"""
        center = self.peak_seq_zarr_summit if region_type == "peak" else self.nonpeak_seq_zarr_summit
        start = center - self.seq_width // 2
        end = start + self.seq_width
        return seq_data[start:end, :].copy()

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        """
        Returns:
            seq_tensor: (seq_width, 4) - DNA sequence
            region_type: str - "peak" or "nonpeak"
            identifier: str - region identifier
            zarr_idx: int - index in zarr store (for mask lookup)
        """
        region_type, zarr_idx, identifier = self.indices[idx]

        # Get sequence data
        if region_type == 'peak':
            seq_data = self.peak_seq_data[zarr_idx]
        else:
            seq_data = self.nonpeak_seq_data[zarr_idx]

        # Extract centered window
        seq_window = self._extract_window(seq_data, region_type)

        # Convert to tensor
        seq_tensor = torch.from_numpy(seq_window).float()

        return seq_tensor, region_type, identifier, zarr_idx
