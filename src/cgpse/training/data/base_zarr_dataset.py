import zarr
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from typing import List, Optional, Tuple
import json
import os
import time
import gc
from filelock import FileLock

# Global flag to prevent multiple set_start_method calls
_FORK_CONTEXT_SET = False

class BasePeakNonpeakZarrDataset(Dataset):
    """
    Ultra-fast in-memory PyTorch Dataset for pre-extracted multi-track genomics data.
    Supports dual zarr sources (peaks + nonpeaks) with epoch-based nonpeak resampling.
    Loads data once globally and shares across all splits (training/validation/test)
    to prevent 3x memory usage when creating multiple dataset instances.
    Optional metadata loading for downstream analysis while keeping training efficient.
    Automatically chooses optimal strategy:
    - Single GPU: Direct in-memory loading (shared across splits)
    - Multi-GPU: Memory-mapped files for sharing across ranks AND splits
    """
    
    # Global shared cache - keyed by (peaks_zarr_path, nonpeaks_zarr_path, fold_name, seq_dataset_path, tuple(peaks_track_dataset_paths), tuple(nonpeaks_track_dataset_paths))
    _global_data_cache = {}
    
    def __init__(
        self,
        # NEW: dual zarr dataset parameters
        peaks_zarr_path: str,
        nonpeaks_zarr_path: str,
        fold_name: str,
        seq_dataset_path: List[str],
        # Track dataset paths (predicted and/or experimental). Each is a list of zarr-key paths; all
        # provided paths for a given set are concatenated along channel dim.
        predicted_peaks_track_dataset_paths: Optional[List[List[str]]],
        predicted_nonpeaks_track_dataset_paths: Optional[List[List[str]]],
        experimental_peaks_track_dataset_paths: Optional[List[List[str]]],
        experimental_nonpeaks_track_dataset_paths: Optional[List[List[str]]],
        split_name: str,
        # peak/nonpeak sampling parameters
        peak_to_nonpeak_ratio: int = 10,
        base_sampling_seed: int = 42,
        # sequence and track parameters
        seq_width: int = 1000,
        track_width: int = 1000,
        max_shift: int = 0,
        shift_aug: bool = False,
        rc_aug: bool = False,
        rc_strand_flip: bool = False, # reverse-complement strand handling (optional)
        strand_channel_pairs: Optional[List[Tuple[int, int]]] = None,
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
                seq_dataset_path: List of nested keys to sequence dataset (e.g., ["reference_dna", "dna_name"])
                peaks_track_dataset_paths: List of lists, each containing nested keys to peak track datasets
                nonpeaks_track_dataset_paths: List of lists, each containing nested keys to nonpeak track datasets
                split_name: Name of the split ('training', 'validation', 'test')
            Nonpeak sampling parameters:
                peak_to_nonpeak_ratio: Ratio of peaks to nonpeaks (default: 10)
                base_sampling_seed: Base seed for reproducible nonpeak sampling
            Sequence and track parameters:
                seq_width: Width of sequence to extract (can be different from track_width)
                track_width: Width of tracks to extract (can be different from seq_width)
                max_shift: Maximum shift in bp for shift augmentation (only applied to 'training')
                rc_aug: Enable reverse complement augmentation (only applied to 'training')
                shift_aug: Enable shift augmentation (only applied to 'training')
            Memory, ddp and caching parameters:
                ddp_safe: "auto" (detect), "single" (in-memory), "multi" (memory-mapped)
                cache_dir: Directory to store .npy cache files for memory-mapped mode

        Important NOTE:
        - This class always loads the entire dataset into memory.
            Unless it is already cached before for this process.
        - This class operates on sequence and tracks of peaks and nonpeaks
            peak sequence shape (n_regions, seq_width, 4)
            peak track shape (n_regions, track_width, n_tracks)
            nonpeak sequence shape (n_regions, seq_width, 4)
            nonpeak track shape (n_regions, track_width, n_tracks)
        """

        self.fold_name = fold_name
        self.split_name = split_name

        self.peaks_zarr_path = peaks_zarr_path
        self.nonpeaks_zarr_path = nonpeaks_zarr_path
 
        self.seq_dataset_path = seq_dataset_path
        
        # Track path sets
        self.predicted_peaks_track_dataset_paths = predicted_peaks_track_dataset_paths or []
        self.predicted_nonpeaks_track_dataset_paths = predicted_nonpeaks_track_dataset_paths or []
        self.experimental_peaks_track_dataset_paths = experimental_peaks_track_dataset_paths or []
        self.experimental_nonpeaks_track_dataset_paths = experimental_nonpeaks_track_dataset_paths or []
        
        self.peak_to_nonpeak_ratio = peak_to_nonpeak_ratio
        self.base_sampling_seed = base_sampling_seed

        self.seq_width = seq_width
        self.track_width = track_width
        self.max_shift = max_shift
        
        # Auto-detect DDP scenario if needed
        if ddp_safe == "auto":
            world_size = int(os.environ.get('WORLD_SIZE', 1))
            local_world_size = int(os.environ.get('LOCAL_WORLD_SIZE', world_size))
            
            if world_size == 1:
                self.ddp_safe = False  # Single GPU
                print(f"Auto-detected: Single GPU mode (split: {split_name})")
            elif local_world_size == 1:
                self.ddp_safe = False  # One rank per node
                print(f"Auto-detected: Multi-node DDP - 1 rank/node (split: {split_name})")
            else:
                self.ddp_safe = True   # Multiple ranks per node
                print(f"Auto-detected: Multi-rank per node DDP (split: {split_name})")
        else:
            self.ddp_safe = (ddp_safe == "multi")
        
        # Set cache directory
        if cache_dir is None:
            self.cache_dir = os.path.join(peaks_zarr_path, "_cache")  # Use peaks zarr for cache
        else:
            self.cache_dir = cache_dir
        
        # Set fork method ONCE globally 
        global _FORK_CONTEXT_SET
        if not _FORK_CONTEXT_SET:
            try:
                torch.multiprocessing.set_start_method("fork", force=True)
                _FORK_CONTEXT_SET = True
                print("✓ Set multiprocessing to fork mode")
            except RuntimeError:
                pass  # Already set
        
        # Handle augmentation settings based on split_name
        if "training" in split_name:
            self.rc_aug = rc_aug
            self.shift_aug = shift_aug
            self.max_shift = max_shift
        else:
            self.rc_aug = False
            self.shift_aug = False
            self.max_shift = 0
            if rc_aug or shift_aug:
                print(f"Warning: Augmentations disabled for '{split_name}' split (only enabled for 'training')")

        # Strand handling
        self.rc_strand_flip = bool(rc_strand_flip)
        if strand_channel_pairs is not None:
            self.strand_channel_pairs = [(int(a), int(b)) for (a, b) in strand_channel_pairs]
        elif self.rc_strand_flip:
            self.strand_channel_pairs = [(0, 1)]
        else:
            self.strand_channel_pairs = None
        
        print(f"=== Initializing Dual Zarr Dataset ({split_name}) ===")
        print(f"Peaks zarr: {peaks_zarr_path}")
        print(f"Nonpeaks zarr: {nonpeaks_zarr_path}")
        print(f"Fold: {fold_name}")
        print(f"Peak:Nonpeak ratio: {peak_to_nonpeak_ratio}:1")
        
        # Open both zarr stores
        self.peak_store = zarr.open(peaks_zarr_path, mode='r')
        self.nonpeak_store = zarr.open(nonpeaks_zarr_path, mode='r')

        # get zarr widths and summits
        # peak, seq
        temp = self._navigate_to_dataset(self.peak_store, self.seq_dataset_path)
        self.peak_seq_zarr_width = temp.shape[1]
        self.peak_seq_zarr_summit = self.peak_seq_zarr_width // 2
        # nonpeak, seq
        temp = self._navigate_to_dataset(self.nonpeak_store, self.seq_dataset_path)
        self.nonpeak_seq_zarr_width = temp.shape[1]
        self.nonpeak_seq_zarr_summit = self.nonpeak_seq_zarr_width // 2
        # peak, track: validate widths across all provided sets
        track_widths = []
        for path_list in (self.predicted_peaks_track_dataset_paths + self.experimental_peaks_track_dataset_paths):
            temp = self._navigate_to_dataset(self.peak_store, path_list)
            track_widths.append(temp.shape[1])
        if track_widths:
            assert len(set(track_widths)) == 1, "All provided peak track datasets must have the same width"
            self.peak_track_zarr_width = track_widths[0]
        else:
            self.peak_track_zarr_width = self.peak_seq_zarr_width
        self.peak_track_zarr_summit = self.peak_track_zarr_width // 2
        # nonpeak, track: validate widths
        track_widths = []
        for path_list in (self.predicted_nonpeaks_track_dataset_paths + self.experimental_nonpeaks_track_dataset_paths):
            temp = self._navigate_to_dataset(self.nonpeak_store, path_list)
            track_widths.append(temp.shape[1])
        if track_widths:
            assert len(set(track_widths)) == 1, "All provided nonpeak track datasets must have the same width"
            self.nonpeak_track_zarr_width = track_widths[0]
        else:
            self.nonpeak_track_zarr_width = self.nonpeak_seq_zarr_width
        self.nonpeak_track_zarr_summit = self.nonpeak_track_zarr_width // 2
        
        # Validate dimensions while accounting for the shift augmentation boundaries
        if self.shift_aug:
            assert seq_width + 2*max_shift <= self.peak_seq_zarr_width, f"(train, peak) seq_width ({seq_width}) + max_shift ({max_shift}) must be <= zarr_width ({self.peak_seq_zarr_width})"
            assert seq_width + 2*max_shift <= self.nonpeak_seq_zarr_width, f"(train, nonpeak) seq_width ({seq_width}) + max_shift ({max_shift}) must be <= zarr_width ({self.nonpeak_seq_zarr_width})"
            assert track_width + 2*max_shift <= self.peak_track_zarr_width, f"(train, peak) track_width ({track_width}) + max_shift ({max_shift}) must be <= zarr_width ({self.peak_track_zarr_width})"
            assert track_width + 2*max_shift <= self.nonpeak_track_zarr_width, f"(train, nonpeak) track_width ({track_width}) + max_shift ({max_shift}) must be <= zarr_width ({self.nonpeak_track_zarr_width})"
        else:
            assert seq_width <= self.peak_seq_zarr_width, f"(train, peak) seq_width ({seq_width}) must be <= zarr_width ({self.peak_seq_zarr_width})"
            assert seq_width <= self.nonpeak_seq_zarr_width, f"(train, nonpeak) seq_width ({seq_width}) must be <= zarr_width ({self.nonpeak_seq_zarr_width})"
            assert track_width <= self.peak_track_zarr_width, f"(train, peak) track_width ({track_width}) must be <= zarr_width ({self.peak_track_zarr_width})"
            assert track_width <= self.nonpeak_track_zarr_width, f"(train, nonpeak) track_width ({track_width}) must be <= zarr_width ({self.nonpeak_track_zarr_width})"
        
        # Load regions_df from _auxiliary
        # peaks
        temp = os.path.join(peaks_zarr_path, "_auxiliary", "regions_df.tsv")
        self.peak_regions_df = pd.read_csv(temp, sep='\t')
        
        # nonpeaks
        temp = os.path.join(nonpeaks_zarr_path, "_auxiliary", "regions_df.tsv")
        self.nonpeak_regions_df = pd.read_csv(temp, sep='\t')

        # Validate split_name is one of the expected values
        valid_splits = {'training', 'validation', 'test'}
        if split_name not in valid_splits:
            raise ValueError(f"split_name must be one of {valid_splits}, got '{split_name}'")

        # Load split indices (no mapping needed)
        # peaks: use fold_name prefix
        temp = os.path.join(peaks_zarr_path, "_auxiliary", "split_indices.npz")
        split_key = f"{fold_name}__{split_name}"
        self.peak_split_indices = np.load(temp)[split_key]
        # nonpeaks
        temp = os.path.join(nonpeaks_zarr_path, "_auxiliary", "split_indices.npz")
        nonpeak_splits = np.load(temp)
        if split_key in nonpeak_splits:
            self.nonpeak_split_indices = nonpeak_splits[split_key]
        elif split_name in nonpeak_splits:
            self.nonpeak_split_indices = nonpeak_splits[split_name]
        else:
            available_keys = list(nonpeak_splits.keys())
            raise KeyError(f"Neither {split_key} nor {split_name} found in nonpeak splits. Available keys: {available_keys}")
        
        # Generate initial mixed indices
        self.indices = self._generate_mixed_indices(epoch_offset=0)

        print(f"Loaded {len(self.peak_split_indices) + len(self.nonpeak_split_indices):,} samples for {split_name} split")
        print(f"Extraction windows: seq_width={seq_width}, track_width={track_width}")
        print(f"Augmentations: RC={self.rc_aug}, shift={self.shift_aug} (max_shift={self.max_shift})")
        print(f"DDP mode: {'Multi-rank safe' if self.ddp_safe else 'Single-rank optimized'}")
        
        # CRITICAL: Load data ONCE globally, share across all splits
        # NOTE: Always loads full dataset into memory unless cached before for this process.
        if self.ddp_safe:
            # Multi-rank DDP safe: use memory-mapped .npy files
            self._load_ddp_safe_shared()
        else:
            # Single-rank: direct in-memory loading (shared across splits)
            self._load_single_rank_shared()

    def _navigate_to_dataset(self, store, path_list):
        """Navigate through nested zarr groups using path list"""
        current = store
        for key in path_list:
            if key not in current:
                raise ValueError(f"Key '{key}' not found in zarr group. Available keys: {list(current.keys())}")
            current = current[key]
        return current

    def _load_ddp_safe_shared(self):
        # Global cache key includes dataset combination
        import os
        cache_key = (
            self.peaks_zarr_path,
            self.nonpeaks_zarr_path,
            self.fold_name,
            tuple(self.seq_dataset_path),
            tuple([tuple(t) for t in self.predicted_peaks_track_dataset_paths]),
            tuple([tuple(t) for t in self.predicted_nonpeaks_track_dataset_paths]),
            tuple([tuple(t) for t in self.experimental_peaks_track_dataset_paths]),
            tuple([tuple(t) for t in self.experimental_nonpeaks_track_dataset_paths]),
        )

        if cache_key in self._global_data_cache:
            print(f"♻️  Using shared DDP-safe cache for {self.split_name} split")
            cached = self._global_data_cache[cache_key]
            self.peak_seq_data = cached['peak_seq_data']
            self.nonpeak_seq_data = cached['nonpeak_seq_data']
            self.pred_peak_track_data = cached.get('pred_peak_track_data')
            self.pred_nonpeak_track_data = cached.get('pred_nonpeak_track_data')
            self.exp_peak_track_data = cached.get('exp_peak_track_data')
            self.exp_nonpeak_track_data = cached.get('exp_nonpeak_track_data')
            return

        print(f"Loading dataset with DDP-safe memory mapping (first split: {self.split_name})...")

        os.makedirs(self.cache_dir, exist_ok=True)

        def _tag(paths):
            return ('_'.join(['_'.join(t) for t in paths]) if paths else 'none')
        cache_filename = f"data_{self.fold_name}_{'_'.join(self.seq_dataset_path)}_predP_{_tag(self.predicted_peaks_track_dataset_paths)}_predN_{_tag(self.predicted_nonpeaks_track_dataset_paths)}_expP_{_tag(self.experimental_peaks_track_dataset_paths)}_expN_{_tag(self.experimental_nonpeaks_track_dataset_paths)}.npy"
        cache_filename = cache_filename.replace('/', '_')

        peak_seq_cache = os.path.join(self.cache_dir, f"peak_seq_{cache_filename}")
        nonpeak_seq_cache = os.path.join(self.cache_dir, f"nonpeak_seq_{cache_filename}")
        pred_peak_track_cache = os.path.join(self.cache_dir, f"pred_peak_track_{cache_filename}")
        pred_nonpeak_track_cache = os.path.join(self.cache_dir, f"pred_nonpeak_track_{cache_filename}")
        exp_peak_track_cache = os.path.join(self.cache_dir, f"exp_peak_track_{cache_filename}")
        exp_nonpeak_track_cache = os.path.join(self.cache_dir, f"exp_nonpeak_track_{cache_filename}")

        need_data = not all([
            os.path.exists(peak_seq_cache),
            os.path.exists(nonpeak_seq_cache),
            os.path.exists(pred_peak_track_cache) if self.predicted_peaks_track_dataset_paths else True,
            os.path.exists(pred_nonpeak_track_cache) if self.predicted_nonpeaks_track_dataset_paths else True,
            os.path.exists(exp_peak_track_cache) if self.experimental_peaks_track_dataset_paths else True,
            os.path.exists(exp_nonpeak_track_cache) if self.experimental_nonpeaks_track_dataset_paths else True,
        ])

        if need_data:
            print("  Creating data cache files (one-time setup)...")
            start_time = time.time()
            with FileLock(peak_seq_cache + '.lock'):
                if not os.path.exists(peak_seq_cache):
                    print("    Loading peak sequence data...")
                    peak_seq_data = np.asarray(self._navigate_to_dataset(self.peak_store, self.seq_dataset_path))
                    np.save(peak_seq_cache, peak_seq_data)
                    os.chmod(peak_seq_cache, 0o664)
                    print(f"      ✓ Saved peak sequences: {peak_seq_data.shape} {peak_seq_data.dtype}")
                    del peak_seq_data

                    print("    Loading nonpeak sequence data...")
                    nonpeak_seq_data = np.asarray(self._navigate_to_dataset(self.nonpeak_store, self.seq_dataset_path))
                    np.save(nonpeak_seq_cache, nonpeak_seq_data)
                    os.chmod(nonpeak_seq_cache, 0o664)
                    print(f"      ✓ Saved nonpeak sequences: {nonpeak_seq_data.shape} {nonpeak_seq_data.dtype}")
                    del nonpeak_seq_data

                    if self.predicted_peaks_track_dataset_paths:
                        print("    Loading predicted peak track data...")
                        peak_list = []
                        for path_list in self.predicted_peaks_track_dataset_paths:
                            arr = np.asarray(self._navigate_to_dataset(self.peak_store, path_list))
                            peak_list.append(arr)
                        pred_peak = np.concatenate(peak_list, axis=2)
                        np.save(pred_peak_track_cache, pred_peak)
                        os.chmod(pred_peak_track_cache, 0o664)
                        print(f"      ✓ Saved predicted peak tracks: {pred_peak.shape} {pred_peak.dtype}")
                        del peak_list, pred_peak

                    if self.predicted_nonpeaks_track_dataset_paths:
                        print("    Loading predicted nonpeak track data...")
                        np_list = []
                        for path_list in self.predicted_nonpeaks_track_dataset_paths:
                            arr = np.asarray(self._navigate_to_dataset(self.nonpeak_store, path_list))
                            np_list.append(arr)
                        pred_non = np.concatenate(np_list, axis=2)
                        np.save(pred_nonpeak_track_cache, pred_non)
                        os.chmod(pred_nonpeak_track_cache, 0o664)
                        print(f"      ✓ Saved predicted nonpeak tracks: {pred_non.shape} {pred_non.dtype}")
                        del np_list, pred_non

                    if self.experimental_peaks_track_dataset_paths:
                        print("    Loading experimental peak track data...")
                        peak_list = []
                        for path_list in self.experimental_peaks_track_dataset_paths:
                            arr = np.asarray(self._navigate_to_dataset(self.peak_store, path_list))
                            peak_list.append(arr)
                        exp_peak = np.concatenate(peak_list, axis=2)
                        np.save(exp_peak_track_cache, exp_peak)
                        os.chmod(exp_peak_track_cache, 0o664)
                        print(f"      ✓ Saved experimental peak tracks: {exp_peak.shape} {exp_peak.dtype}")
                        del peak_list, exp_peak

                    if self.experimental_nonpeaks_track_dataset_paths:
                        print("    Loading experimental nonpeak track data...")
                        np_list = []
                        for path_list in self.experimental_nonpeaks_track_dataset_paths:
                            arr = np.asarray(self._navigate_to_dataset(self.nonpeak_store, path_list))
                            np_list.append(arr)
                        exp_non = np.concatenate(np_list, axis=2)
                        np.save(exp_nonpeak_track_cache, exp_non)
                        os.chmod(exp_nonpeak_track_cache, 0o664)
                        print(f"      ✓ Saved experimental nonpeak tracks: {exp_non.shape} {exp_non.dtype}")
                        del np_list, exp_non

                    gc.collect()

            cache_time = time.time() - start_time
            print(f"    Cache creation completed in {cache_time:.1f}s")

        print("  Loading memory-mapped data...")
        self.peak_seq_data = np.load(peak_seq_cache, mmap_mode='r')
        self.nonpeak_seq_data = np.load(nonpeak_seq_cache, mmap_mode='r')
        self.pred_peak_track_data = np.load(pred_peak_track_cache, mmap_mode='r') if os.path.exists(pred_peak_track_cache) else None
        self.pred_nonpeak_track_data = np.load(pred_nonpeak_track_cache, mmap_mode='r') if os.path.exists(pred_nonpeak_track_cache) else None
        self.exp_peak_track_data = np.load(exp_peak_track_cache, mmap_mode='r') if os.path.exists(exp_peak_track_cache) else None
        self.exp_nonpeak_track_data = np.load(exp_nonpeak_track_cache, mmap_mode='r') if os.path.exists(exp_nonpeak_track_cache) else None

        self._global_data_cache[cache_key] = {
            'peak_seq_data': self.peak_seq_data,
            'nonpeak_seq_data': self.nonpeak_seq_data,
            'pred_peak_track_data': self.pred_peak_track_data,
            'pred_nonpeak_track_data': self.pred_nonpeak_track_data,
            'exp_peak_track_data': self.exp_peak_track_data,
            'exp_nonpeak_track_data': self.exp_nonpeak_track_data,
        }

        # Calculate memory usage
        peak_seq_gb = self.peak_seq_data.nbytes / (1024**3)
        nonpeak_seq_gb = self.nonpeak_seq_data.nbytes / (1024**3)
        peak_track_gb = sum([
            (self.pred_peak_track_data.nbytes / (1024**3)) if self.pred_peak_track_data is not None else 0,
            (self.exp_peak_track_data.nbytes / (1024**3)) if self.exp_peak_track_data is not None else 0,
        ])
        nonpeak_track_gb = sum([
            (self.pred_nonpeak_track_data.nbytes / (1024**3)) if self.pred_nonpeak_track_data is not None else 0,
            (self.exp_nonpeak_track_data.nbytes / (1024**3)) if self.exp_nonpeak_track_data is not None else 0,
        ])
        total_gb = peak_seq_gb + nonpeak_seq_gb + peak_track_gb + nonpeak_track_gb

        print(f"📊 DDP-safe dataset loaded | Total: {total_gb:.1f} GB (shared across ranks + splits)")

    def _load_single_rank_shared(self):
        """Load data directly into memory (single-rank, shared across splits)."""
        # Global cache key includes dataset combination (including mounted track sets)
        cache_key = (
            self.peaks_zarr_path,
            self.nonpeaks_zarr_path,
            self.fold_name,
            tuple(self.seq_dataset_path),
            tuple([tuple(t) for t in self.predicted_peaks_track_dataset_paths]),
            tuple([tuple(t) for t in self.predicted_nonpeaks_track_dataset_paths]),
            tuple([tuple(t) for t in self.experimental_peaks_track_dataset_paths]),
            tuple([tuple(t) for t in self.experimental_nonpeaks_track_dataset_paths]),
        )
        
        if cache_key in self._global_data_cache:
            print(f"♻️  Using shared in-memory cache for {self.split_name} split")
            cached = self._global_data_cache[cache_key]
            self.peak_seq_data = cached['peak_seq_data']
            self.nonpeak_seq_data = cached['nonpeak_seq_data']
            # rehydrate mounted track arrays if present
            self.pred_peak_track_data = cached.get('pred_peak_track_data')
            self.pred_nonpeak_track_data = cached.get('pred_nonpeak_track_data')
            self.exp_peak_track_data = cached.get('exp_peak_track_data')
            self.exp_nonpeak_track_data = cached.get('exp_nonpeak_track_data')
            return
        
        print(f"🚀 Loading entire dataset into memory (first split: {self.split_name})...")
        start_time = time.time()
        
        # Load sequence data for peaks
        print("  Loading peak sequences...")
        peak_seq_data = self._navigate_to_dataset(self.peak_store, self.seq_dataset_path)
        self.peak_seq_data = np.asarray(peak_seq_data)
        peak_seq_gb = self.peak_seq_data.nbytes / (1024**3)
        del peak_seq_data
        print(f"    ✓ Peak sequences: {self.peak_seq_data.shape} {self.peak_seq_data.dtype} = {peak_seq_gb:.1f} GB")
        
        # Load sequence data for nonpeaks
        print("  Loading nonpeak sequences...")
        nonpeak_seq_data = self._navigate_to_dataset(self.nonpeak_store, self.seq_dataset_path)
        self.nonpeak_seq_data = np.asarray(nonpeak_seq_data)
        nonpeak_seq_gb = self.nonpeak_seq_data.nbytes / (1024**3)
        del nonpeak_seq_data
        print(f"    ✓ Nonpeak sequences: {self.nonpeak_seq_data.shape} {self.nonpeak_seq_data.dtype} = {nonpeak_seq_gb:.1f} GB")
        
        # Load predicted/experimental tracks for peaks
        if self.predicted_peaks_track_dataset_paths:
            lst = []
            for path_list in self.predicted_peaks_track_dataset_paths:
                arr = np.asarray(self._navigate_to_dataset(self.peak_store, path_list))
                lst.append(arr)
                print(f"    ✓ Pred peak {path_list}: {arr.shape} {arr.dtype} = {arr.nbytes/(1024**3):.1f} GB")
            self.pred_peak_track_data = np.concatenate(lst, axis=2)
            del lst
        else:
            self.pred_peak_track_data = None

        if self.experimental_peaks_track_dataset_paths:
            lst = []
            for path_list in self.experimental_peaks_track_dataset_paths:
                arr = np.asarray(self._navigate_to_dataset(self.peak_store, path_list))
                lst.append(arr)
                print(f"    ✓ Exp peak {path_list}: {arr.shape} {arr.dtype} = {arr.nbytes/(1024**3):.1f} GB")
            self.exp_peak_track_data = np.concatenate(lst, axis=2)
            del lst
        else:
            self.exp_peak_track_data = None

        # Load predicted/experimental tracks for nonpeaks
        if self.predicted_nonpeaks_track_dataset_paths:
            lst = []
            for path_list in self.predicted_nonpeaks_track_dataset_paths:
                arr = np.asarray(self._navigate_to_dataset(self.nonpeak_store, path_list))
                lst.append(arr)
                print(f"    ✓ Pred nonpeak {path_list}: {arr.shape} {arr.dtype} = {arr.nbytes/(1024**3):.1f} GB")
            self.pred_nonpeak_track_data = np.concatenate(lst, axis=2)
            del lst
        else:
            self.pred_nonpeak_track_data = None

        if self.experimental_nonpeaks_track_dataset_paths:
            lst = []
            for path_list in self.experimental_nonpeaks_track_dataset_paths:
                arr = np.asarray(self._navigate_to_dataset(self.nonpeak_store, path_list))
                lst.append(arr)
                print(f"    ✓ Exp nonpeak {path_list}: {arr.shape} {arr.dtype} = {arr.nbytes/(1024**3):.1f} GB")
            self.exp_nonpeak_track_data = np.concatenate(lst, axis=2)
            del lst
        else:
            self.exp_nonpeak_track_data = None

        peak_seq_gb = self.peak_seq_data.nbytes / (1024**3)
        nonpeak_seq_gb = self.nonpeak_seq_data.nbytes / (1024**3)
        peak_track_gb = sum([
            (self.pred_peak_track_data.nbytes / (1024**3)) if self.pred_peak_track_data is not None else 0,
            (self.exp_peak_track_data.nbytes / (1024**3)) if self.exp_peak_track_data is not None else 0,
        ])
        nonpeak_track_gb = sum([
            (self.pred_nonpeak_track_data.nbytes / (1024**3)) if self.pred_nonpeak_track_data is not None else 0,
            (self.exp_nonpeak_track_data.nbytes / (1024**3)) if self.exp_nonpeak_track_data is not None else 0,
        ])
        total_gb = peak_seq_gb + nonpeak_seq_gb + peak_track_gb + nonpeak_track_gb
        print(f"    ✓ Peak sequences: {self.peak_seq_data.shape} {self.peak_seq_data.dtype} = {peak_seq_gb:.1f} GB")
        print(f"    ✓ Nonpeak sequences: {self.nonpeak_seq_data.shape} {self.nonpeak_seq_data.dtype} = {nonpeak_seq_gb:.1f} GB")
        if self.pred_peak_track_data is not None:
            print(f"    ✓ Pred peak tracks: {self.pred_peak_track_data.shape} {self.pred_peak_track_data.dtype}")
        if self.exp_peak_track_data is not None:
            print(f"    ✓ Exp peak tracks: {self.exp_peak_track_data.shape} {self.exp_peak_track_data.dtype}")
        if self.pred_nonpeak_track_data is not None:
            print(f"    ✓ Pred nonpeak tracks: {self.pred_nonpeak_track_data.shape} {self.pred_nonpeak_track_data.dtype}")
        if self.exp_nonpeak_track_data is not None:
            print(f"    ✓ Exp nonpeak tracks: {self.exp_nonpeak_track_data.shape} {self.exp_nonpeak_track_data.dtype}")
        print(f"    ✓ Total: {total_gb:.1f} GB")
        
        load_time = time.time() - start_time
        print(f"🎉 Dataset loaded in {load_time:.1f}s")
        
        # Store in global cache for other splits to reuse
        self._global_data_cache[cache_key] = {
            'peak_seq_data': self.peak_seq_data,
            'nonpeak_seq_data': self.nonpeak_seq_data,
            'pred_peak_track_data': self.pred_peak_track_data,
            'pred_nonpeak_track_data': self.pred_nonpeak_track_data,
            'exp_peak_track_data': self.exp_peak_track_data,
            'exp_nonpeak_track_data': self.exp_nonpeak_track_data,
        }
        
        # Clean up to free memory
        gc.collect()

    def _apply_augmentations(self, seq_data, track_data, region_type, *, forced_shift=None, forced_rc=None, return_params=False):
        """
        Apply shift and reverse complement augmentations.
        
        Assumes:
        - seq_data: (peak_seq_zarr_width, 4) or (nonpeak_seq_zarr_width, 4)
            one-hot encoded DNA sequence for peak or nonpeak dataset
        - track_data: (peak_track_zarr_width, n_tracks) or (nonpeak_track_zarr_width, n_tracks)
            track data for peak or nonpeak dataset or None
        - region_type: "peak" or "nonpeak"
        
        Windows are extracted centered on the (possibly shifted) summit (shared location on ref genome).
        
        For reverse complement, sequence and track handling is different:
        - sequence: apply reverse complement (reverse both dimensions)  
        - tracks: apply reverse (reverse only the position dimension)

        Returns:
        - augmented seq_data: (seq_width, 4) one-hot encoded DNA sequence
        - augmented track_data: (track_width, n_tracks) track data or None
        """
        # Generate shift ONCE and use for both sequence and track
        if forced_shift is not None:
            shift = int(forced_shift)
        elif self.shift_aug:  
            shift = np.random.randint(-self.max_shift, self.max_shift + 1)
        else:
            shift = 0
        
        ###### Sequence ######
        seq_center_pos = self.peak_seq_zarr_summit if region_type == "peak" \
            else self.nonpeak_seq_zarr_summit
        ### 1. Apply shift augmentation - use the shared shift
        seq_center_pos = seq_center_pos + shift
        ### 2. Compute sequence window bounds centered on (possibly shifted) summit
        start = seq_center_pos - self.seq_width // 2
        end = start + self.seq_width
        ### 3. Extract windows (both summit-centered)
        augmented_seq_data = seq_data[start:end, :].copy()  # (seq_width, 4)

        ###### Track ######
        if track_data is not None:
            track_center_pos = self.peak_track_zarr_summit if region_type == "peak" \
                else self.nonpeak_track_zarr_summit
            ### 1. Apply shift augmentation - use the SAME shift
            track_center_pos = track_center_pos + shift
            ### 2. Compute track window bounds centered on (possibly shifted) summit
            start = track_center_pos - self.track_width // 2
            end = start + self.track_width
            ### 3. Extract windows (both summit-centered)
            augmented_track_data = track_data[start:end, :].copy()  # (track_width, n_tracks)
        else:
            augmented_track_data = None
        
        # 4. Apply reverse complement augmentation
        if forced_rc is not None:
            rc_flip = bool(forced_rc)
        else:
            rc_flip = (self.rc_aug and np.random.random() < 0.5)

        if rc_flip:  # Already filtered by split_name

            # Sequence: reverse the augmented data along the position and nucleotide dimensions
            augmented_seq_data = augmented_seq_data[::-1, ::-1].copy()  # Need copy for in-place operations

            # Track: reverse the augmented data along the position dimension
            if augmented_track_data is not None:
                augmented_track_data = augmented_track_data[::-1, :].copy()
                # If strand channel pairs are defined, swap them
                if self.strand_channel_pairs is not None:
                    T = augmented_track_data.shape[1]
                    for a, b in self.strand_channel_pairs:
                        if 0 <= a < T and 0 <= b < T and a != b:
                            augmented_track_data[:, [a, b]] = augmented_track_data[:, [b, a]]
        
        if return_params:
            return augmented_seq_data, augmented_track_data, (shift, rc_flip)
        return augmented_seq_data, augmented_track_data 

    def _generate_mixed_indices(self, epoch_offset=0):
        """Generate mixed indices combining peaks and nonpeaks"""
        # Test split: return ALL regions, no subsampling
        if self.split_name == "test":
            mixed_indices = []
            # Add all peak indices - direct DataFrame access (guaranteed aligned)
            for peak_zarr_idx in self.peak_split_indices:
                peak_identifier = self.peak_regions_df.iloc[peak_zarr_idx]['identifier']
                mixed_indices.append(('peak', peak_zarr_idx, peak_identifier))
        
            # Add all nonpeak indices - direct DataFrame access (guaranteed aligned)
            for nonpeak_zarr_idx in self.nonpeak_split_indices:
                nonpeak_identifier = self.nonpeak_regions_df.iloc[nonpeak_zarr_idx]['identifier']
                mixed_indices.append(('nonpeak', nonpeak_zarr_idx, nonpeak_identifier))
            
            # No shuffle for test - deterministic order
            return mixed_indices
        
        # Training/Validation: Apply ratio-based sampling
        seed = self.base_sampling_seed + (epoch_offset if self.split_name == "training" else 0)
        rng = np.random.RandomState(seed)
        
        # Calculate how many nonpeaks we need
        n_peaks = len(self.peak_split_indices)
        n_nonpeaks_needed = n_peaks // self.peak_to_nonpeak_ratio
        
        # Sample nonpeaks randomly
        available_nonpeaks = len(self.nonpeak_split_indices)
        sampled_nonpeak_local_indices = rng.choice(
            available_nonpeaks, 
            size=n_nonpeaks_needed, 
            replace=(n_nonpeaks_needed > available_nonpeaks)
        )
        
        mixed_indices = []
        
        # Add all peak indices with identifiers
        for peak_zarr_idx in self.peak_split_indices:
            peak_identifier = self.peak_regions_df.iloc[peak_zarr_idx]['identifier']
            mixed_indices.append(('peak', peak_zarr_idx, peak_identifier))
        
        # Add sampled nonpeak indices with identifiers
        for nonpeak_local_idx in sampled_nonpeak_local_indices:
            nonpeak_zarr_idx = self.nonpeak_split_indices[nonpeak_local_idx]
            nonpeak_identifier = self.nonpeak_regions_df.iloc[nonpeak_zarr_idx]['identifier']
            mixed_indices.append(('nonpeak', nonpeak_zarr_idx, nonpeak_identifier))
        
        # Shuffle for training, but keep deterministic for validation
        if self.split_name == "training":
            # NOTE: unconditionally shuffles for training
            rng.shuffle(mixed_indices)
        # Validation: no shuffle - deterministic order
        
        return mixed_indices

    def subsample_nonpeaks_for_epoch(self, epoch_id, verbose=False):
        """
        Called by Lightning module to resample nonpeaks for new epoch.
        This allows for dynamic resampling of negative examples during training.
        """
        if self.split_name != "training":
            return  # Only resample for training
        
        old_len = len(self.indices)
        self.indices = self._generate_mixed_indices(epoch_offset=epoch_id)
        new_len = len(self.indices)
        
        if verbose:
            print(f"🔄 Resampling nonpeaks for epoch {epoch_id}")
            print(f"   Dataset size: {old_len:,} → {new_len:,} samples")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        """Ultra-fast access; returns sequence and both predicted/experimental track windows if available.

        Output:
            seq_tensor: (seq_width, 4)
            pred_track_tensor: (track_width, T) or None
            exp_track_tensor: (track_width, T) or None
            region_type: 'peak' | 'nonpeak'
            identifier: str
        """
        region_type, zarr_idx, identifier = self.indices[idx]

        # Select arrays by region
        if region_type == 'peak':
            seq_data = self.peak_seq_data[zarr_idx]
            pred_track_full = self.pred_peak_track_data[zarr_idx] if getattr(self, 'pred_peak_track_data', None) is not None else None
            exp_track_full = self.exp_peak_track_data[zarr_idx] if getattr(self, 'exp_peak_track_data', None) is not None else None
        else:
            seq_data = self.nonpeak_seq_data[zarr_idx]
            pred_track_full = self.pred_nonpeak_track_data[zarr_idx] if getattr(self, 'pred_nonpeak_track_data', None) is not None else None
            exp_track_full = self.exp_nonpeak_track_data[zarr_idx] if getattr(self, 'exp_nonpeak_track_data', None) is not None else None

        # Decide augmentation params once (based on sequence) and reuse
        seq_aug, _, (shift, rc_flip) = self._apply_augmentations(seq_data, None, region_type, return_params=True)

        # Extract track windows consistently for predicted/experimental using forced decisions
        pred_win = None
        exp_win = None
        if pred_track_full is not None:
            _, pred_win = self._apply_augmentations(seq_data, pred_track_full, region_type, forced_shift=shift, forced_rc=rc_flip)
        if exp_track_full is not None:
            _, exp_win = self._apply_augmentations(seq_data, exp_track_full, region_type, forced_shift=shift, forced_rc=rc_flip)

        # Convert to Float32 tensors efficiently
        seq_tensor = torch.from_numpy(seq_aug).float()
        # Emit empty-channel tensors for missing tracks to allow default collation
        if pred_win is not None:
            pred_tensor = torch.from_numpy(pred_win).float()
        else:
            pred_tensor = torch.empty(self.track_width, 0, dtype=torch.float32)
        if exp_win is not None:
            exp_tensor = torch.from_numpy(exp_win).float()
        else:
            exp_tensor = torch.empty(self.track_width, 0, dtype=torch.float32)

        return seq_tensor, pred_tensor, exp_tensor, region_type, identifier

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
    rc_strand_flip=False,
    strand_channel_pairs=None,
    **kwargs,
):
    """
    Smart dataset factory for dual zarr (peaks + nonpeaks) that loads data once and shares across splits.
    
    Args:
        peaks_zarr_path: Path to peaks zarr store (contains all folds)
        nonpeaks_zarr_path: Path to nonpeaks zarr store (fold-specific)
        fold_name: Fold identifier (e.g., "fold_0")
        seq_dataset_path: List of nested keys to sequence dataset
        peaks_track_dataset_paths: List of lists, each containing nested keys to peak track datasets
        nonpeaks_track_dataset_paths: List of lists, each containing nested keys to nonpeak track datasets
        split_name: Name of the split ('training', 'validation', 'test')
        **kwargs: Additional arguments passed to dataset constructor
    
    Usage:
        train_ds = create_dataset(
            peaks_zarr_path="/path/to/peaks.zarr",
            nonpeaks_zarr_path="/path/to/nonpeaks_fold_0.zarr", 
            fold_name="fold_0",
            seq_dataset_path=["reference_dna", "dna_seq"],
            peaks_track_dataset_paths=[["fold_0__model_based_tracks", "atac_track"]],
            nonpeaks_track_dataset_paths=[["model_based_tracks", "atac_track"]],
            split_name="training"
        )
    """
    return BasePeakNonpeakZarrDataset(
        peaks_zarr_path=peaks_zarr_path,
        nonpeaks_zarr_path=nonpeaks_zarr_path,
        fold_name=fold_name,
        seq_dataset_path=seq_dataset_path,
        predicted_peaks_track_dataset_paths=predicted_peaks_track_dataset_paths,
        predicted_nonpeaks_track_dataset_paths=predicted_nonpeaks_track_dataset_paths,
        experimental_peaks_track_dataset_paths=experimental_peaks_track_dataset_paths,
        experimental_nonpeaks_track_dataset_paths=experimental_nonpeaks_track_dataset_paths,
        split_name=split_name,
        rc_strand_flip=rc_strand_flip,
        strand_channel_pairs=strand_channel_pairs,
        **kwargs
    )
