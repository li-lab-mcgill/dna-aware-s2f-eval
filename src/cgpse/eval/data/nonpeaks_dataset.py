import zarr
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from typing import List, Optional, Tuple
import os
import time
import gc
from filelock import FileLock

# Global flag to prevent multiple set_start_method calls
_FORK_CONTEXT_SET = False

class NonpeaksZarrDataset(Dataset):
    """
    Simplified PyTorch Dataset for pre-extracted genomics nonpeak data.
    - Nonpeaks only (no peaks)
    - Optional track loading (predicted and/or experimental)
    - Flexible: can load any subset of {DNA, predicted tracks, experimental tracks}
    - Loads data once globally and shares across splits to prevent memory duplication
    - Automatic DDP-safe strategy (single GPU: in-memory, multi-GPU: memory-mapped)
    """

    # Global shared cache
    _global_data_cache = {}

    def __init__(
        self,
        # Dataset parameters
        nonpeaks_zarr_path: str,
        fold_name: str,
        seq_dataset_path: List[str],
        # Optional track dataset paths
        predicted_track_dataset_paths: Optional[List[List[str]]] = None,
        experimental_track_dataset_paths: Optional[List[List[str]]] = None,
        # Optional mask dataset path
        mask_dataset_path: Optional[List[str]] = None,
        mask_replicate_id: Optional[int] = None,
        split_name: str = "training",
        n_debug: Optional[int] = None,
        # Sequence and track parameters
        seq_width: int = 1000,
        track_width: int = 1000,
        max_shift: int = 0,
        shift_aug: bool = False,
        rc_aug: bool = False,
        rc_strand_flip: bool = False,
        strand_channel_pairs: Optional[List[Tuple[int, int]]] = None,
        # Memory, DDP and caching parameters
        ddp_safe: str = "single",  # "auto", "single", "multi"
        cache_dir: Optional[str] = None,
    ):
        """
        Args:
            Dataset parameters:
                nonpeaks_zarr_path: Path to nonpeaks zarr store
                fold_name: Fold identifier (e.g., "fold_0", "fold_1")
                seq_dataset_path: List of nested keys to sequence dataset (e.g., ["reference_dna", "dna_name"])
                predicted_track_dataset_paths: Optional list of lists, each containing nested keys to predicted track datasets
                experimental_track_dataset_paths: Optional list of lists, each containing nested keys to experimental track datasets
                split_name: Name of the split ('training', 'validation', 'test')
            Sequence and track parameters:
                seq_width: Width of sequence to extract
                track_width: Width of tracks to extract
                max_shift: Maximum shift in bp for shift augmentation (only applied to 'training')
                rc_aug: Enable reverse complement augmentation (only applied to 'training')
                shift_aug: Enable shift augmentation (only applied to 'training')
            Memory, DDP and caching parameters:
                ddp_safe: "auto" (detect), "single" (in-memory), "multi" (memory-mapped)
                cache_dir: Directory to store .npy cache files for memory-mapped mode
        """
        self.fold_name = fold_name
        self.split_name = split_name
        self.nonpeaks_zarr_path = nonpeaks_zarr_path
        self.seq_dataset_path = seq_dataset_path
        self.mask_dataset_path = mask_dataset_path
        self.mask_replicate_id = mask_replicate_id
        self.n_debug = n_debug

        # Track path sets (optional)
        self.predicted_track_dataset_paths = predicted_track_dataset_paths or []
        self.experimental_track_dataset_paths = experimental_track_dataset_paths or []

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
            self.cache_dir = os.path.join(nonpeaks_zarr_path, "_cache")
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

        print(f"=== Initializing Nonpeaks Dataset ({split_name}) ===")
        print(f"Nonpeaks zarr: {nonpeaks_zarr_path}")
        print(f"Fold: {fold_name}")

        # Open zarr store
        self.nonpeak_store = zarr.open(nonpeaks_zarr_path, mode='r')

        # Get zarr widths and summits
        # Sequence
        temp = self._navigate_to_dataset(self.nonpeak_store, self.seq_dataset_path)
        self.seq_zarr_width = temp.shape[1]
        self.seq_zarr_summit = self.seq_zarr_width // 2

        # Track: validate widths across all provided sets
        track_widths = []
        for path_list in (self.predicted_track_dataset_paths + self.experimental_track_dataset_paths):
            temp = self._navigate_to_dataset(self.nonpeak_store, path_list)
            track_widths.append(temp.shape[1])
        if track_widths:
            assert len(set(track_widths)) == 1, "All provided track datasets must have the same width"
            self.track_zarr_width = track_widths[0]
        else:
            self.track_zarr_width = self.seq_zarr_width
        self.track_zarr_summit = self.track_zarr_width // 2

        # Validate dimensions while accounting for the shift augmentation boundaries
        if self.shift_aug:
            assert seq_width + 2*max_shift <= self.seq_zarr_width, \
                f"seq_width ({seq_width}) + max_shift ({max_shift}) must be <= zarr_width ({self.seq_zarr_width})"
            assert track_width + 2*max_shift <= self.track_zarr_width, \
                f"track_width ({track_width}) + max_shift ({max_shift}) must be <= zarr_width ({self.track_zarr_width})"
        else:
            assert seq_width <= self.seq_zarr_width, \
                f"seq_width ({seq_width}) must be <= zarr_width ({self.seq_zarr_width})"
            assert track_width <= self.track_zarr_width, \
                f"track_width ({track_width}) must be <= zarr_width ({self.track_zarr_width})"

        # Load regions_df from _auxiliary
        temp = os.path.join(nonpeaks_zarr_path, "_auxiliary", "regions_df.tsv")
        self.regions_df = pd.read_csv(temp, sep='\t')

        # Validate split_name is one of the expected values
        valid_splits = {'training', 'validation', 'test'}
        if split_name not in valid_splits:
            raise ValueError(f"split_name must be one of {valid_splits}, got '{split_name}'")

        # Load split indices
        temp = os.path.join(nonpeaks_zarr_path, "_auxiliary", "split_indices.npz")
        nonpeak_splits = np.load(temp)
        split_key = f"{fold_name}__{split_name}"

        if split_key in nonpeak_splits:
            self.split_indices = nonpeak_splits[split_key]
        elif split_name in nonpeak_splits:
            self.split_indices = nonpeak_splits[split_name]
        else:
            available_keys = list(nonpeak_splits.keys())
            raise KeyError(f"Neither {split_key} nor {split_name} found in nonpeak splits. Available keys: {available_keys}")

        if self.n_debug is not None:
            if self.n_debug < 0:
                raise ValueError(f"n_debug must be >= 0, got {self.n_debug}")
            self.split_indices = self.split_indices[: self.n_debug]
            print(f"Debug subset enabled: first {len(self.split_indices)} samples")

        print(f"Loaded {len(self.split_indices):,} samples for {split_name} split")
        print(f"Extraction windows: seq_width={seq_width}, track_width={track_width}")
        print(f"Augmentations: RC={self.rc_aug}, shift={self.shift_aug} (max_shift={self.max_shift})")
        print(f"DDP mode: {'Multi-rank safe' if self.ddp_safe else 'Single-rank optimized'}")

        # Load data ONCE globally, share across all splits
        if self.ddp_safe:
            self._load_ddp_safe_shared()
        else:
            self._load_single_rank_shared()
        self.update_mask_source(self.mask_dataset_path, self.mask_replicate_id)

    def _navigate_to_dataset(self, store, path_list):
        """Navigate through nested zarr groups using path list"""
        current = store
        for key in path_list:
            if key not in current:
                raise ValueError(f"Key '{key}' not found in zarr group. Available keys: {list(current.keys())}")
            current = current[key]
        return current

    def _load_ddp_safe_shared(self):
        """Load data with memory-mapped files (DDP-safe)"""
        cache_key = (
            self.nonpeaks_zarr_path,
            self.fold_name,
            tuple(self.seq_dataset_path),
            tuple([tuple(t) for t in self.predicted_track_dataset_paths]),
            tuple([tuple(t) for t in self.experimental_track_dataset_paths]),
        )

        if cache_key in self._global_data_cache:
            print(f"♻️  Using shared DDP-safe cache for {self.split_name} split")
            cached = self._global_data_cache[cache_key]
            self.seq_data = cached['seq_data']
            self.pred_track_data = cached.get('pred_track_data')
            self.exp_track_data = cached.get('exp_track_data')
            return

        print(f"Loading dataset with DDP-safe memory mapping (first split: {self.split_name})...")

        os.makedirs(self.cache_dir, exist_ok=True)

        def _tag(paths):
            return ('_'.join(['_'.join(t) for t in paths]) if paths else 'none')

        cache_filename = f"data_{self.fold_name}_{'_'.join(self.seq_dataset_path)}_pred_{_tag(self.predicted_track_dataset_paths)}_exp_{_tag(self.experimental_track_dataset_paths)}.npy"
        cache_filename = cache_filename.replace('/', '_')

        seq_cache = os.path.join(self.cache_dir, f"seq_{cache_filename}")
        pred_track_cache = os.path.join(self.cache_dir, f"pred_track_{cache_filename}")
        exp_track_cache = os.path.join(self.cache_dir, f"exp_track_{cache_filename}")

        need_data = not all([
            os.path.exists(seq_cache),
            os.path.exists(pred_track_cache) if self.predicted_track_dataset_paths else True,
            os.path.exists(exp_track_cache) if self.experimental_track_dataset_paths else True,
        ])

        if need_data:
            print("  Creating data cache files (one-time setup)...")
            start_time = time.time()
            with FileLock(seq_cache + '.lock'):
                if not os.path.exists(seq_cache):
                    print("    Loading sequence data...")
                    seq_data = np.asarray(self._navigate_to_dataset(self.nonpeak_store, self.seq_dataset_path))
                    np.save(seq_cache, seq_data)
                    os.chmod(seq_cache, 0o664)
                    print(f"      ✓ Saved sequences: {seq_data.shape} {seq_data.dtype}")
                    del seq_data

                    if self.predicted_track_dataset_paths:
                        print("    Loading predicted track data...")
                        track_list = []
                        for path_list in self.predicted_track_dataset_paths:
                            arr = np.asarray(self._navigate_to_dataset(self.nonpeak_store, path_list))
                            track_list.append(arr)
                        pred_track = np.concatenate(track_list, axis=2)
                        np.save(pred_track_cache, pred_track)
                        os.chmod(pred_track_cache, 0o664)
                        print(f"      ✓ Saved predicted tracks: {pred_track.shape} {pred_track.dtype}")
                        del track_list, pred_track

                    if self.experimental_track_dataset_paths:
                        print("    Loading experimental track data...")
                        track_list = []
                        for path_list in self.experimental_track_dataset_paths:
                            arr = np.asarray(self._navigate_to_dataset(self.nonpeak_store, path_list))
                            track_list.append(arr)
                        exp_track = np.concatenate(track_list, axis=2)
                        np.save(exp_track_cache, exp_track)
                        os.chmod(exp_track_cache, 0o664)
                        print(f"      ✓ Saved experimental tracks: {exp_track.shape} {exp_track.dtype}")
                        del track_list, exp_track

                    gc.collect()

            cache_time = time.time() - start_time
            print(f"    Cache creation completed in {cache_time:.1f}s")

        print("  Loading memory-mapped data...")
        self.seq_data = np.load(seq_cache, mmap_mode='r')
        self.pred_track_data = np.load(pred_track_cache, mmap_mode='r') if os.path.exists(pred_track_cache) else None
        self.exp_track_data = np.load(exp_track_cache, mmap_mode='r') if os.path.exists(exp_track_cache) else None

        self._global_data_cache[cache_key] = {
            'seq_data': self.seq_data,
            'pred_track_data': self.pred_track_data,
            'exp_track_data': self.exp_track_data,
        }

        # Calculate memory usage
        seq_gb = self.seq_data.nbytes / (1024**3)
        pred_gb = (self.pred_track_data.nbytes / (1024**3)) if self.pred_track_data is not None else 0
        exp_gb = (self.exp_track_data.nbytes / (1024**3)) if self.exp_track_data is not None else 0
        total_gb = seq_gb + pred_gb + exp_gb

        print(f"📊 DDP-safe dataset loaded | Total: {total_gb:.1f} GB (shared across ranks + splits)")

    def _load_single_rank_shared(self):
        """Load data directly into memory (single-rank, shared across splits)"""
        cache_key = (
            self.nonpeaks_zarr_path,
            self.fold_name,
            tuple(self.seq_dataset_path),
            tuple([tuple(t) for t in self.predicted_track_dataset_paths]),
            tuple([tuple(t) for t in self.experimental_track_dataset_paths]),
        )

        if cache_key in self._global_data_cache:
            print(f"♻️  Using shared in-memory cache for {self.split_name} split")
            cached = self._global_data_cache[cache_key]
            self.seq_data = cached['seq_data']
            self.pred_track_data = cached.get('pred_track_data')
            self.exp_track_data = cached.get('exp_track_data')
            return

        print(f"🚀 Loading entire dataset into memory (first split: {self.split_name})...")
        start_time = time.time()

        # Load sequence data
        print("  Loading sequences...")
        seq_data = self._navigate_to_dataset(self.nonpeak_store, self.seq_dataset_path)
        self.seq_data = np.asarray(seq_data)
        seq_gb = self.seq_data.nbytes / (1024**3)
        del seq_data
        print(f"    ✓ Sequences: {self.seq_data.shape} {self.seq_data.dtype} = {seq_gb:.1f} GB")

        # Load predicted tracks (optional)
        if self.predicted_track_dataset_paths:
            print("  Loading predicted tracks...")
            lst = []
            for path_list in self.predicted_track_dataset_paths:
                arr = np.asarray(self._navigate_to_dataset(self.nonpeak_store, path_list))
                lst.append(arr)
                print(f"    ✓ Pred track {path_list}: {arr.shape} {arr.dtype} = {arr.nbytes/(1024**3):.1f} GB")
            self.pred_track_data = np.concatenate(lst, axis=2)
            del lst
        else:
            self.pred_track_data = None

        # Load experimental tracks (optional)
        if self.experimental_track_dataset_paths:
            print("  Loading experimental tracks...")
            lst = []
            for path_list in self.experimental_track_dataset_paths:
                arr = np.asarray(self._navigate_to_dataset(self.nonpeak_store, path_list))
                lst.append(arr)
                print(f"    ✓ Exp track {path_list}: {arr.shape} {arr.dtype} = {arr.nbytes/(1024**3):.1f} GB")
            self.exp_track_data = np.concatenate(lst, axis=2)
            del lst
        else:
            self.exp_track_data = None

        # Summary
        seq_gb = self.seq_data.nbytes / (1024**3)
        pred_gb = (self.pred_track_data.nbytes / (1024**3)) if self.pred_track_data is not None else 0
        exp_gb = (self.exp_track_data.nbytes / (1024**3)) if self.exp_track_data is not None else 0
        total_gb = seq_gb + pred_gb + exp_gb

        print(f"    ✓ Sequences: {self.seq_data.shape} {self.seq_data.dtype} = {seq_gb:.1f} GB")
        if self.pred_track_data is not None:
            print(f"    ✓ Predicted tracks: {self.pred_track_data.shape} {self.pred_track_data.dtype} = {pred_gb:.1f} GB")
        if self.exp_track_data is not None:
            print(f"    ✓ Experimental tracks: {self.exp_track_data.shape} {self.exp_track_data.dtype} = {exp_gb:.1f} GB")
        print(f"    ✓ Total: {total_gb:.1f} GB")

        load_time = time.time() - start_time
        print(f"🎉 Dataset loaded in {load_time:.1f}s")

        # Store in global cache for other splits to reuse
        self._global_data_cache[cache_key] = {
            'seq_data': self.seq_data,
            'pred_track_data': self.pred_track_data,
            'exp_track_data': self.exp_track_data,
        }

        gc.collect()

    def update_mask_source(self, mask_dataset_path=None, mask_replicate_id=None):
        if (mask_dataset_path is None) != (mask_replicate_id is None):
            raise ValueError("mask_dataset_path and mask_replicate_id must be provided together.")
        if mask_replicate_id is not None and mask_replicate_id < 0:
            raise ValueError(f"mask_replicate_id must be >= 0, got {mask_replicate_id}")

        self.mask_dataset_path = mask_dataset_path
        self.mask_replicate_id = mask_replicate_id

        if self.mask_dataset_path is None:
            self.mask_data = None
            return

        mask_arr = self._navigate_to_dataset(self.nonpeak_store, self.mask_dataset_path)
        if mask_arr.ndim != 3:
            raise ValueError("mask dataset must be 3D (N, L, K)")
        if self.mask_replicate_id >= mask_arr.shape[2]:
            raise IndexError(
                f"mask_replicate_id {self.mask_replicate_id} out of range for K={mask_arr.shape[2]}"
            )

        mask_data = np.asarray(mask_arr[:, :, self.mask_replicate_id])
        self.mask_data = mask_data

    def _apply_augmentations(self, seq_data, track_data, *, forced_shift=None, forced_rc=None, return_params=False):
        """
        Apply shift and reverse complement augmentations.

        Args:
            seq_data: (seq_zarr_width, 4) one-hot encoded DNA sequence
            track_data: (track_zarr_width, n_tracks) track data or None
            forced_shift: Optional fixed shift value
            forced_rc: Optional fixed reverse complement flag
            return_params: If True, return (seq, track, (shift, rc_flip))

        Returns:
            augmented seq_data: (seq_width, 4)
            augmented track_data: (track_width, n_tracks) or None
            (shift, rc_flip): if return_params=True
        """
        # Generate shift ONCE and use for both sequence and track
        if forced_shift is not None:
            shift = int(forced_shift)
        elif self.shift_aug:
            shift = np.random.randint(-self.max_shift, self.max_shift + 1)
        else:
            shift = 0

        # Sequence
        seq_center_pos = self.seq_zarr_summit + shift
        start = seq_center_pos - self.seq_width // 2
        end = start + self.seq_width
        augmented_seq_data = seq_data[start:end, :].copy()

        # Track
        if track_data is not None:
            track_center_pos = self.track_zarr_summit + shift
            start = track_center_pos - self.track_width // 2
            end = start + self.track_width
            augmented_track_data = track_data[start:end, :].copy()
        else:
            augmented_track_data = None

        # Apply reverse complement augmentation
        if forced_rc is not None:
            rc_flip = bool(forced_rc)
        else:
            rc_flip = (self.rc_aug and np.random.random() < 0.5)

        if rc_flip:
            # Sequence: reverse both position and nucleotide dimensions
            augmented_seq_data = augmented_seq_data[::-1, ::-1].copy()

            # Track: reverse position dimension only
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

    def __len__(self):
        return len(self.split_indices)

    def __getitem__(self, idx):
        """
        Returns sequence and optionally predicted/experimental track windows.

        Output:
            seq_tensor: (seq_width, 4) - DNA sequence
            pred_track_tensor: (track_width, T) or None - predicted tracks
            exp_track_tensor: (track_width, T) or None - experimental tracks
            identifier: str - region identifier
        """
        zarr_idx = self.split_indices[idx]
        identifier = self.regions_df.iloc[zarr_idx]['identifier']

        # Load sequence (always present)
        seq_data = self.seq_data[zarr_idx]

        # Load tracks (optional)
        pred_track_full = self.pred_track_data[zarr_idx] if self.pred_track_data is not None else None
        exp_track_full = self.exp_track_data[zarr_idx] if self.exp_track_data is not None else None
        mask_full = self.mask_data[zarr_idx] if self.mask_data is not None else None

        # Apply augmentations to sequence and get params
        seq_aug, _, (shift, rc_flip) = self._apply_augmentations(seq_data, None, return_params=True)

        # Apply same augmentations to tracks (if present)
        pred_win = None
        exp_win = None
        if pred_track_full is not None:
            _, pred_win = self._apply_augmentations(seq_data, pred_track_full, forced_shift=shift, forced_rc=rc_flip)
        if exp_track_full is not None:
            _, exp_win = self._apply_augmentations(seq_data, exp_track_full, forced_shift=shift, forced_rc=rc_flip)

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

        if mask_full is not None:
            mask_tensor = torch.from_numpy(mask_full).float().unsqueeze(-1)
        else:
            mask_tensor = torch.empty(self.track_width, 0, dtype=torch.float32)

        return seq_tensor, pred_tensor, exp_tensor, mask_tensor, "nonpeak", identifier


def create_dataset(
    nonpeaks_zarr_path,
    fold_name,
    seq_dataset_path,
    predicted_track_dataset_paths=None,
    experimental_track_dataset_paths=None,
    mask_dataset_path=None,
    mask_replicate_id=None,
    split_name="training",
    n_debug=None,
    rc_strand_flip=False,
    strand_channel_pairs=None,
    **kwargs,
):
    """
    Factory for creating simplified nonpeaks-only dataset.

    Args:
        nonpeaks_zarr_path: Path to nonpeaks zarr store
        fold_name: Fold identifier (e.g., "fold_0")
        seq_dataset_path: List of nested keys to sequence dataset
        predicted_track_dataset_paths: Optional list of lists for predicted track datasets
        experimental_track_dataset_paths: Optional list of lists for experimental track datasets
        split_name: Name of the split ('training', 'validation', 'test')
        **kwargs: Additional arguments passed to dataset constructor

    Usage:
        # DNA only
        ds = create_dataset(
            nonpeaks_zarr_path="/path/to/nonpeaks.zarr",
            fold_name="fold_0",
            seq_dataset_path=["reference_dna", "dna_seq"],
            split_name="training"
        )

        # DNA + predicted tracks
        ds = create_dataset(
            nonpeaks_zarr_path="/path/to/nonpeaks.zarr",
            fold_name="fold_0",
            seq_dataset_path=["reference_dna", "dna_seq"],
            predicted_track_dataset_paths=[["model_tracks", "atac"]],
            split_name="training"
        )

        # All three
        ds = create_dataset(
            nonpeaks_zarr_path="/path/to/nonpeaks.zarr",
            fold_name="fold_0",
            seq_dataset_path=["reference_dna", "dna_seq"],
            predicted_track_dataset_paths=[["model_tracks", "atac"]],
            experimental_track_dataset_paths=[["experimental_tracks", "atac"]],
            split_name="training"
        )
    """
    return NonpeaksZarrDataset(
        nonpeaks_zarr_path=nonpeaks_zarr_path,
        fold_name=fold_name,
        seq_dataset_path=seq_dataset_path,
        predicted_track_dataset_paths=predicted_track_dataset_paths,
        experimental_track_dataset_paths=experimental_track_dataset_paths,
        mask_dataset_path=mask_dataset_path,
        mask_replicate_id=mask_replicate_id,
        split_name=split_name,
        n_debug=n_debug,
        rc_strand_flip=rc_strand_flip,
        strand_channel_pairs=strand_channel_pairs,
        **kwargs
    )
