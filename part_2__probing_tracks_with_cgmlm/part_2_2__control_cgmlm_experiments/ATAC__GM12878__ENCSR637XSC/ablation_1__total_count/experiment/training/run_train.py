import zarr
import json
import pdb
import time
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pathlib import Path
from ml_collections import ConfigDict
from copy import deepcopy

# Extend sys.path  
import sys
import os 
import os.path as osp
sys.path.append(os.path.join(
    os.path.dirname(__file__), 
    "../../../../../../.."
))
from config import *  # noqa 
sys.path.append(MANUSCRIPT_SECTION_2_DIR)

# dataset 
from section_2_utils.data.peak_nonpeak.latest.base_zarr_dataset import create_dataset
# batch preparer
from section_2_utils.data.peak_nonpeak.latest.prepare_batch import BatchPreparer
# tCLM model 
from section_2_utils.models.tCLM.legacy_model.unet import MaskedGenomeUNet
# tCLM training lightning model
from section_2_utils.lightning.tCLM.model import LightningModel
# loss function
from section_2_utils.losses.masked_CE import MaskedCELoss


def main(config_dict):
    # Set seed for reproducibility
    pl.seed_everything(config_dict.task.seed)
    
    # Initialize a global rank variable for DDP awareness
    global_rank = 0
    local_rank = 0
    world_size = 1
    
    # Check if we're in a DDP environment
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        global_rank = int(os.environ["RANK"]) if "RANK" in os.environ else local_rank
        world_size = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    
    # Helper function to print only on the main process
    def main_process_print(*args, **kwargs):
        if global_rank == 0:
            print(*args, **kwargs)
    
    main_process_print(f"Starting process {global_rank}/{world_size-1}")
    
    """ Init datasets """
    # train
    main_process_print("Creating train dataset...")
    train_data_config = deepcopy(config_dict.data)
    train_data_config.split_name = "training"
    train_dataset = create_dataset(**train_data_config.to_dict())
    main_process_print(f"Train dataset length: {len(train_dataset)}")
    
    # valid dataset
    main_process_print("Creating valid dataset...")
    valid_data_config = deepcopy(config_dict.data)
    valid_data_config.split_name = "validation"
    valid_data_config.rc_aug = False
    valid_data_config.shift_aug = False
    valid_data_config.max_shift = 0
    valid_dataset = create_dataset(**valid_data_config.to_dict())
    main_process_print(f"Valid dataset length: {len(valid_dataset)}")
    
    """ Init dataloaders """
    # train
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config_dict.lightning.batch_size,
        shuffle=True,
        num_workers=config_dict.lightning.num_workers,
        pin_memory=config_dict.lightning.pin_memory,
        persistent_workers=config_dict.lightning.persistent_workers
    )
    main_process_print(f"Train dataloader created with {len(train_dataloader)} batches")
    
    # valid (not used in current tests but kept for future)
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=config_dict.lightning.batch_size,
        shuffle=False,
        num_workers=config_dict.lightning.num_workers,
        pin_memory=config_dict.lightning.pin_memory,
        persistent_workers=config_dict.lightning.persistent_workers
    )
    main_process_print(f"Valid dataloader created with {len(valid_dataloader)} batches")

    """ Init model """
    main_process_print("Creating model...")
    model = MaskedGenomeUNet(**config_dict.model.to_dict())
    main_process_print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    """ Init loss function """
    main_process_print("Creating loss function...")
    loss_fn = MaskedCELoss(**config_dict.loss.to_dict())
    main_process_print("Loss function created")

    """ Init batch preparer """
    batch_preparer = BatchPreparer(**config_dict.batch_preparer.to_dict())

    """ Init Lightning model"""
    # build model
    lightning_model = LightningModel(
        config_dict,
        model,
        loss_fn,
        batch_preparer
    )

    """ Train model """
    # train and validate model 
    trainer = lightning_model.train_on_dataset(train_dataloader, valid_dataloader)
    # NOTE: Do we save the trainer?
    

if __name__ == "__main__":
    # Create configuration
    config_dict = ConfigDict()

    # dynamic track key for lighnting model
    config_dict.track_key_in_batch = "exp_tracks"

    """ Task config and params """
    config_dict.task = ConfigDict()
    config_dict.task.seed = 42 #35
    config_dict.task.fold_name = "fold_0"
    config_dict.task.assay_type = "ATAC"
    config_dict.task.assay_experiment_identifier = "GM12878__ENCSR637XSC"
    config_dict.task.ablation_type = "ablation_1__count_equalization"

    """ Model config and params """
    
    # Model configuration
    config_dict.model = ConfigDict()
    config_dict.model.T = 1 + 4  # 1 functional track + 4 DNA one-hot
    config_dict.model.depths = [128, 192, 256]  # Tiny model depths
    config_dict.model.input_kernel_size = 23
    config_dict.model.input_conv_channels = 64
    config_dict.model.num_groups = 8
    config_dict.model.conv_kernel_size = 3
    config_dict.model.dropout = 0.1
    config_dict.model.padding_mode = 'same'
    
    # Bottleneck configuration
    config_dict.model.bottleneck_config = {
        'num_layers': 2,
        'heads': 4,
        'ff_mult': 2,
        'dropout': 0.1,
        'use_gqa': True,
        'use_flash': False,
        'rotary_emb_fraction': 0.5,
        'rotary_emb_base': 20000.0
    }

    """ Loss config and params """
    
    # Loss configuration
    config_dict.loss = ConfigDict()
    config_dict.loss.ignore_index = -100

    """ Batch preparer config and params """

    # Mask configuration - row masking
    p_mask_dna_only = 1.00
    p_mask_track_only = 0.00
    mask_pct_min = 0.10
    mask_pct_max = 0.20
    mask_config = {
        "variable_rate_uniform_partial_row_sampler": {
            "prob": 1.0,
            "kwargs": {
                "mask_pct_min": mask_pct_min,
                "mask_pct_max": mask_pct_max,
                "p_mask_dna_only": p_mask_dna_only,
                "p_mask_track_only": p_mask_track_only,
            }
        },
    }
    if mask_pct_min == mask_pct_max:
        mask_dir = f"mask_{int(100*mask_pct_min)}pct"
    else:
        mask_dir = f"mask_{int(100*mask_pct_min)}to{int(100*mask_pct_max)}pct"

    # batch preparer configuration
    config_dict.batch_preparer = ConfigDict()
    config_dict.batch_preparer.region_lengths = (-1,) # use full sequences
    config_dict.batch_preparer.bin_width = 1  # no binning
    config_dict.batch_preparer.equalize_counts = False
    config_dict.batch_preparer.use_logits = False
    config_dict.batch_preparer.mask_config = mask_config

    """ Data config and params """

    # Data configuration
    config_dict.data = ConfigDict()
    config_dict.data.fold_name = config_dict.task.fold_name
    config_dict.data.peaks_zarr_path = osp.join(
        MANUSCRIPT_EXPERIMENTS_DIR,
        "section_1__ConditionalLM_as_diagnostic_probe",
        "workspace",
        "data",
        config_dict.task.assay_type,
        config_dict.task.assay_experiment_identifier,
        "preprocessed",
        "zarr_datasets",
        "peaks.all_folds.zarr"
    )
    config_dict.data.nonpeaks_zarr_path = osp.join(
        MANUSCRIPT_EXPERIMENTS_DIR,
        "section_1__ConditionalLM_as_diagnostic_probe",
        "workspace",
        "data",
        config_dict.task.assay_type,
        config_dict.task.assay_experiment_identifier,
        "preprocessed",
        "zarr_datasets",
        f"nonpeaks.{config_dict.data.fold_name}.zarr"
    )
    config_dict.data.seq_dataset_path = [
        "reference_dna", 
        "dna_summitcentered1000bp_symmetric500bpshift"
    ]
    config_dict.data.predicted_peaks_track_dataset_paths = None
    config_dict.data.predicted_nonpeaks_track_dataset_paths = None
    config_dict.data.experimental_peaks_track_dataset_paths = [
        [
            "observed_tracks", 
            "observed_track_summitcentered1000bp_symmetric500bpshift"
        ]
    ]
    config_dict.data.experimental_nonpeaks_track_dataset_paths = [
        [
            "observed_tracks", 
            "observed_track_summitcentered1000bp_symmetric500bpshift"
        ]
    ]
    # peak/nonpeak sampling parameters
    config_dict.data.peak_to_nonpeak_ratio = 10
    config_dict.data.base_sampling_seed = config_dict.task.seed
    # sequence and track parameters
    config_dict.data.seq_width = 1000
    config_dict.data.track_width = 1000
    config_dict.data.max_shift = 500
    config_dict.data.rc_aug = False
    config_dict.data.shift_aug = True
    config_dict.data.ddp_safe = "auto"
    
    # Lightning configuration 
    config_dict.lightning = ConfigDict()
    config_dict.lightning.seed = config_dict.task.seed
    # device and ddp
    config_dict.lightning.accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    config_dict.lightning.devices = [8] if torch.cuda.is_available() else 1  # Use 1 CPU device instead of None
    config_dict.lightning.strategy = \
        pl.strategies.DDPStrategy(find_unused_parameters=False) \
        if isinstance(config_dict.lightning.devices, list) and len(config_dict.lightning.devices) > 1 and torch.cuda.is_available() else "auto"
    # dataloader 
    num_devices = len(config_dict.lightning.devices) if isinstance(config_dict.lightning.devices, list) else config_dict.lightning.devices
    config_dict.lightning.batch_size = int(128*3.5) * num_devices
    config_dict.lightning.num_workers = 4 // num_devices
    config_dict.lightning.pin_memory = True
    config_dict.lightning.persistent_workers = True
    # early stopping
    config_dict.lightning.es_patience = 5
    # weight decay 
    config_dict.lightning.weight_decay = 0.001
    # learning rate and max epochs
    config_dict.lightning.max_epochs = 200
    config_dict.lightning.base_lr = 1e-3 * num_devices
    config_dict.lightning.min_lr = 1e-5
    config_dict.lightning.warmup_type = None
    config_dict.lightning.warmup_epochs = 0
    config_dict.lightning.cooldown_type = None
    config_dict.lightning.cooldown_epochs = config_dict.lightning.max_epochs - config_dict.lightning.warmup_epochs
    # trainer
    config_dict.lightning.deterministic = False
    config_dict.lightning.precision = "32-true"
    config_dict.lightning.gradient_clip_val = 1.0
    config_dict.lightning.val_check_interval = 1.0

    # Experiment directory
    config_dict.task.experiment_dir = osp.join(
        MANUSCRIPT_EXPERIMENTS_DIR,
        "section_2__ConditionalLM_controlled_experiments",
        "workspace",
        "runs",
        config_dict.task.assay_type,
        config_dict.task.assay_experiment_identifier,
        config_dict.task.ablation_type,
        "observed",
        config_dict.task.fold_name,
        mask_dir,
        f"seed_{config_dict.task.seed}"
    )
    os.makedirs(config_dict.task.experiment_dir, exist_ok=True)
    
    # Run main function
    main(config_dict)
