import zarr
import json
import pdb
import os
import sys
import time
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pathlib import Path
from ml_collections import ConfigDict
from copy import deepcopy

# Add project root to path
print(os.path.dirname(__file__))
# for local_utils import 
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../.."))
from local_utils.datasets.v3.with_nonpeak_sampling.masked_multimodal_input_dna_output_zarr_dataset import create_dataset
from local_utils.models.masked_unet.multimodal_input_dna_output import MaskedGenomeUNet
from local_utils.losses.masked_dna_lm_loss import MaskedDNALMLoss
from local_utils.lightning_models.masked_unet.multimodal_input_dna_output import LightningModel
# for gloabl utils and config import
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../../../.."))
from config import * 


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
    train_data_config.mask_config = train_mask_config
    train_dataset = create_dataset(**train_data_config.to_dict())
    main_process_print(f"Train dataset length: {len(train_dataset)}")
    
    # valid dataset
    main_process_print("Creating valid dataset...")
    valid_data_config = deepcopy(config_dict.data)
    valid_data_config.split_name = "validation"
    valid_data_config.rc_aug = False
    valid_data_config.shift_aug = False
    valid_data_config.max_shift = 0
    valid_data_config.mask_config = valid_mask_config
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
    loss_fn = MaskedDNALMLoss(**config_dict.loss.to_dict())
    main_process_print("Loss function created")

    """ Init Lightning model"""
    # build model
    lightning_model = LightningModel(
        config_dict,
        model,
        loss_fn,
    )

    """ Train model """
    # train and validate model 
    trainer = lightning_model.train_on_dataset(train_dataloader, valid_dataloader)
    # NOTE: Do we save the trainer?
    

if __name__ == "__main__":
    # Create configuration
    config_dict = ConfigDict()
    
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
    
    # Loss configuration
    config_dict.loss = ConfigDict()
    config_dict.loss.w_dna = 1.0  # Weight for DNA loss
    config_dict.loss.ignore_index = -100
    
    # Task configuration
    config_dict.task = ConfigDict()
    config_dict.task.seed = 42 #35
    config_dict.task.fold_name = "fold_0"
    config_dict.task.cell_type = "GM12878"  # Match your dataset

    # Data configuration
    config_dict.data = ConfigDict()
    config_dict.data.fold_name = config_dict.task.fold_name
    config_dict.data.peaks_zarr_path = osp.join(
        MANUSCRIPT_EXPERIMENTS_DIR,
        "section_1__ConditionalLM_as_diagnostic_probe",
        "workspace",
        "data",
        "ATAC",
        "GM12878__ENCSR637XSC",
        "preprocessed",
        "zarr_datasets",
        "peaks.all_folds.zarr"
    )
    config_dict.data.nonpeaks_zarr_path = osp.join(
        MANUSCRIPT_EXPERIMENTS_DIR,
        "section_1__ConditionalLM_as_diagnostic_probe",
        "workspace",
        "data",
        "ATAC",
        "GM12878__ENCSR637XSC",
        "preprocessed",
        "zarr_datasets",
        f"nonpeaks.{config_dict.data.fold_name}.zarr"
    )
    config_dict.data.seq_dataset_path = [
        "reference_dna", 
        "dna_summitcentered1000bp_symmetric500bpshift"
    ]
    config_dict.data.peaks_track_dataset_paths = [
        [
            f"{config_dict.data.fold_name}__model_based_tracks", 
            "scaled_bias_predicted_track_summitcentered1000bp_symmetric500bpshift"
        ]
    ]
    config_dict.data.nonpeaks_track_dataset_paths = [
        [
            f"{config_dict.data.fold_name}__model_based_tracks", 
            "scaled_bias_predicted_track_summitcentered1000bp_symmetric500bpshift"
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
    
    # Mask configuration - row masking
    p_mask_dna_only = 1.00
    p_mask_track_only = 0.00
    mask_pct_min = 1.00
    mask_pct_max = 1.00
    train_mask_config = {
        "variable_rate_uniform_partial_row_sampler": {
            "prob": 1.0,
            "kwargs": {
                "mask_percentage": None, # NOTE: unused
                "mask_pct_min": mask_pct_min,
                "mask_pct_max": mask_pct_max,
                "p_mask_dna_only": p_mask_dna_only,
                "p_mask_track_only": p_mask_track_only,
            }
        },
    }
    valid_mask_config = deepcopy(train_mask_config)

    # Experiment directory
    config_dict.task.model_name = \
        f"tiny__variable_rate_partial_row_mask__" + \
        f"from_{mask_pct_min}_to_{mask_pct_max}_pct__" + \
        f"p_dna_{p_mask_dna_only}__p_track_{p_mask_track_only}__"
    config_dict.task.experiment_dir = osp.join(
        MANUSCRIPT_EXPERIMENTS_DIR,
        "section_1__ConditionalLM_as_diagnostic_probe",
        "workspace",
        "runs",
        "ATAC",
        "GM12878__ENCSR637XSC",
        "ConditionalLM",
        "chrombpnet__bias_model__predicted",
        config_dict.task.fold_name,
        config_dict.task.model_name,
        f"DNA_{config_dict.loss.w_dna}"
    )
    os.makedirs(config_dict.task.experiment_dir, exist_ok=True)
    
    # Lightning configuration 
    config_dict.lightning = ConfigDict()
    config_dict.lightning.seed = config_dict.task.seed
    # device and ddp
    config_dict.lightning.accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    config_dict.lightning.devices = [2] if torch.cuda.is_available() else 1  # Use 1 CPU device instead of None
    config_dict.lightning.strategy = \
        pl.strategies.DDPStrategy(find_unused_parameters=False) \
        if isinstance(config_dict.lightning.devices, list) and len(config_dict.lightning.devices) > 1 and torch.cuda.is_available() else "auto"
    # dataloader 
    num_devices = len(config_dict.lightning.devices) if isinstance(config_dict.lightning.devices, list) else config_dict.lightning.devices
    config_dict.lightning.batch_size = 512 * num_devices
    config_dict.lightning.num_workers = 8 // num_devices
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
    
    # Run main function
    main(config_dict)