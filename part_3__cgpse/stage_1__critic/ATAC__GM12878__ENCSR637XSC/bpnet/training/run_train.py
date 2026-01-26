import json
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
    "../../../../../../../.."
))
from config import *  # noqa
sys.path.append(MANUSCRIPT_SECTION_3_DIR)
sys.path.append(MULTITASK_CRITIC_PRETRAINING_DIR)

# dataset / batch preparer
from utils.data.peak_nonpeak.base_zarr_dataset import create_dataset
from utils.data.peak_nonpeak.prepare_batch import BatchPreparer

# Split-stem single-channel multi-head critic with DISC, orchestrator, lightning wrapper
from utils.critic.architecture.splitstem_singlechannel_multihead_with_disc_critic import (
    SplitStemSingleHeadRouterCritic,
)
from utils.critic.pretraining.splitstem_singlechannel_multihead_with_disc_critic import (
    SingleStemCriticPretrainingOrchestrator,
)
from utils.critic.pretraining.splitstem_singlechannel_multihead_with_disc_critic.lightning_multiple_vald_masks import (
    LightningSingleStemCriticPretrain,
)

from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping


# ---------------------- Mask configuration utilities ----------------------

def build_uniform_row_mask_config(
    mask_pct_min: float,
    mask_pct_max: float,
    p_mask_dna_only: float = 1.00,
    p_mask_track_only: float = 0.00,
) -> dict:
    """
    Build uniform partial row mask configuration for MaskSampler.

    Parameters
    ----------
    mask_pct_min : float
        Minimum mask percentage (0.0 to 1.0)
    mask_pct_max : float
        Maximum mask percentage (0.0 to 1.0)
    p_mask_dna_only : float
        Probability of masking DNA only (default: 1.0)
    p_mask_track_only : float
        Probability of masking track only (default: 0.0)

    Returns
    -------
    mask_config : dict
        Configuration dict for MaskSampler
    """
    mask_config = {
        "variable_rate_uniform_partial_row_sampler": {
            "prob": 1.0,
            "kwargs": {
                "mask_pct_min": mask_pct_min,
                "mask_pct_max": mask_pct_max,
                "p_mask_dna_only": p_mask_dna_only,
                "p_mask_track_only": p_mask_track_only,
            }
        }
    }
    return mask_config


def build_validation_mask_configs() -> dict:
    """
    Build validation mask configurations for different mask rate bins.

    Returns
    -------
    val_mask_configs : dict
        Dictionary mapping bin names to mask configurations
    """
    p_mask_dna_only = 1.00
    p_mask_track_only = 0.00

    val_mask_configs = {
        "10to20": build_uniform_row_mask_config(0.10, 0.20, p_mask_dna_only, p_mask_track_only),
        "20to40": build_uniform_row_mask_config(0.20, 0.40, p_mask_dna_only, p_mask_track_only),
        "40to60": build_uniform_row_mask_config(0.40, 0.60, p_mask_dna_only, p_mask_track_only),
        "60to80": build_uniform_row_mask_config(0.60, 0.80, p_mask_dna_only, p_mask_track_only),
        "80to100": build_uniform_row_mask_config(0.80, 1.00, p_mask_dna_only, p_mask_track_only),
        "100": build_uniform_row_mask_config(1.00, 1.00, p_mask_dna_only, p_mask_track_only),
    }
    return val_mask_configs


# Helper functions for critic configs
def make_cglm_backbone_default(
    *,
    T: int = 1 + 4,
    depths: list = [128, 192, 256],
    input_kernel_size: int = 23,
    input_conv_channels: int = 64,
    num_groups: int = 8,
    conv_kernel_size: int = 3,
    dropout: float = 0.1,
    padding_mode: str = 'same',
    mask_aware_padding: bool = True,
    apply_shared_token_ln: bool = True,
    shared_token_ln_affine: bool = False,
    bottleneck_config: dict = None,
) -> ConfigDict:
    if bottleneck_config is None:
        bottleneck_config = {
            'num_layers': 2,
            'heads': 4,
            'ff_mult': 2,
            'dropout': 0.1,
            'use_gqa': True,
            'use_flash': False,
            'rotary_emb_fraction': 0.5,
            'rotary_emb_base': 20000.0,
        }
    cfg = ConfigDict()
    cfg.T = int(T)
    cfg.depths = list(depths)
    cfg.input_kernel_size = int(input_kernel_size)
    cfg.input_conv_channels = int(input_conv_channels)
    cfg.num_groups = int(num_groups)
    cfg.conv_kernel_size = int(conv_kernel_size)
    cfg.dropout = float(dropout)
    cfg.padding_mode = str(padding_mode)
    cfg.mask_aware_padding = bool(mask_aware_padding)
    cfg.apply_shared_token_ln = bool(apply_shared_token_ln)
    cfg.shared_token_ln_affine = bool(shared_token_ln_affine)
    cfg.bottleneck_config = dict(bottleneck_config)
    return cfg

def make_cglm_mlm_proj_default(
    *,
    in_channels: int,
    out_channels: int = 4,
    expansion_factor: int = 2,
    dropout_prelogits: float = 0.0,
) -> ConfigDict:
    """
    Create MLM head config.
    """
    cfg = ConfigDict()
    cfg.in_channels = int(in_channels)
    cfg.out_channels = int(out_channels)
    cfg.expansion_factor = int(expansion_factor)
    cfg.dropout_prelogits = float(dropout_prelogits)
    return cfg


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

    # valid
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=config_dict.lightning.batch_size,
        shuffle=False,
        num_workers=config_dict.lightning.num_workers,
        pin_memory=config_dict.lightning.pin_memory,
        persistent_workers=config_dict.lightning.persistent_workers
    )
    main_process_print(f"Valid dataloader created with {len(valid_dataloader)} batches")

    """ Init critic model """
    main_process_print("Creating single-stem single-channel multi-head critic model (with DISC)...")
    backbone_kwargs = config_dict.model.backbone.to_dict()
    mlm_head_cfg = config_dict.model.mlm_head.to_dict()
    disc_head_cfg = config_dict.model.disc_head.to_dict()
    head_names = ["GT", "S2F", "DISC"]
    per_head_kwargs = {
        "GT": dict(mlm_head_cfg),
        "S2F": dict(mlm_head_cfg),
        "DISC": dict(disc_head_cfg),
    }

    critic = SplitStemSingleHeadRouterCritic(
        backbone_kwargs=backbone_kwargs,
        head_names=head_names,
        per_head_kwargs=per_head_kwargs,
    )
    main_process_print(f"Model created with {sum(p.numel() for p in critic.parameters()):,} parameters")

    """ Init batch preparers """
    batch_preparer_cfg = config_dict.batch_preparer.to_dict()
    train_batch_preparer = BatchPreparer(**batch_preparer_cfg)

    # Validation: create one batch preparer per mask configuration
    val_mask_configs = build_validation_mask_configs()
    val_batch_preparers = {}
    for mask_name, mask_cfg in val_mask_configs.items():
        val_bp_cfg = dict(batch_preparer_cfg)
        val_bp_cfg["mask_config"] = mask_cfg
        val_batch_preparers[mask_name] = BatchPreparer(**val_bp_cfg)
    main_process_print(f"Created {len(val_batch_preparers)} validation batch preparers: {list(val_batch_preparers.keys())}")

    """ Init orchestrator """
    orchestrator = SingleStemCriticPretrainingOrchestrator(
        critic=critic,
        lambda_GT=float(config_dict.training.lambda_GT),
        lambda_S2F=float(config_dict.training.lambda_S2F),
        lambda_DISC=float(config_dict.training.lambda_DISC),
    )

    """ Init Lightning model"""
    lightning_model = LightningSingleStemCriticPretrain(
        config_dict=config_dict,
        critic=critic,
        orchestrator=orchestrator,
        train_batch_preparer=train_batch_preparer,
        val_batch_preparers=val_batch_preparers,  # Now a dict!
    )

    """ Train model """
    logger = CSVLogger(save_dir=config_dict.task.experiment_dir)
    os.makedirs(logger.log_dir, exist_ok=True)
    config_save_path = osp.join(logger.log_dir, "config_dict.json")
    config_json = json.loads(config_dict.to_json_best_effort())
    with open(config_save_path, "w") as f:
        json.dump(config_json, f, indent=4)

    # Checkpoints: per validation mask bin and per head (GT, S2F)
    callbacks = [LearningRateMonitor(logging_interval="epoch")]

    # Create checkpoints for each validation mask bin
    for mask_name in val_mask_configs.keys():
        # Total loss per mask bin
        callbacks.append(
            ModelCheckpoint(
                monitor=f"val_{mask_name}/loss_total",
                mode="min",
                filename=f"best_val_{mask_name}_total_loss-{{epoch}}-{{step}}",
            )
        )
        # GT head per mask bin
        callbacks.append(
            ModelCheckpoint(
                monitor=f"val_{mask_name}/GT",
                mode="min",
                filename=f"best_val_{mask_name}_GT_loss-{{epoch}}-{{step}}",
            )
        )
        # S2F head per mask bin
        callbacks.append(
            ModelCheckpoint(
                monitor=f"val_{mask_name}/S2F",
                mode="min",
                filename=f"best_val_{mask_name}_S2F_loss-{{epoch}}-{{step}}",
            )
        )

    trainer = pl.Trainer(
        max_epochs=config_dict.lightning.max_epochs,
        accelerator=config_dict.lightning.accelerator,
        devices=config_dict.lightning.devices,
        strategy=config_dict.lightning.strategy,
        logger=logger,
        callbacks=callbacks,
        default_root_dir=config_dict.task.experiment_dir,
        val_check_interval=config_dict.lightning.val_check_interval,
        gradient_clip_val=config_dict.lightning.gradient_clip_val,
        deterministic=config_dict.lightning.deterministic,
        precision=config_dict.lightning.precision,
    )

    trainer.fit(
        model=lightning_model,
        train_dataloaders=train_dataloader,
        val_dataloaders=valid_dataloader,
    )

    # Save CPU-safe PyTorch critic checkpoints for downstream use
    ckpt_dir = os.path.join(logger.log_dir, "checkpoints")
    if os.path.isdir(ckpt_dir):
        import glob
        for ckpt_path in glob.glob(os.path.join(ckpt_dir, "*.ckpt")):
            ckpt = torch.load(ckpt_path, map_location="cpu")
            state_dict = ckpt.get("state_dict", ckpt)
            critic_state = {
                k[len("critic."):] : v
                for k, v in state_dict.items()
                if k.startswith("critic.")
            }
            # Rebuild critic instance and save full object
            backbone_kwargs = config_dict.model.backbone.to_dict()
            mlm_head_cfg = config_dict.model.mlm_head.to_dict()
            disc_head_cfg = config_dict.model.disc_head.to_dict()
            head_names = ["GT", "S2F", "DISC"]
            per_head_kwargs = {
                "GT": dict(mlm_head_cfg),
                "S2F": dict(mlm_head_cfg),
                "DISC": dict(disc_head_cfg),
            }
            critic_cpu = SplitStemSingleHeadRouterCritic(
                backbone_kwargs=backbone_kwargs,
                head_names=head_names,
                per_head_kwargs=per_head_kwargs,
            )
            critic_cpu.load_state_dict(critic_state)
            out_path = ckpt_path.replace(".ckpt", ".critic.pt")
            torch.save(critic_cpu, out_path)

    return trainer


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=None, help="GPU id to use (single device).")
    parser.add_argument(
        "--bucket-len",
        type=int,
        default=512,
        help="Override fixed input length per bucket (default: 512).",
    )
    args = parser.parse_args()

    # Create configuration
    config_dict = ConfigDict()

    """ Task config and params """
    config_dict.task = ConfigDict()
    config_dict.task.seed = 42
    config_dict.task.fold_name = "fold_0"
    config_dict.task.assay_type = "ATAC"
    config_dict.task.assay_experiment_identifier = "GM12878__ENCSR637XSC"
    config_dict.task.task_type = "fixed_length__single_window__singlestem_multihead_critic_pretraining"

    """ Model config and params """

    # Critic backbone + MLM head configuration
    config_dict.model = ConfigDict()
    # Backbone configuration
    config_dict.model.backbone = make_cglm_backbone_default(
        T=1 + 4,  # 1 track channel + 4 DNA
        depths=[128, 192, 256],
        input_kernel_size=23,
        input_conv_channels=64,
        num_groups=8,
        conv_kernel_size=3,
        dropout=0.1,
        padding_mode='same',
        mask_aware_padding=True,
        apply_shared_token_ln=True,
        shared_token_ln_affine=False,
        bottleneck_config={
            'num_layers': 2,
            'heads': 4,
            'ff_mult': 2,
            'dropout': 0.1,
            'use_gqa': True,
            'use_flash': False,
            'rotary_emb_fraction': 0.5,
            'rotary_emb_base': 20000.0
        }
    )
    # Model name for logging
    config_dict.model.name = "SplitStemSingleChannelMultiheadWithDiscCritic"

    # Base head configuration (used for GT/S2F MLM heads and DISC head template)
    C_backbone = int(config_dict.model.backbone.input_conv_channels)
    config_dict.model.mlm_head = make_cglm_mlm_proj_default(
        in_channels=C_backbone,  # 64 - backbone output channels
        out_channels=4,  # DNA nucleotides
        expansion_factor=2,
        dropout_prelogits=0.0,
    )
    # DISC head shares the same internal structure but outputs 2 classes (GT vs S2F)
    config_dict.model.disc_head = make_cglm_mlm_proj_default(
        in_channels=C_backbone,
        out_channels=2,
        expansion_factor=2,
        dropout_prelogits=0.0,
    )

    """ Loss config and params """
    config_dict.loss = ConfigDict()
    config_dict.loss.ignore_index = -100

    """ Training config """
    config_dict.training = ConfigDict()
    config_dict.training.lambda_GT = 1.0  # GT head loss weight
    config_dict.training.lambda_S2F = 1.0  # S2F head loss weight
    config_dict.training.lambda_DISC = 1.0  # DISC head loss weight

    """ Batch preparer config and params """

    BUCKET_INPUT_LEN = int(getattr(args, "bucket_len", 512))

    # Build training mask configuration: uniform 10-100% masking
    train_mask_config = build_uniform_row_mask_config(
        mask_pct_min=0.10,
        mask_pct_max=1.00,
        p_mask_dna_only=1.00,
        p_mask_track_only=0.00,
    )

    run_identifier = f"{BUCKET_INPUT_LEN}bp__mask_10to100pct"

    # Batch preparer configuration - fixed input length per bucket
    config_dict.batch_preparer = ConfigDict()
    config_dict.batch_preparer.region_lengths = (BUCKET_INPUT_LEN,)
    config_dict.batch_preparer.region_length_weights = None
    config_dict.batch_preparer.bin_width = 1
    config_dict.batch_preparer.equalize_counts = False
    config_dict.batch_preparer.use_logits = False
    config_dict.batch_preparer.mask_config = train_mask_config

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
    config_dict.data.predicted_peaks_track_dataset_paths = [
        [
            f"{config_dict.data.fold_name}__model_based_tracks",
            "bpnet_predicted_track_summitcentered1000bp_symmetric500bpshift"
        ]
    ]
    config_dict.data.predicted_nonpeaks_track_dataset_paths = [
        [
            f"{config_dict.data.fold_name}__model_based_tracks",
            "bpnet_predicted_track_summitcentered1000bp_symmetric500bpshift"
        ]
    ]
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
    config_dict.lightning.devices = [1] if torch.cuda.is_available() else 1  # default single device
    if args.device is not None:
        config_dict.lightning.devices = [int(args.device)]
    config_dict.lightning.strategy = \
        pl.strategies.DDPStrategy(find_unused_parameters=False) \
        if isinstance(config_dict.lightning.devices, list) and len(config_dict.lightning.devices) > 1 and torch.cuda.is_available() else "auto"
    # dataloader
    num_devices = len(config_dict.lightning.devices) if isinstance(config_dict.lightning.devices, list) else config_dict.lightning.devices
    base_batch_size = int(128 * 9.0) * num_devices
    if BUCKET_INPUT_LEN > 600:
        base_batch_size = max(1, base_batch_size // 2)
    config_dict.lightning.batch_size = base_batch_size
    config_dict.lightning.num_workers = 4 // num_devices
    config_dict.lightning.pin_memory = True
    config_dict.lightning.persistent_workers = True
    # early stopping
    config_dict.lightning.es_patience = 10
    # weight decay
    config_dict.lightning.weight_decay = 0.001
    # learning rate and max epochs
    config_dict.lightning.max_epochs = 2000
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

    # Experiment directory (keep separate from original singlestem_multihead runs)
    config_dict.task.model_name = config_dict.model.name
    config_dict.task.experiment_dir = osp.join(
        MULTITASK_CRITIC_PRETRAINING_DIR,
        "workspace",
        "runs",
        "1__critic_pretraining",
        config_dict.task.assay_type,
        config_dict.task.assay_experiment_identifier,
        "splitstem_singlechannel_multihead_with_disc_critic",
        "bpnet__predicted",
        run_identifier,
        f"seed_{config_dict.task.seed}",
    )
    os.makedirs(config_dict.task.experiment_dir, exist_ok=True)

    # Run main function
    main(config_dict)
