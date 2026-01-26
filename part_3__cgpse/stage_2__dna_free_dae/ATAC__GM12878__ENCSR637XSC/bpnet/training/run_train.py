import sys
import os
import os.path as osp

# Keep Lightning/torch temp files off /tmp (often small on shared systems).
os.environ.setdefault("TMPDIR", "/home/mcb/users/dcakma3/mytmp")
os.makedirs(os.environ["TMPDIR"], exist_ok=True)

# Extend sys.path (do this before importing torch/lightning).
sys.path.append(os.path.join(
    os.path.dirname(__file__),
    "../../../../.."
))
from config import *  # noqa

# Ensure v2 Debiasing root is importable so pickled critics referencing
# `utils.critic.*` can be loaded without any legacy section_3__Debiasing hack.
if MANUSCRIPT_SECTION_3_V2_DIR not in sys.path:
    sys.path.insert(0, MANUSCRIPT_SECTION_3_V2_DIR)

import json
import time
from copy import deepcopy
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from ml_collections import ConfigDict
# -----------------------------------------------------------------------------
# Final DNA-free DAE training script
# -----------------------------------------------------------------------------
# This trains:
#   - DNAFreeDAE (trainable)
# using:
#   - pretrained critic (frozen, GT head)
#
# Model order (track-only DAE):
#   TrackEncoderRouter (gt, s2f) -> BottleneckEncoder -> Bottleneck -> BottleneckDecoder -> TrackDecoder
#
# Masking contract (iteration-2):
#   - BatchPreparer produces (critic_input_mask=M_ci, critic_output_mask=M_co).
#   - Training mask uses nested masks: nested_beta_headroom_mask_sampler -> (M_ci, M_co) with M_co ⊆ M_ci
#   - Validation mask uses tied masks: variable_rate_uniform_partial_row_sampler -> M_ci == M_co
# -----------------------------------------------------------------------------

# dataset / batch preparer (v2)
from utils.data.peak_nonpeak.base_zarr_dataset import create_dataset
from utils.data.batch_preparer import BatchPreparer

# Final DNA-free DAE model
from utils.final_debiaser_v2.model import DNAFreeDAE

# Orchestrator, adapter, losses for final DNA-free DAE
from utils.final_debiaser_v2.orchestrator.critic_adapter import CriticAdapter
from utils.final_debiaser_v2.orchestrator.dae_orchestrator import TrainingOrchestrator
from utils.final_debiaser_v2.orchestrator.dae_lightning import LightningWrapper
from utils.final_debiaser_v2.losses.fidelity_losses import FidelityLossBundle
from utils.final_debiaser_v2.losses.critic_losses import CriticLossBundle
from utils.final_debiaser_v2.losses import GradientRatioController

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

    Note (iteration-2):
    - The batch preparer produces BOTH `critic_input_mask` (M_ci) and `critic_output_mask` (M_co).
    - This mask strategy returns a single (L, 4+T) mask, so BatchPreparer will set:
        critic_input_mask == critic_output_mask
      which corresponds to the "tied masks" regime (no nested masks).

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


def build_nested_beta_headroom_mask_config(
    rco_min: float,
    rco_max: float,
    rci_max: float,
    alpha: float = 2.0,
    beta: float = 3.0,
) -> dict:
    """
    Build nested mask configuration for training:
      - returns (critic_input_mask=M_ci, critic_output_mask=M_co) with M_co ⊆ M_ci

    This uses `nested_beta_headroom_mask_sampler`.
    """
    return {
        "nested_beta_headroom_mask_sampler": {
            "prob": 1.0,
            "kwargs": {
                "rco_min": rco_min,
                "rco_max": rco_max,
                "rci_max": rci_max,
                "alpha": alpha,
                "beta": beta,
            },
        }
    }


def build_validation_mask_configs() -> dict:
    """
    Build validation mask configurations for different mask rate bins.
    Identical to critic pretraining validation configuration.

    Returns
    -------
    val_mask_configs : dict
        Dictionary mapping bin names to mask configurations
    """
    p_mask_dna_only = 1.00
    p_mask_track_only = 0.00

    val_mask_configs = {
        "10to20": build_uniform_row_mask_config(0.10, 0.20, p_mask_dna_only, p_mask_track_only),
        "100": build_uniform_row_mask_config(1.00, 1.00, p_mask_dna_only, p_mask_track_only),
    }
    return val_mask_configs


# ---------------------- Pretrained critic loader ----------------------

CRITIC_MODEL_PATH = (
    "/home/mcb/users/dcakma3/multi_res_bench_v2/multimodal_masked_contrastive_main/"
    "manuscript_experiments/section_3_v2__Debiasing/workspace/runs/1__critic/"
    "ATAC/GM12878__ENCSR637XSC/splitstem_singlechannel_multihead_with_disc_critic/"
    # NOTE(bpnet): match the bpnet critic checkpoint used by the legacy blind-DAE training script.
    "bpnet__predicted/1000bp__mask_10to100pct/seed_42/lightning_logs/version_0/"
    "checkpoints/best_val_100_GT_loss-epoch=611-step=227052.critic.pt"
)


def load_frozen_critic_from_pretraining():
    """
    Load a pretrained critic object and freeze it.

    NOTE: We intentionally avoid reconstructing the critic architecture here.
    This follows the "blind denoise model training" loader pattern:
        torch.load(..., weights_only=False) on a `.critic.pt` file.
    """
    if not os.path.exists(CRITIC_MODEL_PATH):
        raise FileNotFoundError(f"Missing critic checkpoint: {CRITIC_MODEL_PATH}")

    critic = torch.load(CRITIC_MODEL_PATH, map_location="cpu", weights_only=False)
    critic.eval()
    for p in critic.parameters():
        p.requires_grad = False
    return critic


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

    """ Init critic model (load pretrained and freeze) """
    main_process_print("Loading pretrained critic model...")
    critic = load_frozen_critic_from_pretraining()
    main_process_print(f"Critic loaded with {sum(p.numel() for p in critic.parameters()):,} parameters (frozen)")

    """ Init batch preparers """
    # Train batch preparer (mask configuration controls critic input/output masks)
    train_batch_preparer_cfg = config_dict.train_batch_preparer.to_dict()
    train_batch_preparer = BatchPreparer(**train_batch_preparer_cfg)

    # Multiple validation batch preparers for different DNA masking rates
    val_mask_configs = build_validation_mask_configs()
    val_batch_preparers = {}

    for mask_name, mask_config in val_mask_configs.items():
        main_process_print(f"Creating validation batch preparer for {mask_name}")
        val_cfg = deepcopy(config_dict.train_batch_preparer).to_dict()
        val_cfg["mask_config"] = mask_config
        val_batch_preparers[mask_name] = BatchPreparer(**val_cfg)

    """ Infer number of tracks T from a prepared training batch """
    first_batch = next(iter(train_dataloader))
    prepared_example = train_batch_preparer(first_batch, device="cpu")
    T_tracks = prepared_example["exp_tracks"].shape[-1]
    main_process_print(f"Inferred T={T_tracks} track channels from training batch")

    """ Init DNA-free DAE (trainable) """
    dae_cfg = config_dict.dae
    dae = DNAFreeDAE(
        in_channels=T_tracks,
        base_channels=int(dae_cfg.base_channels),
        conv_latent_channels=int(dae_cfg.conv_latent_channels),
        encoder_kernel_size=int(dae_cfg.encoder_kernel_size),
        stem_kernel_size=int(dae_cfg.stem_kernel_size),
        bottleneck_channels=int(dae_cfg.bottleneck_channels),
        bottleneck_swiglu_use_layernorm=bool(dae_cfg.bottleneck_swiglu_use_layernorm),
        model_dim=int(dae_cfg.model_dim) if getattr(dae_cfg, "model_dim", None) is not None else None,
        transformer_depth=int(dae_cfg.transformer_depth),
        transformer_heads=int(dae_cfg.transformer_heads),
        transformer_dim_head=(
            int(dae_cfg.transformer_dim_head) if dae_cfg.transformer_dim_head is not None else None
        ),
        transformer_kv_heads=(
            int(dae_cfg.transformer_kv_heads) if dae_cfg.transformer_kv_heads is not None else None
        ),
        transformer_attn_dropout=float(dae_cfg.transformer_attn_dropout),
        transformer_proj_dropout=float(dae_cfg.transformer_proj_dropout),
        transformer_ff_mult=float(dae_cfg.transformer_ff_mult),
        transformer_ff_dropout=float(dae_cfg.transformer_ff_dropout),
        transformer_rope_fraction=float(dae_cfg.transformer_rope_fraction),
        transformer_rope_theta=float(dae_cfg.transformer_rope_theta),
        bottleneck_decoder_swiglu_use_layernorm=bool(dae_cfg.bottleneck_decoder_swiglu_use_layernorm),
        decoder_kernel_size=int(dae_cfg.decoder_kernel_size),
    )
    main_process_print(f"DNA-free DAE initialized with {sum(p.numel() for p in dae.parameters()):,} parameters")

    """ Init fidelity losses """
    fid_cfg = config_dict.fidelity_loss
    fidelity_losses = FidelityLossBundle(
        lambdas=dict(fid_cfg.lambdas),
        count_coeffs_configs=dict(fid_cfg.count_coeffs),
        eps=float(fid_cfg.eps),
        huber_delta=float(fid_cfg.huber_delta),
    )

    """ Init critic losses (GT head for DAE) """
    crit_cfg = config_dict.critic_loss
    critic_losses = CriticLossBundle(
        lambdas=dict(crit_cfg.lambdas),
        count_coeffs=dict(crit_cfg.count_coeffs),
        temp=float(crit_cfg.temp),
        ref_ce_delta=float(crit_cfg.ref_ce_delta),
        huber_delta=float(crit_cfg.huber_delta),
        head_prefix=str(getattr(crit_cfg, "head_prefix", "gt_head")),
        eps=float(crit_cfg.eps),
    )

    """ Init gradient-ratio controller """
    grad_cfg = config_dict.grad_ratio
    grad_ratio_controller = GradientRatioController(
        gamma=float(grad_cfg.gamma),
        beta=float(grad_cfg.beta),
        target_ratio=float(grad_cfg.target_ratio),
        lambda_min=float(grad_cfg.lambda_min),
        lambda_max=float(grad_cfg.lambda_max),
        epsilon=float(grad_cfg.epsilon),
        lambda_init=float(grad_cfg.lambda_init),
    )

    """ Init critic adapter (GT head) """
    critic_adapter = CriticAdapter(
        critic=critic,
        head_name="GT",
    )

    """ Init DAE + critic orchestrator """
    orchestrator = TrainingOrchestrator(
        dae=dae,
        critic_adapter=critic_adapter,
        critic_losses=critic_losses,
        fidelity_losses=fidelity_losses,
        mode_probs=tuple(config_dict.orchestrator.mode_probs),
        grad_ratio_controller=grad_ratio_controller,
        eps=float(config_dict.orchestrator.eps),
    )

    """ Init Lightning model """
    lightning_model = LightningWrapper(
        config_dict=config_dict,
        orchestrator=orchestrator,
        train_batch_preparer=train_batch_preparer,
        val_batch_preparers=val_batch_preparers,
    )

    """ Train model """
    logger = CSVLogger(save_dir=config_dict.task.experiment_dir)
    os.makedirs(logger.log_dir, exist_ok=True)
    config_save_path = osp.join(logger.log_dir, "config_dict.json")
    config_json = json.loads(config_dict.to_json_best_effort())
    with open(config_save_path, "w") as f:
        json.dump(config_json, f, indent=4)

    # Checkpoints: per validation mask bin for total loss and fidelity
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

    # Optional EarlyStopping callback, controlled by es_patience.
    es_patience = getattr(config_dict.lightning, "es_patience", None)
    if isinstance(es_patience, int) and es_patience > 0:
        # Use the mid-range bin (40-60%) fidelity loss as the canonical monitor if available.
        if "40to60" in val_mask_configs:
            es_monitor_metric = "val_40to60/fid/total"
        else:
            first_mask = next(iter(val_mask_configs.keys()))
            es_monitor_metric = f"val_{first_mask}/fid/total"
        callbacks.append(
            EarlyStopping(
                monitor=es_monitor_metric,
                mode="min",
                patience=int(es_patience),
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
        log_every_n_steps=int(getattr(config_dict.lightning, "log_every_n_steps", 10)),
    )

    trainer.fit(
        model=lightning_model,
        train_dataloaders=train_dataloader,
        val_dataloaders=valid_dataloader,
    )

    return trainer


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=None, help="GPU id to use (single device).")
    args = parser.parse_args()

    # Load count-coeff lower bound from metadata (nonpeaks p75).
    metadata_path = Path(__file__).resolve().parent.parent / "metadata" / "count_coeff_bounds__training.json"
    with open(metadata_path, "r") as f:
        count_meta = json.load(f)
    count_coeff_cmin = float(count_meta["counts"]["nonpeaks"]["p75"])

    # Create configuration
    config_dict = ConfigDict()

    """ Task config and params """
    config_dict.task = ConfigDict()
    config_dict.task.seed = 42
    config_dict.task.fold_name = "fold_0"
    config_dict.task.assay_type = "ATAC"
    config_dict.task.assay_experiment_identifier = "GM12878__ENCSR637XSC"
    config_dict.task.task_type = "final_dna_free_dae_training"

    """ Model / checkpoint config """
    config_dict.model = ConfigDict()
    config_dict.model.name = "FinalDNAFreeDAETraining"

    """ DNA-free DAE config (DNABlindDAE hyperparams) """
    config_dict.dae = ConfigDict()
    config_dict.dae.base_channels = 16
    config_dict.dae.conv_latent_channels = 64
    config_dict.dae.encoder_kernel_size = 5
    config_dict.dae.stem_kernel_size = 23
    config_dict.dae.bottleneck_channels = 32
    config_dict.dae.bottleneck_swiglu_use_layernorm = False
    config_dict.dae.model_dim = config_dict.dae.conv_latent_channels
    config_dict.dae.transformer_depth = 1
    config_dict.dae.transformer_heads = 4
    config_dict.dae.transformer_dim_head = None
    config_dict.dae.transformer_kv_heads = None
    config_dict.dae.transformer_attn_dropout = 0.1
    config_dict.dae.transformer_proj_dropout = 0.1
    config_dict.dae.transformer_ff_mult = 2.0
    config_dict.dae.transformer_ff_dropout = 0.1
    config_dict.dae.transformer_rope_fraction = 0.5
    config_dict.dae.transformer_rope_theta = 20000.0
    config_dict.dae.bottleneck_decoder_swiglu_use_layernorm = False
    config_dict.dae.decoder_kernel_size = 5

    def _format_coeff_value(value: float | None) -> str:
        if value is None:
            return ""
        if float(value).is_integer():
            return str(int(value))
        return str(value).replace(".", "p")

    def _build_count_coeff_suffix(count_coeff_config: dict) -> str:
        coeff_type = str(count_coeff_config.get("type", "none"))
        parts = [coeff_type]
        if coeff_type in {"pow", "pow_softcap"}:
            alpha = count_coeff_config.get("alpha", None)
            if alpha is not None:
                parts.append(f"alpha_{_format_coeff_value(float(alpha))}")
        lower, upper = count_coeff_config.get("bounds", (None, None))
        if lower is not None:
            parts.append(f"lb_nonpeak_{_format_coeff_value(float(lower))}")
        if upper is not None:
            parts.append(f"ub_peak_{_format_coeff_value(float(upper))}")
        return "__".join(parts)

    # Shared count scaling config for all losses.
    count_coeff_config = {
        "type": "sqrt",
        "bounds": (count_coeff_cmin, None),
    }

    """ Fidelity loss config """
    config_dict.fidelity_loss = ConfigDict()
    config_dict.fidelity_loss.lambdas = {
        "ce": 1.0,
        "manifold_align": 1.0,
    }
    config_dict.fidelity_loss.count_coeffs = {
        "ce": dict(count_coeff_config),
        "manifold_align": dict(count_coeff_config),
    }
    config_dict.fidelity_loss.eps = 1e-8
    config_dict.fidelity_loss.huber_delta = 1.0

    """ Critic loss config (GT head) """
    config_dict.critic_loss = ConfigDict()
    config_dict.critic_loss.lambdas = {
        "lm_head": 1.0,
        "track_embed_align": 0.0,
    }
    config_dict.critic_loss.count_coeffs = {
        "lm_head": dict(count_coeff_config),
        "track_embed_align": dict(count_coeff_config),
    }
    config_dict.critic_loss.temp = 1.0
    config_dict.critic_loss.ref_ce_delta = 1.0
    config_dict.critic_loss.huber_delta = 1.0
    config_dict.critic_loss.head_prefix = "gt_head"
    config_dict.critic_loss.eps = 1e-8

    """ Gradient-ratio controller config """
    config_dict.grad_ratio = ConfigDict()
    config_dict.grad_ratio.gamma = 1.0
    config_dict.grad_ratio.beta = 0.9
    config_dict.grad_ratio.target_ratio = 1.0
    config_dict.grad_ratio.lambda_min = 0.1
    config_dict.grad_ratio.lambda_max = 2000.0
    config_dict.grad_ratio.epsilon = 1e-8
    config_dict.grad_ratio.lambda_init = 1.0

    """ Orchestrator config """
    config_dict.orchestrator = ConfigDict()
    config_dict.orchestrator.eps = 1e-8
    config_dict.orchestrator.mode_probs = (0.1, 0.6, 0.3)

    """ Batch preparer config and params """

    BUCKET_INPUT_LEN = 1000

    # Training batch preparer: nested masks (M_co ⊆ M_ci).
    train_rco_min = 0.10
    train_rco_max = 0.60
    train_rci_max = 0.80
    train_mask_config = build_nested_beta_headroom_mask_config(
        rco_min=train_rco_min,
        rco_max=train_rco_max,
        rci_max=train_rci_max,
    )

    config_dict.train_batch_preparer = ConfigDict()
    config_dict.train_batch_preparer.mask_config = train_mask_config

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
    config_dict.data.seq_width = BUCKET_INPUT_LEN
    config_dict.data.track_width = BUCKET_INPUT_LEN
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
    config_dict.lightning.batch_size = int(128*4) * num_devices
    config_dict.lightning.num_workers = 8 // num_devices
    config_dict.lightning.pin_memory = True
    config_dict.lightning.persistent_workers = True
    # early stopping: None disables ES; positive int enables ES on val fidelity loss
    config_dict.lightning.es_patience = None
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
    # Log every N steps for batch-level logging
    config_dict.lightning.log_every_n_steps = 25

    # Loss configuration identifier for directory name
    fid_parts = []
    if config_dict.fidelity_loss.lambdas.get("ce", 0.0) > 0:
        fid_parts.append("ce")
    if config_dict.fidelity_loss.lambdas.get("manifold_align", 0.0) > 0:
        fid_parts.append("align")
    fidelity_dir = "fidelity__" + "+".join(fid_parts) if fid_parts else "no_fidelity"
    fidelity_dir += (
        "__critic_lm"
        f"{_format_coeff_value(config_dict.critic_loss.lambdas.get('lm_head', 0.0))}"
        "__track"
        f"{_format_coeff_value(config_dict.critic_loss.lambdas.get('track_embed_align', 0.0))}"
    )

    # Experiment directory
    config_dict.task.model_name = config_dict.model.name
    coeff_suffix = _build_count_coeff_suffix(count_coeff_config)
    config_dict.task.experiment_dir = osp.join(
        MANUSCRIPT_SECTION_3_V2_DIR,
        "workspace",
        "runs",
        "6_1__final_debiaser",
        config_dict.task.assay_type,
        config_dict.task.assay_experiment_identifier,
        "dna_free_dae",
        "bpnet__predicted",
        (
            f"{BUCKET_INPUT_LEN}bp__mask_rco_{int(train_rco_min*100)}to{int(train_rco_max*100)}pct"
            f"__rci_max_{int(train_rci_max*100)}pct"
        ),
        f"latentC{config_dict.dae.conv_latent_channels}__{coeff_suffix}",
        fidelity_dir,
        f"seed_{config_dict.task.seed}",
    )
    os.makedirs(config_dict.task.experiment_dir, exist_ok=True)

    # Run main function
    main(config_dict)
