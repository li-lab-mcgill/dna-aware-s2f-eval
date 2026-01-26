import json
import time
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pathlib import Path
from ml_collections import ConfigDict

# Extend sys.path
import sys
import os
import os.path as osp
sys.path.append(os.path.join(
    os.path.dirname(__file__),
    "../../../../../../../.."
))
from config import *  # noqa

# -----------------------------------------------------------------------------
# IMPORTANT: Critic pickle portability
# -----------------------------------------------------------------------------
# The serialized `.critic.pt` object is a pickled Python module instance, and the
# pickle stores the class import path as `utils.critic.*`.
#
# To ensure the resulting `.critic.pt` loads in the v2 environment without relying
# on legacy `section_3__Debiasing`, we must serialize against the v2 critic code:
#   `MANUSCRIPT_SECTION_3_V2_DIR/utils/critic/...`
#
# Therefore:
#   - Put `MANUSCRIPT_SECTION_3_V2_DIR` (the *parent* of `utils/`) on sys.path.
#   - Do NOT add legacy `section_3__Debiasing` to sys.path in this serializer.
# -----------------------------------------------------------------------------
if MANUSCRIPT_SECTION_3_V2_DIR not in sys.path:
    sys.path.insert(0, MANUSCRIPT_SECTION_3_V2_DIR)

# Critic architecture
from utils.critic.architecture.splitstem_singlechannel_multihead_with_disc_critic import (
    SplitStemSingleHeadRouterCritic,
)


# Helper functions for critic configs (mirrors critic pretraining + noise-model usage)
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
    cfg = ConfigDict()
    cfg.in_channels = int(in_channels)
    cfg.out_channels = int(out_channels)
    cfg.expansion_factor = int(expansion_factor)
    cfg.dropout_prelogits = float(dropout_prelogits)
    return cfg


def main(config_dict):
    """Rebuild and serialize SplitStemSingleHeadRouterCritic checkpoints."""

    # Sanity check: confirm we are serializing against the v2 `utils.critic` module.
    import utils.critic as _critic_pkg
    print(f"[serialize_checkpoints] utils.critic resolves to: {_critic_pkg.__file__}")

    # Save CPU-safe PyTorch critic checkpoints for downstream use
    import glob
    for ckpt_path in glob.glob(os.path.join(config_dict.checkpoints_dir, "*.ckpt")):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state_dict = ckpt.get("state_dict", ckpt)
        critic_state = {
            k[len("critic."):] : v
            for k, v in state_dict.items()
            if k.startswith("critic.")
        }

        # Rebuild critic instance (mirrors run_train_10to100pct_mask + noise-model loader)
        backbone_cfg = make_cglm_backbone_default(
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
                'rotary_emb_base': 20000.0,
            },
        )
        C_backbone = int(backbone_cfg.input_conv_channels)
        mlm_head_cfg = make_cglm_mlm_proj_default(
            in_channels=C_backbone,
            out_channels=4,
            expansion_factor=2,
            dropout_prelogits=0.0,
        )
        disc_head_cfg = make_cglm_mlm_proj_default(
            in_channels=C_backbone,
            out_channels=2,
            expansion_factor=2,
            dropout_prelogits=0.0,
        )
        head_names = ["GT", "S2F", "DISC"]
        per_head_kwargs = {
            "GT": dict(mlm_head_cfg),
            "S2F": dict(mlm_head_cfg),
            "DISC": dict(disc_head_cfg),
        }
        critic = SplitStemSingleHeadRouterCritic(
            backbone_kwargs=backbone_cfg.to_dict(),
            head_names=head_names,
            per_head_kwargs=per_head_kwargs,
        )

        # Keep a copy of the randomly initialized parameters for comparison.
        initial_state = {
            k: v.detach().clone()
            for k, v in critic.state_dict().items()
        }

        critic.load_state_dict(critic_state)

        # Report how much the checkpoint load changed each tensor.
        max_diff = 0.0
        num_changed = 0
        for name, init_tensor in initial_state.items():
            loaded_tensor = critic.state_dict()[name]
            diff = (loaded_tensor - init_tensor).abs().max().item()
            if diff > 0:
                num_changed += 1
                if diff > max_diff:
                    max_diff = diff
        print(
            f"[{os.path.basename(ckpt_path)}] "
            f"changed {num_changed}/{len(initial_state)} tensors, "
            f"max |Δ| = {max_diff:.3e}"
        )

        out_path = ckpt_path.replace(".ckpt", ".critic.pt")
        torch.save(critic, out_path)



if __name__ == "__main__":

    # Create configuration
    config_dict = ConfigDict()

    # v2 critic-pretraining checkpoints for bpnet__predicted
    config_dict.checkpoints_dir = (
        "/home/mcb/users/dcakma3/multi_res_bench_v2/multimodal_masked_contrastive_main/"
        "manuscript_experiments/section_3_v2__Debiasing/workspace/runs/1__critic/"
        "ATAC/GM12878__ENCSR637XSC/splitstem_singlechannel_multihead_with_disc_critic/"
        "bpnet__predicted/1000bp__mask_10to100pct/seed_42/lightning_logs/version_0/"
        "checkpoints"
    )


    # For critic serialization we only need the checkpoints directory.

    # Run main function
    main(config_dict)
