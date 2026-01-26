#!/usr/bin/env python3
"""
Serialize Lightning checkpoints for the iteration-2 DNA-free DAE.

Why this exists
--------------
Lightning `.ckpt` files contain a full training state; for downstream use we often want
just the DAE weights in a stable/portable format.

Outputs (per input .ckpt)
------------------------
- `<stem>.dae.pt` (the “load out of the box” artifact)
    The full `DNAFreeDAE` object via `torch.save(model, ...)`.
    This satisfies: `model = torch.load(path)` (no manual init), BUT it still
    requires that the class is importable at load time.
    Practical requirement: ensure `.../section_3_v2__Debiasing` is on `sys.path`
    so `import utils.final_debiaser_v2` works.
- `<stem>.dae_state_dict.pt` (portable fallback)
    A plain `torch.save({...})` dict containing `model_kwargs` + `state_dict`.
    This is resilient to refactors, but requires reconstructing the model class.

Usage
-----
    python3 serialize_checkpoints.py
    python3 serialize_checkpoints.py --checkpoints-dir /path/to/lightning_logs/version_X/checkpoints
    python3 serialize_checkpoints.py --checkpoints-dir /path/one --checkpoints-dir /path/two
    python3 serialize_checkpoints.py --out-dir /path/to/serialized
    python3 serialize_checkpoints.py --no-save-full-model
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict

# Keep Lightning/torch temp files off /tmp (often small on shared systems).
os.environ.setdefault("TMPDIR", "/home/mcb/users/dcakma3/mytmp")
os.makedirs(os.environ["TMPDIR"], exist_ok=True)

# Extend sys.path early (before importing torch).
sys.path.append(
    os.path.join(
        os.path.dirname(__file__),
        "../../../../..",
    )
)
from config import *  # noqa

# Ensure v2 Debiasing root is importable so pickled models serialize with
# module paths like `utils.final_debiaser_v2.*` (portable across v2 scripts).
if MANUSCRIPT_SECTION_3_V2_DIR not in sys.path:
    sys.path.insert(0, MANUSCRIPT_SECTION_3_V2_DIR)

import torch

from utils.final_debiaser_v2.model import DNAFreeDAE


DEFAULT_CHECKPOINT_DIRS = [
    (
        "/home/mcb/users/dcakma3/multi_res_bench_v2/multimodal_masked_contrastive_main/manuscript_experiments/section_3_v2__Debiasing/workspace/runs/6_1__final_debiaser/ATAC/GM12878__ENCSR637XSC/dna_free_dae/bpnet__predicted/1000bp__mask_rco_10to60pct__rci_max_80pct/latentC64__sqrt__lb_nonpeak_133/fidelity__ce+align__critic_lm1__track0/seed_42/lightning_logs/version_0/staged_checkpoints"
    ),
]


def _load_run_config(ckpt_dir: Path, config_path: Path | None) -> Dict[str, Any]:
    if config_path is None:
        config_path = ckpt_dir.parent / "config_dict.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config_dict.json: {config_path}")
    return json.loads(config_path.read_text())


def _extract_dae_state_dict(ckpt: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    state_dict = ckpt.get("state_dict", ckpt)
    prefixes = (
        "orchestrator.dae.",
        "model.orchestrator.dae.",
        "dae.",
        "model.dae.",
    )
    dae_state: Dict[str, torch.Tensor] = {}
    for prefix in prefixes:
        dae_state = {k[len(prefix) :]: v for k, v in state_dict.items() if k.startswith(prefix)}
        if dae_state:
            break
    if not dae_state:
        raise KeyError(
            f"No DAE weights found under any of: {prefixes}. "
            f"Example keys: {list(state_dict.keys())[:10]}"
        )
    return dae_state


def _infer_track_in_channels(dae_state: Dict[str, torch.Tensor]) -> int:
    for k, v in dae_state.items():
        if "encoder_router.encoders" in k and k.endswith("stem.0.weight"):
            if getattr(v, "ndim", None) == 3:
                return int(v.shape[1])
    for k, v in dae_state.items():
        if "encoder_router.track_encoders" in k and k.endswith("stem.0.weight"):
            if getattr(v, "ndim", None) == 3:
                return int(v.shape[1])
    for k, v in dae_state.items():
        if "track_encoders" in k and k.endswith("stem.0.weight"):
            if getattr(v, "ndim", None) == 3:
                return int(v.shape[1])
    for k, v in dae_state.items():
        if k.endswith("stem.0.weight") and getattr(v, "ndim", None) == 3:
            return int(v.shape[1])
    for _, v in dae_state.items():
        if getattr(v, "ndim", None) == 3:
            return int(v.shape[1])
    raise RuntimeError("Could not infer track_in_channels (T) from DAE state_dict.")


def _build_model_kwargs(cfg: Dict[str, Any], *, track_in_channels: int) -> Dict[str, Any]:
    dae = cfg.get("dae", {})
    if "base_channels" in dae or "conv_latent_channels" in dae:
        return dict(
            in_channels=int(track_in_channels),
            base_channels=int(dae["base_channels"]),
            conv_latent_channels=int(dae["conv_latent_channels"])
            if dae.get("conv_latent_channels", None) is not None
            else None,
            encoder_kernel_size=int(dae.get("encoder_kernel_size", 5)),
            stem_kernel_size=int(dae.get("stem_kernel_size", 23)),
            bottleneck_channels=int(dae["bottleneck_channels"]),
            bottleneck_swiglu_use_layernorm=bool(dae.get("bottleneck_swiglu_use_layernorm", False)),
            model_dim=int(dae["model_dim"]) if dae.get("model_dim", None) is not None else None,
            transformer_depth=int(dae.get("transformer_depth", 1)),
            transformer_heads=int(dae.get("transformer_heads", 4)),
            transformer_dim_head=(
                int(dae.get("transformer_dim_head"))
                if dae.get("transformer_dim_head", None) is not None
                else None
            ),
            transformer_kv_heads=(
                int(dae.get("transformer_kv_heads"))
                if dae.get("transformer_kv_heads", None) is not None
                else None
            ),
            transformer_attn_dropout=float(dae.get("transformer_attn_dropout", 0.0)),
            transformer_proj_dropout=float(dae.get("transformer_proj_dropout", 0.0)),
            transformer_ff_mult=float(dae.get("transformer_ff_mult", 4.0)),
            transformer_ff_dropout=float(dae.get("transformer_ff_dropout", 0.0)),
            transformer_rope_fraction=float(dae.get("transformer_rope_fraction", 0.0)),
            transformer_rope_theta=float(dae.get("transformer_rope_theta", 10000.0)),
            bottleneck_decoder_swiglu_use_layernorm=bool(
                dae.get("bottleneck_decoder_swiglu_use_layernorm", False)
            ),
            decoder_kernel_size=int(dae.get("decoder_kernel_size", 5)),
        )
    return dict(
        in_channels=int(track_in_channels),
        base_channels=int(dae["track_base_channels"]),
        conv_latent_channels=int(dae["track_conv_latent_channels"])
        if dae.get("track_conv_latent_channels", None) is not None
        else None,
        encoder_kernel_size=int(dae.get("down_kernel_size", 5)),
        stem_kernel_size=int(dae.get("stem_kernel_size", 23)),
        bottleneck_channels=int(dae["bottleneck_channels"]),
        model_dim=int(dae["model_dim"]) if dae.get("model_dim", None) is not None else None,
        decoder_kernel_size=int(dae.get("decoder_kernel_size", 5)),
    )


def _report_load_delta(model: torch.nn.Module, loaded_state: Dict[str, torch.Tensor], label: str) -> None:
    initial = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(loaded_state, strict=True)
    max_diff = 0.0
    num_changed = 0
    for name, init_tensor in initial.items():
        loaded_tensor = model.state_dict()[name]
        diff = (loaded_tensor - init_tensor).abs().max().item()
        if diff > 0:
            num_changed += 1
            max_diff = max(max_diff, diff)
    print(f"[{label}] changed {num_changed}/{len(initial)} tensors, max |Δ| = {max_diff:.3e}")


def _resolve_out_dir(base_out_dir: Path | None, ckpt_dir: Path, multi_dir: bool) -> Path:
    if base_out_dir is None:
        return ckpt_dir
    if not multi_dir:
        return base_out_dir
    return base_out_dir / ckpt_dir.parent.name


def _serialize_dir(
    ckpt_dir: Path,
    *,
    config_path: Path | None,
    base_out_dir: Path | None,
    pattern: str,
    no_save_full_model: bool,
    skip_existing: bool,
    multi_dir: bool,
) -> None:
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Missing checkpoints dir: {ckpt_dir}")

    out_dir = _resolve_out_dir(base_out_dir, ckpt_dir, multi_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = _load_run_config(ckpt_dir, config_path)

    ckpts = sorted(ckpt_dir.glob(pattern))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints matched '{pattern}' under: {ckpt_dir}")

    print(f"[serialize_checkpoints] processing {ckpt_dir}")

    for ckpt_path in ckpts:
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        except TypeError:
            ckpt = torch.load(ckpt_path, map_location="cpu")
        dae_state = _extract_dae_state_dict(ckpt)
        T_tracks = _infer_track_in_channels(dae_state)
        model_kwargs = _build_model_kwargs(cfg, track_in_channels=T_tracks)

        # Rebuild model on CPU and verify strict load.
        model = DNAFreeDAE(**model_kwargs)
        _report_load_delta(model, dae_state, ckpt_path.name)

        payload = {
            "created_at_unix": time.time(),
            "source_ckpt": ckpt_path.name,
            "epoch": ckpt.get("epoch", None),
            "global_step": ckpt.get("global_step", None),
            "model_class": "DNAFreeDAE",
            "model_kwargs": dict(model_kwargs),
            "state_dict": dae_state,
        }

        out_sd = out_dir / f"{ckpt_path.stem}.dae_state_dict.pt"
        if out_sd.exists() and skip_existing:
            print(f"  - skip existing {out_sd.name}")
        else:
            torch.save(payload, out_sd)
            print(f"  ✓ wrote {out_sd.name}")

        if not no_save_full_model:
            out_model = out_dir / f"{ckpt_path.stem}.dae.pt"
            if out_model.exists() and skip_existing:
                print(f"  - skip existing {out_model.name}")
            else:
                torch.save(model, out_model)
                print(f"  ✓ wrote {out_model.name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoints-dir",
        action="append",
        default=None,
        help="Directory containing Lightning checkpoints (*.ckpt). Can be passed multiple times.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional path to config_dict.json (defaults to <checkpoints-dir>/../config_dict.json).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Optional output directory (defaults to each checkpoints dir).",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.ckpt",
        help="Glob pattern for selecting checkpoints within --checkpoints-dir.",
    )
    parser.add_argument("--no-save-full-model", action="store_true")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip writing an output file if it already exists.",
    )
    args = parser.parse_args()

    ckpt_dirs = args.checkpoints_dir or DEFAULT_CHECKPOINT_DIRS
    ckpt_dirs = [Path(path) for path in ckpt_dirs]

    base_out_dir = Path(args.out_dir) if args.out_dir else None
    config_path = Path(args.config) if args.config else None

    multi_dir = len(ckpt_dirs) > 1
    for ckpt_dir in ckpt_dirs:
        _serialize_dir(
            ckpt_dir,
            config_path=config_path,
            base_out_dir=base_out_dir,
            pattern=args.pattern,
            no_save_full_model=args.no_save_full_model,
            skip_existing=args.skip_existing,
            multi_dir=multi_dir,
        )


if __name__ == "__main__":
    main()
