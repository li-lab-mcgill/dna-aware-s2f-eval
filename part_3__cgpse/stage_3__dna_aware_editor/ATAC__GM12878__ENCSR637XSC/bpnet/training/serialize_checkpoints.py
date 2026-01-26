#!/usr/bin/env python3
"""
Serialize Lightning checkpoints for the iteration-2 DNA-aware editor (v2).

Why this exists
--------------
Lightning `.ckpt` files contain full training state. For downstream use we want a
portable artifact with the trained editor attached to a frozen DAE.

Outputs (per input .ckpt)
------------------------
- `<stem>.editor.pt` (the “load out of the box” artifact)
    The full `DNAAwareEditorAugmentedDAE` object via `torch.save(model, ...)`.
    This satisfies: `model = torch.load(path)` (no manual init), BUT it still
    requires that the class is importable at load time.
    Practical requirement: ensure `.../section_3_v2__Debiasing` is on `sys.path`
    so `import utils.final_debiaser_v2` works.
- `<stem>.editor_state_dict.pt` (portable fallback)
    A plain `torch.save({...})` dict containing:
      - `reencoder_kwargs` + `editor_kwargs`
      - `dae_checkpoint_path` (recorded for provenance)
      - `reencoder_state_dict` + `editor_state_dict`

Usage
-----
    python3 serialize_checkpoints.py
    python3 serialize_checkpoints.py --checkpoints-dir /path/to/lightning_logs/version_X/checkpoints
    python3 serialize_checkpoints.py --checkpoints-dir /path/one --checkpoints-dir /path/two
    python3 serialize_checkpoints.py --out-dir /path/to/serialized
    python3 serialize_checkpoints.py --dae-checkpoint /path/to/frozen_dae.pt
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
from utils.final_debiaser_v2.model.dna_aware_editor import DNAAwareEditorAugmentedDAE


DEFAULT_CHECKPOINT_DIRS = [
    (
        "/home/mcb/users/dcakma3/multi_res_bench_v2/multimodal_masked_contrastive_main/manuscript_experiments/section_3_v2__Debiasing/workspace/runs/7_1__final_dna_aware_editor/ATAC/GM12878__ENCSR637XSC/dna_aware_editor/bpnet__predicted/1000bp__mask_rco_10to60pct__rci_max_80pct/latentC64__sqrt__lb_nonpeak_133/editorH64/fidelity__ce+align__critic_lm1__track0/seed_42/lightning_logs/version_0/staged_checkpoints"
    ),
]

DAE_MODEL_PATH = (
    "/home/mcb/users/dcakma3/multi_res_bench_v2/multimodal_masked_contrastive_main/"
    "manuscript_experiments/section_3_v2__Debiasing/workspace/runs/6_1__final_debiaser/"
    "ATAC/GM12878__ENCSR637XSC/dna_free_dae/bpnet__predicted/"
    "1000bp__mask_rco_10to60pct__rci_max_80pct/latentC64__sqrt__lb_nonpeak_133/"
    "fidelity__ce+align__critic_lm1__track0/seed_42/lightning_logs/version_0/"
    "staged_checkpoints/best_val_100_total_loss-epoch=137-step=57546.dae.pt"
)


def _load_run_config(ckpt_dir: Path, config_path: Path | None) -> Dict[str, Any]:
    if config_path is None:
        config_path = ckpt_dir.parent / "config_dict.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config_dict.json: {config_path}")
    return json.loads(config_path.read_text())


def _extract_editor_state_dict(ckpt: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    state_dict = ckpt.get("state_dict", ckpt)
    prefixes = (
        "orchestrator.dae.",
        "model.orchestrator.dae.",
        "dae.",
        "model.dae.",
    )
    editor_state: Dict[str, torch.Tensor] = {}
    for prefix in prefixes:
        editor_state = {k[len(prefix) :]: v for k, v in state_dict.items() if k.startswith(prefix)}
        if editor_state:
            break
    if not editor_state:
        raise KeyError(
            f"No editor weights found under any of: {prefixes}. "
            f"Example keys: {list(state_dict.keys())[:10]}"
        )
    return editor_state


def _split_editor_state_dict(
    editor_state: Dict[str, torch.Tensor],
) -> tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    dae_state = {
        k[len("dae.") :]: v for k, v in editor_state.items() if k.startswith("dae.")
    }
    reencoder_state = {
        k[len("reencoder.") :]: v for k, v in editor_state.items() if k.startswith("reencoder.")
    }
    editor_state = {
        k[len("editor.") :]: v for k, v in editor_state.items() if k.startswith("editor.")
    }
    if not reencoder_state:
        raise KeyError("No reencoder weights found under editor state_dict.")
    if not editor_state:
        raise KeyError("No editor weights found under editor state_dict.")
    return dae_state, reencoder_state, editor_state


def _infer_track_in_channels(reencoder_state: Dict[str, torch.Tensor]) -> int:
    for k, v in reencoder_state.items():
        if k.endswith("track_encoder.stem.0.weight") and getattr(v, "ndim", None) == 3:
            return int(v.shape[1])
    for k, v in reencoder_state.items():
        if k.endswith("stem.0.weight") and getattr(v, "ndim", None) == 3:
            return int(v.shape[1])
    for _, v in reencoder_state.items():
        if getattr(v, "ndim", None) == 3:
            return int(v.shape[1])
    raise RuntimeError("Could not infer track_in_channels (T) from reencoder state_dict.")


def _build_reencoder_kwargs(cfg: Dict[str, Any], *, track_in_channels: int) -> Dict[str, Any]:
    reencoder = cfg.get("reencoder", {})
    return dict(
        track_in_channels=int(track_in_channels),
        track_base_channels=int(reencoder["track_base_channels"]),
        track_conv_latent_channels=int(reencoder["track_conv_latent_channels"]),
        track_encoder_kernel_size=int(reencoder["track_encoder_kernel_size"]),
        track_stem_kernel_size=int(reencoder["track_stem_kernel_size"]),
        track_down_kernel_size=int(reencoder["track_down_kernel_size"]),
        dna_num_modalities=int(reencoder.get("dna_num_modalities", 4)),
        dna_base_channels=int(reencoder["dna_base_channels"]),
        dna_latent_channels=int(reencoder["dna_latent_channels"]),
        dna_stem_kernel_size=int(reencoder["dna_stem_kernel_size"]),
        dna_down_kernel_size=int(reencoder["dna_down_kernel_size"]),
        dna_padding_mode=str(reencoder.get("dna_padding_mode", "same")),
        dna_mask_aware_padding=bool(reencoder.get("dna_mask_aware_padding", True)),
        bottleneck_channels=int(reencoder["bottleneck_channels"]),
        bottleneck_depth=int(reencoder.get("bottleneck_depth", 1)),
        bottleneck_heads=int(reencoder.get("bottleneck_heads", 4)),
        bottleneck_dim_head=(
            int(reencoder["bottleneck_dim_head"])
            if reencoder.get("bottleneck_dim_head", None) is not None
            else None
        ),
        bottleneck_kv_heads=(
            int(reencoder["bottleneck_kv_heads"])
            if reencoder.get("bottleneck_kv_heads", None) is not None
            else None
        ),
        bottleneck_attn_dropout=float(reencoder.get("bottleneck_attn_dropout", 0.0)),
        bottleneck_proj_dropout=float(reencoder.get("bottleneck_proj_dropout", 0.0)),
        bottleneck_ff_mult=float(reencoder.get("bottleneck_ff_mult", 4.0)),
        bottleneck_ff_dropout=float(reencoder.get("bottleneck_ff_dropout", 0.0)),
        bottleneck_rope_fraction=float(reencoder.get("bottleneck_rope_fraction", 0.25)),
        bottleneck_rope_theta=float(reencoder.get("bottleneck_rope_theta", 10000.0)),
        bottleneck_swiglu_use_layernorm=bool(
            reencoder.get("bottleneck_swiglu_use_layernorm", False)
        ),
    )


def _build_editor_kwargs(cfg: Dict[str, Any]) -> Dict[str, Any]:
    editor = cfg.get("editor", {})
    return dict(
        bottleneck_channels=int(editor["bottleneck_channels"]),
        context_channels=int(editor["context_channels"]),
        hidden_channels=int(editor["hidden_channels"]),
        use_layernorm=bool(editor.get("use_layernorm", False)),
    )


def _load_dae_from_checkpoint(dae_ckpt_path: Path) -> torch.nn.Module:
    try:
        dae_obj = torch.load(dae_ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        dae_obj = torch.load(dae_ckpt_path, map_location="cpu")

    if isinstance(dae_obj, dict):
        if "state_dict" not in dae_obj or "model_kwargs" not in dae_obj:
            raise ValueError(
                f"DAE checkpoint did not contain state_dict/model_kwargs: {dae_ckpt_path}"
            )
        dae = DNAFreeDAE(**dae_obj["model_kwargs"])
        dae.load_state_dict(dae_obj["state_dict"], strict=True)
    else:
        dae = dae_obj

    dae.eval()
    for p in dae.parameters():
        p.requires_grad = False
    return dae


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


def _compare_dae_state(
    dae: torch.nn.Module,
    dae_state: Dict[str, torch.Tensor],
    label: str,
) -> None:
    if not dae_state:
        print(f"[{label}] dae_check: no DAE weights found in editor checkpoint")
        return
    ref_state = dae.state_dict()
    common_keys = [k for k in dae_state.keys() if k in ref_state]
    if not common_keys:
        print(f"[{label}] dae_check: no overlapping DAE keys for comparison")
        return
    max_diff = 0.0
    num_changed = 0
    for name in common_keys:
        diff = (dae_state[name] - ref_state[name]).abs().max().item()
        if diff > 0:
            num_changed += 1
            max_diff = max(max_diff, diff)
    print(
        f"[{label}] dae_check: changed {num_changed}/{len(common_keys)} tensors, "
        f"max abs diff = {max_diff:.3e}"
    )


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
    dae_checkpoint_path: Path,
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

    dae = _load_dae_from_checkpoint(dae_checkpoint_path)

    for ckpt_path in ckpts:
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        except TypeError:
            ckpt = torch.load(ckpt_path, map_location="cpu")
        editor_state = _extract_editor_state_dict(ckpt)
        dae_state, reencoder_state, editor_state = _split_editor_state_dict(editor_state)
        _compare_dae_state(dae, dae_state, ckpt_path.name)
        T_tracks = _infer_track_in_channels(reencoder_state)
        reencoder_kwargs = _build_reencoder_kwargs(cfg, track_in_channels=T_tracks)
        editor_kwargs = _build_editor_kwargs(cfg)

        # Rebuild model on CPU and verify strict load for trainable submodules.
        model = DNAAwareEditorAugmentedDAE(
            dae=dae,
            editor_kwargs=editor_kwargs,
            reencoder_kwargs=reencoder_kwargs,
        )
        _report_load_delta(model.reencoder, reencoder_state, f"{ckpt_path.name} reencoder")
        _report_load_delta(model.editor, editor_state, f"{ckpt_path.name} editor")

        payload = {
            "created_at_unix": time.time(),
            "source_ckpt": ckpt_path.name,
            "epoch": ckpt.get("epoch", None),
            "global_step": ckpt.get("global_step", None),
            "model_class": "DNAAwareEditorAugmentedDAE",
            "reencoder_kwargs": dict(reencoder_kwargs),
            "editor_kwargs": dict(editor_kwargs),
            "dae_checkpoint_path": str(dae_checkpoint_path),
            "reencoder_state_dict": reencoder_state,
            "editor_state_dict": editor_state,
        }

        out_sd = out_dir / f"{ckpt_path.stem}.editor_state_dict.pt"
        if out_sd.exists() and skip_existing:
            print(f"  - skip existing {out_sd.name}")
        else:
            torch.save(payload, out_sd)
            print(f"  ✓ wrote {out_sd.name}")

        if not no_save_full_model:
            out_model = out_dir / f"{ckpt_path.stem}.editor.pt"
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
    parser.add_argument(
        "--dae-checkpoint",
        type=str,
        default=None,
        help="Path to the frozen DNA-free DAE checkpoint (.dae.pt or state_dict payload).",
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

    dae_checkpoint_path = Path(args.dae_checkpoint) if args.dae_checkpoint else Path(DAE_MODEL_PATH)
    if not dae_checkpoint_path.exists():
        raise FileNotFoundError(f"Missing DAE checkpoint: {dae_checkpoint_path}")

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
            dae_checkpoint_path=dae_checkpoint_path,
        )


if __name__ == "__main__":
    main()
