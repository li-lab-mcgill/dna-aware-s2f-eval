from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from ..metrics.critic.adapter import CriticAdapter


def _ensure_mask_2d(mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if mask is None:
        return None
    if mask.dim() == 3 and mask.size(-1) == 1:
        return mask.squeeze(-1)
    if mask.dim() == 2:
        return mask
    raise ValueError(f"mask must be (B,L) or (B,L,1), got {tuple(mask.shape)}")


def extract_tracks_and_logits(
    *,
    dataloader: Iterable,
    debiaser: torch.nn.Module,
    editor_debiaser: torch.nn.Module | None = None,
    critic_gt_head_adapter: CriticAdapter,
    critic_s2f_head_adapter: CriticAdapter | None,
    device: torch.device | str,
    eps: float = 1e-8,
    verbose: bool = False,
) -> Dict[str, np.ndarray | list]:
    """
    Extract tracks and critic logits for GT/S2F heads.

    Records:
      - gt_counts, s2f_counts (B,T,L)
      - dae_logprobs (B,T,L) using log_softmax
      - editor_full_dna_logprobs (B,T,L) using log_softmax
      - editor_partial_dna_logprobs (B,T,L) using log_softmax
      - dna_mask (B,L) for critic evaluation
      - critic logits for GT head (teacher=GT, students=GT,S2F,DAE,editors)
      - critic logits for S2F head (teacher=S2F eq to GT, students=DAE,editors)
    """
    critic_gt_head_adapter.critic.eval()
    if critic_s2f_head_adapter is not None:
        critic_s2f_head_adapter.critic.eval()
    debiaser.eval()
    if editor_debiaser is not None:
        editor_debiaser.eval()

    gt_counts_list: List[torch.Tensor] = []
    s2f_counts_list: List[torch.Tensor] = []
    dna_mask_list: List[torch.Tensor] = []
    ref_onehot_list: List[torch.Tensor] = []
    region_types: List[str] = []
    identifiers: List[str] = []

    dae__logprobs_list: List[torch.Tensor] = []
    editor_full_dna__logprobs_list: List[torch.Tensor] = []
    editor_partial_dna__logprobs_list: List[torch.Tensor] = []

    gt_teacher__logits_list: List[torch.Tensor] = []
    gt_student__s2f_logits_list: List[torch.Tensor] = []
    gt_student__dae_logits_list: List[torch.Tensor] = []
    gt_student__gt_logits_list: List[torch.Tensor] = []
    gt_student__editor_full_dna_logits_list: List[torch.Tensor] = []
    gt_student__editor_partial_dna_logits_list: List[torch.Tensor] = []

    s2f_teacher_logits_list: List[torch.Tensor] = []
    s2f_student__dae_logits_list: List[torch.Tensor] = []
    s2f_student__editor_full_dna_logits_list: List[torch.Tensor] = []
    s2f_student__editor_partial_dna_logits_list: List[torch.Tensor] = []

    with torch.no_grad():
        batch_iter = (
            tqdm(dataloader, desc="Extracting tracks/logits")
            if verbose
            else dataloader
        )
        for raw_batch in batch_iter:
            dna, s2f_tracks, gt_tracks, mask, batch_region_types, batch_identifiers = raw_batch

            dna = dna.to(device=device, dtype=torch.float32)
            s2f_tracks = s2f_tracks.to(device=device, dtype=torch.float32)
            gt_tracks = gt_tracks.to(device=device, dtype=torch.float32)
            mask = mask.to(device=device, dtype=torch.float32)

            if s2f_tracks.size(-1) == 0 or gt_tracks.size(-1) == 0:
                raise ValueError("Both pred_tracks and exp_tracks must be present for evaluation.")

            mask_2d = _ensure_mask_2d(mask)
            if mask_2d is None:
                raise ValueError("mask is required for critic evaluation.")

            # DNA-free DAE debiasing (single forward pass)
            dae_debias_logits = debiaser(
                track=s2f_tracks, route="s2f"
            )[0]
            dae_logprobs = F.log_softmax(dae_debias_logits, dim=-1)

            # Convert DAE outputs to (B, L, T) probabilities for critic.
            dae_probs_blt = dae_logprobs.exp().permute(0, 2, 1).contiguous()

            editor_full_logprobs = None
            editor_partial_logprobs = None
            editor_full_probs_blt = None
            editor_partial_probs_blt = None
            if editor_debiaser is not None:
                mask_partial = ~mask_2d.to(torch.bool)
                editor_full_logits, _ = editor_debiaser(
                    dna=dna, track=s2f_tracks, mask="full_dna", route="s2f"
                )
                editor_partial_logits, _ = editor_debiaser(
                    dna=dna, track=s2f_tracks, mask=mask_partial, route="s2f"
                )
                editor_full_logprobs = F.log_softmax(editor_full_logits, dim=-1)
                editor_partial_logprobs = F.log_softmax(editor_partial_logits, dim=-1)
                editor_full_probs_blt = (
                    editor_full_logprobs.exp().permute(0, 2, 1).contiguous()
                )
                editor_partial_probs_blt = (
                    editor_partial_logprobs.exp().permute(0, 2, 1).contiguous()
                )

            # Equalize S2F to GT for S2F head teacher.
            s2f_eq_to_gt = CriticAdapter._global_total_equalize(
                s2f_tracks, gt_tracks, eps=eps
            )

            # Critic logits for GT head.
            gt_students = {
                "gt": gt_tracks,
                "s2f": s2f_tracks,
                "dae": dae_probs_blt,
            }
            if editor_full_probs_blt is not None and editor_partial_probs_blt is not None:
                gt_students["editor_full_dna"] = editor_full_probs_blt
                gt_students["editor_partial_dna"] = editor_partial_probs_blt
            gt_out = critic_gt_head_adapter.compute_logits_and_embeddings(
                dna=dna,
                teacher_track=gt_tracks,
                student_tracks=gt_students,
                dna_mask=mask_2d,
            )

            gt_teacher__logits_list.append(
                gt_out["teacher__logits"].cpu()
            )
            gt_student__s2f_logits_list.append(
                gt_out["student__s2f__logits"].cpu()
            )
            gt_student__dae_logits_list.append(
                gt_out["student__dae__logits"].cpu()
            )
            gt_student__gt_logits_list.append(
                gt_out["student__gt__logits"].cpu()
            )
            if "student__editor_full_dna__logits" in gt_out:
                gt_student__editor_full_dna_logits_list.append(
                    gt_out["student__editor_full_dna__logits"].cpu()
                )
            if "student__editor_partial_dna__logits" in gt_out:
                gt_student__editor_partial_dna_logits_list.append(
                    gt_out["student__editor_partial_dna__logits"].cpu()
                )

            # Critic logits for S2F head.
            if critic_s2f_head_adapter is not None:
                s2f_students = {
                    "dae": dae_probs_blt,
                }
                if editor_full_probs_blt is not None and editor_partial_probs_blt is not None:
                    s2f_students["editor_full_dna"] = editor_full_probs_blt
                    s2f_students["editor_partial_dna"] = editor_partial_probs_blt
                s2f_out = critic_s2f_head_adapter.compute_logits_and_embeddings(
                    dna=dna,
                    teacher_track=s2f_eq_to_gt,
                    student_tracks=s2f_students,
                    dna_mask=mask_2d,
                )

                s2f_teacher_logits_list.append(s2f_out["teacher__logits"].cpu())
                s2f_student__dae_logits_list.append(
                    s2f_out["student__dae__logits"].cpu()
                )
                if "student__editor_full_dna__logits" in s2f_out:
                    s2f_student__editor_full_dna_logits_list.append(
                        s2f_out["student__editor_full_dna__logits"].cpu()
                    )
                if "student__editor_partial_dna__logits" in s2f_out:
                    s2f_student__editor_partial_dna_logits_list.append(
                        s2f_out["student__editor_partial_dna__logits"].cpu()
                    )

            # Accumulate tracks and logprobs
            gt_counts_list.append(gt_tracks.permute(0, 2, 1).cpu())
            s2f_counts_list.append(s2f_tracks.permute(0, 2, 1).cpu())
            dae__logprobs_list.append(dae_logprobs.cpu())
            if editor_full_logprobs is not None and editor_partial_logprobs is not None:
                editor_full_dna__logprobs_list.append(editor_full_logprobs.cpu())
                editor_partial_dna__logprobs_list.append(editor_partial_logprobs.cpu())

            dna_mask_list.append(mask_2d.cpu())
            ref_onehot_list.append(dna.cpu())
            region_types.extend(batch_region_types)
            identifiers.extend(batch_identifiers)

    extracted = {
        "gt_counts": torch.cat(gt_counts_list, dim=0).numpy(),
        "s2f_counts": torch.cat(s2f_counts_list, dim=0).numpy(),
        "dae_logprobs": torch.cat(
            dae__logprobs_list, dim=0
        ).numpy(),
        "editor_full_dna_logprobs": (
            torch.cat(editor_full_dna__logprobs_list, dim=0).numpy()
            if editor_full_dna__logprobs_list
            else None
        ),
        "editor_partial_dna_logprobs": (
            torch.cat(editor_partial_dna__logprobs_list, dim=0).numpy()
            if editor_partial_dna__logprobs_list
            else None
        ),
        "dna_mask": torch.cat(dna_mask_list, dim=0).numpy(),
        "ref_onehot": torch.cat(ref_onehot_list, dim=0).numpy(),
        "region_types": list(region_types),
        "identifiers": list(identifiers),
        "critic_gt_head": {
            "teacher_logits": torch.cat(gt_teacher__logits_list, dim=0).numpy(),
            "student_s2f_logits": torch.cat(
                gt_student__s2f_logits_list, dim=0
            ).numpy(),
            "student_dae_logits": torch.cat(
                gt_student__dae_logits_list, dim=0
            ).numpy(),
            "student_gt_logits": torch.cat(
                gt_student__gt_logits_list, dim=0
            ).numpy(),
            "student_editor_full_dna_logits": (
                torch.cat(gt_student__editor_full_dna_logits_list, dim=0).numpy()
                if gt_student__editor_full_dna_logits_list
                else None
            ),
            "student_editor_partial_dna_logits": (
                torch.cat(gt_student__editor_partial_dna_logits_list, dim=0).numpy()
                if gt_student__editor_partial_dna_logits_list
                else None
            ),
        },
    }

    if s2f_teacher_logits_list:
        extracted["critic_s2f_head"] = {
            "teacher_logits": torch.cat(s2f_teacher_logits_list, dim=0).numpy(),
            "student_dae_logits": torch.cat(
                s2f_student__dae_logits_list, dim=0
            ).numpy(),
            "student_editor_full_dna_logits": (
                torch.cat(s2f_student__editor_full_dna_logits_list, dim=0).numpy()
                if s2f_student__editor_full_dna_logits_list
                else None
            ),
            "student_editor_partial_dna_logits": (
                torch.cat(s2f_student__editor_partial_dna_logits_list, dim=0).numpy()
                if s2f_student__editor_partial_dna_logits_list
                else None
            ),
        }
    else:
        extracted["critic_s2f_head"] = None

    return extracted


__all__ = ["extract_tracks_and_logits"]
