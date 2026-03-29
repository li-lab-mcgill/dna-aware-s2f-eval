from __future__ import annotations

from typing import Any, Dict

import numpy as np

from ..metrics.critic.accuracy import MaskedAccuracyMetric
from ..metrics.critic.ce import MaskedCEMetric
from ..metrics.critic.entropy import MaskedEntropyMetric
from ..metrics.critic.forward_kl import MaskedForwardKLMetric


def _softmax(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits)
    logits = logits - logits.max(axis=-1, keepdims=True)
    exp_logits = np.exp(logits)
    return exp_logits / exp_logits.sum(axis=-1, keepdims=True)


def _compute_accuracy(
    *,
    p_probs: np.ndarray,
    q_probs: np.ndarray,
    mask: np.ndarray,
) -> float:
    acc_metric = MaskedAccuracyMetric()
    out = acc_metric.evaluate(
        p_probs=p_probs,
        q_probs=q_probs,
        mask=mask,
        prefix="",
    )
    return float(out["accuracy"])


def _compute_entropy(
    *,
    probs: np.ndarray,
    mask: np.ndarray,
) -> Dict[str, Any]:
    ent_metric = MaskedEntropyMetric()
    out = ent_metric.evaluate(
        p_probs=probs,
        q_probs=probs,
        mask=mask,
        return_stats=True,
        return_values=False,
        return_mean=True,
        prefix="",
    )
    return out


def _compute_kl(
    *,
    p_probs: np.ndarray,
    q_probs: np.ndarray,
    mask: np.ndarray,
) -> float:
    kl_metric = MaskedForwardKLMetric()
    out = kl_metric.evaluate(
        p_probs=p_probs,
        q_probs=q_probs,
        mask=mask,
        return_stats=False,
        return_values=False,
        return_mean=True,
        prefix="",
    )
    return float(out["mean"])


def _compute_ce(
    *,
    p_probs: np.ndarray,
    q_probs: np.ndarray,
    mask: np.ndarray,
) -> float:
    ce_metric = MaskedCEMetric()
    out = ce_metric.evaluate(
        p_probs=p_probs,
        q_probs=q_probs,
        mask=mask,
        return_stats=False,
        return_values=False,
        return_mean=True,
        prefix="",
    )
    return float(out["mean"])


def compute_critic_metrics(
    extracted: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Compute critic metrics from extracted logits.

    For each head and student, metrics are computed over four masked subsets:
      - masked
      - masked & (teacher == REF)
      - masked & (teacher != REF)
      - masked & (teacher argmax == student argmax)

    Metrics per subset:
      - accuracy vs teacher
      - accuracy vs REF
      - entropy (student)
      - KL(teacher || student)
      - CE(teacher || student)
    """
    mask = np.asarray(extracted["dna_mask"])
    if mask.ndim == 3 and mask.shape[-1] == 1:
        mask = mask[:, :, 0]
    mask = mask.astype(bool)

    ref_probs = np.asarray(extracted["ref_onehot"])
    if ref_probs.ndim != 3 or ref_probs.shape[2] != 4:
        raise ValueError(f"ref_onehot must be (B, L, 4), got {tuple(ref_probs.shape)}")

    gt_head = extracted["critic_gt_head"]
    s2f_head = extracted.get("critic_s2f_head")

    out: Dict[str, Any] = {"gt_head": {}, "s2f_head": {}}

    def _compute_head_metrics(
        head: Dict[str, Any],
        students: Dict[str, np.ndarray],
    ) -> Dict[str, Any]:
        teacher_probs = _softmax(np.asarray(head["teacher_logits"]))
        teacher_argmax = np.argmax(teacher_probs, axis=-1)
        ref_argmax = np.argmax(ref_probs, axis=-1)

        head_out: Dict[str, Any] = {}

        for student_name, student_logits in students.items():
            student_probs = _softmax(np.asarray(student_logits))
            student_argmax = np.argmax(student_probs, axis=-1)

            mask_sets = {
                "masked": mask,
                "masked_teacher_eq_ref": mask & (teacher_argmax == ref_argmax),
                "masked_teacher_neq_ref": mask & (teacher_argmax != ref_argmax),
                "masked_teacher_student_match": mask
                & (teacher_argmax == student_argmax),
            }

            metrics: Dict[str, Any] = {}
            for mask_name, mask_value in mask_sets.items():
                metrics[mask_name] = {
                    "acc_teacher": _compute_accuracy(
                        p_probs=student_probs,
                        q_probs=teacher_probs,
                        mask=mask_value,
                    ),
                    "acc_ref": _compute_accuracy(
                        p_probs=student_probs,
                        q_probs=ref_probs,
                        mask=mask_value,
                    ),
                    "entropy": _compute_entropy(
                        probs=student_probs,
                        mask=mask_value,
                    ),
                    "kl_teacher_student": _compute_kl(
                        p_probs=teacher_probs,
                        q_probs=student_probs,
                        mask=mask_value,
                    ),
                    "ce_teacher_student": _compute_ce(
                        p_probs=teacher_probs,
                        q_probs=student_probs,
                        mask=mask_value,
                    ),
                }

            head_out[student_name] = metrics

        return head_out

    dae_gt_logits = gt_head.get("student_dae_logits")
    if dae_gt_logits is None:
        dae_gt_logits = gt_head.get("student_dae_debiased_logits")

    gt_students = {
        "gt": gt_head.get("student_gt_logits", gt_head["teacher_logits"]),
        "s2f": gt_head["student_s2f_logits"],
        "dae": dae_gt_logits,
    }
    gt_students = {k: v for k, v in gt_students.items() if v is not None}
    if gt_head.get("student_editor_full_dna_logits") is not None:
        gt_students["editor_full_dna"] = gt_head["student_editor_full_dna_logits"]
    if gt_head.get("student_editor_partial_dna_logits") is not None:
        gt_students["editor_partial_dna"] = gt_head["student_editor_partial_dna_logits"]

    out["gt_head"] = _compute_head_metrics(gt_head, gt_students)

    if s2f_head is not None:
        dae_s2f_logits = s2f_head.get("student_dae_logits")
        if dae_s2f_logits is None:
            dae_s2f_logits = s2f_head.get("student_dae_debiased_logits")

        s2f_students = {
            "s2f": s2f_head["teacher_logits"],
            "dae": dae_s2f_logits,
        }
        s2f_students = {k: v for k, v in s2f_students.items() if v is not None}
        if s2f_head.get("student_editor_full_dna_logits") is not None:
            s2f_students["editor_full_dna"] = s2f_head["student_editor_full_dna_logits"]
        if s2f_head.get("student_editor_partial_dna_logits") is not None:
            s2f_students["editor_partial_dna"] = s2f_head["student_editor_partial_dna_logits"]
        out["s2f_head"] = _compute_head_metrics(s2f_head, s2f_students)
    else:
        out["s2f_head"] = None

    return out


__all__ = ["compute_critic_metrics"]
