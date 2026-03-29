# -*- coding: utf-8 -*-
"""
Critic (distinct-channel variant) with a shared backbone and two fixed MLM heads (GT and S2F).

- Expects backbone input T = 2 (two track channels). For each head pass we build
  a two-channel track tensor where the non-target track is zeroed out (and mask is adjusted) so the
  backbone never sees both inputs simultaneously.
- Accepts the original single-channel inputs `s2f_tracks` and `gt_tracks` and a
  mask of shape (B, L, 4+T_in). If T_in != 2, the track mask is replicated or
  padded to produce a mask with last dim 4+2 (DNA mask left unchanged).

The backbone maps to (B, C, L); each head projects to per-token logits over DNA
(B, L, 4). Forward returns logits for self and optional cross inference and can
optionally return backbone features.
"""

from typing import Dict, Any, Tuple
import os

import torch
import torch.nn as nn
import torch.nn.functional as F


def masked_ce_loss(
    logits: torch.Tensor,          # (B, L, 4) logits
    ref_genome_1hot: torch.Tensor, # (B, L, 4)
    mask: torch.Tensor,            # (B, L) or (B, L, 4) or (B, L, 4+T)
) -> torch.Tensor:
    """
    Masked mean cross-entropy w.r.t. reference DNA from predicted logits.

    Notes on mask:
    - `mask` is 1 where the token is masked, 0 otherwise.
    - Accepts shapes (B,L), (B,L,4), or (B,L,4+T). If last dim > 4, slice to 4.
    - Reduces to token-wise mask (B,L) by OR over the last channel dim when 3D.
    """
    log_probs = F.log_softmax(logits, dim=-1)
    ce = -(ref_genome_1hot * log_probs).sum(dim=-1)  # (B,L)

    # normalize mask to (B,L)
    if mask.dim() == 3:
        if mask.size(-1) > 4:
            mask = mask[..., :4]
        mask = mask.any(dim=-1)
    mask = (mask > 0).to(logits.dtype)  # (B,L)

    total = (ce * mask).sum()
    denom = mask.sum()
    # Branchless guard: normalize by denom (clamped) and zero-out if denom==0
    loss = (total / denom.clamp_min(1.0)) * (denom > 0).to(ce.dtype)
    return loss


def masked_token_ce_from_labels(
    logits: torch.Tensor,  # (B, L, C) logits
    labels: torch.Tensor,  # (B, L) int64 class indices
    mask: torch.Tensor,    # (B, L) or (B, L, 4) or (B, L, 4+T)
) -> torch.Tensor:
    """
    Masked mean cross-entropy for generic per-token classification heads.

    Parameters
    ----------
    logits : (B, L, C)
        Per-token logits over C classes.
    labels : (B, L)
        Integer class indices in [0, C-1].
    mask :
        Token mask where 1 marks supervised positions. Accepts shapes
        (B,L), (B,L,4), or (B,L,4+T); when 3D, the last dim is reduced
        with an OR over channels and, if >4, sliced to the first 4
        channels (DNA mask layout).
    """
    if logits.dim() != 3:
        raise ValueError(f"logits must be (B,L,C), got {tuple(logits.shape)}")
    if labels.dim() != 2:
        raise ValueError(f"labels must be (B,L), got {tuple(labels.shape)}")

    B, L, C = logits.shape
    if labels.shape != (B, L):
        raise ValueError(
            f"labels shape {tuple(labels.shape)} does not match logits (B,L)={(B, L)}"
        )

    # Normalize mask to (B,L)
    if mask.dim() == 3:
        if mask.size(-1) > 4:
            mask = mask[..., :4]
        mask = mask.any(dim=-1)
    elif mask.dim() != 2:
        raise ValueError(f"mask must be (B,L) or (B,L,K), got {tuple(mask.shape)}")
    mask_bool = (mask > 0)

    # Flatten for per-token CE
    logits_flat = logits.reshape(B * L, C)
    labels_flat = labels.reshape(B * L).to(torch.long)
    mask_flat = mask_bool.reshape(B * L).to(logits.dtype)

    log_probs_flat = F.log_softmax(logits_flat, dim=-1)
    nll_flat = F.nll_loss(
        log_probs_flat,
        labels_flat,
        reduction="none",
    )  # (B*L,)

    total = (nll_flat * mask_flat).sum()
    denom = mask_flat.sum()
    loss = (total / denom.clamp_min(1.0)) * (denom > 0).to(logits.dtype)
    return loss
