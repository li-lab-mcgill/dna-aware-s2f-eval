from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class GradientStats:
    g_fid: float
    g_mi: float


class GradientRatioController:
    """
    Gradient-ratio controller for balancing fidelity vs critic losses.

    Maintains EMAs of gradient norms on the full 1000bp interface and
    updates a critic weight lambda_t to target a desired ratio:

        lambda_t ≈ gamma * (g_fid / g_mi)

    with clipping to [lambda_min, lambda_max].

    This module is intended adaptively scale critic losses relative to fidelity. Usage:
      1) Compute g_fid, g_mi via `compute_grad_norm` on the fidelity and
         critic losses w.r.t. z_edit (no masking).
      2) Call `update(g_fid, g_mi)` once per step to obtain lambda_t.
      3) Scale the critic loss total (returned by CriticLossBundle) by
         lambda_t before backprop.
    """

    def __init__(
        self,
        *,
        gamma: float = 1.0,
        beta: float = 0.9,
        target_ratio: Optional[float] = None,
        lambda_min: float = 0.1,
        lambda_max: float = 10.0,
        epsilon: float = 1e-8,
        lambda_init: float = 1.0,
    ):
        self.gamma = float(gamma)
        self.beta = float(beta)
        # Optional explicit target ratio for ||g_critic_scaled|| / ||g_fid||.
        # When not None, this overrides `gamma` as the effective target; when
        # None, behavior is identical to the original implementation.
        self.target_ratio: Optional[float] = (
            float(target_ratio) if target_ratio is not None else None
        )
        if self.target_ratio is not None and self.target_ratio <= 0.0:
            # Degenerate / invalid target ratios fall back to legacy behavior.
            self.target_ratio = None

        self.lambda_min = float(lambda_min)
        self.lambda_max = float(lambda_max)
        self.epsilon = float(epsilon)

        self.lambda_t: float = float(lambda_init)
        self._g_fid_ema: Optional[float] = None
        self._g_mi_ema: Optional[float] = None
        self.last_stats = GradientStats(g_fid=0.0, g_mi=0.0)

    def compute_grad_norm(
        self,
        loss: torch.Tensor,
        z_edit: torch.Tensor,        # (B,L)
    ) -> float:
        """
        Compute gradient norm of loss w.r.t. z_edit over the full window,
        normalized by sqrt(B*L).
        """
        grads = torch.autograd.grad(
            outputs=loss,
            inputs=z_edit,
            retain_graph=True,
            create_graph=False,
            allow_unused=True,
        )[0]
        if grads is None:
            return 0.0

        if grads.numel() == 0:
            return 0.0

        norm = grads.norm(p=2).item()
        band_size = float(z_edit.numel())
        denom = max(band_size, 1.0) ** 0.5
        return norm / denom

    def update(self, g_fid: float, g_mi: float) -> float:
        """
        Update EMAs and lambda_t based on new gradient norms.
        """
        if self._g_fid_ema is None:
            self._g_fid_ema = g_fid
            self._g_mi_ema = g_mi
        else:
            self._g_fid_ema = self.beta * self._g_fid_ema + (1.0 - self.beta) * g_fid
            self._g_mi_ema = self.beta * self._g_mi_ema + (1.0 - self.beta) * g_mi

        g_fid_ema = self._g_fid_ema
        g_mi_ema = self._g_mi_ema if self._g_mi_ema is not None else 0.0

        if g_mi_ema > 0.0:
            # If a target_ratio is provided, we treat it as the desired
            # post-scaling gradient ratio and let it override gamma's role
            # as the effective target. Otherwise, we fall back to the legacy
            # behavior where gamma directly controls the target ratio.
            effective_gamma = self.target_ratio if self.target_ratio is not None else self.gamma
            ratio = effective_gamma * g_fid_ema / (g_mi_ema + self.epsilon)
        else:
            ratio = self.lambda_max

        self.lambda_t = float(max(self.lambda_min, min(self.lambda_max, ratio)))

        self.last_stats = GradientStats(g_fid=g_fid, g_mi=g_mi)
        return self.lambda_t


__all__ = ["GradientRatioController", "GradientStats"]
