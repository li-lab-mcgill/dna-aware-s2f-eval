import torch
import torch.nn as nn

from .bottleneck_helpers import SwiGLUProjector


class LatentEditor(nn.Module):
    """
    DNA-aware FiLM editor operating at the DAE bottleneck code resolution.

    This module takes:
      - an anchor bottleneck code z_anchor (typically z from the frozen DAE),
      - a context bottleneck embedding z_ctx (from the trainable track+DNA encoder),

    and applies a lightweight two-layer SwiGLU MLP to produce per-channel,
    per-position FiLM parameters (s, gamma), which define:

        alpha = 1 + s
        z_edit = alpha * z_anchor + gamma

    Identity initialization:
      - all head weights = 0
      - all head biases  = 0
      so at initialization alpha = 1 and gamma = 0, i.e., z_edit == z_anchor.

    Shapes (per batch):
        z_anchor : (B, C_bottleneck, L_lat)
        z_ctx    : (B, C_context,   L_lat)
        z_edit   : (B, C_bottleneck, L_lat)
    """

    def __init__(
        self,
        bottleneck_channels: int,
        context_channels: int,
        *,
        hidden_channels: int | None = None,
        use_layernorm: bool = False,
    ):
        super().__init__()

        self.bottleneck_channels = int(bottleneck_channels)
        self.context_channels = int(context_channels)
        if hidden_channels is None:
            hidden_channels = bottleneck_channels
        self.hidden_channels = int(hidden_channels)

        in_channels = self.bottleneck_channels + self.context_channels

        # Core editor operates in a hidden channel space.
        self.core_proj_in = SwiGLUProjector(
            in_channels=in_channels,
            out_channels=self.hidden_channels,
            use_layernorm=use_layernorm,
            use_residual=True,
        )
        self.core_proj_mid = SwiGLUProjector(
            in_channels=self.hidden_channels,
            out_channels=self.hidden_channels,
            use_layernorm=use_layernorm,
            use_residual=True,
        )

        # Readout head from hidden space to (s, gamma) in bottleneck space.
        # We predict 2 * C_bottleneck channels and split into (s, gamma).
        self.readout_head = nn.Conv1d(
            self.hidden_channels,
            2 * self.bottleneck_channels,
            kernel_size=1,
            bias=True,
        )
        # Identity init: start from z_edit = z_anchor.
        nn.init.zeros_(self.readout_head.weight)
        if self.readout_head.bias is not None:
            nn.init.zeros_(self.readout_head.bias)

    def forward(
        self,
        z_anchor: torch.Tensor,
        z_ctx: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute a FiLM-style edit of the anchor bottleneck code.

        Args:
            z_anchor: (B, C_bottleneck, L_lat) anchor bottleneck code (e.g., DAE z).
            z_ctx:    (B, C_context,   L_lat) context embedding from track+DNA encoder.

        Returns:
            z_edit:   (B, C_bottleneck, L_lat) edited bottleneck code.
        """
        if z_anchor.dim() != 3:
            raise ValueError(f"z_anchor must be (B,C,L), got {tuple(z_anchor.shape)}")
        if z_ctx.dim() != 3:
            raise ValueError(f"z_ctx must be (B,C,L), got {tuple(z_ctx.shape)}")
        if z_anchor.shape[-1] != z_ctx.shape[-1]:
            raise ValueError(
                f"z_anchor and z_ctx must share spatial length, got {z_anchor.shape} vs {z_ctx.shape}"
            )

        feat = torch.cat([z_anchor, z_ctx], dim=1)  # (B, C_bottleneck+C_ctx, L)

        h_core = self.core_proj_in(feat)
        h_core = self.core_proj_mid(h_core)

        s_gamma = self.readout_head(h_core)  # (B, 2*C_bottleneck, L)
        s, gamma = torch.split(s_gamma, self.bottleneck_channels, dim=1)

        alpha = 1.0 + s
        z_edit = alpha * z_anchor + gamma
        return z_edit


__all__ = ["DNAAwareLatentEditor"]
