import torch
import torch.nn as nn

from .bottleneck_helpers import (
    SwiGLUProjector,
    TransformerBottleneck1D,
)


class Bottleneck(nn.Module):
    """
    Simple channel bottleneck:

        h: (B, C_in,  L) -> z: (B, C_lat, L) -> h_out: (B, C_out, L)

    A tiny SwiGLU MLP used as a reusable
        C_in -> C_lat -> C_out
    projector. For now the noise AE mostly uses the bottleneck latent z, but
    returning both h_out and z from forward() is important for later DAE/editors.
    """

    def __init__(
        self,
        in_channels: int,
        bottleneck_channels: int,
        out_channels: int,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.bottleneck_channels = int(bottleneck_channels)
        self.out_channels = int(out_channels)

        # C_in -> C_lat using the lightweight SwiGLU projector (no residual branch).
        self.to_latent = SwiGLUProjector(
            in_channels=self.in_channels,
            out_channels=self.bottleneck_channels,
            use_layernorm=False,
            use_residual=True,
        )
        # C_lat -> C_out using the same selector.
        self.from_latent = SwiGLUProjector(
            in_channels=self.bottleneck_channels,
            out_channels=self.out_channels,
            use_layernorm=False,
            use_residual=True,
        )

    def encode(self, h: torch.Tensor) -> torch.Tensor:
        """
        C_in -> C_lat
        """
        return self.to_latent(h)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        C_lat -> C_out
        """
        return self.from_latent(z)

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Full C_in -> C_lat -> C_out bottleneck.

        Args:
            h: (B, C_in, L)

        Returns:
            h_out: (B, C_out, L)
            z:     (B, C_lat, L)  -- the actual bottleneck code
        """
        z = self.encode(h)
        h_out = self.decode(z)
        return h_out, z


class BottleneckEncoder(nn.Module):
    """
    Bottleneck encoder operating at the conv-latent resolution (L_lat).

    This takes the conv features from the multimodal encoder:

        h_conv: (B, C_conv_lat, L_lat)

    and produces a mixed conv-latent representation at the same width:

        h_enc: (B, C_conv_lat, L_lat)

    New structure (no dilated convs), symmetric in depth with the decoder:
      1) Channel-only SwiGLU projector C_conv_lat -> C_conv_lat (mixing).
      2) TransformerBottleneck1D stack at C_conv_lat, depth = transformer_depth.

    The explicit C_in -> C_lat -> C_out bottleneck is implemented separately by
    the Bottleneck class and is wired in at the DNAMaskedNoiseModel level.
    """

    def __init__(
        self,
        conv_latent_channels: int,      # C_conv_lat (e.g., 64)
        bottleneck_channels: int,       # C_lat (e.g., 16)
        transformer_depth: int = 1,
        transformer_heads: int = 8,
        transformer_dim_head: int | None = None,
        transformer_kv_heads: int | None = None,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
        ff_mult: float = 4.0,
        ff_dropout: float = 0.0,
        rope_fraction: float = 0.25,
        rope_theta: float = 10000.0,
        swiglu_use_layernorm: bool = False,
    ):
        super().__init__()

        self.conv_latent_channels = int(conv_latent_channels)
        self.bottleneck_channels = int(bottleneck_channels)

        # SwiGLU mixing at conv-latent resolution (C_conv_lat -> C_conv_lat)
        self.pre_swiglu = SwiGLUProjector(
            in_channels=self.conv_latent_channels,
            out_channels=self.conv_latent_channels,
            use_layernorm=swiglu_use_layernorm,
        )

        # Transformer stack at conv-latent resolution (symmetric depth with decoder)
        self.transformer = TransformerBottleneck1D(
            dim=self.conv_latent_channels,
            depth=transformer_depth,
            heads=transformer_heads,
            dim_head=transformer_dim_head,
            kv_heads=transformer_kv_heads,
            attn_dropout=attn_dropout,
            proj_dropout=proj_dropout,
            ff_mult=ff_mult,
            ff_dropout=ff_dropout,
            rope_fraction=rope_fraction,
            rope_theta=rope_theta,
        )

    def forward(self, h_conv: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h_conv: (B, C_conv_lat, L_lat)

        Returns:
            h_enc: (B, C_conv_lat, L_lat)
        """
        h = self.pre_swiglu(h_conv)          # (B, C_conv_lat, L_lat)
        h = self.transformer(h)              # (B, C_conv_lat, L_lat)
        return h


class BottleneckDecoder(nn.Module):
    """
    Bottleneck decoder that operates at the post-bottleneck model_dim:

        h_in:   (B, d_model, L_lat)
        h_bott: (B, d_model, L_lat)

    Structure:
      1) TransformerBottleneck1D stack at d_model (global mixing at L_lat).
      2) SwiGLUProjector d_model -> d_model for additional channel-wise mixing.

    This is the shared global mixer for the GT manifold. In the GT AE:
        z_GT -> BottleneckDecoder -> h_bott_GT -> GTTrackDecoder

    Later, S2F latents (after their own encoder + gating) will pass through
    the *same* BottleneckDecoder + GTTrackDecoder during denoising.
    """

    def __init__(
        self,
        bottleneck_channels: int,       # C_lat (e.g., 8)
        model_dim: int,                 # d_model (e.g., 64)
        transformer_depth: int = 1,
        transformer_heads: int = 8,
        transformer_dim_head: int | None = None,
        transformer_kv_heads: int | None = None,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
        ff_mult: float = 4.0,
        ff_dropout: float = 0.0,
        rope_fraction: float = 0.25,
        rope_theta: float = 10000.0,
        swiglu_use_layernorm: bool = False,
    ):
        super().__init__()

        self.model_dim = int(model_dim)

        # Transformer stack at latent resolution d_model (symmetric depth with encoder)
        self.transformer = TransformerBottleneck1D(
            dim=self.model_dim,
            depth=transformer_depth,
            heads=transformer_heads,
            dim_head=transformer_dim_head,
            kv_heads=transformer_kv_heads,
            attn_dropout=attn_dropout,
            proj_dropout=proj_dropout,
            ff_mult=ff_mult,
            ff_dropout=ff_dropout,
            rope_fraction=rope_fraction,
            rope_theta=rope_theta,
        )

        # SwiGLU mixing d_model -> d_model after attention
        self.post_swiglu = SwiGLUProjector(
            in_channels=self.model_dim,
            out_channels=self.model_dim,
            use_layernorm=swiglu_use_layernorm,
        )

    def forward(self, h_in: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h_in: (B, d_model, L_lat)

        Returns:
            h_bott: (B, d_model, L_lat)
        """
        h = self.transformer(h_in)           # (B, d_model, L_lat)
        h_bott = self.post_swiglu(h)         # (B, d_model, L_lat)
        return h_bott
