import torch
import torch.nn as nn

from .encoder import TrackEncoder, MaskedDNAEncoder
from .bottleneck import BottleneckEncoder


class DNAAwareTrackReencoder(nn.Module):
    """
    Track + masked-DNA re-encoder that produces a context embedding at
    conv-latent resolution for the FiLM latent editor.
    """

    def __init__(
        self,
        track_in_channels: int,
        *,
        # Track encoder config
        track_base_channels: int = 16,
        track_conv_latent_channels: int | None = None,
        track_encoder_kernel_size: int = 5,
        track_stem_kernel_size: int | None = None,
        track_down_kernel_size: int | None = None,
        # DNA encoder config
        dna_num_modalities: int = 4,
        dna_base_channels: int = 16,
        dna_latent_channels: int | None = None,
        dna_stem_kernel_size: int = 23,
        dna_down_kernel_size: int = 5,
        dna_padding_mode: str = "same",
        dna_mask_aware_padding: bool = True,
        # Bottleneck encoder config
        bottleneck_channels: int = 32,
        bottleneck_depth: int = 1,
        bottleneck_heads: int = 4,
        bottleneck_dim_head: int | None = None,
        bottleneck_kv_heads: int | None = None,
        bottleneck_attn_dropout: float = 0.0,
        bottleneck_proj_dropout: float = 0.0,
        bottleneck_ff_mult: float = 4.0,
        bottleneck_ff_dropout: float = 0.0,
        bottleneck_rope_fraction: float = 0.25,
        bottleneck_rope_theta: float = 10000.0,
        bottleneck_swiglu_use_layernorm: bool = False,
    ):
        super().__init__()

        if track_conv_latent_channels is None:
            track_conv_latent_channels = track_base_channels * 4
        if dna_latent_channels is None:
            dna_latent_channels = track_conv_latent_channels

        self.track_in_channels = int(track_in_channels)
        self.track_base_channels = int(track_base_channels)
        self.track_conv_latent_channels = int(track_conv_latent_channels)
        self.dna_base_channels = int(dna_base_channels)
        self.dna_latent_channels = int(dna_latent_channels)

        if self.track_conv_latent_channels != self.dna_latent_channels:
            raise ValueError(
                "track_conv_latent_channels must match dna_latent_channels for fusion; "
                f"got {self.track_conv_latent_channels} vs {self.dna_latent_channels}."
            )

        self.track_encoder = TrackEncoder(
            in_channels=self.track_in_channels,
            base_channels=self.track_base_channels,
            latent_channels=self.track_conv_latent_channels,
            kernel_size=int(track_encoder_kernel_size),
            stem_kernel_size=track_stem_kernel_size,
            down_kernel_size=track_down_kernel_size,
        )

        self.dna_encoder = MaskedDNAEncoder(
            num_modalities=int(dna_num_modalities),
            base_channels=self.dna_base_channels,
            latent_channels=self.dna_latent_channels,
            stem_kernel_size=int(dna_stem_kernel_size),
            down_kernel_size=int(dna_down_kernel_size),
            padding_mode=str(dna_padding_mode),
            mask_aware_padding=bool(dna_mask_aware_padding),
        )

        self.bottleneck_encoder = BottleneckEncoder(
            conv_latent_channels=self.track_conv_latent_channels,
            bottleneck_channels=int(bottleneck_channels),
            transformer_depth=int(bottleneck_depth),
            transformer_heads=int(bottleneck_heads),
            transformer_dim_head=(
                int(bottleneck_dim_head) if bottleneck_dim_head is not None else None
            ),
            transformer_kv_heads=(
                int(bottleneck_kv_heads) if bottleneck_kv_heads is not None else None
            ),
            attn_dropout=float(bottleneck_attn_dropout),
            proj_dropout=float(bottleneck_proj_dropout),
            ff_mult=float(bottleneck_ff_mult),
            ff_dropout=float(bottleneck_ff_dropout),
            rope_fraction=float(bottleneck_rope_fraction),
            rope_theta=float(bottleneck_rope_theta),
            swiglu_use_layernorm=bool(bottleneck_swiglu_use_layernorm),
        )

    def forward(
        self,
        track: torch.Tensor,
        masked_dna: torch.Tensor,
        *,
        return_latents: bool = False,
    ):
        """
        Args:
            track: (B, C_in, L) track tensor.
            masked_dna: (B, 2, L, 4) DNA value+mask tensor.
            return_latents: if True, return intermediate latents.

        Returns:
            z_ctx: (B, C_conv_lat, L_lat) context embedding.
            If return_latents is True, also returns a dict with:
                "h_track", "h_dna", "h_fused".
        """
        if track.dim() != 3:
            raise ValueError(f"track must be (B,C,L), got {tuple(track.shape)}")
        if masked_dna.dim() != 4:
            raise ValueError(f"masked_dna must be (B,2,L,4), got {tuple(masked_dna.shape)}")
        if track.size(0) != masked_dna.size(0) or track.size(2) != masked_dna.size(2):
            raise ValueError(
                "masked_dna batch/length must match track: "
                f"got track {tuple(track.shape)} vs dna {tuple(masked_dna.shape)}"
            )

        h_track = self.track_encoder(track)
        h_dna = self.dna_encoder(masked_dna)

        if h_track.shape != h_dna.shape:
            raise ValueError(
                "Track and DNA latents must have same shape for fusion; "
                f"got {tuple(h_track.shape)} vs {tuple(h_dna.shape)}"
            )

        h_fused = h_track + h_dna
        z_ctx = self.bottleneck_encoder(h_fused)

        if not return_latents:
            return z_ctx

        latents = {
            "h_track": h_track,
            "h_dna": h_dna,
            "h_fused": h_fused,
        }
        return z_ctx, latents


__all__ = ["DNAAwareTrackReencoder"]
