################# MASKED INPUT LAYER #################

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedInputConv2DLayer(nn.Module):
    """
    2D Input Convolution based input layer for multimodal masked genome language model
    
    Input: (B, 2, L, T) where:
    - B: batch size
    - 2: channels (mask + value)
    - L: genome positions (height)
    - T: modalities (width) - T functional tracks + 4 DNA one-hot channels

    Output: (B, C, L') where L' depends on padding mode
        This is squeezed version of conv2D output (B, C, L', 1) -> (B, C, L')
    """
    
    def __init__(self, 
                 T,     # number of functional genomic tracks + DNA one-hot
                 C,     # output channels
                 k,     # kernel size along sequence (height) dimension
                 padding_mode='same',   # 'same' or 'valid'
                 mask_aware_padding: bool = False):
        super().__init__()
        
        self.T = T
        self.C = C
        self.k = k
        self.padding_mode = padding_mode
        self.mask_aware_padding = mask_aware_padding
        
        # Calculate padding - only allowed in sequence (height) dimension
        if padding_mode == 'same':
            pad_L = k // 2
        elif padding_mode == 'valid':
            pad_L = 0
        else:
            raise ValueError(f"padding_mode must be 'same' or 'valid', got {padding_mode}")
        
        self.pad_L = pad_L
            
        # Conv2D: (B, 2, L, T) -> (B, C, L', 1) -> (B, C, L')
        # Treats mask/value as channels - more natural and efficient
        self.conv = nn.Conv2d(
            in_channels=2,                # mask and value channels (hardcoded)
            out_channels=C,
            kernel_size=(k, T),           # (sequence, modalities)
            stride=(1, 1),                # stride only effective in sequence dimension
            # If mask-aware padding is enabled (and padding_mode == 'same'), we pre-pad
            # sequence dim ourselves with value=0, mask=1. Then conv uses no internal pad.
            padding=(0 if (mask_aware_padding and padding_mode == 'same') else pad_L, 0)
        )
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (B, 2, L, T)
               - Channel 0: Value channel
               - Channel 1: Mask channel
        
        Returns:
            Output tensor of shape (B, C, L')
        """
        # Input validation
        assert x.dim() == 4, f"Expected 4D input (B, 2, L, T), got {x.dim()}D"
        assert x.shape[1] == 2, f"Expected 2 input channels (mask+value), got {x.shape[1]}"
        assert x.shape[3] == self.T, f"Expected {self.T} modalities, got {x.shape[3]}"
        
        # Optional mask-aware padding for 'same' mode: pre-pad sequence dim
        # with value channel padded by 0 and mask channel padded by 1.
        if self.padding_mode == 'same' and self.mask_aware_padding and self.pad_L > 0:
            # x: (B, 2, L, T); channels: 0=value, 1=mask
            x_value = x[:, 0:1, :, :]
            x_mask = x[:, 1:2, :, :]
            # Pad only height/sequence dim (H), leave width/modalities (W=T) unchanged
            # F.pad pads last dims first: (pad_w_left, pad_w_right, pad_h_top, pad_h_bottom)
            x_value = F.pad(x_value, (0, 0, self.pad_L, self.pad_L), mode='constant', value=0.0)
            x_mask = F.pad(x_mask, (0, 0, self.pad_L, self.pad_L), mode='constant', value=1.0)
            x = torch.cat([x_value, x_mask], dim=1)

        out = self.conv(x)  # (B, C, L', 1) since kernel consumes all T modalities
        return out.squeeze(-1)  # (B, C, L')
    
    def get_output_length(self, input_length):
        """Calculate output sequence length given input length"""
        if self.padding_mode == 'same':
            return input_length
        else:  # valid padding
            return input_length - self.k + 1
    
    def __repr__(self):
        return (f"InputConv2D(T={self.T}, C={self.C}, k={self.k}, "
                f"padding_mode='{self.padding_mode}', mask_aware_padding={self.mask_aware_padding})\n"
                f"  Input shape: (B, 2, L, {self.T})\n"
                f"  Output shape: (B, {self.C}, L')")



################ ENCODER ###############


import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_group_norm(num_channels: int, max_groups: int = 8) -> nn.GroupNorm:
    """
    Create a GroupNorm layer with a sensible number of groups for 1D convs.

    - Uses up to `max_groups` groups, but always divides `num_channels`.
    - Falls back to 1 group (i.e., LayerNorm-like over channels) if needed.
    """
    groups = min(max_groups, num_channels)
    while groups > 1 and (num_channels % groups) != 0:
        groups -= 1
    return nn.GroupNorm(num_groups=groups, num_channels=num_channels)


class DownBlock1D(nn.Module):
    """
    Strided depthwise + pointwise conv block for 1D tracks.

    Input:  (B, C_in, L)
    Output: (B, C_out, L_down)  with L_down ≈ L / 2

    Structure:
      - DepthwiseConv1d(C_in -> C_in, stride=2, k=kernel_size)
      - PointwiseConv1d(C_in -> C_out, k=1)
      - GroupNorm(C_out)
      - SiLU
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 5,
    ):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be odd, got {kernel_size}")

        padding = kernel_size // 2

        # Depthwise conv with stride 2 for downsampling
        self.depthwise = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=2,
            padding=padding,
            groups=in_channels,
            bias=True,
        )

        # Pointwise conv for channel mixing
        self.pointwise = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=True,
        )

        self.norm = _make_group_norm(out_channels)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C_in, L)

        Returns:
            (B, C_out, L_down)
        """
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class TrackEncoder(nn.Module):
    """
    1D convolutional encoder for 1kbp log-simplex tracks.

    Design:
      - Input length ~1000bp, T track channels (e.g., 1 or 2).
      - 3 strided DownBlock1D blocks → spatial downsampling by 8.
      - Channel progression (default base_channels = 16, latent_channels = 64):
          C_in → base
          base → 2*base
          2*base → latent
          latent → latent   (third down keeps same channels)
      - No skip connections; output is a compact conv feature map at 125bp.

    Shapes (for L_in=1000, base=16, latent=64):
      x:        (B, C_in, 1000)
      stem:     (B, 16,   1000)
      down1:    (B, 32,   500)
      down2:    (B, 64,   250)
      down3:    (B, 64,   125)  ← this is the conv latent that goes into the bottleneck.

    Args:
        in_channels:    number of track channels (e.g. 1 for single-strand).
        base_channels:  width of the first conv stage.
        latent_channels:channels at the conv-latent resolution (L_lat).
        kernel_size:    spatial kernel size for convs (must be odd).
    """

    def __init__(
        self,
        in_channels: int,
        base_channels: int = 16,
        latent_channels: int | None = None,
        kernel_size: int = 5,
        stem_kernel_size: int | None = None,
        down_kernel_size: int | None = None,
    ):
        super().__init__()

        # Backwards-compatible handling:
        # - If stem_kernel_size/down_kernel_size are not provided, fall back to kernel_size.
        if stem_kernel_size is None:
            stem_kernel_size = kernel_size
        if down_kernel_size is None:
            down_kernel_size = kernel_size

        if stem_kernel_size % 2 == 0:
            raise ValueError(f"stem_kernel_size must be odd, got {stem_kernel_size}")
        if down_kernel_size % 2 == 0:
            raise ValueError(f"down_kernel_size must be odd, got {down_kernel_size}")

        if latent_channels is None:
            latent_channels = base_channels * 4  # e.g., 64 when base=16

        self.in_channels = int(in_channels)
        self.base_channels = int(base_channels)
        self.latent_channels = int(latent_channels)
        self.stem_kernel_size = int(stem_kernel_size)
        self.down_kernel_size = int(down_kernel_size)

        # Stem: simple conv + GN + SiLU
        self.stem = nn.Sequential(
            nn.Conv1d(
                in_channels,
                base_channels,
                kernel_size=self.stem_kernel_size,
                padding=self.stem_kernel_size // 2,
                bias=True,
            ),
            _make_group_norm(base_channels),
            nn.SiLU(),
        )

        # 3x downsampling: L -> L/2 -> L/4 -> L/8
        # Channels: base -> 2*base -> latent -> latent
        self.down1 = DownBlock1D(
            in_channels=base_channels,
            out_channels=base_channels * 2,
            kernel_size=self.down_kernel_size,
        )
        self.down2 = DownBlock1D(
            in_channels=base_channels * 2,
            out_channels=self.latent_channels,
            kernel_size=self.down_kernel_size,
        )
        self.down3 = DownBlock1D(
            in_channels=self.latent_channels,
            out_channels=self.latent_channels,
            kernel_size=self.down_kernel_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C_in, L_in) — typically L_in ≈ 1000.

        Returns:
            z_conv: (B, latent_channels, L_lat) — typically L_lat ≈ L_in / 8 (e.g., 125).
        """
        h = self.stem(x)    # (B, base_channels, L)
        h = self.down1(h)   # (B, 2*base, L/2)
        h = self.down2(h)   # (B, latent_channels, L/4)
        z = self.down3(h)   # (B, latent_channels, L/8)
        return z


class MaskedDNAEncoder(nn.Module):
    """
    Masked 1D convolutional encoder for 1kbp DNA.

    This encoder is explicitly *mask aware* via a 2D input stem:

        x_dna_masked: (B, 2, L, 4)
          - channel 0: DNA value channel (one‑hot over 4 bases, flattened into width)
          - channel 1: DNA mask channel (1 = masked, 0 = visible)

    Pipeline (for L_in ≈ 1000bp):

        x_dna_masked: (B, 2, 1000, 4)
          └─> MaskedInputConv2DLayer  -> stem: (B, base,   1000)
                └─> DownBlock1D x 3   -> z:    (B, latent, 125)

    Design mirrors TrackEncoder so that the conv-latent resolutions and channel
    counts line up for simple additive fusion in MultimodalEncoder.

    Args:
        num_modalities: number of per-position DNA channels (default: 4).
        base_channels:  width of the first conv stage.
        latent_channels:channels at the conv-latent resolution (L_lat).
        stem_kernel_size:
            spatial kernel size for the masked 2D stem conv (must be odd;
            typically larger, e.g. 23, to mirror cgLM input kernels).
        down_kernel_size:
            spatial kernel size for 1D DownBlock1D convs (must be odd; this
            is usually kept in sync with the track encoder kernel size, e.g. 5).
        padding_mode:   'same' or 'valid' for the masked 2D stem.
        mask_aware_padding:
            If True and padding_mode=='same', pad sequence dimension with
            value=0, mask=1 before the 2D stem conv.
    """

    def __init__(
        self,
        num_modalities: int = 4,
        base_channels: int = 16,
        latent_channels: int | None = None,
        stem_kernel_size: int = 23,
        down_kernel_size: int = 5,
        padding_mode: str = "same",
        mask_aware_padding: bool = True,
    ):
        super().__init__()
        if stem_kernel_size % 2 == 0:
            raise ValueError(f"stem_kernel_size must be odd, got {stem_kernel_size}")
        if down_kernel_size % 2 == 0:
            raise ValueError(f"down_kernel_size must be odd, got {down_kernel_size}")

        if latent_channels is None:
            latent_channels = base_channels * 4  # e.g., 64 when base=16

        self.num_modalities = int(num_modalities)
        self.base_channels = int(base_channels)
        self.latent_channels = int(latent_channels)

        # Stem: masked 2D conv over (value, mask) × DNA modalities.
        self.stem = nn.Sequential(
            MaskedInputConv2DLayer(
                T=self.num_modalities,
                C=self.base_channels,
                k=stem_kernel_size,
                padding_mode=padding_mode,
                mask_aware_padding=mask_aware_padding,
            ),
            _make_group_norm(self.base_channels),
            nn.SiLU(),
        )

        # 3x downsampling: L -> L/2 -> L/4 -> L/8
        # Channels: base -> 2*base -> latent -> latent
        self.down1 = DownBlock1D(
            in_channels=base_channels,
            out_channels=base_channels * 2,
            kernel_size=down_kernel_size,
        )
        self.down2 = DownBlock1D(
            in_channels=base_channels * 2,
            out_channels=self.latent_channels,
            kernel_size=down_kernel_size,
        )
        self.down3 = DownBlock1D(
            in_channels=self.latent_channels,
            out_channels=self.latent_channels,
            kernel_size=down_kernel_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 2, L_in, 4) — fused DNA value + mask, L_in ≈ 1000.

        Returns:
            z_conv: (B, latent_channels, L_lat) — typically L_lat ≈ L_in / 8 (e.g., 125).
        """
        # Stem collapses modalities and mask/value into a 1D feature map.
        h = self.stem(x)    # (B, base_channels, L)
        h = self.down1(h)   # (B, 2*base, L/2)
        h = self.down2(h)   # (B, latent_channels, L/4)
        z = self.down3(h)   # (B, latent_channels, L/8)
        return z


class TrackEncoderRouter(nn.Module):
    """
    Router over multiple TrackEncoder branches (e.g. GT vs S2F).

    This is the Stage-2 blind DAE replacement for the multimodal
    track+DNA encoder: it is track-only and simply selects which
    TrackEncoder to use based on a string route key.

    Typical usage:
        router = TrackEncoderRouter(track_in_channels=T)
        z_gt  = router(track_GT, route="gt")
        z_s2f = router(track_s2f, route="s2f")

    All branches share the same hyperparameters (base_channels,
    conv_latent_channels, kernel sizes) but have independent weights.
    """

    def __init__(
        self,
        track_in_channels: int,
        *,
        base_channels: int = 16,
        conv_latent_channels: int | None = None,
        encoder_kernel_size: int = 5,
        stem_kernel_size: int | None = None,
        down_kernel_size: int | None = None,
        routes: tuple[str, ...] = ("gt", "s2f"),
    ):
        super().__init__()

        if conv_latent_channels is None:
            conv_latent_channels = base_channels * 4
        if stem_kernel_size is None:
            stem_kernel_size = encoder_kernel_size
        if down_kernel_size is None:
            down_kernel_size = encoder_kernel_size

        self.track_in_channels = int(track_in_channels)
        self.base_channels = int(base_channels)
        self.conv_latent_channels = int(conv_latent_channels)

        # Register a TrackEncoder per route key (e.g. "gt", "s2f").
        self.encoders = nn.ModuleDict()
        for name in routes:
            self.encoders[name] = TrackEncoder(
                in_channels=self.track_in_channels,
                base_channels=self.base_channels,
                latent_channels=self.conv_latent_channels,
                kernel_size=encoder_kernel_size,
                stem_kernel_size=stem_kernel_size,
                down_kernel_size=down_kernel_size,
            )

    def forward(
        self,
        track: torch.Tensor,
        route: str,
    ) -> torch.Tensor:
        """
        Args:
            track : (B, C_trk, L) — input tracks (GT, S2F, etc.).
            route : str, one of the keys passed at init time
                    (default: "gt" or "s2f").

        Returns:
            z_conv: (B, latent_channels, L_lat) conv-latent features.
        """
        if track.dim() != 3:
            raise ValueError(f"track must be (B, C_trk, L), got {tuple(track.shape)}")
        if route not in self.encoders:
            raise ValueError(f"Unknown route '{route}'. Available: {list(self.encoders.keys())}")
        return self.encoders[route](track)
