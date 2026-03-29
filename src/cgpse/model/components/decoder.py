import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_group_norm(num_channels: int, max_groups: int = 8) -> nn.GroupNorm:
    """
    Create a GroupNorm layer with a sensible number of groups for 1D convs.

    - Uses up to `max_groups` groups, but always divides `num_channels`.
    - Falls back to 1 group if needed.
    """
    groups = min(max_groups, num_channels)
    while groups > 1 and (num_channels % groups) != 0:
        groups -= 1
    return nn.GroupNorm(num_groups=groups, num_channels=num_channels)


class UpBlock1D(nn.Module):
    """
    Upsampling depthwise + pointwise conv block for 1D tracks.

    Input:  (B, C_in, L)
    Output: (B, C_out, L_up) with L_up ≈ 2 * L

    Structure:
      - Linear upsample by factor 2 along L.
      - DepthwiseConv1d(C_in -> C_in, k=kernel_size)
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

        self.upsample = nn.Upsample(
            scale_factor=2.0,
            mode="linear",
            align_corners=False,
        )

        self.depthwise = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=in_channels,
            bias=True,
        )
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
            (B, C_out, L_up)
        """
        x = self.upsample(x)
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class TrackDecoder(nn.Module):
    """
    1D convolutional decoder for 1kbp log-simplex tracks.

    Design:
      - Input: latent conv feature map at L_lat ≈ 125, channels = latent_channels
      - 3 upsampling blocks → spatial upsampling by 8.
      - Channel progression (for base_channels = 16, latent_channels = 64):
          latent -> latent
          latent -> 2*base
          2*base -> base
          base   -> C_out
      - No skip connections from encoder; all information flows through this
        decoder from the shared bottleneck representation.

    Shapes (for L_lat=125, base=16, latent=64):
      h_lat:    (B, 64, 125)
      up1:      (B, 64, 250)
      up2:      (B, 32, 500)
      up3:      (B, 16, 1000)
      head:     (B, C_out, 1000)

    Args:
        out_channels:    number of output track channels (e.g. 1 or 2).
        base_channels:   "base" width to mirror GTTrackEncoder.
        latent_channels: channels at the conv-latent resolution (must match encoder).
        kernel_size:     spatial kernel size for convs (must be odd).
    """

    def __init__(
        self,
        out_channels: int,
        base_channels: int = 16,
        latent_channels: int | None = None,
        kernel_size: int = 5,
    ):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be odd, got {kernel_size}")

        if latent_channels is None:
            latent_channels = base_channels * 4  # e.g., 64 when base=16

        self.out_channels = int(out_channels)
        self.base_channels = int(base_channels)
        self.latent_channels = int(latent_channels)

        # 3x upsampling: L_lat -> 2*L_lat -> 4*L_lat -> 8*L_lat (~1000 for 125)
        # Channels: latent -> latent -> 2*base -> base
        self.up1 = UpBlock1D(
            in_channels=self.latent_channels,
            out_channels=self.latent_channels,
            kernel_size=kernel_size,
        )
        self.up2 = UpBlock1D(
            in_channels=self.latent_channels,
            out_channels=base_channels * 2,
            kernel_size=kernel_size,
        )
        self.up3 = UpBlock1D(
            in_channels=base_channels * 2,
            out_channels=base_channels,
            kernel_size=kernel_size,
        )

        # Final 1x1 conv to project back to track channels (log-space)
        self.head = nn.Conv1d(
            in_channels=base_channels,
            out_channels=out_channels,
            kernel_size=1,
            bias=True,
        )

        self._init_head()

    def _init_head(self) -> None:
        # Small init so the decoder starts near-zero and learns corrections.
        with torch.no_grad():
            self.head.weight.mul_(0.01)
            if self.head.bias is not None:
                self.head.bias.zero_()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (B, latent_channels, L_lat)

        Returns:
            y_hat: (B, out_channels, L_out) — typically L_out ≈ 1000.
        """
        h = self.up1(z)
        h = self.up2(h)
        h = self.up3(h)
        y_hat = self.head(h)
        return y_hat
