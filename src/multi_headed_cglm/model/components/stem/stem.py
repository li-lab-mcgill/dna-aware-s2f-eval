import torch
import torch.nn as nn

from typing import Tuple

from .masked_input_layer import MaskedInputConv2DLayer
from ..core.unet_helpers import DownBlock


class MaskConditionalStem(nn.Module):
    """
    Regime-specific stem: masked input convolution + first down block.

    This mirrors the initial part of the MaskConditionalUNet:
      (B, 2, L, T) --MaskedInputConv2DLayer--> (B, C_in, L)
                --DownBlock--> (B, C_out, L/2) with a skip at L/2.

    The stem is responsible for:
      - handling the 2D masked input convolution over (mask, value),
      - performing the first encoder down block,
      - returning both the downsampled features and the high-res skip
        that will later be used by the final up block.
    """

    def __init__(
        self,
        T: int,
        first_depth: int,
        *,
        input_kernel_size: int = 23,
        input_conv_channels: int = 128,
        num_groups: int = 8,
        conv_kernel_size: int = 3,
        dropout: float = 0.1,
        padding_mode: str = "same",
        mask_aware_padding: bool = False,
    ):
        super().__init__()

        self.T = int(T)
        self.first_depth = int(first_depth)
        self.input_conv_channels = int(input_conv_channels)

        # Always use masked input convolution here (hard design choice).
        self.input_conv = MaskedInputConv2DLayer(
            T=self.T,
            C=self.input_conv_channels,
            k=input_kernel_size,
            padding_mode=padding_mode,
            mask_aware_padding=mask_aware_padding,
        )

        # First encoder block (downsample=True) as in the UNet backbone.
        self.down_block = DownBlock(
            in_channels=self.input_conv_channels,
            out_channels=self.first_depth,
            kernel_size=conv_kernel_size,
            num_groups=num_groups,
            downsample=True,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x
            Input tensor of shape (B, 2, L, T):
              - channel 0: value
              - channel 1: mask

        Returns
        -------
        features
            Downsampled stem features of shape (B, first_depth, L/2).
        skip
            Skip tensor from the first down block (B, first_depth, L/2),
            used later by the final up block.
        """
        if x.dim() != 4:
            raise ValueError(f"Expected input of shape (B, 2, L, T), got {tuple(x.shape)}")
        if x.size(1) != 2:
            raise ValueError(f"Expected 2 channels (value+mask) on dim=1, got {x.size(1)}")
        if x.size(3) != self.T:
            raise ValueError(f"Expected {self.T} modalities on dim=3, got {x.size(3)}")

        # (B, 2, L, T) -> (B, C_in, L)
        x = self.input_conv(x)

        # First down block: returns (features, skip)
        features, skip = self.down_block(x)
        return features, skip
