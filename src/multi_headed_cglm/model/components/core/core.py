import torch
import torch.nn as nn

from typing import List, Dict, Any

from .unet_helpers import DownBlock, Bottleneck, UpBlock


class UNetCore(nn.Module):
    """
    Shared UNet core operating *after* the first down block.

    Expected usage:
      - A regime-specific stem applies the masked 2D input conv and first
        DownBlock, producing features of shape (B, depths[0], L/2).
      - UNetCore then applies the remaining down blocks, bottleneck, and all
        but the final up block, returning features again of shape
        (B, depths[0], L/2).

    The final up block that consumes the stem skip and core output is expected
    to live in the head/critic wrapper.
    """

    def __init__(
        self,
        depths: List[int],
        *,
        num_groups: int = 8,
        bottleneck_config: Dict[str, Any] | None = None,
        conv_kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()

        if len(depths) < 1:
            raise ValueError("UNetCore requires depths with at least one element.")

        self.depths = list(depths)
        self.num_groups = int(num_groups)

        # Core operates on all depths beyond the first one from the stem.
        self.num_core_blocks = max(len(self.depths) - 1, 0)

        # Additional down blocks (beyond the stem)
        self.encoder_core = nn.ModuleList()
        for i in range(self.num_core_blocks):
            in_ch = self.depths[i]
            out_ch = self.depths[i + 1]
            self.encoder_core.append(
                DownBlock(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=conv_kernel_size,
                    num_groups=self.num_groups,
                    downsample=True,
                    dropout=dropout,
                )
            )

        # Bottleneck (transformer) at deepest depth
        if bottleneck_config is None:
            bottleneck_config = {}
        default_config = {
            "dim": self.depths[-1],
            "num_layers": 2,
            "heads": 8,
            "ff_mult": 4,
            "dropout": 0.1,
            "use_gqa": True,
            "use_flash": True,
            "rotary_emb_fraction": 0.5,
            "rotary_emb_base": 20000.0,
            "rotary_emb_scale_base": None,
        }
        default_config.update(bottleneck_config)
        self.bottleneck = Bottleneck(**default_config)

        # Up blocks within the core (all but the final one)
        self.decoder_core = nn.ModuleList()
        for k in range(self.num_core_blocks):
            # Mirror the deepest-to-shallower path, excluding the final up block
            in_ch = self.depths[len(self.depths) - k - 1]
            skip_ch = in_ch
            out_ch = self.depths[len(self.depths) - k - 2]
            self.decoder_core.append(
                UpBlock(
                    in_channels=in_ch,
                    skip_channels=skip_ch,
                    out_channels=out_ch,
                    kernel_size=conv_kernel_size,
                    num_groups=self.num_groups,
                    dropout=dropout,
                )
            )

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Conv1d, nn.ConvTranspose1d)):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.GroupNorm, nn.LayerNorm)):
            if getattr(module, "weight", None) is not None:
                nn.init.ones_(module.weight)
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor):
        """
        Forward through the shared core.

        Parameters
        ----------
        x
            Tensor of shape (B, depths[0], L/2) coming from the stem.

        Returns
        -------
        Tensor of shape (B, depths[0], L/2) to be consumed by the final up block.
        """
        if x.dim() != 3:
            raise ValueError(f"UNetCore expects input of shape (B, C, L), got {tuple(x.shape)}")
        if x.size(1) != self.depths[0]:
            raise ValueError(
                f"UNetCore expected channel dim {self.depths[0]} from stem, got {x.size(1)}"
            )

        # Down path within the core
        skips: list[torch.Tensor] = []
        for down_block in self.encoder_core:
            x, skip = down_block(x)
            skips.append(skip)

        # Bottleneck
        x = self.bottleneck(x)

        # Up path within the core (mirror core skips)
        skips = skips[::-1]
        for up_block, skip in zip(self.decoder_core, skips):
            x = up_block(x, skip)

        return x

