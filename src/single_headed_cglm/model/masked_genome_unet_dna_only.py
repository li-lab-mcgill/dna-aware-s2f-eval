import torch
import torch.nn as nn

from typing import List, Optional, Dict, Any, Union, Tuple

from .unet.unet_helpers import (
    DownBlock, Bottleneck, UpBlock, FinalOutputBlock,
    calculate_output_length, get_channel_progression
)


class MaskedGenomeUNet(nn.Module):
    """
    UNet for DNA-only masked genome language model

    Architecture:
    Input: (B, L, 4) -> InputConv1D -> (B, input_channels, L)
    Encoder: DownBlocks with skip connections
    Bottleneck: Transformer with RoPE + GQA
    Decoder: UpBlocks with skip connections
    Output: (B, L, 4) - reconstructed values (no mask channel)
        For DNA, it is the logits of a softmax over the 4 DNA bases, for language modelling

    Key Features:
    - GroupNorm throughout for mask pattern robustness
    - Transformer bottleneck with RoPE and GQA
    - Scalable architecture via depth configuration
    - Skip connections for fine-grained reconstruction
    """

    def __init__(self,
                 T: int,                           # number of functional genomic tracks + DNA one-hot
                 depths: List[int],                # encoder channel progression excluding input conv
                 # input conv
                 input_kernel_size: int = 23,      # InputConv2D kernel size
                 input_conv_channels: int = 128,   # channels from InputConv2D
                 # norm
                 num_groups: int = 8,              # GroupNorm groups (critical for masks)
                 # bottleneck transformer
                 bottleneck_config: Dict[str, Any] = None,
                 # encoder/decoder convs
                 conv_kernel_size: int = 3,        # UNet conv kernel size
                 dropout: float = 0.1,             # dropout rate
                 padding_mode: str = 'same'        # InputConv2D padding
    ):
        super().__init__()

        self.T = T
        self.input_conv_channels = input_conv_channels
        self.depths = depths
        self.num_groups = num_groups
        self.num_down_blocks = len(depths)

        # Default bottleneck config
        if bottleneck_config is None:
            bottleneck_config = {}

        default_config = {
            'num_layers': 2,
            'heads': 8,
            'ff_mult': 4,
            'dropout': 0.1,
            'use_gqa': True,
            'use_flash': True,
            'rotary_emb_fraction': 0.5,
            'rotary_emb_base': 20000.0,
            'rotary_emb_scale_base': None
        }
        default_config.update(bottleneck_config)
        self.bottleneck_config = default_config

        # Input convolution layer (DNA-only: 4 channels)
        self.input_conv = nn.Conv1d(
            in_channels=4,
            out_channels=input_conv_channels,
            kernel_size=input_kernel_size,
            padding=padding_mode
        )

        # Calculate channel progressions
        encoder_channels, decoder_channels = get_channel_progression(
            input_conv_channels, depths
        )

        # Encoder (downsampling path)
        self.encoder = nn.ModuleList()
        for i in range(self.num_down_blocks):
            in_ch = encoder_channels[i]
            out_ch = encoder_channels[i + 1]

            down_block = DownBlock(
                in_channels=in_ch,
                out_channels=out_ch,
                kernel_size=conv_kernel_size,
                num_groups=num_groups,
                downsample=True,
                dropout=dropout
            )
            self.encoder.append(down_block)

        # Bottleneck (transformer with RoPE + GQA)
        self.bottleneck_config['dim'] = depths[-1]  # deepest encoder dimension
        self.bottleneck = Bottleneck(**self.bottleneck_config)

        # Decoder (upsampling path)
        self.decoder = nn.ModuleList()
        for i in range(self.num_down_blocks):
            # Decoder mirrors encoder in reverse
            in_ch = depths[-(i+1)]           # current decoder input
            skip_ch = depths[-(i+1)]         # skip connection channels
            out_ch = decoder_channels[i+1]   # decoder output

            up_block = UpBlock(
                in_channels=in_ch,
                skip_channels=skip_ch,
                out_channels=out_ch,
                kernel_size=conv_kernel_size,
                num_groups=num_groups,
                dropout=dropout
            )
            self.decoder.append(up_block)

        # Final output layer (reconstruction)
        self.final_conv = FinalOutputBlock(
            in_channels=input_conv_channels,
            out_channels=T,  # T functional tracks + DNA one-hot
            num_groups=num_groups,
            dropout=dropout
        )

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize model weights"""
        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.ConvTranspose1d)):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.GroupNorm, nn.LayerNorm)):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self,
                x: torch.Tensor,
                return_bottleneck_representations: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass

        Args:
            x: Input tensor of shape (B, L, 4) - DNA one-hot encoded
            return_bottleneck_representations: Whether to return bottleneck representations

        Returns:
            If return_bottleneck_representations=False: output tensor (B, L, T)
            If return_bottleneck_representations=True: (output, bottleneck_repr)
                - bottleneck_repr: (B, timesteps, C) where timesteps = num_layers + 1
        """

        # Input convolution: (B, L, 4) -> (B, input_conv_channels, L)
        x = self.input_conv(x.transpose(1, 2))

        # Store skip connections
        skip_connections = []

        # Encoder path
        for down_block in self.encoder:
            x, skip = down_block(x) # (B,C,L)
            skip_connections.append(skip)

        # Bottleneck (transformer)
        if return_bottleneck_representations:
            representations = []
            # Store encoder output (timestep 0)
            representations.append(x.transpose(1, 2).mean(dim=1))  # (B, C)

            # Apply transformer layers and collect
            x_attn = x.transpose(1, 2)  # (B, L, C)
            for layer in self.bottleneck.layers:
                x_attn = layer(x_attn)
                representations.append(x_attn.mean(dim=1))  # (B, C)

            x = x_attn.transpose(1, 2)  # Back to (B, C, L)
            bottleneck_repr = torch.stack(representations, dim=1)  # (B, timesteps, C)
        else:
            x = self.bottleneck(x)

        # Decoder path (reverse order of skip connections)
        skip_connections = skip_connections[::-1]
        for up_block, skip in zip(self.decoder, skip_connections):
            x = up_block(x, skip)

        # Final reconstruction: (B, input_conv_channels, L) -> (B, T, L)
        x = self.final_conv(x)
        x = x.transpose(1, 2)  # (B, L, T)

        if return_bottleneck_representations:
            return x, bottleneck_repr
        else:
            return x

    def get_model_info(self) -> Dict[str, Any]:
        """Get model architecture information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            'model_name': 'MaskedGenomeUNet',
            'input_tracks': self.T,
            'input_conv_channels': self.input_conv_channels,
            'encoder_depths': self.depths,
            'num_down_blocks': self.num_down_blocks,
            'num_groups': self.num_groups,
            'bottleneck_config': self.bottleneck_config,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'architecture': self._get_architecture_summary()
        }

    def _get_architecture_summary(self) -> str:
        """Get a string summary of the architecture"""
        summary = []
        summary.append(f"Input: (B, L, 4)")
        summary.append(f"InputConv1D: -> (B, {self.input_conv_channels}, L)")

        # Encoder
        channels = [self.input_conv_channels] + self.depths
        for i, (in_ch, out_ch) in enumerate(zip(channels[:-1], channels[1:])):
            summary.append(f"DownBlock{i+1}: (B, {in_ch}, L/{2**i}) -> (B, {out_ch}, L/{2**(i+1)})")

        # Bottleneck
        summary.append(f"Bottleneck: (B, {self.depths[-1]}, L/{2**self.num_down_blocks}) [{self.bottleneck_config['num_layers']} Transformer layers]")

        # Decoder
        decoder_channels = self.depths[::-1] + [self.input_conv_channels]
        for i, (in_ch, out_ch) in enumerate(zip(decoder_channels[:-1], decoder_channels[1:])):
            scale = 2**(self.num_down_blocks - i - 1)
            summary.append(f"UpBlock{i+1}: (B, {in_ch}, L/{scale*2}) -> (B, {out_ch}, L/{scale})")

        summary.append(f"FinalConv: (B, {self.input_conv_channels}, L) -> (B, L, {self.T})")

        return "\n".join(summary)

    @classmethod
    def create_mini_model(cls, T: int = 4, **kwargs) -> 'MaskedGenomeUNet':
        """Create a mini UNet model"""
        default_bottleneck_config = {
            'heads': 8,
            'dim_head': 64,
            'ff_mult': 4,
            'dropout': 0.1,
            'use_gqa': True,
            'rotary_emb_dim': 32
        }

        # Merge with any provided bottleneck_config
        if 'bottleneck_config' in kwargs:
            default_bottleneck_config.update(kwargs['bottleneck_config'])
        kwargs['bottleneck_config'] = default_bottleneck_config

        return cls(
            T=T,
            input_conv_channels=128,
            depths=[256, 512],
            bottleneck_layers=2,
            **kwargs
        )

    @classmethod
    def create_medium_model(cls, T: int = 4, **kwargs) -> 'MaskedGenomeUNet':
        """Create a medium UNet model"""
        default_bottleneck_config = {
            'heads': 12,
            'dim_head': 64,
            'ff_mult': 4,
            'dropout': 0.1,
            'use_gqa': True,
            'rotary_emb_dim': 32
        }

        if 'bottleneck_config' in kwargs:
            default_bottleneck_config.update(kwargs['bottleneck_config'])
        kwargs['bottleneck_config'] = default_bottleneck_config

        return cls(
            T=T,
            input_conv_channels=128,
            depths=[256, 512, 1024],
            bottleneck_layers=4,
            **kwargs
        )

    @classmethod
    def create_large_model(cls, T: int = 4, **kwargs) -> 'MaskedGenomeUNet':
        """Create a large UNet model"""
        default_bottleneck_config = {
            'heads': 16,
            'dim_head': 64,
            'ff_mult': 4,
            'dropout': 0.1,
            'use_gqa': True,
            'rotary_emb_dim': 32
        }

        if 'bottleneck_config' in kwargs:
            default_bottleneck_config.update(kwargs['bottleneck_config'])
        kwargs['bottleneck_config'] = default_bottleneck_config

        return cls(
            T=T,
            input_conv_channels=256,
            depths=[512, 1024, 2048],
            bottleneck_layers=6,
            **kwargs
        )
