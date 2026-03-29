import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List
import math

from .transformer import TransformerBlock


class Residual(nn.Module):
    """Residual connection wrapper"""
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class DownBlock(nn.Module):
    """
    Encoder block for UNet
    
    Structure: Conv1D → GroupNorm → GELU → Conv1D → GroupNorm → GELU → Downsample
    
    Args:
        in_channels: Input channels
        out_channels: Output channels  
        kernel_size: Convolution kernel size
        num_groups: Number of groups for GroupNorm (critical for mask robustness)
        downsample: Whether to downsample (stride=2)
        dropout: Dropout rate
    """
    
    def __init__(self, 
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 num_groups: int = 8,
                 downsample: bool = True,
                 dropout: float = 0.1):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.downsample = downsample
        
        # Ensure num_groups divides channels evenly
        self.num_groups_in = min(num_groups, in_channels)
        while in_channels % self.num_groups_in != 0:
            self.num_groups_in -= 1
            
        self.num_groups_out = min(num_groups, out_channels)
        while out_channels % self.num_groups_out != 0:
            self.num_groups_out -= 1
        
        # First conv block
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, 
            kernel_size=kernel_size, 
            stride=1, 
            padding=kernel_size//2  # equivalent to padding="same"
        )
        self.norm1 = nn.GroupNorm(self.num_groups_out, out_channels)
        self.act1 = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        
        # Second conv block
        self.conv2 = nn.Conv1d(
            out_channels, out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size//2
        )
        self.norm2 = nn.GroupNorm(self.num_groups_out, out_channels)
        self.act2 = nn.GELU()
        self.dropout2 = nn.Dropout(dropout)
        
        # Downsampling
        if downsample:
            self.downsample_layer = nn.MaxPool1d(kernel_size=2, stride=2)
        else:
            self.downsample_layer = nn.Identity()
            
        # Residual connection (if channel dimensions match)
        if in_channels == out_channels:
            self.residual_proj = nn.Identity()
        else:
            self.residual_proj = nn.Conv1d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        # Store input for residual
        identity = x
        
        # First conv block
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act1(out)
        out = self.dropout1(out)
        
        # Second conv block
        out = self.conv2(out)
        out = self.norm2(out)
        
        # Residual connection
        identity = self.residual_proj(identity)
        out = out + identity
        
        out = self.act2(out)
        out = self.dropout2(out)
        
        # Store skip connection before downsampling
        skip = out
        
        # Downsample
        out = self.downsample_layer(out)
        
        return out, skip


class UpBlock(nn.Module):
    """
    Decoder block for UNet
    
    Structure: Upsample → Conv1D → GroupNorm → GELU → Skip Connection → Conv1D → GroupNorm → GELU
    
    Args:
        in_channels: Input channels from previous layer
        skip_channels: Channels from skip connection
        out_channels: Output channels
        kernel_size: Convolution kernel size
        num_groups: Number of groups for GroupNorm
        dropout: Dropout rate
    """
    
    def __init__(self,
                 in_channels: int,
                 skip_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 num_groups: int = 8,
                 dropout: float = 0.1):
        super().__init__()
        
        self.in_channels = in_channels
        self.skip_channels = skip_channels
        self.out_channels = out_channels
        
        # Upsample
        self.upsample = nn.ConvTranspose1d(
            in_channels, in_channels,
            kernel_size=2, stride=2
        )
        
        # Combined channels after skip connection
        combined_channels = in_channels + skip_channels
        
        # Ensure num_groups divides channels evenly
        self.num_groups_combined = min(num_groups, combined_channels)
        while combined_channels % self.num_groups_combined != 0:
            self.num_groups_combined -= 1
            
        self.num_groups_out = min(num_groups, out_channels)
        while out_channels % self.num_groups_out != 0:
            self.num_groups_out -= 1
        
        # First conv after skip connection
        self.conv1 = nn.Conv1d(
            combined_channels, out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size//2
        )
        self.norm1 = nn.GroupNorm(self.num_groups_out, out_channels)
        self.act1 = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        
        # Second conv block
        self.conv2 = nn.Conv1d(
            out_channels, out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size//2
        )
        self.norm2 = nn.GroupNorm(self.num_groups_out, out_channels)
        self.act2 = nn.GELU()
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x, skip):
        # Upsample
        x = self.upsample(x)
        
        # Handle potential size mismatch with skip connection
        if x.size(-1) != skip.size(-1):
            # Pad or crop to match skip connection size
            diff = skip.size(-1) - x.size(-1)
            if diff > 0:
                x = F.pad(x, (diff//2, diff - diff//2))
            else:
                start = (-diff) // 2
                x = x[..., start:start + skip.size(-1)]
        
        # Concatenate with skip connection
        x = torch.cat([x, skip], dim=1)
        
        # First conv block
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act1(out)
        out = self.dropout1(out)
        
        # Second conv block
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.act2(out)
        out = self.dropout2(out)
        
        return out


class Bottleneck(nn.Module):
    """
    Transformer bottleneck with configurable attention
    
    Structure: LayerNorm → Attention → Residual → LayerNorm → FFN → Residual
    
    Args:
        dim: Model dimension (dim_head calculated automatically as dim // heads)
        num_layers: Number of transformer layers  
        heads: Number of attention heads
        ff_mult: FFN expansion multiplier
        dropout: Dropout rate
        use_gqa: Whether to use Grouped Query Attention
        use_flash: Whether to use FlashAttention
        rotary_emb_fraction: RoPE embedding fraction (of dim_head)
        rotary_emb_base: RoPE base frequency
        rotary_emb_scale_base: RoPE scale base
    """
    
    def __init__(self,
                 dim: int,
                 num_layers: int = 2,
                 heads: int = 8,
                 ff_mult: int = 4,
                 dropout: float = 0.1,
                 use_gqa: bool = True,
                 use_flash: bool = True,
                 rotary_emb_fraction: float = 0.5,
                 rotary_emb_base: float = 20000.0,
                 rotary_emb_scale_base: Optional[float] = None):
        super().__init__()
        
        self.dim = dim
        self.num_layers = num_layers
        
        # Build transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(
                dim=dim,
                heads=heads,
                ff_mult=ff_mult,
                dropout=dropout,
                use_gqa=use_gqa,
                use_flash=use_flash,
                rotary_emb_fraction=rotary_emb_fraction,
                rotary_emb_base=rotary_emb_base,
                rotary_emb_scale_base=rotary_emb_scale_base
            )
            for _ in range(num_layers)
        ])
        
        print(f"Initialized Bottleneck with {num_layers} layers, dim={dim}")
        if use_gqa:
            print(f"Using GQA + RoPE {rotary_emb_fraction*100}%", end=" ")
        else:
            print(f"Using RoPE {rotary_emb_fraction*100}%", end=" ")
        if use_flash:
            print(f"with FlashAttention backend")
        else:
            print("Using standard Attention backend")
    
    def forward(self, x):
        # x shape: (B, dim, L) -> need (B, L, dim) for attention
        x = x.transpose(1, 2)  # (B, L, dim)
        
        # Apply transformer layers
        for layer in self.layers:
            x = layer(x)
        
        # Convert back to conv format
        x = x.transpose(1, 2)  # (B, dim, L)
        
        return x


class FinalOutputBlock(nn.Module):
    """
    Final output block for reconstruction
    
    Structure: Conv1D → GroupNorm → GELU → Conv1D (no final activation)
    """
    
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 num_groups: int = 8,
                 dropout: float = 0.1):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Intermediate channels - ensure at least 2
        mid_channels = max(in_channels // 2, 1)
        
        # Ensure num_groups divides channels evenly
        self.num_groups_in = min(num_groups, in_channels)
        while in_channels % self.num_groups_in != 0:
            self.num_groups_in -= 1
            
        self.num_groups_mid = min(num_groups, mid_channels)
        while mid_channels % self.num_groups_mid != 0:
            self.num_groups_mid -= 1
        
        # First conv block
        self.conv1 = nn.Conv1d(
            in_channels, mid_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.norm1 = nn.GroupNorm(self.num_groups_mid, mid_channels)
        self.act1 = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        
        # Final projection (no activation for reconstruction)
        self.conv2 = nn.Conv1d(
            mid_channels, out_channels,
            kernel_size=1,
            stride=1
        )
    
    def forward(self, x):
        # First conv block
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act1(out)
        out = self.dropout1(out)
        
        # Final projection
        out = self.conv2(out)
        
        return out


def calculate_output_length(input_length: int, num_down_blocks: int) -> int:
    """Calculate the output length after downsampling and upsampling"""
    # Each down block halves the length
    length = input_length
    for _ in range(num_down_blocks):
        length = length // 2
    
    # Each up block doubles the length back
    for _ in range(num_down_blocks):
        length = length * 2
    
    return length


def get_channel_progression(input_channels: int, depths: List[int]) -> List[int]:
    """Get the channel progression for encoder and decoder"""
    encoder_channels = [input_channels] + depths
    decoder_channels = depths[::-1] + [input_channels]
    return encoder_channels, decoder_channels
