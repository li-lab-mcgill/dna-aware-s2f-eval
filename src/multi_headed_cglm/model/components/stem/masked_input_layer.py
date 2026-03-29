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

