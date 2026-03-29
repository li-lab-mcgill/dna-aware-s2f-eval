import torch
import torch.nn as nn


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
                 padding_mode='same'):  # 'same' or 'valid'
        super().__init__()

        self.T = T
        self.C = C
        self.k = k
        self.padding_mode = padding_mode

        # Calculate padding - only allowed in sequence (height) dimension
        if padding_mode == 'same':
            pad_L = k // 2
        elif padding_mode == 'valid':
            pad_L = 0
        else:
            raise ValueError(f"padding_mode must be 'same' or 'valid', got {padding_mode}")

        # Conv2D: (B, 2, L, T) -> (B, C, L', 1) -> (B, C, L')
        # Treats mask/value as channels - more natural and efficient
        self.conv = nn.Conv2d(
            in_channels=2,                # mask and value channels (hardcoded)
            out_channels=C,
            kernel_size=(k, T),           # (sequence, modalities)
            stride=(1, 1),                # stride only effective in sequence dimension
            padding=(pad_L, 0)            # only pad sequence dimension
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
                f"padding_mode='{self.padding_mode}')\n"
                f"  Input shape: (B, 2, L, {self.T})\n"
                f"  Output shape: (B, {self.C}, L')")
