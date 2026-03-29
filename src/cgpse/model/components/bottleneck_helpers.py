import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------

class ChannelwiseLayerNorm1d(nn.Module):
    """
    LayerNorm over channels for each token position on (B, C, L):
      x: (B, C, L) -> transpose to (B, L, C) -> LN(C) -> back to (B, C, L)
    """
    def __init__(self, num_channels: int, elementwise_affine: bool = True, eps: float = 1e-5):
        super().__init__()
        self.ln = nn.LayerNorm(num_channels, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ln(x.transpose(1, 2)).transpose(1, 2)


def _make_group_norm(num_channels: int, max_groups: int = 8) -> nn.GroupNorm:
    """
    Create a GroupNorm layer with a sensible number of groups for 1D convs.

    - Uses up to `max_groups` groups, but always divides `num_channels`.
    - Falls back to 1 group (LayerNorm-like over channels) if needed.
    """
    groups = min(max_groups, num_channels)
    while groups > 1 and (num_channels % groups) != 0:
        groups -= 1
    return nn.GroupNorm(num_groups=groups, num_channels=num_channels)


# ---------------------------------------------------------------------------
# SwiGLU projector (generic, channel-only)
# ---------------------------------------------------------------------------

class SwiGLUProjector(nn.Module):
    """
    Lightweight channel-only SwiGLU + residual projector using a single 1x1 conv.

    Structure (x: (B, C_in, L)):
      1. Optional ChannelwiseLayerNorm1d on the inputs.
      2. 1x1 Conv projecting to (2 or 3) * C_out.
         - first C_out: gate a
         - second C_out: value b
         - optional third C_out: residual contribution c
      3. SwiGLU: SiLU(a) * b + c (if residual enabled).
      4. Optional dropout.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_layernorm: bool = False,
        use_residual: bool = True,
    ):
        super().__init__()
        if out_channels <= 0:
            raise ValueError(f"out_channels must be positive, got {out_channels}")

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.use_residual = bool(use_residual)

        proj_channels = 2 * self.out_channels + (self.out_channels if self.use_residual else 0)

        self.ln = ChannelwiseLayerNorm1d(self.in_channels) if use_layernorm else nn.Identity()
        self.proj = nn.Conv1d(self.in_channels, proj_channels, kernel_size=1, stride=1)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.kaiming_normal_(self.proj.weight, nonlinearity="linear")
        with torch.no_grad():
            H = self.out_channels
            self.proj.weight[:H].mul_(2 ** -0.5)
            self.proj.weight[H:2 * H].mul_(2 ** -0.5)
            if self.proj.bias is not None:
                self.proj.bias[:H].fill_(-0.5)
                self.proj.bias[H:2 * H].zero_()
                if self.use_residual:
                    self.proj.bias[2 * H:].zero_()

            if self.use_residual:
                w = self.proj.weight[2 * H:]     # (C_out, C_in, 1)
                w.zero_()
                diag = min(self.in_channels, self.out_channels)
                eye = torch.eye(diag, dtype=w.dtype, device=w.device)
                w[:diag, :diag, 0].copy_(eye)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C_in, L)

        Returns:
            (B, C_out, L)
        """
        y = self.ln(x)
        proj = self.proj(y)

        if self.use_residual:
            a, b, c = torch.split(proj, self.out_channels, dim=1)
            out = F.silu(a) * b + c
        else:
            a, b = torch.chunk(proj, 2, dim=1)
            out = F.silu(a) * b

        return out


# ---------------------------------------------------------------------------
# Dilated conv block for medium-range mixing at latent resolution
# ---------------------------------------------------------------------------

class DilatedConvBlock1D(nn.Module):
    """
    Simple residual dilated conv block at fixed spatial resolution (L_lat).

    Structure:
      x: (B, C, L)
        -> depthwise Conv1d(C->C, k, dilation)
        -> pointwise Conv1d(C->C)
        -> GroupNorm(C)
        -> SiLU
        -> + residual
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 5,
        dilation: int = 1,
    ):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be odd, got {kernel_size}")
        self.channels = channels
        self.kernel_size = kernel_size
        self.dilation = dilation

        padding = (kernel_size // 2) * dilation

        self.depthwise = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            groups=channels,
            bias=True,
        )
        self.pointwise = nn.Conv1d(
            channels,
            channels,
            kernel_size=1,
            bias=True,
        )

        self.norm = _make_group_norm(channels)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, L)

        Returns:
            (B, C, L)
        """
        residual = x
        h = self.depthwise(x)
        h = self.pointwise(h)
        h = self.norm(h)
        h = self.act(h)
        return residual + h


# ---------------------------------------------------------------------------
# Multi-head self-attention with optional GQA and RoPE on a subset of dims
# ---------------------------------------------------------------------------

def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Helper for RoPE: split last dim into pairs and rotate.
    """
    x = x.view(*x.shape[:-1], x.shape[-1] // 2, 2)
    x1, x2 = x[..., 0], x[..., 1]
    # (-x2, x1)
    return torch.stack([-x2, x1], dim=-1).reshape(*x.shape[:-2], x.shape[-2] * 2)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    seq_len: int,
    rotary_dim: int,
    rope_theta: float = 10000.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply RoPE to the first `rotary_dim` of the last dimension of q, k.

    q, k: (B, n_heads, L, dim_head)
    rotary_dim: must be <= dim_head and even.
    """
    if rotary_dim == 0:
        return q, k

    dim_head = q.shape[-1]
    if rotary_dim > dim_head:
        raise ValueError(f"rotary_dim {rotary_dim} > dim_head {dim_head}")
    if rotary_dim % 2 != 0:
        raise ValueError(f"rotary_dim must be even, got {rotary_dim}")

    # Positions
    device = q.device
    dtype = q.dtype
    positions = torch.arange(seq_len, device=device, dtype=dtype)  # (L,)

    # Frequencies
    # Standard RoPE: base^{ -2i/d } for i in [0, rotary_dim/2)
    half_dim = rotary_dim // 2
    freqs = torch.arange(half_dim, device=device, dtype=dtype)
    inv_freq = rope_theta ** (-2 * freqs / rotary_dim)  # (half_dim,)

    # (L, half_dim)
    sinusoid_inp = torch.einsum("n,d->nd", positions, inv_freq)
    sin = sinusoid_inp.sin()[None, None, :, :]  # (1,1,L,half_dim)
    cos = sinusoid_inp.cos()[None, None, :, :]  # (1,1,L,half_dim)

    # Expand to full rotary_dim via stacking pairs
    sin = sin.unsqueeze(-1)  # (1,1,L,half_dim,1)
    cos = cos.unsqueeze(-1)  # (1,1,L,half_dim,1)
    sin = sin.repeat(1, 1, 1, 1, 2).reshape(1, 1, seq_len, rotary_dim)
    cos = cos.repeat(1, 1, 1, 1, 2).reshape(1, 1, seq_len, rotary_dim)

    # Split q,k into rotary and pass-through parts
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

    q_rot = (q_rot * cos) + (_rotate_half(q_rot) * sin)
    k_rot = (k_rot * cos) + (_rotate_half(k_rot) * sin)

    q = torch.cat([q_rot, q_pass], dim=-1)
    k = torch.cat([k_rot, k_pass], dim=-1)
    return q, k


class MultiheadSelfAttention1D(nn.Module):
    """
    Multi-head self-attention operating on (B, C, L) with:

      - Optional RoPE applied to a subset of the head dimension.
      - Optional GQA (Grouped Query Attention) via separate kv_heads.

    Inputs:
      x: (B, C, L)

    Outputs:
      y: (B, C, L)
    """

    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: Optional[int] = None,
        kv_heads: Optional[int] = None,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
        rope_fraction: float = 0.0,
        rope_theta: float = 10000.0,
    ):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.kv_heads = kv_heads if kv_heads is not None else heads
        self.dim_head = dim // heads if dim_head is None else dim_head

        if self.dim_head * heads != dim:
            raise ValueError(
                f"dim must be divisible by heads; got dim={dim}, heads={heads}, dim_head={self.dim_head}"
            )

        if self.kv_heads <= 0 or self.kv_heads > self.heads:
            raise ValueError(
                f"kv_heads must be in [1, heads]; got kv_heads={self.kv_heads}, heads={self.heads}"
            )

        # How many dims of the head to apply RoPE to (must be even)
        rotary_dim = int(self.dim_head * rope_fraction)
        if rotary_dim % 2 == 1:
            rotary_dim -= 1
        self.rotary_dim = max(rotary_dim, 0)
        self.rope_theta = rope_theta

        # Per-head scaling
        self.scale = self.dim_head ** -0.5

        # Projection layers
        self.to_q = nn.Linear(dim, heads * self.dim_head, bias=False)
        self.to_kv = nn.Linear(dim, self.kv_heads * self.dim_head * 2, bias=False)
        self.to_out = nn.Linear(heads * self.dim_head, dim, bias=False)

        self.attn_dropout = nn.Dropout(attn_dropout) if attn_dropout > 0.0 else nn.Identity()
        self.proj_dropout = nn.Dropout(proj_dropout) if proj_dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, L)

        Returns:
            out: (B, C, L)
        """
        B, C, L = x.shape
        # Work in (B, L, C) for linear layers
        x_t = x.transpose(1, 2)  # (B, L, C)

        # Project to q, kv
        q = self.to_q(x_t)  # (B, L, heads*dim_head)
        kv = self.to_kv(x_t)  # (B, L, kv_heads*dim_head*2)

        # Reshape
        q = q.view(B, L, self.heads, self.dim_head).transpose(1, 2)  # (B, heads, L, dim_head)
        kv = kv.view(B, L, self.kv_heads, 2 * self.dim_head).transpose(1, 2)  # (B, kv_heads, L, 2*dim_head)
        k, v = torch.chunk(kv, 2, dim=-1)  # each (B, kv_heads, L, dim_head)

        # Apply RoPE to subset of dimensions (q,k only)
        if self.rotary_dim > 0:
            q, k = apply_rotary_pos_emb(
                q,
                k,
                seq_len=L,
                rotary_dim=self.rotary_dim,
                rope_theta=self.rope_theta,
            )

        # If kv_heads < heads, repeat k, v to match q heads (GQA-style)
        if self.kv_heads != self.heads:
            repeat_factor = self.heads // self.kv_heads
            if repeat_factor * self.kv_heads != self.heads:
                raise ValueError(
                    f"heads ({self.heads}) must be a multiple of kv_heads ({self.kv_heads})"
                    " for grouped-query attention."
                )
            k = k.repeat_interleave(repeat_factor, dim=1)
            v = v.repeat_interleave(repeat_factor, dim=1)

        # Scaled dot-product attention (no Flash)
        q = q * self.scale
        attn = torch.matmul(q, k.transpose(-2, -1))  # (B, heads, L, L)
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        # Apply attention to values
        out = torch.matmul(attn, v)  # (B, heads, L, dim_head)

        # Merge heads and project out
        out = out.transpose(1, 2).contiguous().view(B, L, self.heads * self.dim_head)  # (B, L, dim)
        out = self.to_out(out)  # (B, L, dim)
        out = self.proj_dropout(out)

        # Back to (B, C, L)
        return out.transpose(1, 2)


# ---------------------------------------------------------------------------
# Transformer block and stack for bottleneck
# ---------------------------------------------------------------------------

class TransformerBlock1D(nn.Module):
    """
    Pre-LN Transformer block operating on (B, C, L):

      x -> x + Attn(LN(x))
        -> x + SwiGLUProjector(LN(x))

    The attention is MultiheadSelfAttention1D, and the FFN is a SwiGLUProjector
    in channel-space (no spatial mixing inside the FFN).
    """

    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: Optional[int] = None,
        kv_heads: Optional[int] = None,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
        ff_mult: float = 4.0,
        ff_dropout: float = 0.0,
        rope_fraction: float = 0.0,
        rope_theta: float = 10000.0,
    ):
        super().__init__()
        self.dim = dim

        self.norm1 = ChannelwiseLayerNorm1d(dim)
        self.norm2 = ChannelwiseLayerNorm1d(dim)

        self.attn = MultiheadSelfAttention1D(
            dim=dim,
            heads=heads,
            dim_head=dim_head,
            kv_heads=kv_heads,
            attn_dropout=attn_dropout,
            proj_dropout=proj_dropout,
            rope_fraction=rope_fraction,
            rope_theta=rope_theta,
        )

        self.ff = SwiGLUProjector(
            in_channels=dim,
            out_channels=dim,
            use_layernorm=False,  # we already do LN outside in norm2
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, L)

        Returns:
            (B, C, L)
        """
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x


class TransformerBottleneck1D(nn.Module):
    """
    Stack of TransformerBlock1D at the latent resolution (L_lat).

    This is the global mixer for the GT manifold and later shared with S2F.
    """

    def __init__(
        self,
        dim: int,
        depth: int = 1,
        heads: int = 8,
        dim_head: Optional[int] = None,
        kv_heads: Optional[int] = None,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
        ff_mult: float = 4.0,
        ff_dropout: float = 0.0,
        rope_fraction: float = 0.0,
        rope_theta: float = 10000.0,
    ):
        super().__init__()
        blocks = []
        for _ in range(depth):
            blocks.append(
                TransformerBlock1D(
                    dim=dim,
                    heads=heads,
                    dim_head=dim_head,
                    kv_heads=kv_heads,
                    attn_dropout=attn_dropout,
                    proj_dropout=proj_dropout,
                    ff_mult=ff_mult,
                    ff_dropout=ff_dropout,
                    rope_fraction=rope_fraction,
                    rope_theta=rope_theta,
                )
            )
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, L)

        Returns:
            (B, C, L)
        """
        return self.blocks(x)
