import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Dict, Any

from rotary_embedding_torch import RotaryEmbedding


class Attention(nn.Module):
    """
    Multi-Head Attention with RoPE and optional GQA (Grouped Query Attention)

    Equivalent to FlashAttention but using standard PyTorch operations
    """

    def __init__(self,
                 dim: int,
                 heads: int = 8,
                 dropout: float = 0.1,
                 use_gqa: bool = True,
                 rotary_emb_fraction: float = 0.5):  # Fraction of dim_head to use for RoPE
        super().__init__()

        self.heads = heads
        self.dim_head = dim // heads
        self.use_gqa = use_gqa

        # Calculate rotary embedding dimension as fraction of dim_head
        self.rotary_emb_dim = int(self.dim_head * rotary_emb_fraction)
        self.use_rope = self.rotary_emb_dim > 0

        # GQA: fewer key/value heads than query heads
        if use_gqa:
            self.kv_heads = self.heads // 2
        else:
            self.kv_heads = self.heads

        self.scale = self.dim_head ** -0.5

        # Q/K/V projections
        self.to_q = nn.Linear(dim, self.heads * self.dim_head, bias=False)
        self.to_k = nn.Linear(dim, self.kv_heads * self.dim_head, bias=False)
        self.to_v = nn.Linear(dim, self.kv_heads * self.dim_head, bias=False)

        # Output projection
        self.to_out = nn.Linear(self.heads * self.dim_head, dim, bias=True)

        # RoPE for positional encoding (only if rotary_emb_dim > 0)
        if self.use_rope:
            self.rotary_emb = RotaryEmbedding(dim=self.rotary_emb_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Initialize weights (similar to FlashAttention)
        nn.init.xavier_uniform_(self.to_q.weight)
        nn.init.xavier_uniform_(self.to_k.weight)
        nn.init.xavier_uniform_(self.to_v.weight)
        nn.init.zeros_(self.to_out.weight)
        nn.init.zeros_(self.to_out.bias)

    def forward(self, x):
        b, n, _ = x.shape

        # Project to Q, K, V
        q = self.to_q(x)  # (B, N, heads * dim_head)
        k = self.to_k(x)  # (B, N, kv_heads * dim_head)
        v = self.to_v(x)  # (B, N, kv_heads * dim_head)

        # Reshape for multi-head attention
        q = q.view(b, n, self.heads, self.dim_head).transpose(1, 2)      # (B, heads, N, dim_head)
        k = k.view(b, n, self.kv_heads, self.dim_head).transpose(1, 2)   # (B, kv_heads, N, dim_head)
        v = v.view(b, n, self.kv_heads, self.dim_head).transpose(1, 2)   # (B, kv_heads, N, dim_head)

        # Apply RoPE (only if enabled and only to the rotary portion)
        if self.use_rope:
            # Split q, k into rotary and non-rotary parts
            q_rot, q_pass = q[..., :self.rotary_emb_dim], q[..., self.rotary_emb_dim:]
            k_rot, k_pass = k[..., :self.rotary_emb_dim], k[..., self.rotary_emb_dim:]

            # Apply RoPE to rotary parts
            q_rot = self.rotary_emb.rotate_queries_or_keys(q_rot)
            k_rot = self.rotary_emb.rotate_queries_or_keys(k_rot)

            # Concatenate back
            q = torch.cat([q_rot, q_pass], dim=-1)
            k = torch.cat([k_rot, k_pass], dim=-1)

        # Handle GQA: repeat k, v to match q heads if needed
        if self.use_gqa and self.kv_heads != self.heads:
            k = k.repeat_interleave(self.heads // self.kv_heads, dim=1)
            v = v.repeat_interleave(self.heads // self.kv_heads, dim=1)

        # Scaled dot-product attention
        q = q * self.scale
        attn = torch.matmul(q, k.transpose(-2, -1))  # (B, heads, N, N)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        out = torch.matmul(attn, v)  # (B, heads, N, dim_head)

        # Reshape and project output
        out = out.transpose(1, 2).contiguous().view(b, n, self.heads * self.dim_head)
        out = self.to_out(out)

        return out


class FlashAttention(nn.Module):
    """
    FlashAttention wrapper with GQA and RoPE
    """
    def __init__(self,
                 dim: int = 1536,
                 heads: int = 8,
                 dropout: float = 0.15,
                 rotary_emb_fraction: float = 0.5,  # Fraction of dim_head to use for RoPE
                 rotary_emb_base: float = 20000.0,
                 rotary_emb_scale_base: Optional[float] = None):
        super().__init__()

        # Calculate rotary embedding dimension as fraction of dim_head
        dim_head = dim // heads
        rotary_emb_dim = int(dim_head * rotary_emb_fraction)

        try:
            from flash_attn.modules.mha import MHA
        except ImportError:
            raise ImportError("flash_attn is not installed. Use Attention instead.")

        self.mha = MHA(
            use_flash_attn=True,
            embed_dim=dim,
            num_heads=heads,
            num_heads_kv=(heads//2),  # GQA: half the KV heads
            qkv_proj_bias=True,
            out_proj_bias=True,
            dropout=dropout,
            softmax_scale=(dim/heads) ** -0.5,
            causal=False,
            rotary_emb_dim=rotary_emb_dim,  # Now proportional to dim_head
            rotary_emb_base=rotary_emb_base,
            rotary_emb_scale_base=rotary_emb_scale_base,
            fused_bias_fc=False,
        )

        # Initialize weights (same as borzoi-pytorch)
        nn.init.kaiming_normal_(self.mha.Wqkv.weight, nonlinearity='relu')
        nn.init.zeros_(self.mha.out_proj.weight)
        nn.init.zeros_(self.mha.out_proj.bias)
        nn.init.ones_(self.mha.Wqkv.bias)

    def forward(self, x):
        out = self.mha(x)
        return out


class TransformerBlock(nn.Module):
    """
    Standard Transformer block with pre-normalization

    Structure: LayerNorm → Attention → Residual → LayerNorm → FFN → Residual
    """

    def __init__(self,
                 dim: int,
                 heads: int = 8,
                 ff_mult: int = 4,
                 dropout: float = 0.1,
                 use_gqa: bool = True,
                 use_flash: bool = True,
                 rotary_emb_fraction: float = 0.5,  # Fraction of dim_head for RoPE
                 rotary_emb_base: float = 20000.0,
                 rotary_emb_scale_base: Optional[float] = None):
        super().__init__()

        # Pre-normalization layers
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        # Choose attention implementation
        if use_flash:
            try:
                self.attn = FlashAttention(
                    dim=dim,
                    heads=heads,
                    dropout=dropout,
                    rotary_emb_fraction=rotary_emb_fraction,
                    rotary_emb_base=rotary_emb_base,
                    rotary_emb_scale_base=rotary_emb_scale_base
                )
                print(f"Using FlashAttention with rotary_emb_dim={int(dim // heads * rotary_emb_fraction)}")
            except ImportError:
                print("FlashAttention not available, falling back to standard Attention")
                self.attn = Attention(
                    dim=dim,
                    heads=heads,
                    dropout=dropout,
                    use_gqa=use_gqa,
                    rotary_emb_fraction=rotary_emb_fraction
                )
        else:
            self.attn = Attention(
                dim=dim,
                heads=heads,
                dropout=dropout,
                use_gqa=use_gqa,
                rotary_emb_fraction=rotary_emb_fraction
            )
            print(f"Using standard Attention with rotary_emb_dim={int(dim // heads * rotary_emb_fraction)}")

        # Feed-forward network
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * ff_mult, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # Pre-norm attention with residual
        x = x + self.attn(self.norm1(x))

        # Pre-norm FFN with residual
        x = x + self.ff(self.norm2(x))

        return x
