import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------- Utilities -------------------------

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

    
# ------------------------- Projection -------------------------

class MLMHead(nn.Module):
    """
    Project a single set of backbone embeddings to per-token logits.

    Structure (token-only; no positional mixing inside the head):
      Selector             LN(C) -> 1x1 SwiGLU (canonical)
      Residual             identity or 1x1 Conv (if C!=H)
      Combine              residual + SwiGLU (no learned scale)
      Readout              1x1 Conv (hidden -> vocab=4)

    Notes:
    - All spatial/sequence mixing is expected in the backbone (e.g., U-Net).
    - This head is intentionally light-weight and channel-only for stability.
    - All grads are live; no detachment in this head.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expansion_factor: int = 1,
        dropout_prelogits: float = 0.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = in_channels * int(expansion_factor)

        # Residual projection always on (1x1 conv)
        self.res_proj = nn.Conv1d(in_channels, self.hidden_channels, kernel_size=1, stride=1)

        # -------- tokenwise selector (SwiGLU) --------
        # In this code path we already apply a tokenwise LN after the final
        # decoder, so we do not add another LN here; the SwiGLU sees the raw
        # channels from the backbone.
        self.ln_sel = nn.Identity()
        self.pw = nn.Conv1d(in_channels, 2 * self.hidden_channels, kernel_size=1, stride=1)

        # No residual scaling parameters — simple sum residual + SwiGLU

        # Optional dropout before logits
        self.dropout_prelogits = nn.Dropout(dropout_prelogits) if dropout_prelogits and dropout_prelogits > 0 else nn.Identity()

        # -------- channel-only head mixer (no positional mixing) --------
        # Simple 1x1 projection to logits; all spatial mixing is in the backbone
        self.to_logits = nn.Conv1d(self.hidden_channels, out_channels, kernel_size=1, stride=1)

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(self.pw.weight, nonlinearity='linear')
        with torch.no_grad():
            H = self.hidden_channels
            # temper product variance
            self.pw.weight[:H].mul_(2 ** -0.5)   # 'a' branch
            self.pw.weight[H:].mul_(2 ** -0.5)   # 'b' branch
            if self.pw.bias is not None:
                # bias ACTIVATED half (a) slightly negative; b to 0
                self.pw.bias[:H].fill_(-0.5)
                self.pw.bias[H:].zero_()
        # Residual projection init: identity when possible; [I;0] for expansion; row-truncation for reduction
        with torch.no_grad():
            if self.res_proj.bias is not None:
                nn.init.zeros_(self.res_proj.bias)
            w = self.res_proj.weight  # (H, C_in, 1)
            w.zero_()
            C_in = self.in_channels
            H = self.hidden_channels
            diag = min(C_in, H)
            eye = torch.eye(diag, dtype=w.dtype, device=w.device)
            w[:diag, :diag, 0].copy_(eye)

        nn.init.xavier_uniform_(self.to_logits.weight, gain=1.0)
        if self.to_logits.bias is not None:
            nn.init.zeros_(self.to_logits.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, C, L)
        # SwiGLU (token-only) selector
        y = self.ln_sel(x)
        main = self.pw(y)
        a, b = torch.chunk(main, 2, dim=1)
        sel = F.silu(a) * b

        # Residual: always 1x1 projection (initialized close to identity)
        residual = self.res_proj(x)
        x = residual + sel
        x = self.dropout_prelogits(x)

        # channel-only projection to logits per token
        out = self.to_logits(x)  # (B, H|C, L) -> (B, 4, L)
        return out.transpose(1, 2)  # (B, L, 4) logits
