import torch
import torch.nn as nn

from typing import Dict, Iterable, List, Optional, Union, Any, Tuple

from .mlm_head import MLMHead
from ..core.unet_helpers import UpBlock


class HeadRouter(nn.Module):
    """
    Lightweight router over multiple MLM heads.

    Responsibilities
    ----------------
    - Hold a set of `MLMHead` modules in a `ModuleDict`.
    - Given backbone features `x` of shape (B, C, L), run one or more heads.
    - Return a dictionary mapping head names to logits of shape (B, L, 4).

    Notes
    -----
    - This class is agnostic to masking regime or modality; it only routes.
    - A higher-level critic will decide which stem was used and which heads
      to query on a given forward pass.
    """

    def __init__(
        self,
        in_channels: int,
        head_names: Iterable[str],
        *,
        vocab_size: int = 4,
        mlm_head_kwargs: Optional[Dict[str, Any]] = None,
        per_head_kwargs: Optional[Dict[str, Dict[str, Any]]] = None,
        # final up block kwargs
        core_channels: Optional[int] = None,
        skip_channels: Optional[int] = None,
        conv_kernel_size: int = 3,
        num_groups: int = 8,
        dropout: float = 0.1,
        apply_shared_token_ln: bool = False,
        shared_token_ln_affine: bool = False,
    ):
        """
        Parameters
        ----------
        in_channels
            Channel dimension of backbone features (B, C, L) fed into the heads.
        head_names
            Iterable of head identifiers (e.g. ["GT_vis", "GT_abs", "S2F_vis", "S2F_abs"]).
        vocab_size
            Output vocabulary size per token (default: 4 nucleotides).
        expansion_factor
            Hidden expansion factor inside each `MLMHead`.
        dropout_prelogits
            Dropout rate applied inside each `MLMHead` before logits.
        per_head_kwargs
            Optional dict mapping head name -> extra kwargs for that head.
            These are merged on top of the shared kwargs above.
        """
        super().__init__()

        self.in_channels = int(in_channels)
        self.vocab_size = int(vocab_size)

        # Normalize head names to a stable list
        head_names = list(head_names)
        if len(head_names) == 0:
            raise ValueError("HeadRouter requires at least one head name.")

        self.head_names: List[str] = head_names

        # Final up block from core output + stem skip to head input channels
        if core_channels is None:
            raise ValueError("HeadRouter requires core_channels for the final up block.")
        if skip_channels is None:
            skip_channels = core_channels

        self.final_up = UpBlock(
            in_channels=int(core_channels),
            skip_channels=int(skip_channels),
            out_channels=self.in_channels,
            kernel_size=conv_kernel_size,
            num_groups=num_groups,
            dropout=dropout,
        )

        # Optional shared tokenwise LN over channels per position
        self.shared_token_ln: Optional[nn.LayerNorm]
        if apply_shared_token_ln:
            self.shared_token_ln = nn.LayerNorm(
                self.in_channels, elementwise_affine=bool(shared_token_ln_affine)
            )
        else:
            self.shared_token_ln = None

        # MLM heads
        self.heads = nn.ModuleDict()
        base_kwargs: Dict[str, Any] = dict(mlm_head_kwargs or {})
        base_kwargs.setdefault("in_channels", self.in_channels)
        base_kwargs.setdefault("out_channels", self.vocab_size)

        per_head_kwargs = per_head_kwargs or {}

        for name in head_names:
            extra = dict(per_head_kwargs.get(name, {}))
            cfg = dict(base_kwargs)
            cfg.update(extra)
            self.heads[name] = MLMHead(**cfg)

    def forward(
        self,
        core_out: torch.Tensor,  # (B, C_core, L/2)
        stem_skip: torch.Tensor,  # (B, C_skip, L/2)
        heads: Optional[Union[str, Iterable[str]]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Run one or more heads on the shared backbone features.

        Parameters
        ----------
        core_out
            Core output features of shape (B, C_core, L/2).
        stem_skip
            First skip tensor from the stem (B, C_skip, L/2).
        heads
            Which heads to evaluate. If None, all registered heads are used.
            If a string, only that head is used. If an iterable, all listed
            heads are used in the given order.

        Returns
        -------
        out
            Dictionary mapping head name -> logits tensor of shape (B, L, vocab_size).
        """
        if core_out.dim() != 3 or stem_skip.dim() != 3:
            raise ValueError(
                f"Expected core_out and stem_skip as (B, C, L), "
                f"got {tuple(core_out.shape)} and {tuple(stem_skip.shape)}"
            )

        # Resolve which heads to run
        if heads is None:
            active_heads: List[str] = self.head_names
        elif isinstance(heads, str):
            active_heads = [heads]
        else:
            active_heads = list(heads)

        # Final up block + optional LN to get head input features
        x = self.final_up(core_out, stem_skip)  # (B, in_channels, L)
        if self.shared_token_ln is not None:
            x = self.shared_token_ln(x.transpose(1, 2)).transpose(1, 2)

        out: Dict[str, torch.Tensor] = {}
        for name in active_heads:
            if name not in self.heads:
                raise KeyError(f"Requested head '{name}' not found in router.heads (keys={list(self.heads.keys())})")
            out[name] = self.heads[name](x)  # (B, L, vocab_size)

        return out

    def forward_with_features(
        self,
        core_out: torch.Tensor,  # (B, C_core, L/2)
        stem_skip: torch.Tensor,  # (B, C_skip, L/2)
        heads: Optional[Union[str, Iterable[str]]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Variant of forward that also returns the decoder features fed into heads.

        Returns
        -------
        features : torch.Tensor
            Decoder features of shape (B, in_channels, L) after final up block
            and optional LayerNorm.
        out : Dict[str, torch.Tensor]
            Mapping head name -> logits tensor of shape (B, L, vocab_size).
        """
        if core_out.dim() != 3 or stem_skip.dim() != 3:
            raise ValueError(
                f"Expected core_out and stem_skip as (B, C, L), "
                f"got {tuple(core_out.shape)} and {tuple(stem_skip.shape)}"
            )

        # Resolve which heads to run
        if heads is None:
            active_heads: List[str] = self.head_names
        elif isinstance(heads, str):
            active_heads = [heads]
        else:
            active_heads = list(heads)

        # Final up block + optional LN to get head input features
        x = self.final_up(core_out, stem_skip)  # (B, in_channels, L)
        if self.shared_token_ln is not None:
            x = self.shared_token_ln(x.transpose(1, 2)).transpose(1, 2)

        out: Dict[str, torch.Tensor] = {}
        for name in active_heads:
            if name not in self.heads:
                raise KeyError(f"Requested head '{name}' not found in router.heads (keys={list(self.heads.keys())})")
            out[name] = self.heads[name](x)

        return x, out
