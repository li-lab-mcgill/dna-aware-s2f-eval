import torch
import torch.nn as nn

from typing import Dict, Any, List, Tuple, Optional

from .components.stem.stem import MaskConditionalStem
from .components.core.core import UNetCore
from .components.head.router import HeadRouter


class MultiHeadedCritic(nn.Module):
    """
    Split-stem critic with:
      - separate DNA and track mask-conditional stems (no routing),
      - a shared UNet core,
      - one head router containing multiple MLM heads (e.g., GT, S2F).

    This is a streamlined version of MultiStemMultiHeadCritic:
      - No regime parameter (no vis/abs split)
      - No stem routing (StemRouter)
      - No multiple head routers per regime
      - Architecture: stem → core → head_router (with multiple heads)

    Forward contracts
    ------------------
    - Input: x of shape (B, 2, L, T) (value + mask, 4+2 channels in T).
    - The stem processes masked input and produces features + skip connection.
    - The core runs the UNet encoder-decoder.
    - The head router combines core output with stem skip and produces logits.
    - Output: dict mapping head name -> logits (B, L, 4), e.g., {"GT": ..., "S2F": ...}.
    """

    def __init__(
        self,
        backbone_kwargs: Dict[str, Any],
        *,
        head_names: List[str],
        per_head_kwargs: Optional[Dict[str, Dict[str, Any]]] = None,
    ):
        """
        Parameters
        ----------
        backbone_kwargs
            kwargs for the UNet backbone (depths, num_groups, etc.).
            Expected keys (subset): 'depths', 'num_groups', 'conv_kernel_size',
            'dropout', 'input_conv_channels', 'T', 'padding_mode',
            'input_kernel_size', 'mask_aware_padding', 'bottleneck_config'.
        mlm_head_kwargs
            Shared kwargs forwarded to each MLM head via HeadRouter.
            `in_channels` and `out_channels` are filled automatically.
        head_names
            List of head identifiers (e.g., ["GT", "S2F"]).
        """
        super().__init__()

        bb = dict(backbone_kwargs)
        depths = list(bb["depths"])
        num_groups = int(bb.get("num_groups", 8))
        conv_kernel_size = int(bb.get("conv_kernel_size", 3))
        dropout = float(bb.get("dropout", 0.1))

        T_total = int(bb["T"])  # total input channels = 4 DNA + T_tracks
        input_conv_channels = int(bb.get("input_conv_channels", 128))
        input_kernel_size = int(bb.get("input_kernel_size", 23))
        padding_mode = bb.get("padding_mode", "same")
        mask_aware_padding = bool(bb.get("mask_aware_padding", False))

        if T_total < 4:
            raise ValueError(f"Expected at least 4 channels (DNA), got T={T_total}")
        T_tracks = T_total - 4

        # -------- Split stems (DNA / track, mask-conditional) --------
        # DNA stem sees only the 4 DNA channels and their mask.
        self.dna_stem = MaskConditionalStem(
            T=4,
            first_depth=depths[0],
            input_kernel_size=input_kernel_size,
            input_conv_channels=input_conv_channels,
            num_groups=num_groups,
            conv_kernel_size=conv_kernel_size,
            dropout=dropout,
            padding_mode=padding_mode,
            mask_aware_padding=mask_aware_padding,
        )
        # Track stem sees only the track channels and their mask.
        self.track_stem = MaskConditionalStem(
            T=T_tracks,
            first_depth=depths[0],
            input_kernel_size=input_kernel_size,
            input_conv_channels=input_conv_channels,
            num_groups=num_groups,
            conv_kernel_size=conv_kernel_size,
            dropout=dropout,
            padding_mode=padding_mode,
            mask_aware_padding=mask_aware_padding,
        )

        # -------- Shared core (UNet encoder-decoder) --------
        self.core = UNetCore(
            depths=depths,
            num_groups=num_groups,
            bottleneck_config=bb.get("bottleneck_config", None),
            conv_kernel_size=conv_kernel_size,
            dropout=dropout,
        )

        # -------- Single head router (final up block + multiple MLM heads) --------
        apply_shared_token_ln = bool(bb.get("apply_shared_token_ln", False))
        shared_token_ln_affine = bool(bb.get("shared_token_ln_affine", False))

        # Optional per-head kwargs (e.g., DISC head with different out_channels).
        # For this critic variant we configure all heads via `per_head_kwargs`
        # and do not rely on a shared mlm_head_kwargs dict.
        if per_head_kwargs is None:
            per_head_kwargs = {}

        self.head_router = HeadRouter(
            in_channels=input_conv_channels,
            head_names=head_names,
            mlm_head_kwargs=None,
            per_head_kwargs=per_head_kwargs,
            core_channels=depths[0],
            skip_channels=depths[0],
            conv_kernel_size=conv_kernel_size,
            num_groups=num_groups,
            dropout=dropout,
            apply_shared_token_ln=apply_shared_token_ln,
            shared_token_ln_affine=shared_token_ln_affine,
        )
        self.head_names = list(head_names)

    def forward(
        self,
        x: torch.Tensor,                    # (B, 2, L, T)
        heads: List[str] | None = None,    # which heads to evaluate
    ) -> Dict[str, torch.Tensor]:
        """
        Run stem, core, and head router to produce DNA MLM logits.

        Parameters
        ----------
        x : torch.Tensor
            Masked input of shape (B, 2, L, T) where:
            - x[:, 0] = masked values (DNA + tracks)
            - x[:, 1] = mask (1.0 = masked, 0.0 = visible)
        heads : List[str] | None
            Which heads to evaluate, e.g., ["GT"] or ["S2F"] or ["GT", "S2F"].
            If None, evaluates all heads.

        Returns
        -------
        logits_dict : Dict[str, torch.Tensor]
            Mapping head name -> logits (B, L, 4).
            Example: {"GT": (B, L, 4), "S2F": (B, L, 4)}
        """
        if x.dim() != 4:
            raise ValueError(f"Expected input of shape (B, 2, L, T), got {tuple(x.shape)}")
        if x.size(1) != 2:
            raise ValueError(f"Expected 2 channels (value+mask) on dim=1, got {x.size(1)}")

        # Split into DNA (first 4 channels) and track channels along T
        if x.size(-1) < 4:
            raise ValueError(f"Expected at least 4 DNA channels in last dim, got {x.size(-1)}")

        x_dna = x[:, :, :, :4]   # (B,2,L,4)
        x_trk = x[:, :, :, 4:]   # (B,2,L,T_tracks)

        T_tracks = x_trk.size(-1)
        if T_tracks <= 0:
            raise ValueError("Track stem expects at least one track channel, got 0")

        # Stems: modality-specific masked input conv + first down block
        dna_features, dna_skip = self.dna_stem(x_dna)
        trk_features, trk_skip = self.track_stem(x_trk)

        # Fuse stems before shared UNet core
        stem_features = dna_features + trk_features
        stem_skip = dna_skip + trk_skip

        # Core: UNet encoder-decoder
        core_out = self.core(stem_features)

        # Head router: final up block + MLM heads
        logits_dict = self.head_router(core_out, stem_skip, heads=heads)

        return logits_dict

    def forward_with_features(
        self,
        x: torch.Tensor,                    # (B, 2, L, T)
        heads: List[str] | None = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Run stem, core, and head router, returning both decoder features and logits.

        Parameters
        ----------
        x : torch.Tensor
            Masked input of shape (B, 2, L, T).
        heads : List[str] | None
            Which heads to evaluate. If None, evaluates all heads.

        Returns
        -------
        features : torch.Tensor
            Decoder features after final up block (and optional LN),
            shape (B, C_dec, L).
        logits_dict : Dict[str, torch.Tensor]
            Mapping head name -> logits (B, L, 4).
        """
        if x.dim() != 4:
            raise ValueError(f"Expected input of shape (B, 2, L, T), got {tuple(x.shape)}")
        if x.size(1) != 2:
            raise ValueError(f"Expected 2 channels (value+mask) on dim=1, got {x.size(1)}")

        if x.size(-1) < 4:
            raise ValueError(f"Expected at least 4 DNA channels in last dim, got {x.size(-1)}")

        x_dna = x[:, :, :, :4]   # (B,2,L,4)
        x_trk = x[:, :, :, 4:]   # (B,2,L,T_tracks)

        T_tracks = x_trk.size(-1)
        if T_tracks <= 0:
            raise ValueError("Track stem expects at least one track channel, got 0")

        dna_features, dna_skip = self.dna_stem(x_dna)
        trk_features, trk_skip = self.track_stem(x_trk)

        stem_features = dna_features + trk_features
        stem_skip = dna_skip + trk_skip
        core_out = self.core(stem_features)

        features, logits_dict = self.head_router.forward_with_features(
            core_out, stem_skip, heads=heads
        )

        return features, logits_dict

    def forward_with_stem_features(
        self,
        x: torch.Tensor,                    # (B, 2, L, T)
        heads: List[str] | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Run stem, core, and head router, returning:
          - track-only stem features (before DNA fusion),
          - fused stem features (after DNA+track fusion),
          - logits for the requested heads.

        This is intended for denoising Phase-B, where separate cosine losses
        can be applied to track-only vs fused features.
        """
        if x.dim() != 4:
            raise ValueError(f"Expected input of shape (B, 2, L, T), got {tuple(x.shape)}")
        if x.size(1) != 2:
            raise ValueError(f"Expected 2 channels (value+mask) on dim=1, got {x.size(1)}")

        if x.size(-1) < 4:
            raise ValueError(f"Expected at least 4 DNA channels in last dim, got {x.size(-1)}")

        x_dna = x[:, :, :, :4]   # (B,2,L,4)
        x_trk = x[:, :, :, 4:]   # (B,2,L,T_tracks)

        T_tracks = x_trk.size(-1)
        if T_tracks <= 0:
            raise ValueError("Track stem expects at least one track channel, got 0")

        dna_features, dna_skip = self.dna_stem(x_dna)
        trk_features, trk_skip = self.track_stem(x_trk)

        stem_features = dna_features + trk_features
        stem_skip = dna_skip + trk_skip
        core_out = self.core(stem_features)

        _, logits_dict = self.head_router.forward_with_features(
            core_out, stem_skip, heads=heads
        )

        return trk_features, stem_features, logits_dict
