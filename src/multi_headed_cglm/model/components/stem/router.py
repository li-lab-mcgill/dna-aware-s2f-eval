import torch
import torch.nn as nn

from typing import Dict, List, Any

from .stem import MaskConditionalStem


class StemRouter(nn.Module):
    """
    Router over regime-specific stems (e.g. vis / abs).

    - Instantiates `MaskConditionalStem` modules from per-regime configs.
    - Given an input (B, 2, L, T) and a regime key, runs the corresponding stem.
    - Returns the downsampled features and the first skip tensor.
    """

    def __init__(self, stem_cfgs: Dict[str, Dict[str, Any]]):
        super().__init__()

        if not stem_cfgs:
            raise ValueError("StemRouter requires at least one stem config.")

        stems: Dict[str, MaskConditionalStem] = {}
        for name, cfg in stem_cfgs.items():
            stems[name] = MaskConditionalStem(**cfg)

        # Normalize keys to a stable list for introspection.
        self.stem_names: List[str] = list(stems.keys())
        self.stems = nn.ModuleDict(stems)

    def forward(self, x: torch.Tensor, regime: str):
        """
        Parameters
        ----------
        x
            Input tensor (B, 2, L, T) to feed into the selected stem.
        regime
            Key of the stem to use (e.g. 'vis', 'abs').

        Returns
        -------
        features
            Downsampled stem features for the selected regime.
        skip
            First skip tensor to be passed to the final up block.
        """
        if regime not in self.stems:
            raise KeyError(f"Requested stem '{regime}' not found. Available: {self.stem_names}")

        stem = self.stems[regime]
        return stem(x)
