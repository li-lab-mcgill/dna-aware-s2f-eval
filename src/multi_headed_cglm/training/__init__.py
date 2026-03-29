"""Critic pretraining utilities."""

"""
Single-stem multi-head critic pretraining module.

Simplified from the multi-stem multi-head (vis/abs) approach:
  - Single stem (no regime routing)
  - Multiple heads (GT, S2F)
  - Masks come from BatchPreparer
  - All masked positions are supervised
"""

from .orchestrator import SingleStemCriticPretrainingOrchestrator
from .lightning import LightningSingleStemCriticPretrain

__all__ = [
    "SingleStemCriticPretrainingOrchestrator",
    "LightningSingleStemCriticPretrain",
]