"""Critic architecture components."""

from .model import MultiHeadedCritic
from .checkpoint_loaders import load_checkpoint

__all__ = [
    "MultiHeadedCritic",
    "load_checkpoint"
]