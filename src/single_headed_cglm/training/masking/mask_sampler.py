"""
Lightweight mask-type sampler with minimal registry.

Usage
-----
>>> from cglm_utils.training.masking.mask_sampler import MaskSampler
>>> sampler = MaskSampler({
...     "variable_rate_uniform_partial_row_sampler": {
...         "prob": 1.0,
...         "kwargs": {
...             "mask_percentage": None,
...             "mask_pct_min": 0.15,
...             "mask_pct_max": 1.0,
...             "p_mask_dna_only": 0.6,
...             "p_mask_track_only": 0.2,
...         }
...     }
... })
>>> mask_fn = sampler()  # returns an instance of the chosen mask
>>> mask = mask_fn.sample_mask((L, T_plus_4))
"""

import torch
from torch.distributions import Categorical

from .masks import VariableRateUniformPartialRowSampler


# Minimal registry with only the needed mask type
MASK_REGISTRY = {
    "variable_rate_uniform_partial_row_sampler": VariableRateUniformPartialRowSampler,
}


class MaskSampler:
    """
    Randomly chooses among pre-instantiated mask objects according to a
    user-supplied probability dictionary.

    Parameters
    ----------
    mask_cfg : dict
      Keys are *names* in `MASK_REGISTRY`.
      Values must be dict with:
        {
            "prob"  : float,
            "kwargs": dict, (keyword args for the mask class)
        }
    """

    def __init__(self, mask_config):
        if not mask_config:
            raise ValueError("mask_cfg must contain at least one entry.")

        probs = []
        instances = []

        for name, spec in mask_config.items():

            # validate mask type given in key
            if name not in MASK_REGISTRY:
                raise ValueError(
                    f"Unknown mask '{name}'. "
                    f"Available options: {list(MASK_REGISTRY.keys())}"
                )

            # parse config
            if isinstance(spec, dict):
                # validate prob and kwargs
                assert "prob" in spec, "prob must be in spec"
                assert "kwargs" in spec, "kwargs must be in spec"

                # parse prob and kwargs
                prob = float(spec["prob"])
                kwargs = dict(spec["kwargs"])
            else:
                raise TypeError(
                    "Each mask config must be a dict with 'prob' and 'kwargs' keys. "
                    "kwargs must contain the arguments needed by the chosen mask."
                )

            if prob <= 0.0:
                continue  # skip zero-prob masks

            # instantiate mask class
            cls = MASK_REGISTRY[name]
            instances.append(cls(**kwargs))
            probs.append(prob)

        if not instances:
            raise ValueError("No valid mask configurations found.")

        self._instances = instances

        # Normalise probabilities and build a categorical sampler
        prob_tensor = torch.tensor(probs, dtype=torch.float32)
        prob_tensor = prob_tensor / prob_tensor.sum()
        self._cat = Categorical(prob_tensor)

    def __call__(self):
        """Return *one* mask class instance chosen according to the configured probs."""
        idx = self._cat.sample().item()
        return self._instances[idx]

    def __repr__(self):
        lines = ["MaskSampler("]
        for inst, p in zip(self._instances, self._cat.probs.tolist()):
            lines.append(f"  {inst.__class__.__name__:>25s}: p={p:.3f}")
        lines.append(")")
        return "\n".join(lines)


# Backwards-compatible alias (for BatchPreparer compatibility)
MaskStrategySampler = MaskSampler

__all__ = ["MaskSampler", "MaskStrategySampler"]
