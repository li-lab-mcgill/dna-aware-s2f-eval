"""
Fast mask‑type sampler with a lightweight registry.

Usage
-----
>>> from utils.data.masking.mask_sampler import MaskSampler
>>> sampler = MaskSampler(
...     {
...         "row": 0.3,
...         "column": 0.2,
...         "row_span": 0.3,
...         "random": 0.2,
...     },
...     default_mask_percentage=0.15,
... )
>>> mask_fn = sampler()           # returns an *instance* of the chosen mask
>>> mask = mask_fn.sample_mask((L, T_plus_4))
"""

import torch
from torch.distributions import Categorical

# ---------------------------------------------------------------------
# 1.  Import available mask classes
# ---------------------------------------------------------------------
from .random_mask import (
    VariableRateUniformPartialRowSampler,
    NestedVariableRateUniformMaskSampler,
    NestedBetaHeadroomMaskSampler,
)

# ---------------------------------------------------------------------
# 2.  Registry   name -> class
# ---------------------------------------------------------------------
MASK_REGISTRY = {
    "variable_rate_uniform_partial_row_sampler": VariableRateUniformPartialRowSampler,
    "nested_variable_rate_uniform_mask_sampler": NestedVariableRateUniformMaskSampler,
    "nested_beta_headroom_mask_sampler": NestedBetaHeadroomMaskSampler,
}

# ---------------------------------------------------------------------
# 3.  Main interface
# ---------------------------------------------------------------------
class MaskStrategySampler:
    """
    Randomly chooses among pre‑instantiated mask objects according to a
    user‑supplied probability dictionary.

    Parameters
    ----------
    mask_cfg : dict
      Keys are *names* in `MASK_REGISTRY`.
      Values can be either

      • float  →  probability of drawing that mask‑type
      • dict   →  {
                      "prob"  : float,
                      "kwargs": dict, (every arg is keyword args)
                   }

      Any mask class expecting a `mask_percentage` argument will receive
      the keyword supplied in ``kwargs``; otherwise the `default_mask_percentage`
      is used. 
    """

    def __init__(self, mask_config):
        if not mask_config:
            raise ValueError("mask_cfg must contain at least one entry.")

        probs = []
        entries = []

        for name, spec in mask_config.items():
            parts = name.split("@", 1)
            registry_name = parts[0]
            alias = parts[1] if len(parts) == 2 else registry_name

            # validate mask type given in key
            if registry_name not in MASK_REGISTRY:
                raise ValueError(
                    f"Unknown mask '{name}'. "
                    f"Available options: {list(MASK_REGISTRY.keys())}"
                )
            elif registry_name == "column_group":
                raise ValueError("Column group mask is not supported.")
            elif registry_name == "row_span_column_group":
                raise ValueError("Row span column group mask is not supported.")

            # parse config
            if isinstance(spec, dict):

                # validate prob and kwargs
                assert "prob" in spec, "prob must be in spec"
                assert "kwargs" in spec, "kwargs must be in spec"

                # parse prob and kwargs (no mandatory mask_percentage any more)
                prob = float(spec["prob"])
                kwargs = dict(spec["kwargs"])
                kwargs.pop("mask_percentage", None)
            else:
                raise TypeError(
                    "Each mask config must be a dict with 'prob' and 'kwargs' keys."
                )

            if prob <= 0.0:
                continue  # skip zero‑prob masks

            # instantiate mask class
            cls = MASK_REGISTRY[registry_name]
            instance = cls(**kwargs)
            # Attach lightweight alias + kwargs metadata for downstream consumers
            setattr(instance, "_mask_alias", alias)
            setattr(instance, "_mask_registry_name", registry_name)
            setattr(instance, "_mask_init_kwargs", dict(kwargs))
            entries.append(
                {
                    "instance": instance,
                    "alias": alias,
                    "registry_name": registry_name,
                    "kwargs": dict(kwargs),
                }
            )
            probs.append(prob)

        if not entries:
            raise ValueError("No valid mask configurations found.")

        self._entries = entries

        # Normalise probabilities and build a categorical sampler
        prob_tensor = torch.tensor(probs, dtype=torch.float32)
        prob_tensor = prob_tensor / prob_tensor.sum()
        self._cat = Categorical(prob_tensor)

    # ------------------------------------------------------------------
    def __call__(self):
        """Return *one* mask class instance chosen according to the configured probs."""
        idx = self._cat.sample().item()
        return self._entries[idx]["instance"]

    # ------------------------------------------------------------------
    def sample_with_metadata(self):
        """Return (instance, metadata) for callers that need alias/kwargs."""
        idx = self._cat.sample().item()
        entry = self._entries[idx]
        metadata = {
            "mask_type": entry["alias"],
            "registry_name": entry["registry_name"],
            "kwargs": dict(entry["kwargs"]),
        }
        return entry["instance"], metadata

    # ------------------------------------------------------------------
    def __repr__(self):
        lines = ["MaskSampler("]
        for entry, p in zip(self._entries, self._cat.probs.tolist()):
            name = entry["alias"]
            lines.append(f"  {name:>25s}: p={p:.3f}")
        lines.append(")")
        return "\n".join(lines)


# Backwards-compatible name (preferred name is MaskStrategySampler).
MaskSampler = MaskStrategySampler

__all__ = ["MaskStrategySampler", "MaskSampler"]
