import torch
import math
import sys
import os 

from .base import BaseMask
    
class VariableRateUniformPartialRowSampler(BaseMask):
    """
    • Draw a row‑masking fraction p ~ Uniform(mask_pct_min , mask_pct_max)
    • Pick ⌊p·L⌋ rows       (vectorised `randperm`)
    • For each chosen row sample a category:
          0 = DNA-only      (mask columns 0-3)
          1 = Track-only    (mask columns 4…T+3)
          2 = Both          (mask all columns)

      Probabilities are user-set:  p_dna , p_track , 1-p_dna-p_track.

    Parameters
    ----------
    mask_pct_min   : float   – lower bound for row-mask rate  (0 < … ≤ 1)
    mask_pct_max   : float   – upper bound for row-mask rate  (mask_pct_max ≤ 1)
    p_dna_only     : float   – P[row is DNA-only]
    p_track_only   : float   – P[row is Track-only]
                           (Both = 1 - p_dna_only - p_track_only)

    Notes
    -----
    • `mask_percentage` from the parent class is *ignored*.
    • DNA is assumed to occupy columns 0,1,2,3.
    """

    def __init__(
        self,
        mask_pct_min: float = 0.15,
        mask_pct_max: float = 1.00,
        p_mask_dna_only  : float = 0.60,
        p_mask_track_only: float = 0.20,
    ):
        super().__init__()
        assert 0.0 <= mask_pct_min <= mask_pct_max <= 1.0
        assert 0.0 <= p_mask_dna_only <= 1.0
        assert 0.0 <= p_mask_track_only <= 1.0
        assert p_mask_dna_only + p_mask_track_only <= 1.0

        self.mask_pct_min = mask_pct_min
        self.mask_pct_max = mask_pct_max

        self.cat_probs = torch.tensor(
            [p_mask_dna_only, p_mask_track_only, 1.0 - p_mask_dna_only - p_mask_track_only],
            dtype=torch.float32,
        )

    # ------------------------------------------------------------------ #

    def sample_mask(self, shape: tuple) -> torch.Tensor:
        """
        Args
        ----
        shape : (L , T_plus_4)

        Returns
        -------
        mask  : float32 tensor (L , T_plus_4)  with 1.0 on masked cells.
        """
        L, T_plus_4 = shape
        device      = self.cat_probs.device

        # ---- 1. draw row‑mask fraction p ------------
        u   = torch.rand(1, device=device).item()
        p   = self.mask_pct_min + u * (self.mask_pct_max - self.mask_pct_min)
        n   = max(1, int(math.floor(p * L)))          # at least one row

        # ---- 2. choose rows to mask -------------
        rows = torch.randperm(L, device=device)[:n]

        # ---- 3. sample category per row ---------
        cats = torch.multinomial(self.cat_probs, n, replacement=True)  # (n,)

        # ---- 4. build the mask (vectorised) -----
        mask = torch.zeros((L, T_plus_4), dtype=torch.float32, device=device)

        dna_cols    = slice(0, 4)
        track_cols  = slice(4, T_plus_4)

        mask_dna_rows    = rows[cats == 0]
        mask_trk_rows    = rows[cats == 1]
        mask_both_rows   = rows[cats == 2]

        if mask_dna_rows.numel():
            mask[mask_dna_rows, dna_cols]   = 1.0
        if mask_trk_rows.numel():
            mask[mask_trk_rows, track_cols] = 1.0
        if mask_both_rows.numel():
            mask[mask_both_rows, :]         = 1.0

        return mask
    

class NestedVariableRateUniformMaskSampler(BaseMask):
    """
    Sample nested DNA masks: M_co ⊂ M_ci (inner ⊂ outer)
      r_outer ~ Uniform(mask_pct_min, mask_pct_max)
      r_inner ~ Uniform(mask_pct_min, r_outer)

    Returns:
      mask_in  = M_ci  (critic input DNA mask)
      mask_out = M_co  (critic output/audit DNA mask)  with mask_out ⊆ mask_in
    Track is always visible (never masked).
    DNA is columns 0..3.
    """

    def __init__(self, mask_pct_min: float = 0.10, mask_pct_max: float = 1.00):
        super().__init__()
        assert 0.0 <= mask_pct_min <= mask_pct_max <= 1.0
        self.mask_pct_min = mask_pct_min
        self.mask_pct_max = mask_pct_max

        # device anchor (works even if BaseMask isn't an nn.Module)
        self._device_anchor = torch.tensor(0.0)

    def sample_mask(self, shape: tuple):
        """
        Returns a nested pair (mask_in, mask_out) where:
          - mask_in  corresponds to M_ci (critic input DNA mask)
          - mask_out corresponds to M_co (critic output/audit DNA mask)
        and mask_out <= mask_in elementwise.
        """
        L, DNA_plus_T = shape
        device = self._device_anchor.device

        # ---- 1) sample outer rate ----
        u_outer = torch.rand((), device=device).item()
        r_outer = self.mask_pct_min + u_outer * (self.mask_pct_max - self.mask_pct_min)
        n_outer = max(1, int(math.floor(r_outer * L)))

        # ---- 2) sample inner rate (nested within outer) ----
        u_inner = torch.rand((), device=device).item()
        r_inner = self.mask_pct_min + u_inner * (r_outer - self.mask_pct_min)
        n_inner = max(1, int(math.floor(r_inner * L)))
        n_inner = min(n_inner, n_outer)

        # ---- 3) sample positions ----
        perm = torch.randperm(L, device=device)
        outer_pos = perm[:n_outer]
        inner_pos = perm[:n_inner]  # subset of outer_pos automatically

        dna_cols = slice(0, 4)

        outer_mask = torch.zeros((L, DNA_plus_T), dtype=torch.float32, device=device)
        inner_mask = torch.zeros((L, DNA_plus_T), dtype=torch.float32, device=device)

        # DNA masked for critic *inputs* at outer positions
        outer_mask[outer_pos, dna_cols] = 1.0

        # DNA audited/predicted at inner positions
        inner_mask[inner_pos, dna_cols] = 1.0

        # assert nesting
        assert torch.all(inner_mask <= outer_mask), "inner_mask must be a subset of outer_mask"

        return outer_mask, inner_mask


class NestedBetaHeadroomMaskSampler(BaseMask):
    """
    Sample nested DNA masks: M_co ⊂ M_ci (audit ⊂ critic-input-hidden)

    Mode B scheme:
      r_co ~ Uniform(rco_min, rco_max)                      (audit rate)
      δ    ~ Beta(alpha, beta)
      r_ci = r_co + (rci_max - r_co) * δ                   (critic-input DNA hide rate; >= r_co, <= rci_max)

    Returns:
      mask_in  = M_ci  (critic input DNA mask; where critic does NOT see DNA)
      mask_out = M_co  (critic output/audit mask; where critic is scored) with mask_out ⊆ mask_in
    Track is always visible (never masked).
    DNA is columns 0..3.
    """

    def __init__(
        self,
        rco_min: float = 0.10,
        rco_max: float = 0.60,
        rci_max: float = 0.80,
        alpha: float = 2.0,
        beta: float = 3.0,
    ):
        super().__init__()
        assert 0.0 <= rco_min <= rco_max <= 1.0
        assert 0.0 <= rci_max <= 1.0
        assert rci_max >= rco_min, "rci_max must be >= rco_min"
        assert alpha > 0.0 and beta > 0.0

        self.rco_min = rco_min
        self.rco_max = rco_max
        self.rci_max = rci_max
        self.alpha = alpha
        self.beta = beta

        self._device_anchor = torch.tensor(0.0)

    def sample_mask(self, shape: tuple):
        L, DNA_plus_T = shape
        device = self._device_anchor.device

        # ---- 1) sample audit rate r_co ----
        u = torch.rand((), device=device).item()
        r_co = self.rco_min + u * (self.rco_max - self.rco_min)
        n_co = max(1, int(math.floor(r_co * L)))

        # ---- 2) sample critic-input hide rate r_ci via Beta headroom up to rci_max ----
        # If rci_max == r_co (can happen at edge), headroom is 0 and r_ci = r_co.
        headroom = max(0.0, self.rci_max - r_co)
        if headroom == 0.0:
            r_ci = r_co
        else:
            delta = torch.distributions.Beta(self.alpha, self.beta).sample(()).to(device).item()
            r_ci = r_co + headroom * delta

        n_ci = max(n_co, int(math.floor(r_ci * L)))  # ensure nesting size
        n_ci = min(n_ci, L)

        # ---- 3) sample positions: first M_ci, then choose M_co subset ----
        perm = torch.randperm(L, device=device)
        ci_pos = perm[:n_ci]
        co_pos = ci_pos[:n_co]  # guaranteed subset

        dna_cols = slice(0, 4)

        mask_in = torch.zeros((L, DNA_plus_T), dtype=torch.float32, device=device)
        mask_out = torch.zeros((L, DNA_plus_T), dtype=torch.float32, device=device)

        # Critic cannot see DNA at M_ci
        mask_in[ci_pos, dna_cols] = 1.0
        # Critic is scored/audits at M_co (subset)
        mask_out[co_pos, dna_cols] = 1.0

        assert torch.all(mask_out <= mask_in), "mask_out must be a subset of mask_in"
        return mask_in, mask_out

