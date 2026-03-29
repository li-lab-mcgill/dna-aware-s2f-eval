import torch
import math

from .base import BaseMask


class VariableRateUniformPartialRowSampler(BaseMask):
    """
    Variable rate row masking with uniform distribution.

    - Draw a row-masking *rate* r ~ Uniform(mask_pct_min, mask_pct_max)
    - Pick floor(r*L) rows (vectorised randperm)
    - For each chosen row sample a category:
          0 = DNA-only      (mask columns 0-3)
          1 = Track-only    (mask columns 4...T+3)
          2 = Both          (mask all columns)

      Probabilities are user-set: p_dna, p_track, 1-p_dna-p_track.

    Parameters
    ----------
    mask_pct_min   : float   - lower bound for row-mask rate (0 < ... <= 1)
    mask_pct_max   : float   - upper bound for row-mask rate (mask_pct_max <= 1)
    p_dna_only     : float   - P[row is DNA-only]
    p_track_only   : float   - P[row is Track-only]
                           (Both = 1 - p_dna_only - p_track_only)

    Notes
    -----
    - `mask_percentage` from the parent class is *ignored*.
    - DNA is assumed to occupy columns 0,1,2,3.
    """

    def __init__(
        self,
        mask_percentage: float = None,
        mask_pct_min: float = 0.15,
        mask_pct_max: float = 1.00,
        p_mask_dna_only: float = 0.60,
        p_mask_track_only: float = 0.20,
    ):
        super().__init__(mask_percentage=mask_percentage)  # ignored
        assert 0.0 < mask_pct_min <= mask_pct_max <= 1.0
        assert 0.0 <= p_mask_dna_only <= 1.0
        assert 0.0 <= p_mask_track_only <= 1.0
        assert p_mask_dna_only + p_mask_track_only <= 1.0

        self.mask_pct_min = mask_pct_min
        self.mask_pct_max = mask_pct_max

        self.cat_probs = torch.tensor(
            [p_mask_dna_only, p_mask_track_only, 1.0 - p_mask_dna_only - p_mask_track_only],
            dtype=torch.float32,
        )

    def sample_mask(self, shape: tuple) -> torch.Tensor:
        """
        Args
        ----
        shape : (L, T_plus_4)

        Returns
        -------
        mask  : float32 tensor (L, T_plus_4) with 1.0 on masked cells.
        """
        L, T_plus_4 = shape
        device = self.cat_probs.device

        # 1. draw row-mask rate r
        u = torch.rand(1, device=device).item()
        r = self.mask_pct_min + u * (self.mask_pct_max - self.mask_pct_min)
        n = max(1, int(math.floor(r * L)))  # at least one row

        # 2. choose rows to mask
        rows = torch.randperm(L, device=device)[:n]

        # 3. sample category per row
        cats = torch.multinomial(self.cat_probs, n, replacement=True)  # (n,)

        # 4. build the mask (vectorised)
        mask = torch.zeros((L, T_plus_4), dtype=torch.float32, device=device)

        dna_cols = slice(0, 4)
        track_cols = slice(4, T_plus_4)

        mask_dna_rows = rows[cats == 0]
        mask_trk_rows = rows[cats == 1]
        mask_both_rows = rows[cats == 2]

        if mask_dna_rows.numel():
            mask[mask_dna_rows, dna_cols] = 1.0
        if mask_trk_rows.numel():
            mask[mask_trk_rows, track_cols] = 1.0
        if mask_both_rows.numel():
            mask[mask_both_rows, :] = 1.0

        return mask
