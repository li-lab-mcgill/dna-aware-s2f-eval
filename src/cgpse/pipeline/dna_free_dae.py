import torch
import torch.nn as nn

        
class TrackDebiasingDAEWrapper(nn.Module):
    """
    Shared utilities for DAE-based debiasing wrappers (inference only).
    """

    def __init__(
        self,
        *,
        dae: nn.Module,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.dae = dae
        self._freeze_modules()
        self.eps = eps
        
    def _freeze_modules(self) -> None:
        for module in (self.dae,):
            module.eval()
            for p in module.parameters():
                p.requires_grad = False

    def train(self, mode: bool = True):
        if mode:
            print("Debiasing wrappers are eval-only; forcing every module to eval mode.")
        super().train(False)
        self._freeze_modules()
        return self

    def forward(
            self,
            track: torch.Tensor, 
            route: str ="s2f"
    ) -> torch.Tensor:
        
        # convert track to profile logprobs and log1p_counts
        # convert B,L,T to B,T,L if needed
        assert track.dim() == 3
        track_nonneg = track.clamp_min_(0.0)
        if track_nonneg.size(2) < track_nonneg.size(1): # likely B,L,T
            track_nonneg = track_nonneg.transpose(1,2) # to B,T,L
        # counts (per-track total across positions)
        denom = track_nonneg.sum(dim=-1, keepdim=True).clamp_min(self.eps)   # (B, T, 1)
        # profile
        profile_prob_tracks = track_nonneg / denom  # (B, T, L)
        # log profile
        profile_logprobs_tracks = torch.log(profile_prob_tracks + self.eps)  # (B, T, L)
        # log1p counts
        log1p_counts = torch.log1p(denom.squeeze(-1))  # (B, T)
        
        # inference with DAE
        debiased_profile_logits = self.dae(
            profile_logprobs_tracks,
            route=route,
            return_latents=False
        )

        # return debiased logits
        return debiased_profile_logits, log1p_counts


__all__ = [
    "TrackDebiasingDAEWrapper",
]
