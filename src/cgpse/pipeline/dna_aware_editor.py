import torch
import torch.nn as nn

        
class TrackDebiasingEditorWrapper(nn.Module):
    """
    DNA-aware editor wrapper for precomputed tracks (inference only).
    """

    def __init__(
        self,
        *,
        editor: nn.Module,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.editor = editor
        self._freeze_modules()
        self.eps = eps
        
    def _freeze_modules(self) -> None:
        self.editor.eval()
        for p in self.editor.parameters():
            p.requires_grad = False

    def train(self, mode: bool = True):
        if mode:
            print("Debiasing wrappers are eval-only; forcing every module to eval mode.")
        super().train(False)
        self._freeze_modules()
        return self

    def _center_crop_dna(self, dna: torch.Tensor, target_len: int) -> torch.Tensor:
        if dna.dim() != 3:
            raise ValueError(f"dna must be (B,L,4) or (B,4,L), got {tuple(dna.shape)}")
        if dna.size(-1) != 4:
            if dna.size(1) == 4:
                dna = dna.transpose(1, 2)
            else:
                raise ValueError(f"dna must end with 4 channels, got {tuple(dna.shape)}")
        if dna.size(1) == target_len:
            return dna
        if dna.size(1) < target_len:
            raise ValueError(
                f"dna length {dna.size(1)} is shorter than track length {target_len}"
            )
        offset = (dna.size(1) - target_len) // 2
        return dna[:, offset : offset + target_len, :]

    def _build_masked_dna_input(
        self,
        dna: torch.Tensor,
        mask: torch.Tensor,
        *,
        target_len: int,
    ) -> torch.Tensor:
        # center crop DNA
        dna = self._center_crop_dna(dna, target_len)
        if mask.dim() != 2 or mask.size(1) != target_len:
            raise ValueError(f"mask must be (B,L_track), got {tuple(mask.shape)}")
        # prepare mask
        mask_bool = mask.to(torch.bool) if mask.dtype == torch.bool else (mask > 0)
        dna_values_masked = dna.masked_fill(mask_bool.unsqueeze(-1), 0.0)
        mask_channel = mask_bool.to(dna.dtype).unsqueeze(-1).expand(-1, -1, 4)
        return torch.stack([dna_values_masked, mask_channel], dim=1)  # (B,2,L,4)

    def forward(
            self,
            dna: torch.Tensor,
            track: torch.Tensor, 
            mask: torch.Tensor | str | None = None,
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
        
        # prepare dna input for editor
        L_track = profile_logprobs_tracks.size(-1)
        # 1) define the mask
        if mask is None or mask == "full_dna":
            mask = torch.zeros(
                profile_logprobs_tracks.size(0),
                L_track,
                device=profile_logprobs_tracks.device,
                dtype=torch.bool,
            )
        elif mask == "no_dna":
            debiased_profile_logits = self.editor.dae(
                profile_logprobs_tracks,
                route=route,
                return_latents=False,
            )
            return debiased_profile_logits, log1p_counts
        elif mask.dim() == 3 and mask.size(-1) == 1:
            mask = mask.squeeze(-1) # to B,L_track
        # 2) build masked dna input
        masked_dna = self._build_masked_dna_input(
            dna, mask, target_len=L_track
        )

        # inference with editor
        debiased_profile_logits, _ = self.editor(
            profile_logprobs_tracks,
            route=route,
            masked_dna=masked_dna,
            return_latents=False,
        )

        # return debiased logits
        return debiased_profile_logits, log1p_counts


__all__ = [
    "TrackDebiasingEditorWrapper",
]
