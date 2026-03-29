import torch
import torch.nn as nn

from .components.dna_aware_reencoder import DNAAwareTrackReencoder
from .components.editor import LatentEditor


class DNAAwareEditorWrapper(nn.Module):
    """
    Frozen DAE + trainable DNA-aware editor operating on bottleneck codes.
    """

    def __init__(
        self,
        dae: nn.Module,
        *,
        editor_kwargs: dict,
        dna_encoder_kwargs: dict | None = None,
        reencoder_kwargs: dict | None = None,
    ):
        super().__init__()

        # Keep a handle to the DAE and freeze it permanently.
        self.dae = dae
        for p in self.dae.parameters():
            p.requires_grad = False
        self.dae.eval()

        editor_kwargs = dict(editor_kwargs)

        if reencoder_kwargs is None:
            if dna_encoder_kwargs is None:
                raise ValueError("reencoder_kwargs must be provided.")
            reencoder_kwargs = dict(dna_encoder_kwargs)
        else:
            reencoder_kwargs = dict(reencoder_kwargs)

        # Trainable Stage-4 components (assume kwargs are fully specified).
        self.reencoder = DNAAwareTrackReencoder(**reencoder_kwargs)
        self.editor = LatentEditor(**editor_kwargs)

    def train(self, mode: bool = True):
        """
        Override train() to ensure the wrapped DAE remains frozen/eval,
        while the recycler/editor obey the requested mode.
        """
        super().train(mode)
        # keep dae frozen
        self.dae.eval()
        for p in self.dae.parameters():
            p.requires_grad = False
        return self
    
    def encode_to_bottleneck(self, track: torch.Tensor, *, route: str) -> torch.Tensor:
        """
        Partial forward pass of DAE to get the bottleneck code.
        """
        if hasattr(self.dae, "encode_bottleneck"):
            return self.dae.encode_bottleneck(track, route)
        h_conv = self.dae.encode_conv(track, route)      # (B, C_conv_lat, L_lat)
        h_enc = self.dae.bottleneck_encoder(h_conv)      # (B, C_conv_lat, L_lat)
        _, z = self.dae.bottleneck(h_enc)                # (B, d_model, L_lat), (B, C_lat, L_lat)
        return z

    def decode_from_bottleneck(self, z_edit: torch.Tensor) -> torch.Tensor:
        """
        Decode track from an edited bottleneck code.
        """
        if hasattr(self.dae, "decode_from_bottleneck"):
            return self.dae.decode_from_bottleneck(z_edit)
        h_pre = self.dae.bottleneck.decode(z_edit)
        h_bott = self.dae.bottleneck_decoder(h_pre)
        return self.dae.decoder(h_bott)
    
    def forward(
        self,
        track: torch.Tensor,
        route: str,
        masked_dna: torch.Tensor | None = None,
        *,
        return_latents: bool = False,
    ) -> torch.Tensor:
        """
        Full forward pass: encode -> edit -> decode
        """
        if route == "gt":
            with torch.no_grad():
                z_gt = self.encode_to_bottleneck(track, route=route)
            if return_latents:
                return z_gt, {"z_gt": z_gt}
            return z_gt

        if route != "s2f":
            raise ValueError(f"Unknown route '{route}'. Expected 'gt' or 's2f'.")

        if masked_dna is None:
            raise ValueError("masked_dna is required when route='s2f'.")

        with torch.no_grad():
            y_dae, latents = self.dae(track, route="s2f", return_latents=True)
        z_anchor = latents.get("z", None)
        if z_anchor is None:
            raise RuntimeError("DAE latents did not include bottleneck code 'z'.")

        y_dae_log = torch.log_softmax(y_dae, dim=-1)
        z_ctx = self.reencoder(y_dae_log, masked_dna)
        z_edit = self.editor(z_anchor, z_ctx)
        y_edit = self.decode_from_bottleneck(z_edit)

        if not return_latents:
            return y_edit, z_edit

        latents_out = {
            "z_anchor": z_anchor,
            "z_ctx": z_ctx,
            "z_edit": z_edit,
            "y_dae": y_dae,
        }
        return y_edit, latents_out

__all__ = ["DNAAwareEditorAugmentedDAE"]
