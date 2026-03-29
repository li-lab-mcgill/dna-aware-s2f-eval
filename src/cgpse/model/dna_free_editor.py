import json
import torch
import torch.nn as nn

try:
    # Package-relative imports (when imported as a module)
    from .components.encoder import TrackEncoderRouter
    from .components.decoder import TrackDecoder
    from .components.bottleneck import Bottleneck, BottleneckEncoder, BottleneckDecoder
except ImportError:
    # Fallback for running this file as a script: `python model.py`
    from components.encoder import TrackEncoderRouter
    from components.decoder import TrackDecoder
    from components.bottleneck import Bottleneck, BottleneckEncoder, BottleneckDecoder


class DNAFreeDAE(nn.Module):
    """
    DNA-free denoising autoencoder (DAE) over tracks.

    Conceptual role (Stage-2 in the paper):
        (Y_GT)      ->  \hat{Y}_GT
        (Y_s2f)     ->  \hat{Y}_s2f    ≈ Y_GT

    This implementation focuses on the core architectural path:
      1) TrackEncoderRouter:
           - Track-only encoders for different routes (e.g. "gt", "s2f").
           - Each route is a TrackEncoder; no DNA branch or fusion.
      2) BottleneckEncoder:
           - SwiGLU projector + Transformer at conv-latent resolution.
      3) Bottleneck + BottleneckDecoder:
           - Explicit (C_in -> C_lat -> d_model) SwiGLU bottleneck and matching
             Transformer mixer at d_model.
      4) TrackDecoder:
           - 1D upsampling conv decoder back to track space.

    The model is intentionally small and operates purely in track space; any
    losses, critic usage, and mask logic live outside this module.
    """

    def __init__(
        self,
        in_channels: int,
        *,
        # Encoder conv config
        base_channels: int = 16,
        conv_latent_channels: int | None = None,
        encoder_kernel_size: int = 5,
        stem_kernel_size: int | None = None,
        # Bottleneck encoder config
        bottleneck_channels: int = 32,
        bottleneck_swiglu_use_layernorm: bool = False,
        # Bottleneck decoder / transformer config
        model_dim: int | None = None,
        transformer_depth: int = 1,
        transformer_heads: int = 4,
        transformer_dim_head: int | None = None,
        transformer_kv_heads: int | None = None,
        transformer_attn_dropout: float = 0.1,
        transformer_proj_dropout: float = 0.1,
        transformer_ff_mult: float = 2.0,
        transformer_ff_dropout: float = 0.1,
        transformer_rope_fraction: float = 0.5,
        transformer_rope_theta: float = 20000.0,
        bottleneck_decoder_swiglu_use_layernorm: bool = False,
        # Decoder conv config
        decoder_kernel_size: int = 5,
    ):
        super().__init__()

        # --------- Infer defaults ---------
        if conv_latent_channels is None:
            conv_latent_channels = base_channels * 4  # e.g. 64 when base=16

        if model_dim is None:
            # By default, make d_model equal to conv_latent_channels
            model_dim = conv_latent_channels

        self.in_channels = int(in_channels)
        self.base_channels = int(base_channels)
        self.conv_latent_channels = int(conv_latent_channels)
        self.bottleneck_channels = int(bottleneck_channels)
        self.model_dim = int(model_dim)

        # --------- Track-only conv encoder router (1000 -> 125) ---------
        # Routes correspond to different input semantics (e.g. "gt", "s2f").
        self.encoder_router = TrackEncoderRouter(
            track_in_channels=self.in_channels,
            base_channels=self.base_channels,
            conv_latent_channels=self.conv_latent_channels,
            encoder_kernel_size=encoder_kernel_size,
            stem_kernel_size=stem_kernel_size,
            down_kernel_size=encoder_kernel_size,
            routes=("gt", "s2f"),
        )

        # --------- Bottleneck encoder (conv_lat -> mixed conv_lat) ---------
        self.bottleneck_encoder = BottleneckEncoder(
            conv_latent_channels=self.conv_latent_channels,
            bottleneck_channels=self.bottleneck_channels,
            transformer_depth=transformer_depth,
            transformer_heads=transformer_heads,
            transformer_dim_head=transformer_dim_head,
            transformer_kv_heads=transformer_kv_heads,
            attn_dropout=transformer_attn_dropout,
            proj_dropout=transformer_proj_dropout,
            ff_mult=transformer_ff_mult,
            ff_dropout=transformer_ff_dropout,
            rope_fraction=transformer_rope_fraction,
            rope_theta=transformer_rope_theta,
            swiglu_use_layernorm=bottleneck_swiglu_use_layernorm,
        )

        # --------- Explicit channel bottleneck C_conv_lat -> C_lat -> d_model ---------
        self.bottleneck = Bottleneck(
            in_channels=self.conv_latent_channels,
            bottleneck_channels=self.bottleneck_channels,
            out_channels=self.model_dim,
        )

        # --------- Bottleneck decoder (d_model -> d_model) ---------
        self.bottleneck_decoder = BottleneckDecoder(
            bottleneck_channels=self.bottleneck_channels,
            model_dim=self.model_dim,
            transformer_depth=transformer_depth,
            transformer_heads=transformer_heads,
            transformer_dim_head=transformer_dim_head,
            transformer_kv_heads=transformer_kv_heads,
            attn_dropout=transformer_attn_dropout,
            proj_dropout=transformer_proj_dropout,
            ff_mult=transformer_ff_mult,
            ff_dropout=transformer_ff_dropout,
            rope_fraction=transformer_rope_fraction,
            rope_theta=transformer_rope_theta,
            swiglu_use_layernorm=bottleneck_decoder_swiglu_use_layernorm,
        )

        # --------- Track decoder (125 -> 1000) ---------
        self.decoder = TrackDecoder(
            out_channels=self.in_channels,
            base_channels=self.base_channels,
            latent_channels=self.model_dim,
            kernel_size=decoder_kernel_size,
        )

    # ------------------------------------------------------------------
    # Helper methods for latents
    # ------------------------------------------------------------------

    def encode_conv(
        self,
        track: torch.Tensor,
        route: str,
    ) -> torch.Tensor:
        """
        Encode tracks for a given route (e.g. "gt" or "s2f") to a conv latent:

            track    : (B, C_in, L)

        Returns:
            h_conv: (B, C_conv_lat, L_lat)
        """
        return self.encoder_router(track, route)

    def encode_bottleneck(
        self,
        track: torch.Tensor,
        route: str,
    ) -> torch.Tensor:
        """
        Encode tracks all the way to bottleneck z for a given route:

            track   : (B, C_in, L)

        Returns:
            z: (B, C_lat, L_lat)
        """
        h_conv = self.encode_conv(track, route)               # (B, C_conv_lat, L_lat)
        h_enc = self.bottleneck_encoder(h_conv)               # (B, C_conv_lat, L_lat)
        _, z = self.bottleneck(h_enc)                         # (B, d_model, L_lat), (B, C_lat, L_lat)
        return z

    def decode_from_bottleneck(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode from bottleneck z all the way back to track space:

            z: (B, C_lat, L_lat)
                -> h_pre:  (B, d_model, L_lat)  via bottleneck.decode
                -> h_bott: (B, d_model, L_lat)  via BottleneckDecoder
                -> y_hat:  (B, C_in,   L_out)   via TrackDecoder
        """
        # Lift bottleneck code back to model_dim.
        h_pre = self.bottleneck.decode(z)              # (B, d_model, L_lat)
        # Apply transformer + SwiGLU at model_dim.
        h_bott = self.bottleneck_decoder(h_pre)        # (B, d_model, L_lat)
        # Final conv decoder to track space.
        return self.decoder(h_bott)                    # (B, C_in, L_out)

    def forward(
        self,
        track: torch.Tensor,
        route: str = "gt",
        return_latents: bool = False,
    ):
        """
        Forward pass for denoising in a given route.

        Args:
            track : (B, C_in, L_in) — input tracks.
            route : string key selecting which TrackEncoder branch to use
                    (e.g. "gt" or "s2f").
            return_latents: if True, also return intermediate latents.

        Returns:
            If return_latents == False:
                y_hat: (B, C_in, L_out)
            If return_latents == True:
                y_hat, latents_dict where latents_dict contains:
                  - "h_conv": (B, C_conv_lat, L_lat)
                  - "z":      (B, C_lat,      L_lat)
                  - "h_bott": (B, d_model,    L_lat)
        """
        # Conv encoder (route-specific)
        h_conv = self.encode_conv(track, route)             # (B, C_conv_lat, L_lat)
        # SwiGLU + Transformer encoder at conv-latent width
        h_enc = self.bottleneck_encoder(h_conv)             # (B, C_conv_lat, L_lat)
        # Explicit channel bottleneck C_conv_lat -> C_lat -> d_model
        h_pre, z = self.bottleneck(h_enc)                   # (B, d_model,    L_lat), (B, C_lat, L_lat)
        # Transformer + SwiGLU at model_dim
        h_bott = self.bottleneck_decoder(h_pre)             # (B, d_model,    L_lat)
        # Conv decoder back to tracks
        y_hat = self.decoder(h_bott)                        # (B, C_in,       L_out)

        if not return_latents:
            return y_hat

        latents = {
            "h_conv": h_conv,
            "h_enc": h_enc,
            "z": z,
            "h_pre": h_pre,
            "h_bott": h_bott,
        }
        return y_hat, latents

    def describe(self) -> dict:
        """
        Returns a nested dictionary describing the configuration of each component.
        Useful for debugging / audits without inspecting source files.
        """
        def _count_params(module: nn.Module) -> int:
            return sum(p.numel() for p in module.parameters())

        # Expect "gt" and "s2f" routes by construction.
        encoders = self.encoder_router.encoders
        enc_gt = encoders["gt"] if "gt" in encoders else None
        enc_s2f = encoders["s2f"] if "s2f" in encoders else None

        desc = {
            "top_level": {
                "in_channels": self.in_channels,
                "base_channels": self.base_channels,
                "conv_latent_channels": self.conv_latent_channels,
                "bottleneck_channels": self.bottleneck_channels,
                "model_dim": self.model_dim,
            },
            "track_encoder_router": {
                "routes": list(self.encoder_router.encoders.keys()),
                "gt_encoder": (
                    {
                        "in_channels": enc_gt.in_channels,
                        "base_channels": enc_gt.base_channels,
                        "latent_channels": enc_gt.latent_channels,
                        "stem_kernel_size": enc_gt.stem_kernel_size,
                        "down_kernel_size": enc_gt.down_kernel_size,
                    }
                    if enc_gt is not None
                    else None
                ),
                "s2f_encoder": (
                    {
                        "in_channels": enc_s2f.in_channels,
                        "base_channels": enc_s2f.base_channels,
                        "latent_channels": enc_s2f.latent_channels,
                        "stem_kernel_size": enc_s2f.stem_kernel_size,
                        "down_kernel_size": enc_s2f.down_kernel_size,
                    }
                    if enc_s2f is not None
                    else None
                ),
            },
            "bottleneck_encoder": {
                "conv_latent_channels": self.bottleneck_encoder.conv_latent_channels,
                "bottleneck_channels": self.bottleneck_encoder.bottleneck_channels,
            },
            "bottleneck": {
                "in_channels": self.bottleneck.in_channels,
                "bottleneck_channels": self.bottleneck.bottleneck_channels,
                "out_channels": self.bottleneck.out_channels,
            },
            "bottleneck_decoder": {
                "model_dim": self.bottleneck_decoder.model_dim,
            },
            "track_decoder": {
                "out_channels": self.decoder.out_channels,
                "base_channels": self.decoder.base_channels,
                "latent_channels": self.decoder.latent_channels,
            },
            "param_counts": {
                "track_encoder_router": {
                    "gt_encoder": _count_params(enc_gt) if enc_gt is not None else 0,
                    "s2f_encoder": _count_params(enc_s2f) if enc_s2f is not None else 0,
                },
                "bottleneck_encoder": {
                    "total": _count_params(self.bottleneck_encoder),
                    "pre_swiglu": _count_params(self.bottleneck_encoder.pre_swiglu),
                    "transformer": _count_params(self.bottleneck_encoder.transformer),
                },
                "bottleneck": {
                    "total": _count_params(self.bottleneck),
                    "to_latent": _count_params(self.bottleneck.to_latent),
                    "from_latent": _count_params(self.bottleneck.from_latent),
                },
                "bottleneck_decoder": {
                    "total": _count_params(self.bottleneck_decoder),
                    "transformer": _count_params(self.bottleneck_decoder.transformer),
                    "post_swiglu": _count_params(self.bottleneck_decoder.post_swiglu),
                },
                "track_decoder": _count_params(self.decoder),
            },
            "total_parameters": sum(p.numel() for p in self.parameters()),
            "trainable_parameters": sum(p.numel() for p in self.parameters() if p.requires_grad),
        }
        return desc


__all__ = ["DNAFreeDAE"]


if __name__ == "__main__":
    """
    Lightweight smoke test for DNAFreeDAE:
      - Instantiates the model with a small config.
      - Prints total and trainable parameter counts.
      - Runs a single forward pass on mock inputs and prints output/latent shapes.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Mock dimensions: 1 track channel, 1000bp window.
    B = 2
    L = 1000
    C_in = 1

    model = DNAFreeDAE(
        in_channels=C_in,
        base_channels=16,
        conv_latent_channels=64,
        encoder_kernel_size=5,
        bottleneck_channels=32,
        transformer_depth=1,
        transformer_heads=4,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"DNAFreeDAE total params    : {total_params:,}")
    print(f"DNAFreeDAE trainable params: {trainable_params:,}")

    # Mock track input: (B, C_in, L)
    track = torch.randn(B, C_in, L, device=device)

    print("Model specification:")
    print(json.dumps(model.describe(), indent=2))

    with torch.no_grad():
        y_hat, latents = model(track, route="gt", return_latents=True)

    print(f"Input track shape      : {track.shape}")
    print(f"Output y_hat shape     : {y_hat.shape}")
    for name, tensor in latents.items():
        print(f"Latent '{name}' shape    : {tuple(tensor.shape)}")
