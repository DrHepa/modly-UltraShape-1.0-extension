"""Autoencoder model subset for local runtime wiring."""

from __future__ import annotations

from ...utils.checkpoint import checkpoint_tokens
from ...utils.tensors import blend_sequences, clamp_unit, stable_signature


class ShapeVAE:
    scope = 'mc-only'

    def __init__(self, checkpoint_state: object = None):
        self.checkpoint_state = checkpoint_state

    def decode_latents(self, latents: dict[str, object], reference_asset: dict[str, object]) -> dict[str, object]:
        latent_values = latents.get('latents') if isinstance(latents.get('latents'), list) else []
        reference_tokens = reference_asset.get('tokens') if isinstance(reference_asset.get('tokens'), list) else []
        checkpoint_signal = checkpoint_tokens(self.checkpoint_state)
        decoded = blend_sequences(latent_values, reference_tokens, checkpoint_signal)[:8]
        decoded = [clamp_unit((value * 0.8) + 0.05) for value in decoded]
        return {
            'decoded_latents': decoded,
            'decoded_signature': stable_signature(decoded),
        }


class UltraShapeAutoencoder(ShapeVAE):
    pass
