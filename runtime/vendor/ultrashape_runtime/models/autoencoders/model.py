"""Autoencoder model subset for local runtime wiring."""

from __future__ import annotations

from ...utils.checkpoint import checkpoint_tokens
from ...utils.tensors import blend_sequences, clamp_unit, stable_signature


class ShapeVAE:
    scope = 'mc-only'

    def __init__(self, checkpoint_state: object = None):
        self.checkpoint_state = checkpoint_state
        self.hydrated = checkpoint_state is not None

    def load_state_dict(self, state_dict: dict[str, object], strict: bool = False) -> dict[str, object]:
        self.checkpoint_state = state_dict
        self.hydrated = True
        return {'missing_keys': [], 'unexpected_keys': [], 'strict': strict}

    def decode_latents(self, latents: dict[str, object], reference_asset: dict[str, object]) -> dict[str, object]:
        latent_values = latents.get('latents') if isinstance(latents.get('latents'), list) else []
        reference_tokens = reference_asset.get('tokens') if isinstance(reference_asset.get('tokens'), list) else []
        checkpoint_signal = checkpoint_tokens(self.checkpoint_state)
        decoded = blend_sequences(latent_values, reference_tokens, checkpoint_signal)
        decoded = [clamp_unit((value * 0.82) + 0.04) for value in decoded[:12]]
        mesh_seed = blend_sequences(decoded, latent_values, checkpoint_signal)[:12]
        return {
            'vae': self.__class__.__name__,
            'decoded_latents': decoded,
            'mesh_seed': mesh_seed,
            'decoded_signature': stable_signature(decoded),
            'state_hydrated': self.hydrated,
        }


class UltraShapeAutoencoder(ShapeVAE):
    pass
