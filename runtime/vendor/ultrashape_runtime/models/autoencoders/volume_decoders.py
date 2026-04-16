"""Volume decoder placeholder for the vendored runtime subset."""

from __future__ import annotations

from ...utils.tensors import clamp_unit, stable_signature


def decoder_name() -> str:
    return 'mc-volume-decoder'


def decode_volume(decoded_latents: dict[str, object]) -> dict[str, object]:
    values = decoded_latents.get('decoded_latents') if isinstance(decoded_latents.get('decoded_latents'), list) else []
    dense_field = [clamp_unit((value * 0.9) + 0.03) for value in values]
    return {
        'decoder': decoder_name(),
        'dense_field': dense_field,
        'field_density': clamp_unit(sum(dense_field) / len(dense_field) if dense_field else 0.0),
        'field_signature': stable_signature(dense_field),
    }
