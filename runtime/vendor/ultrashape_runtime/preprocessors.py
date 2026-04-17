"""Reference asset preprocessing for the local mc-only runner."""

from __future__ import annotations

from pathlib import Path

from .rembg import maybe_apply_cutout, payload_has_cutout_alpha
from .utils.tensors import blend_sequences, bytes_to_unit_floats, clamp_unit, stable_signature


class ReferencePreprocessError(Exception):
    code = 'LOCAL_RUNTIME_UNAVAILABLE'


class ImageProcessorV2:
    def process(self, path: str) -> dict[str, object]:
        asset_path = Path(path)
        if not asset_path.is_file():
            raise ReferencePreprocessError(f'reference_image is not readable: {path}.')

        content = asset_path.read_bytes()
        processed_bytes, cutout_applied = maybe_apply_cutout(content, require_cutout='conditional')
        normalized = bytes_to_unit_floats(processed_bytes, length=32)
        rgba_quads = [normalized[index : index + 4] for index in range(0, len(normalized), 4) if len(normalized[index : index + 4]) == 4]
        image_tokens = [
            clamp_unit(((red * 0.299) + (green * 0.587) + (blue * 0.114))) for red, green, blue, _ in rgba_quads
        ]
        mask_tokens = [clamp_unit(alpha) for _, _, _, alpha in rgba_quads]
        conditioning_tokens = blend_sequences(image_tokens, mask_tokens)[:8]

        return {
            'path': str(asset_path),
            'processor': self.__class__.__name__,
            'byte_length': len(processed_bytes),
            'has_content': bool(processed_bytes),
            'normalized_channels': 4,
            'pixel_count': len(rgba_quads),
            'tokens': conditioning_tokens,
            'image_tokens': image_tokens,
            'mask_tokens': mask_tokens,
            'image_signature': stable_signature(image_tokens),
            'mask_signature': stable_signature(mask_tokens),
            'signature': stable_signature(conditioning_tokens),
            'mean_intensity': clamp_unit(sum(image_tokens) / len(image_tokens) if image_tokens else 0.0),
            'cutout_applied': cutout_applied,
            'had_cutout_alpha': payload_has_cutout_alpha(content),
        }


def normalize_reference_asset(path: str) -> dict[str, object]:
    return ImageProcessorV2().process(path)
