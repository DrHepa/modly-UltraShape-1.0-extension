"""Reference asset preprocessing for the local mc-only runner."""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

from .rembg import maybe_apply_cutout, payload_has_cutout_alpha
from .utils import blend_sequences, bytes_to_unit_floats, clamp_unit, stable_signature


PNG_SIGNATURE = b'\x89PNG\r\n\x1a\n'


class _CompatState(dict):
    def __init__(self, *args, compat: dict[str, object] | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._compat = compat or {}

    def _resolve(self, key: str):
        value = self._compat[key]
        return value(self) if callable(value) else value

    def get(self, key, default=None):
        if dict.__contains__(self, key):
            return dict.get(self, key, default)
        if key in self._compat:
            return self._resolve(key)
        return default

    def __getitem__(self, key):
        if dict.__contains__(self, key):
            return dict.__getitem__(self, key)
        if key in self._compat:
            return self._resolve(key)
        raise KeyError(key)

    def __contains__(self, key):
        return dict.__contains__(self, key) or key in self._compat


class ReferencePreprocessError(Exception):
    code = 'LOCAL_RUNTIME_UNAVAILABLE'


class ImageProcessorV2:
    def process(self, path: str) -> dict[str, object]:
        asset_path = Path(path)
        if not asset_path.is_file():
            raise ReferencePreprocessError(f'reference_image is not readable: {path}.')

        content = asset_path.read_bytes()
        processed_bytes, cutout_applied = maybe_apply_cutout(content, require_cutout='conditional')
        image_data = _decode_image(processed_bytes)
        image_tokens = image_data['image_features']
        mask_tokens = image_data['mask_features']
        conditioning_tokens = blend_sequences(image_tokens, mask_tokens)[:8]

        conditioning_tokens = blend_sequences(image_tokens, mask_tokens)[:8]
        return _CompatState({
            'path': str(asset_path),
            'processor': self.__class__.__name__,
            'byte_length': len(processed_bytes),
            'has_content': bool(processed_bytes),
            'normalized_channels': 4,
            'pixel_count': image_data['pixel_count'],
            'source_format': image_data['source_format'],
            'image_tensor': image_data['image_tensor'],
            'image_tensor_shape': image_data['image_tensor_shape'],
            'mask_tensor': image_data['mask_tensor'],
            'mask_tensor_shape': image_data['mask_tensor_shape'],
            'image_meta': image_data['image_meta'],
            'image_features': image_tokens,
            'mask_features': mask_tokens,
            'mean_intensity': image_data['mean_intensity'],
            'mask_coverage': image_data['mask_coverage'],
            'cutout_applied': cutout_applied,
            'had_cutout_alpha': payload_has_cutout_alpha(content),
            'evidence': {
                'mean_intensity': image_data['mean_intensity'],
                'mask_coverage': image_data['mask_coverage'],
                'cutout_applied': cutout_applied,
            },
        }, compat={
            'tokens': lambda state: conditioning_tokens,
            'image_tokens': lambda state: list(state['image_features']),
            'mask_tokens': lambda state: list(state['mask_features']),
            'image_signature': lambda state: stable_signature(state['image_features']),
            'mask_signature': lambda state: stable_signature(state['mask_features']),
            'signature': lambda state: stable_signature(conditioning_tokens),
        })


def normalize_reference_asset(path: str) -> dict[str, object]:
    return ImageProcessorV2().process(path)


def _decode_image(payload: bytes) -> dict[str, object]:
    if payload[:8] == PNG_SIGNATURE:
        return _decode_png(payload)

    normalized = bytes_to_unit_floats(payload, length=16)
    rgba_pixels = [normalized[index : index + 4] for index in range(0, len(normalized), 4) if len(normalized[index : index + 4]) == 4]
    return _build_image_data(rgba_pixels, width=max(len(rgba_pixels), 1), height=1, source_format='raw-bytes')


def _decode_png(payload: bytes) -> dict[str, object]:
    if payload[:8] != PNG_SIGNATURE:
        raise ReferencePreprocessError('reference_image must be a readable PNG for this MVP path.')

    width = 0
    height = 0
    bit_depth = 0
    color_type = 0
    interlace = 0
    chunks: list[bytes] = []
    offset = 8
    while offset + 8 <= len(payload):
        length = struct.unpack_from('>I', payload, offset)[0]
        offset += 4
        chunk_type = payload[offset : offset + 4]
        offset += 4
        chunk_data = payload[offset : offset + length]
        offset += length + 4
        if chunk_type == b'IHDR':
            width, height, bit_depth, color_type, _, _, interlace = struct.unpack('>IIBBBBB', chunk_data)
        elif chunk_type == b'IDAT':
            chunks.append(chunk_data)
        elif chunk_type == b'IEND':
            break

    if width <= 0 or height <= 0 or not chunks:
        raise ReferencePreprocessError('reference_image PNG is missing required image data chunks.')
    if bit_depth != 8 or color_type not in {2, 6} or interlace != 0:
        raise ReferencePreprocessError('reference_image PNG must be non-interlaced RGB/RGBA with 8-bit channels.')

    bytes_per_pixel = 4 if color_type == 6 else 3
    stride = width * bytes_per_pixel
    decoded = zlib.decompress(b''.join(chunks))
    rows: list[bytes] = []
    cursor = 0
    previous = bytes(stride)
    for _ in range(height):
        filter_type = decoded[cursor]
        cursor += 1
        filtered = bytearray(decoded[cursor : cursor + stride])
        cursor += stride
        rows.append(_unfilter_scanline(filter_type, filtered, previous, bytes_per_pixel))
        previous = rows[-1]

    rgba_pixels: list[list[float]] = []
    for row in rows:
        for index in range(0, len(row), bytes_per_pixel):
            red = row[index] / 255.0
            green = row[index + 1] / 255.0
            blue = row[index + 2] / 255.0
            alpha = row[index + 3] / 255.0 if bytes_per_pixel == 4 else 1.0
            rgba_pixels.append([round(red, 6), round(green, 6), round(blue, 6), round(alpha, 6)])

    return _build_image_data(rgba_pixels, width=width, height=height, source_format='png')


def _unfilter_scanline(filter_type: int, filtered: bytearray, previous: bytes, bytes_per_pixel: int) -> bytes:
    restored = bytearray(len(filtered))
    for index, value in enumerate(filtered):
        left = restored[index - bytes_per_pixel] if index >= bytes_per_pixel else 0
        up = previous[index] if index < len(previous) else 0
        up_left = previous[index - bytes_per_pixel] if index >= bytes_per_pixel and index < len(previous) else 0
        if filter_type == 0:
            restored[index] = value
        elif filter_type == 1:
            restored[index] = (value + left) & 0xFF
        elif filter_type == 2:
            restored[index] = (value + up) & 0xFF
        elif filter_type == 3:
            restored[index] = (value + ((left + up) // 2)) & 0xFF
        elif filter_type == 4:
            restored[index] = (value + _paeth_predictor(left, up, up_left)) & 0xFF
        else:
            raise ReferencePreprocessError(f'Unsupported PNG filter type: {filter_type}.')
    return bytes(restored)


def _paeth_predictor(left: int, up: int, up_left: int) -> int:
    prediction = left + up - up_left
    left_distance = abs(prediction - left)
    up_distance = abs(prediction - up)
    up_left_distance = abs(prediction - up_left)
    if left_distance <= up_distance and left_distance <= up_left_distance:
        return left
    if up_distance <= up_left_distance:
        return up
    return up_left


def _build_image_data(
    rgba_pixels: list[list[float]], *, width: int, height: int, source_format: str
) -> dict[str, object]:
    image_tensor = [[[list(pixel) for pixel in rgba_pixels[(row * width) : ((row + 1) * width)]] for row in range(height)]]
    mask_tensor = [[[[pixel[3]] for pixel in rgba_pixels[(row * width) : ((row + 1) * width)]] for row in range(height)]]
    image_features = _pool_image_features(rgba_pixels, width, height)
    mask_features = _pool_mask_features(rgba_pixels, width, height)
    mean_intensity = clamp_unit(sum(_pixel_luminance(pixel) for pixel in rgba_pixels) / len(rgba_pixels) if rgba_pixels else 0.0)
    mask_coverage = clamp_unit(sum(pixel[3] for pixel in rgba_pixels) / len(rgba_pixels) if rgba_pixels else 0.0)
    return {
        'source_format': source_format,
        'pixel_count': len(rgba_pixels),
        'image_tensor': image_tensor,
        'image_tensor_shape': [1, height, width, 4],
        'mask_tensor': mask_tensor,
        'mask_tensor_shape': [1, height, width, 1],
        'image_meta': {
            'width': width,
            'height': height,
            'pixel_count': len(rgba_pixels),
            'source_format': source_format,
        },
        'image_features': image_features,
        'mask_features': mask_features,
        'mean_intensity': mean_intensity,
        'mask_coverage': mask_coverage,
    }


def _pool_image_features(rgba_pixels: list[list[float]], width: int, height: int) -> list[float]:
    quadrants = _quadrant_pixels(rgba_pixels, width, height)
    features: list[float] = []
    for pixels in quadrants:
        if not pixels:
            features.extend([0.0, 0.0])
            continue
        luminance = clamp_unit(sum(_pixel_luminance(pixel) for pixel in pixels) / len(pixels))
        chroma = clamp_unit(sum(((pixel[0] * 0.5) + (pixel[2] * 0.5)) for pixel in pixels) / len(pixels))
        features.extend([luminance, chroma])
    return features[:8]


def _pool_mask_features(rgba_pixels: list[list[float]], width: int, height: int) -> list[float]:
    quadrants = _quadrant_pixels(rgba_pixels, width, height)
    features: list[float] = []
    for pixels in quadrants:
        if not pixels:
            features.extend([0.0, 0.0])
            continue
        alpha_mean = clamp_unit(sum(pixel[3] for pixel in pixels) / len(pixels))
        opaque_ratio = clamp_unit(sum(1.0 for pixel in pixels if pixel[3] >= 0.5) / len(pixels))
        features.extend([alpha_mean, opaque_ratio])
    return features[:8]


def _quadrant_pixels(rgba_pixels: list[list[float]], width: int, height: int) -> list[list[list[float]]]:
    quadrants: list[list[list[float]]] = [[], [], [], []]
    for row in range(height):
        for column in range(width):
            pixel = rgba_pixels[(row * width) + column]
            quadrant = (0 if row < max(height / 2, 1) else 2) + (0 if column < max(width / 2, 1) else 1)
            quadrants[quadrant].append(pixel)
    return quadrants


def _pixel_luminance(pixel: list[float]) -> float:
    return clamp_unit((pixel[0] * 0.299) + (pixel[1] * 0.587) + (pixel[2] * 0.114))
