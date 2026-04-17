"""Surface extractor preference for the mc-only MVP."""

from __future__ import annotations

try:
    import cubvh  # type: ignore # pragma: no cover
except ImportError:  # pragma: no cover - expected on degraded installs
    cubvh = None

from ...utils.mesh import build_renderable_mesh_payload


def preferred_surface_extractor() -> str:
    return 'cubvh.sparse_marching_cubes' if cubvh is not None else 'cubvh.missing'


class SurfaceExtractionError(Exception):
    code = 'LOCAL_RUNTIME_UNAVAILABLE'


class SurfaceExtractionDependencyError(Exception):
    code = 'DEPENDENCY_MISSING'


def extract_mc_surface(
    *,
    coarse_surface: dict[str, object],
    reference_asset: dict[str, object],
    decoded_volume: dict[str, object],
    preserve_scale: bool,
) -> dict[str, object]:
    extractor = preferred_surface_extractor()
    if extractor != 'cubvh.sparse_marching_cubes':
        raise SurfaceExtractionDependencyError('Required runtime import is unavailable: cubvh.')

    mesh_payload = coarse_surface.get('mesh')
    if not isinstance(mesh_payload, dict):
        raise SurfaceExtractionError('coarse_surface.mesh must be a structured binary-safe mesh payload.')

    field_signature = decoded_volume.get('field_signature') if isinstance(decoded_volume, dict) else None
    if not isinstance(field_signature, int):
        raise SurfaceExtractionError('decoded_volume.field_signature must be an integer.')

    dense_field = decoded_volume.get('dense_field') if isinstance(decoded_volume, dict) else []
    field_bytes = bytes(int(round(float(value) * 255)) for value in dense_field if isinstance(value, (int, float)))
    payload_bytes = (
        f'ultrashape:{extractor}:{field_signature}:{reference_asset["signature"]}:{preserve_scale}:'.encode('utf8')
        + field_bytes
        + b'\x80\x01'
    )

    renderable_payload = build_renderable_mesh_payload(
        {
            'kind': 'refined-mesh',
            'path': mesh_payload.get('path'),
            'bytes': payload_bytes,
            'byte_length': len(payload_bytes),
            'is_binary_glb': False,
            'mesh_name': 'refined-surface',
        }
    )

    return {
        'extractor': extractor,
        'preserve_scale': preserve_scale,
        'payload': renderable_payload,
        'reference_bytes': reference_asset['byte_length'],
        'payload_bytes': len(payload_bytes),
        'surface_signature': field_signature,
    }


def extract_surface(
    *,
    extraction: str,
    coarse_surface: dict[str, object],
    reference_asset: dict[str, object],
    decoded_volume: dict[str, object],
    preserve_scale: bool,
) -> dict[str, object]:
    if extraction != 'mc':
        raise SurfaceExtractionError(f'UltraShape local runner is mc-only in this MVP, received extraction={extraction}.')

    return extract_mc_surface(
        coarse_surface=coarse_surface,
        reference_asset=reference_asset,
        decoded_volume=decoded_volume,
        preserve_scale=preserve_scale,
    )
