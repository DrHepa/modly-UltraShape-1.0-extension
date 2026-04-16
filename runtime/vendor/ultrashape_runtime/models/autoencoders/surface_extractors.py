"""Surface extractor preference for the mc-only MVP."""

from __future__ import annotations

try:
    import cubvh  # type: ignore # pragma: no cover
except ImportError:  # pragma: no cover - expected on degraded installs
    cubvh = None


def preferred_surface_extractor() -> str:
    return 'cubvh.sparse_marching_cubes' if cubvh is not None else 'skimage.measure.marching_cubes'


class SurfaceExtractionError(Exception):
    code = 'LOCAL_RUNTIME_UNAVAILABLE'


def extract_mc_surface(*, coarse_surface: dict[str, object], reference_asset: dict[str, object], preserve_scale: bool) -> dict[str, object]:
    extractor = preferred_surface_extractor()
    if extractor not in {'cubvh.sparse_marching_cubes', 'skimage.measure.marching_cubes'}:
        raise SurfaceExtractionError(f'Unsupported mc extractor: {extractor}.')

    mesh_payload = coarse_surface.get('mesh')
    if not isinstance(mesh_payload, dict):
        raise SurfaceExtractionError('coarse_surface.mesh must be a structured binary-safe mesh payload.')

    return {
        'extractor': extractor,
        'preserve_scale': preserve_scale,
        'payload': mesh_payload,
        'reference_bytes': reference_asset['byte_length'],
    }


def extract_surface(*, extraction: str, coarse_surface: dict[str, object], reference_asset: dict[str, object], preserve_scale: bool) -> dict[str, object]:
    if extraction != 'mc':
        raise SurfaceExtractionError(f'UltraShape local runner is mc-only in this MVP, received extraction={extraction}.')

    return extract_mc_surface(
        coarse_surface=coarse_surface,
        reference_asset=reference_asset,
        preserve_scale=preserve_scale,
    )
