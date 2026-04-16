"""Surface extractor preference for the mc-only MVP."""

try:
    import cubvh  # type: ignore # pragma: no cover
except ImportError:  # pragma: no cover - expected on degraded installs
    cubvh = None


def preferred_surface_extractor() -> str:
    return 'cubvh.sparse_marching_cubes' if cubvh is not None else 'skimage.measure.marching_cubes'
