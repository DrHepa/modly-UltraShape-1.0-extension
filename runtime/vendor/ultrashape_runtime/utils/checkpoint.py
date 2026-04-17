"""Checkpoint metadata helpers."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import torch

from .tensors import clamp_unit, stable_signature

REQUIRED_SUBTREES = ('vae', 'dit', 'conditioner')


class CheckpointResolutionError(Exception):
    code = 'WEIGHTS_MISSING'


def expected_checkpoint_name() -> str:
    return 'ultrashape_v1.pt'


def expected_checkpoint_subtrees() -> tuple[str, ...]:
    return REQUIRED_SUBTREES


def resolve_checkpoint(checkpoint: str | None, primary_weight: str | None, ext_dir: str) -> str:
    if checkpoint not in (None, ''):
        resolved = Path(checkpoint)
    else:
        relative_weight = primary_weight or f'models/ultrashape/{expected_checkpoint_name()}'
        resolved = Path(ext_dir) / relative_weight

    if not resolved.is_file():
        raise CheckpointResolutionError(f'Required checkpoint is not readable: {resolved}.')

    return str(resolved)


def load_checkpoint_subtrees(
    checkpoint: str | None,
    primary_weight: str | None,
    ext_dir: str,
    required_subtrees: list[str] | tuple[str, ...] | None = None,
) -> dict[str, object]:
    resolved_path = resolve_checkpoint(checkpoint, primary_weight, ext_dir)
    try:
        parsed = torch.load(resolved_path, map_location='cpu')
    except Exception as error:  # pragma: no cover - exercised via runtime seam tests
        raise CheckpointResolutionError(
            f'Checkpoint metadata is unreadable for subtree validation: {resolved_path}.'
        ) from error

    if not isinstance(parsed, Mapping):
        raise CheckpointResolutionError(f'Checkpoint metadata must be a checkpoint object: {resolved_path}.')

    expected = tuple(required_subtrees or expected_checkpoint_subtrees())
    missing = [name for name in expected if name not in parsed]
    if missing:
        raise CheckpointResolutionError(
            f'Required checkpoint subtrees are missing: {", ".join(missing)}.'
        )

    bundle: dict[str, object] = {}
    tensor_values: list[float] = []
    tensor_count = 0
    for name in expected:
        subtree = parsed.get(name)
        if not isinstance(subtree, Mapping):
            raise CheckpointResolutionError(f'Checkpoint subtree {name} must be an object.')

        tensors = _normalize_checkpoint_subtree(subtree)
        if not tensors:
            raise CheckpointResolutionError(f'Checkpoint subtree {name} must include non-empty tensor data.')

        for tensor_name, values in tensors.items():
            if not isinstance(tensor_name, str) or not tensor_name.strip():
                raise CheckpointResolutionError(f'Checkpoint subtree {name} has an invalid tensor key.')
            if not values:
                raise CheckpointResolutionError(f'Checkpoint subtree {name}.{tensor_name} must be a non-empty tensor list.')

            tensor_values.extend(values)
            tensor_count += 1

        bundle[name] = {
            'tensors': tensors,
            'tensor_count': len(tensors),
            'value_count': sum(len(values) for values in tensors.values()),
            'signature': stable_signature([value for values in tensors.values() for value in values]),
        }

    return {
        'path': resolved_path,
        'subtrees_loaded': list(expected),
        'bundle': bundle,
        'summary': {
            'format': 'pytorch-binary-checkpoint',
            'tensor_count': tensor_count,
            'value_count': len(tensor_values),
            'signature': stable_signature(tensor_values),
        },
    }


def _normalize_checkpoint_subtree(subtree: Mapping[str, object]) -> dict[str, list[float]]:
    tensor_source = subtree.get('tensors') if isinstance(subtree.get('tensors'), Mapping) else subtree
    normalized_tensors = _collect_tensor_values(tensor_source)
    return {name: values for name, values in normalized_tensors.items() if values}


def _collect_tensor_values(node: object, prefix: str = '') -> dict[str, list[float]]:
    if not isinstance(node, Mapping):
        if not prefix:
            return {}
        values = _coerce_tensor_values(node)
        return {prefix: values} if values else {}

    tensors: dict[str, list[float]] = {}
    for key, value in node.items():
        if not isinstance(key, str) or not key.strip():
            continue

        tensor_name = f'{prefix}.{key}' if prefix else key
        tensor_values = _coerce_tensor_values(value)
        if tensor_values:
            tensors[tensor_name] = tensor_values
            continue

        tensors.update(_collect_tensor_values(value, tensor_name))

    return tensors


def _coerce_tensor_values(value: object) -> list[float]:
    if isinstance(value, (int, float)):
        return [clamp_unit(float(value))]

    if isinstance(value, (list, tuple)):
        normalized: list[float] = []
        for item in value:
            normalized.extend(_coerce_tensor_values(item))
        return normalized

    tensor_like = value
    for method_name in ('detach', 'cpu'):
        method = getattr(tensor_like, method_name, None)
        if callable(method):
            tensor_like = method()

    flatten = getattr(tensor_like, 'flatten', None)
    if callable(flatten):
        tensor_like = flatten()

    tolist = getattr(tensor_like, 'tolist', None)
    if callable(tolist):
        return _coerce_tensor_values(tolist())

    return []
