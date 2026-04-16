"""Checkpoint metadata helpers."""

from __future__ import annotations

import json
from pathlib import Path

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
    checkpoint_payload = Path(resolved_path).read_text(encoding='utf8').strip()

    try:
        parsed = json.loads(checkpoint_payload)
    except json.JSONDecodeError as error:
        raise CheckpointResolutionError(
            f'Checkpoint metadata is unreadable for subtree validation: {resolved_path}.'
        ) from error

    if not isinstance(parsed, dict):
        raise CheckpointResolutionError(f'Checkpoint metadata must be a JSON object: {resolved_path}.')

    if parsed.get('format') != 'ultrashape-checkpoint-bundle/v1':
        raise CheckpointResolutionError(
            f'Checkpoint bundle must declare format=ultrashape-checkpoint-bundle/v1: {resolved_path}.'
        )

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
        if not isinstance(subtree, dict):
            raise CheckpointResolutionError(f'Checkpoint subtree {name} must be an object.')

        tensors = subtree.get('tensors')
        if not isinstance(tensors, dict) or not tensors:
            raise CheckpointResolutionError(f'Checkpoint subtree {name} must include non-empty tensor data.')

        normalized_tensors: dict[str, list[float]] = {}
        for tensor_name, values in tensors.items():
            if not isinstance(tensor_name, str) or not tensor_name.strip():
                raise CheckpointResolutionError(f'Checkpoint subtree {name} has an invalid tensor key.')
            if not isinstance(values, list) or not values:
                raise CheckpointResolutionError(f'Checkpoint subtree {name}.{tensor_name} must be a non-empty tensor list.')

            normalized: list[float] = []
            for value in values:
                if not isinstance(value, (int, float)):
                    raise CheckpointResolutionError(
                        f'Checkpoint subtree {name}.{tensor_name} must contain only numeric tensor values.'
                    )
                normalized.append(clamp_unit(float(value)))

            normalized_tensors[tensor_name] = normalized
            tensor_values.extend(normalized)
            tensor_count += 1

        bundle[name] = {
            'tensors': normalized_tensors,
            'tensor_count': len(normalized_tensors),
            'value_count': sum(len(values) for values in normalized_tensors.values()),
            'signature': stable_signature([value for values in normalized_tensors.values() for value in values]),
        }

    return {
        'path': resolved_path,
        'subtrees_loaded': list(expected),
        'bundle': bundle,
        'summary': {
            'format': parsed.get('format'),
            'tensor_count': tensor_count,
            'value_count': len(tensor_values),
            'signature': stable_signature(tensor_values),
        },
    }
