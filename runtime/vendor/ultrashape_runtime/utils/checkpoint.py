"""Checkpoint metadata helpers."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import torch

from . import clamp_unit, stable_signature

REQUIRED_SUBTREES = ('vae', 'dit', 'conditioner')
MAX_TENSOR_SAMPLES = 8


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
    summary_tokens: list[float] = []
    tensor_count = 0
    value_count = 0
    for name in expected:
        subtree = parsed.get(name)
        if not isinstance(subtree, Mapping):
            raise CheckpointResolutionError(f'Checkpoint subtree {name} must be an object.')

        tensors = _normalize_checkpoint_subtree(subtree)
        if not tensors:
            raise CheckpointResolutionError(f'Checkpoint subtree {name} must include non-empty tensor data.')

        subtree_tokens: list[float] = []
        subtree_value_count = 0
        for tensor_name, tensor_summary in tensors.items():
            if not isinstance(tensor_name, str) or not tensor_name.strip():
                raise CheckpointResolutionError(f'Checkpoint subtree {name} has an invalid tensor key.')
            if not isinstance(tensor_summary, Mapping):
                raise CheckpointResolutionError(f'Checkpoint subtree {name}.{tensor_name} must be summarized tensor data.')

            sample = tensor_summary.get('sample')
            if not isinstance(sample, list) or not sample:
                raise CheckpointResolutionError(f'Checkpoint subtree {name}.{tensor_name} must provide compact tensor samples.')

            subtree_tokens.extend(float(value) for value in sample if isinstance(value, (int, float)))
            subtree_value_count += int(tensor_summary.get('value_count', 0))
            tensor_count += 1

        compact_tokens = subtree_tokens[:MAX_TENSOR_SAMPLES]
        summary_tokens.extend(compact_tokens)
        value_count += subtree_value_count
        bundle[name] = {
            'state_dict': dict(subtree),
            'tensors': tensors,
            'representation': 'checkpoint-subtree-v1',
            'tensor_count': len(tensors),
            'value_count': subtree_value_count,
            'tokens': compact_tokens,
            'signature': stable_signature(compact_tokens),
            'evidence': {
                'tensor_count': len(tensors),
                'value_count': subtree_value_count,
            },
        }

    return {
        'path': resolved_path,
        'subtrees_loaded': list(expected),
        'bundle': bundle,
        'summary': {
            'format': 'pytorch-binary-checkpoint',
            'representation': 'tensor-summary-v1',
            'tensor_count': tensor_count,
            'value_count': value_count,
            'signature': stable_signature(summary_tokens),
        },
    }


def checkpoint_tokens(checkpoint_state: object, *, limit: int = MAX_TENSOR_SAMPLES) -> list[float]:
    if not isinstance(checkpoint_state, Mapping):
        return []

    direct_tokens = checkpoint_state.get('tokens')
    if isinstance(direct_tokens, list):
        return [clamp_unit(float(value)) for value in direct_tokens[:limit] if isinstance(value, (int, float))]

    tensors = checkpoint_state.get('tensors')
    if not isinstance(tensors, Mapping):
        return []

    tokens: list[float] = []
    for tensor in tensors.values():
        if isinstance(tensor, list):
            tokens.extend(clamp_unit(float(value)) for value in tensor if isinstance(value, (int, float)))
        elif isinstance(tensor, Mapping):
            sample = tensor.get('sample')
            if isinstance(sample, list):
                tokens.extend(clamp_unit(float(value)) for value in sample if isinstance(value, (int, float)))
            elif isinstance(tensor.get('mean'), (int, float)):
                tokens.append(clamp_unit(float(tensor['mean'])))

        if len(tokens) >= limit:
            return tokens[:limit]

    return tokens[:limit]


def checkpoint_signature(checkpoint_state: object) -> int:
    if isinstance(checkpoint_state, Mapping) and isinstance(checkpoint_state.get('signature'), int):
        return int(checkpoint_state['signature'])
    return stable_signature(checkpoint_tokens(checkpoint_state, limit=MAX_TENSOR_SAMPLES))


def checkpoint_tensor_count(checkpoint_state: object) -> int:
    if isinstance(checkpoint_state, Mapping) and isinstance(checkpoint_state.get('tensor_count'), int):
        return int(checkpoint_state['tensor_count'])

    tensors = checkpoint_state.get('tensors') if isinstance(checkpoint_state, Mapping) else None
    return len(tensors) if isinstance(tensors, Mapping) else 0


def checkpoint_value_count(checkpoint_state: object) -> int:
    if isinstance(checkpoint_state, Mapping) and isinstance(checkpoint_state.get('value_count'), int):
        return int(checkpoint_state['value_count'])

    tensors = checkpoint_state.get('tensors') if isinstance(checkpoint_state, Mapping) else None
    if not isinstance(tensors, Mapping):
        return 0

    total = 0
    for tensor in tensors.values():
        if isinstance(tensor, Mapping) and isinstance(tensor.get('value_count'), int):
            total += int(tensor['value_count'])
        elif isinstance(tensor, (list, tuple)):
            total += len(tensor)
    return total


def apply_checkpoint_state(module: object, checkpoint_state: object, *, strict: bool = False) -> dict[str, object]:
    if not isinstance(checkpoint_state, Mapping):
        raise CheckpointResolutionError('Checkpoint hydration requires structured checkpoint state.')

    loader = getattr(module, 'load_state_dict', None)
    if not callable(loader):
        raise CheckpointResolutionError(f'{module.__class__.__name__} does not support load_state_dict hydration.')

    state_dict = checkpoint_state.get('state_dict')
    if not isinstance(state_dict, Mapping):
        raise CheckpointResolutionError('Checkpoint hydration requires a state_dict mapping.')

    loader(dict(state_dict), strict=strict)
    return {
        'module': module.__class__.__name__,
        'load_style': 'load_state_dict',
        'strict': strict,
        'signature': checkpoint_signature(checkpoint_state),
        'tensor_count': checkpoint_tensor_count(checkpoint_state),
        'value_count': checkpoint_value_count(checkpoint_state),
    }


def _normalize_checkpoint_subtree(subtree: Mapping[str, object]) -> dict[str, dict[str, object]]:
    tensor_source = subtree.get('tensors') if isinstance(subtree.get('tensors'), Mapping) else subtree
    normalized_tensors = _collect_tensor_values(tensor_source)
    return {name: values for name, values in normalized_tensors.items() if values.get('sample')}


def _collect_tensor_values(node: object, prefix: str = '') -> dict[str, dict[str, object]]:
    if not isinstance(node, Mapping):
        if not prefix:
            return {}
        values = _coerce_tensor_values(node)
        return {prefix: values} if values else {}

    tensors: dict[str, dict[str, object]] = {}
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


def _coerce_tensor_values(value: object) -> dict[str, object] | None:
    if isinstance(value, (int, float)):
        normalized = clamp_unit(float(value))
        return {
            'sample': [normalized],
            'sample_count': 1,
            'value_count': 1,
            'min': normalized,
            'max': normalized,
            'mean': normalized,
            'signature': stable_signature([normalized]),
        }

    if isinstance(value, (list, tuple)):
        return _summarize_iterable(value)

    tensor_like = value
    for method_name in ('detach', 'cpu', 'float'):
        method = getattr(tensor_like, method_name, None)
        if callable(method):
            tensor_like = method()

    reshape = getattr(tensor_like, 'reshape', None)
    if callable(reshape):
        tensor_like = reshape(-1)
    else:
        flatten = getattr(tensor_like, 'flatten', None)
        if callable(flatten):
            tensor_like = flatten()

    numel = getattr(tensor_like, 'numel', None)
    if callable(numel):
        value_count = int(numel())
        if value_count <= 0:
            return None

        sample_indices = _sample_indices(value_count, MAX_TENSOR_SAMPLES)
        sample = [clamp_unit(_tensor_scalar(tensor_like, index)) for index in sample_indices]

        return {
            'sample': sample,
            'sample_count': len(sample),
            'value_count': value_count,
            'min': clamp_unit(_tensor_reduce_scalar(tensor_like, 'min')),
            'max': clamp_unit(_tensor_reduce_scalar(tensor_like, 'max')),
            'mean': clamp_unit(_tensor_reduce_scalar(tensor_like, 'mean')),
            'signature': stable_signature(sample),
        }

    return None


def _summarize_iterable(values: list[object] | tuple[object, ...]) -> dict[str, object] | None:
    sample: list[float] = []
    value_count = 0
    total = 0.0
    minimum: float | None = None
    maximum: float | None = None

    def visit(node: object) -> None:
        nonlocal value_count, total, minimum, maximum
        if isinstance(node, (int, float)):
            normalized = clamp_unit(float(node))
            value_count += 1
            total += normalized
            minimum = normalized if minimum is None else min(minimum, normalized)
            maximum = normalized if maximum is None else max(maximum, normalized)
            if len(sample) < MAX_TENSOR_SAMPLES:
                sample.append(normalized)
            return

        if isinstance(node, (list, tuple)):
            for item in node:
                visit(item)

    visit(values)
    if value_count <= 0 or not sample:
        return None

    return {
        'sample': sample,
        'sample_count': len(sample),
        'value_count': value_count,
        'min': minimum,
        'max': maximum,
        'mean': clamp_unit(total / value_count),
        'signature': stable_signature(sample),
    }


def _sample_indices(value_count: int, sample_size: int) -> list[int]:
    capped = min(value_count, sample_size)
    if capped <= 0:
        return []
    if value_count <= capped:
        return list(range(value_count))

    last_index = value_count - 1
    return sorted({round((last_index * index) / (capped - 1)) for index in range(capped)})


def _tensor_scalar(tensor_like: object, index: int) -> float:
    try:
        value = tensor_like[index]
    except TypeError:
        value = tensor_like.__getitem__(index)
    return _to_python_float(value)


def _tensor_reduce_scalar(tensor_like: object, method_name: str) -> float:
    method = getattr(tensor_like, method_name)
    return _to_python_float(method())


def _to_python_float(value: object) -> float:
    item = getattr(value, 'item', None)
    if callable(item):
        value = item()
    return float(value)
