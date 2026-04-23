"""Checkpoint metadata helpers."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

try:  # pragma: no cover - optional dependency seam
    import torch  # type: ignore
except ImportError:  # pragma: no cover - expected on degraded installs
    torch = None

from . import clamp_unit, stable_signature

REQUIRED_SUBTREES = ('vae', 'dit', 'conditioner')
MAX_TENSOR_SAMPLES = 8
MODULE_FAMILY_SPECS = {
    'conditioner': {
        'module_family': 'SingleImageEncoder',
        'required_roots': ('main_image_encoder',),
    },
    'dit': {
        'module_family': 'RefineDiT',
        'required_roots': ('x_embedder', 't_embedder', 'final_layer'),
    },
    'vae': {
        'module_family': 'ShapeVAE',
        'required_roots': ('post_kl', 'transformer', 'geo_decoder'),
    },
}
MODULE_STATE_DICT_REPRESENTATION = 'module-state-dict-v2'
LEGACY_SUBTREE_REPRESENTATION = 'checkpoint-subtree-v1'


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
    if torch is None:
        raise CheckpointResolutionError('Required runtime import is unavailable: torch.')
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

        state_dict_source = subtree.get('state_dict') if isinstance(subtree.get('state_dict'), Mapping) else None
        state_dict = _collect_state_dict_entries(state_dict_source if isinstance(state_dict_source, Mapping) else subtree)
        tensors = {tensor_name: _coerce_tensor_values(tensor_value) for tensor_name, tensor_value in state_dict.items()}
        tensors = {tensor_name: tensor_summary for tensor_name, tensor_summary in tensors.items() if tensor_summary}
        if not tensors:
            raise CheckpointResolutionError(f'Checkpoint subtree {name} must include non-empty tensor data.')

        family_spec = MODULE_FAMILY_SPECS.get(name, {})
        module_family = str(family_spec.get('module_family', name))
        module_roots = _module_roots(state_dict)
        required_roots = tuple(root for root in family_spec.get('required_roots', ()) if isinstance(root, str) and root)
        if state_dict_source is not None:
            missing_roots = [root for root in required_roots if root not in module_roots]
            if missing_roots:
                raise CheckpointResolutionError(
                    f'Checkpoint subtree {name} must include module-family roots: {", ".join(missing_roots)}.'
                )

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
        parameter_names = sorted(state_dict.keys())
        state_dict_metadata = {
            'tensor_count': len(tensors),
            'value_count': subtree_value_count,
            'tensor_names': sorted(tensors.keys()),
            'parameter_count': len(parameter_names),
            'parameter_names': parameter_names,
            'module_roots': module_roots,
            'module_family': module_family,
            'required_roots': list(required_roots),
            'representation': MODULE_STATE_DICT_REPRESENTATION if state_dict_source is not None else LEGACY_SUBTREE_REPRESENTATION,
        }
        bundle[name] = {
            'state_dict': state_dict,
            'state_dict_metadata': state_dict_metadata,
            'tensors': tensors,
            'representation': state_dict_metadata['representation'],
            'tensor_count': len(tensors),
            'parameter_count': len(parameter_names),
            'value_count': subtree_value_count,
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
            'representation': MODULE_STATE_DICT_REPRESENTATION if all(
                isinstance(bundle.get(name), Mapping)
                and bundle[name].get('representation') == MODULE_STATE_DICT_REPRESENTATION
                for name in expected
            ) else 'tensor-summary-v1',
            'tensor_count': tensor_count,
            'value_count': value_count,
            'signature': stable_signature(summary_tokens),
            'subtree_representations': {
                subtree_name: bundle[subtree_name].get('representation')
                for subtree_name in expected
                if isinstance(bundle.get(subtree_name), Mapping)
            },
        },
    }


def checkpoint_tokens(checkpoint_state: object, *, limit: int = MAX_TENSOR_SAMPLES) -> list[float]:
    if not isinstance(checkpoint_state, Mapping):
        return []

    state_dict = checkpoint_parameter_map(checkpoint_state)
    if state_dict:
        tokens: list[float] = []
        for parameter_name in sorted(state_dict.keys()):
            tensor_summary = _coerce_tensor_values(state_dict[parameter_name])
            if not tensor_summary:
                continue
            sample = tensor_summary.get('sample')
            if isinstance(sample, list):
                tokens.extend(clamp_unit(float(value)) for value in sample if isinstance(value, (int, float)))
            if len(tokens) >= limit:
                return tokens[:limit]
        return tokens[:limit]

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
    if isinstance(checkpoint_state, Mapping) and isinstance(checkpoint_state.get('parameter_count'), int):
        return int(checkpoint_state['parameter_count'])
    metadata = checkpoint_state.get('state_dict_metadata') if isinstance(checkpoint_state, Mapping) and isinstance(checkpoint_state.get('state_dict_metadata'), Mapping) else {}
    if isinstance(metadata.get('parameter_count'), int):
        return int(metadata['parameter_count'])
    if isinstance(checkpoint_state, Mapping) and isinstance(checkpoint_state.get('tensor_count'), int):
        return int(checkpoint_state['tensor_count'])

    tensors = checkpoint_state.get('tensors') if isinstance(checkpoint_state, Mapping) else None
    return len(tensors) if isinstance(tensors, Mapping) else 0


def checkpoint_value_count(checkpoint_state: object) -> int:
    if isinstance(checkpoint_state, Mapping) and isinstance(checkpoint_state.get('value_count'), int):
        return int(checkpoint_state['value_count'])
    metadata = checkpoint_state.get('state_dict_metadata') if isinstance(checkpoint_state, Mapping) and isinstance(checkpoint_state.get('state_dict_metadata'), Mapping) else {}
    if isinstance(metadata.get('value_count'), int):
        return int(metadata['value_count'])

    state_dict = checkpoint_parameter_map(checkpoint_state)
    if state_dict:
        total = 0
        for parameter_value in state_dict.values():
            tensor_summary = _coerce_tensor_values(parameter_value)
            if isinstance(tensor_summary, Mapping) and isinstance(tensor_summary.get('value_count'), int):
                total += int(tensor_summary['value_count'])
        return total

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

    loader_result = loader(dict(state_dict), strict=strict)
    incompatible = _normalize_loader_result(loader_result)
    if strict and (incompatible['missing_keys'] or incompatible['unexpected_keys']):
        raise CheckpointResolutionError(
            f'{module.__class__.__name__} strict hydration mismatch: '
            f'missing={incompatible["missing_keys"]}, unexpected={incompatible["unexpected_keys"]}.'
        )

    metadata = checkpoint_state.get('state_dict_metadata') if isinstance(checkpoint_state.get('state_dict_metadata'), Mapping) else {}
    return {
        'module': module.__class__.__name__,
        'module_family': metadata.get('module_family', module.__class__.__name__),
        'load_style': 'load_state_dict',
        'strict': strict,
        'representation': checkpoint_state.get('representation', metadata.get('representation', LEGACY_SUBTREE_REPRESENTATION)),
        'parameter_count': metadata.get('parameter_count', checkpoint_tensor_count(checkpoint_state)),
        'module_roots': metadata.get('module_roots', []),
        'signature': checkpoint_signature(checkpoint_state),
        'tensor_count': checkpoint_tensor_count(checkpoint_state),
        'value_count': checkpoint_value_count(checkpoint_state),
        'missing_keys': incompatible['missing_keys'],
        'unexpected_keys': incompatible['unexpected_keys'],
    }


def _normalize_checkpoint_subtree(subtree: Mapping[str, object]) -> dict[str, dict[str, object]]:
    tensor_source = subtree.get('tensors') if isinstance(subtree.get('tensors'), Mapping) else subtree
    normalized_tensors = _collect_tensor_values(tensor_source)
    return {name: values for name, values in normalized_tensors.items() if values.get('sample')}


def checkpoint_parameter_map(checkpoint_state: object) -> dict[str, object]:
    if not isinstance(checkpoint_state, Mapping):
        return {}
    state_dict = checkpoint_state.get('state_dict')
    if not isinstance(state_dict, Mapping):
        return {}
    return {str(name): value for name, value in state_dict.items() if isinstance(name, str) and name.strip()}


def _collect_state_dict_entries(node: object, prefix: str = '') -> dict[str, object]:
    if not isinstance(node, Mapping):
        return {prefix: node} if prefix else {}

    tensors: dict[str, object] = {}
    for key, value in node.items():
        if not isinstance(key, str) or not key.strip():
            continue
        tensor_name = key if not prefix else f'{prefix}.{key}'
        tensor_values = _coerce_tensor_values(value)
        if tensor_values:
            tensors[tensor_name] = value
            continue
        tensors.update(_collect_state_dict_entries(value, tensor_name))
    return tensors


def _module_roots(state_dict: Mapping[str, object]) -> list[str]:
    roots = {name.split('.', 1)[0] for name in state_dict if isinstance(name, str) and name.strip()}
    return sorted(root for root in roots if root)


def _normalize_loader_result(result: object) -> dict[str, list[str]]:
    missing_keys: list[str] = []
    unexpected_keys: list[str] = []
    if isinstance(result, Mapping):
        raw_missing = result.get('missing_keys')
        raw_unexpected = result.get('unexpected_keys')
        if isinstance(raw_missing, list):
            missing_keys = [str(value) for value in raw_missing]
        if isinstance(raw_unexpected, list):
            unexpected_keys = [str(value) for value in raw_unexpected]
        return {'missing_keys': missing_keys, 'unexpected_keys': unexpected_keys}

    raw_missing = getattr(result, 'missing_keys', None)
    raw_unexpected = getattr(result, 'unexpected_keys', None)
    if isinstance(raw_missing, list):
        missing_keys = [str(value) for value in raw_missing]
    if isinstance(raw_unexpected, list):
        unexpected_keys = [str(value) for value in raw_unexpected]
    return {'missing_keys': missing_keys, 'unexpected_keys': unexpected_keys}


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
