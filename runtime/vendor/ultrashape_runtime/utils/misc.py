"""Upstream-style config and instantiation helpers."""

from __future__ import annotations

import importlib
import inspect
from collections.abc import Mapping
from pathlib import Path


class ConfigLoadError(Exception):
    code = 'LOCAL_RUNTIME_UNAVAILABLE'


def get_obj_from_str(path: str):
    module_name, object_name = path.rsplit('.', 1)
    return getattr(importlib.import_module(module_name, package=None), object_name)


def instantiate_from_config(config: Mapping[str, object], /, **kwargs):
    target = config.get('target')
    if not isinstance(target, str) or not target.strip():
        raise KeyError('Expected key `target` to instantiate.')

    cls = get_obj_from_str(target)
    params = config.get('params')
    merged_kwargs = {}
    if isinstance(params, Mapping):
        merged_kwargs.update(dict(params))
    merged_kwargs.update(kwargs)

    signature = inspect.signature(cls)
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values()):
        return cls(**merged_kwargs)

    accepted = {
        name
        for name, parameter in signature.parameters.items()
        if parameter.kind in {inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY}
    }
    filtered_kwargs = {name: value for name, value in merged_kwargs.items() if name in accepted}
    return cls(**filtered_kwargs)


def load_omega_config(config_path: str) -> dict[str, object]:
    path = Path(config_path)
    if not path.is_file():
        raise ConfigLoadError(f'config_path is not readable: {config_path}.')

    try:
        from omegaconf import OmegaConf  # type: ignore
    except ImportError as error:  # pragma: no cover - fallback handled by caller
        raise ConfigLoadError('OmegaConf.load is unavailable.') from error

    loader = getattr(OmegaConf, 'load', None)
    if not callable(loader):
        raise ConfigLoadError('OmegaConf.load is unavailable.')

    return _to_builtin(loader(path))


def _to_builtin(node: object):
    if isinstance(node, Mapping):
        return {str(key): _to_builtin(value) for key, value in node.items()}
    if isinstance(node, list):
        return [_to_builtin(value) for value in node]
    if isinstance(node, tuple):
        return [_to_builtin(value) for value in node]
    return node
