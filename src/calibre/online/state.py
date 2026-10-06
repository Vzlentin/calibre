"""Flat storage form of calibration state: one numpy array per key."""

from typing import Any

import numpy as np

_SEPARATOR = "/"


def flatten(state: dict[str, Any], prefix: str = "") -> dict[str, np.ndarray]:
    """Nested state to `{"a/b": array}`. The result saves with `np.savez` as is."""
    flat = {}
    for key, value in state.items():
        if _SEPARATOR in key:
            raise ValueError(f"state key {key!r} contains {_SEPARATOR!r}")
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(flatten(value, f"{name}{_SEPARATOR}"))
        else:
            flat[name] = np.asarray(value)
    return flat


def unflatten(flat: dict[str, np.ndarray]) -> dict[str, Any]:
    """Inverse of `flatten`."""
    state: dict[str, Any] = {}
    for name, value in flat.items():
        *parents, key = name.split(_SEPARATOR)
        node = state
        for parent in parents:
            node = node.setdefault(parent, {})
        node[key] = np.asarray(value)
    return state
