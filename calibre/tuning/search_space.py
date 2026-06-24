"""Declarative search-space spec parser shared by every Optuna study builder."""

from __future__ import annotations

from typing import Any

import optuna


def suggest_from_spec(trial: optuna.Trial, name: str, spec: dict[str, Any]) -> Any:
    """Sample a parameter from a declarative search-space spec.

    The spec's ``type`` selects the Optuna sampler: ``categorical`` over
    ``choices``, ``int``/``float`` over ``low``/``high`` with optional ``step``
    (and ``log`` for floats). Any other ``type`` raises ``ValueError`` so a
    malformed spec fails loud instead of silently sampling nothing.
    """
    kind = spec["type"]
    if kind == "categorical":
        return trial.suggest_categorical(name, spec["choices"])
    if kind == "int":
        return trial.suggest_int(name, spec["low"], spec["high"], step=spec.get("step", 1))
    if kind == "float":
        return trial.suggest_float(
            name,
            spec["low"],
            spec["high"],
            step=spec.get("step"),
            log=spec.get("log", False),
        )
    raise ValueError(f"Unknown HPO spec type: {kind!r}")
