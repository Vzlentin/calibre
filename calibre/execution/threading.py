"""Canonical CPU/thread-budget helpers shared by the backend and the tuner.

Both the Ray task path (``BackendEngine``) and the Ray Tune trial path
(``calibre.tuning.optimizer``) must keep library-level parallelism inside their
per-worker CPU budget, or budgeted workers oversubscribe the box. The logic
lived in two verbatim private copies; this is the single source of truth.
Callers map their domain-specific budget (``cpu_per_task``, ``cpu_per_trial``)
onto the one ``cpu_budget`` parameter.
"""

from __future__ import annotations

from typing import Any

_THREAD_KEYS = ("n_jobs", "num_threads", "nthread")
_TREE_MODEL_MARKERS = ("lgbm", "lightgbm", "xgb")


def thread_budget(cpu_budget: float | None) -> int:
    """Threads a worker may use given its CPU budget (>= 1; ``None`` -> 1)."""
    if cpu_budget is None:
        return 1
    return max(1, int(cpu_budget))


def cap_threaded_config(config: dict[str, Any], cpu_budget: float | None) -> dict[str, Any]:
    """Clamp a model config's thread knobs to ``cpu_budget``.

    Returns ``config`` unchanged when ``cpu_budget`` is ``None`` (no budget set).
    Tree-model configs (LightGBM/XGBoost) get an explicit ``n_jobs`` so they do
    not silently default to every core inside a budgeted worker.
    """
    if cpu_budget is None:
        return config
    capped = dict(config)
    threads = thread_budget(cpu_budget)
    for key in _THREAD_KEYS:
        if key not in capped:
            continue
        value = capped[key]
        if value is None or int(value) < 1 or int(value) > threads:
            capped[key] = threads
    model_name = str(capped.get("model", "")).lower()
    if any(marker in model_name for marker in _TREE_MODEL_MARKERS):
        capped.setdefault("n_jobs", threads)
    return capped
