from __future__ import annotations

from typing import Any


def _thread_budget(cpu_budget: float | None) -> int:
    if cpu_budget is None:
        return 1
    return max(1, int(cpu_budget))


def _cap_threaded_config(
    config: dict[str, Any],
    cpu_budget: float | None = None,
) -> dict[str, Any]:
    """Keep library-level parallelism inside the assigned CPU budget.

    Callers translate their domain-specific vocabularies (``cpu_per_task``,
    ``cpu_per_trial``) to ``cpu_budget`` before invoking this helper.
    """
    if cpu_budget is None:
        return config

    capped = dict(config)
    threads = _thread_budget(cpu_budget)
    for key in ("n_jobs", "num_threads", "nthread"):
        if key not in capped:
            continue
        value = capped.get(key)
        if value is None or int(value) < 1 or int(value) > threads:
            capped[key] = threads
    model_name = str(capped.get("model", "")).lower()
    if "lgbm" in model_name or "lightgbm" in model_name or "xgb" in model_name:
        capped.setdefault("n_jobs", threads)
    return capped
