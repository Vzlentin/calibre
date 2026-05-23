from __future__ import annotations

from typing import Any


def _thread_budget(cpu_budget: float | None) -> int:
    if cpu_budget is None:
        return 1
    return max(1, int(cpu_budget))


def _cap_threaded_config(
    config: dict[str, Any],
    cpu_budget: float | None = None,
    *,
    cpu_per_task: float | None = None,
    cpu_per_trial: float | None = None,
) -> dict[str, Any]:
    """Keep library-level parallelism inside the assigned CPU budget."""
    explicit_budgets = [
        budget for budget in (cpu_budget, cpu_per_task, cpu_per_trial) if budget is not None
    ]
    if not explicit_budgets:
        return config

    capped = dict(config)
    threads = _thread_budget(explicit_budgets[-1])
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
