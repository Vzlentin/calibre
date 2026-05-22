from __future__ import annotations

from typing import Any


def _thread_budget(cpu_per_task: float | None) -> int:
    if cpu_per_task is None:
        return 1
    return max(1, int(cpu_per_task))


def _cap_threaded_config(config: dict[str, Any], cpu_per_task: float | None) -> dict[str, Any]:
    """Keep library-level parallelism inside the task CPU budget when set."""
    if cpu_per_task is None:
        return config
    capped = dict(config)
    threads = _thread_budget(cpu_per_task)
    for key in ("n_jobs", "num_threads", "nthread"):
        if key not in capped:
            continue
        value = capped[key]
        if value is None or int(value) < 1 or int(value) > threads:
            capped[key] = threads
    model_name = str(capped.get("model", "")).lower()
    if any(name in model_name for name in ("lgbm", "lightgbm", "xgb")):
        capped.setdefault("n_jobs", threads)
    return capped
