from __future__ import annotations

import pandas as pd
import ray

import calibre.execution.backend as backend
from calibre.core.forecast_task import ForecastTaskRef
from calibre.execution.backend import BackendEngine, ExecutionOptions


def _ref(config: dict, idx: int) -> ForecastTaskRef:
    return ForecastTaskRef(
        unique_id=f"sku_{idx}",
        model_config=config,
        horizon=2,
        forecast_origin=None,
        history_uri=f"memory://missing/{idx}.parquet",
    )


class _RemoteFunction:
    def __init__(self, fn):
        self._fn = fn
        self.options_kwargs: dict | None = None

    def options(self, **kwargs) -> _RemoteFunction:
        self.options_kwargs = kwargs
        return self

    def remote(self, *args, **kwargs):
        return self._fn(*args, **kwargs)


class _FakeRay:
    @staticmethod
    def remote(*args, **kwargs):
        del kwargs
        if args and callable(args[0]):
            return _RemoteFunction(args[0])
        return lambda fn: _RemoteFunction(fn)

    @staticmethod
    def get(object_refs):
        return object_refs


def _patch_fake_ray(monkeypatch, engine: BackendEngine) -> None:
    # The fanout methods obtain ray via a local `import ray` after calling
    # `_ensure_ray()` (which no longer returns a handle), so stub the module here.
    monkeypatch.setattr(engine, "_ensure_ray", lambda: None)
    monkeypatch.setattr(ray, "remote", _FakeRay.remote)
    monkeypatch.setattr(ray, "get", _FakeRay.get)


def test_multiple_global_configs_run_in_parallel(monkeypatch) -> None:
    calls: list[tuple[str, int]] = []

    def _fake_process_global_panel(refs, model_config, origin):
        del origin
        calls.append((str(model_config["name"]), len(refs)))
        return backend._empty_forecast_frame()

    monkeypatch.setattr(backend, "_process_global_panel", _fake_process_global_panel)
    engine = BackendEngine(execution=ExecutionOptions(backend="ray", max_concurrency=2))
    _patch_fake_ray(monkeypatch, engine)

    config_a = {"backend": "stub", "model": "stub", "scope": "global", "name": "a"}
    config_b = {"backend": "stub", "model": "stub", "scope": "global", "name": "b"}
    engine._run_global_scope(
        [_ref(config_a, 1), _ref(config_a, 2), _ref(config_b, 3)],
        pd.Timestamp("2024-01-01"),
    )

    assert sorted(calls) == [("a", 2), ("b", 1)]


def test_global_fanout_applies_cpu_per_task_options(monkeypatch) -> None:
    def _fake_process_global_panel(refs, model_config, origin):
        del origin
        return pd.DataFrame({"name": [str(model_config["name"])], "count": [len(refs)]})

    monkeypatch.setattr(backend, "_process_global_panel", _fake_process_global_panel)
    engine = BackendEngine(
        execution=ExecutionOptions(backend="ray", max_concurrency=2, cpu_per_task=1.0)
    )
    _patch_fake_ray(monkeypatch, engine)

    config_a = {"backend": "stub", "model": "stub", "scope": "global", "name": "a"}
    config_b = {"backend": "stub", "model": "stub", "scope": "global", "name": "b"}
    result = engine._run_global_scope(
        [_ref(config_a, 1), _ref(config_a, 2), _ref(config_b, 3)],
        pd.Timestamp("2024-01-01"),
    )

    assert sorted(zip(result["name"], result["count"], strict=True)) == [("a", 2), ("b", 1)]
    assert engine._remote_process_global_panel.options_kwargs == {"num_cpus": 1.0}


def test_local_fanout_applies_cpu_per_task_options(monkeypatch) -> None:
    def _fake_process_task_ref(ref, origin, local_scope):
        del origin, local_scope
        return pd.DataFrame({"name": [ref.unique_id]})

    monkeypatch.setattr(backend, "_process_task_ref", _fake_process_task_ref)
    engine = BackendEngine(
        execution=ExecutionOptions(backend="ray", max_concurrency=2, cpu_per_task=2.0)
    )
    _patch_fake_ray(monkeypatch, engine)

    config = {"backend": "stub", "model": "stub", "scope": "local"}
    result = engine._run_local_scope(
        [_ref(config, 1), _ref(config, 2)],
        pd.Timestamp("2024-01-01"),
    )

    assert sorted(result["name"]) == ["sku_1", "sku_2"]
    assert engine._remote_process_task.options_kwargs == {"num_cpus": 2.0}
