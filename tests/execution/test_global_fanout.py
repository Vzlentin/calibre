from __future__ import annotations

import pandas as pd

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


def test_multiple_global_configs_run_in_parallel(monkeypatch) -> None:
    calls: list[tuple[str, int]] = []

    def _fake_process_global_panel(refs, model_config, origin):
        del origin
        calls.append((str(model_config["name"]), len(refs)))
        return backend._empty_forecast_frame()

    class _RemoteFunction:
        def __init__(self, fn):
            self._fn = fn

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

    monkeypatch.setattr(backend, "_process_global_panel", _fake_process_global_panel)
    engine = BackendEngine(execution=ExecutionOptions(backend="ray", max_concurrency=2))
    monkeypatch.setattr(engine, "_ensure_ray", lambda: _FakeRay())

    config_a = {"backend": "stub", "model": "stub", "scope": "global", "name": "a"}
    config_b = {"backend": "stub", "model": "stub", "scope": "global", "name": "b"}
    engine._run_global_scope(
        [_ref(config_a, 1), _ref(config_a, 2), _ref(config_b, 3)],
        pd.Timestamp("2024-01-01"),
    )

    assert sorted(calls) == [("a", 2), ("b", 1)]
