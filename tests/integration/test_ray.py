from __future__ import annotations

import builtins
import importlib
import sys
from pathlib import Path

import pandas as pd
import pytest

from benchmarks.vn2.run_benchmark import run_benchmark
from calibre.cli.commands import run
from calibre.core.forecast_frame import DS, FORECAST_ORIGIN, UNIQUE_ID, H, Y
from calibre.core.forecast_task import ForecastTask
from calibre.execution.backend import BackendEngine, BackendResult, ExecutionOptions
from calibre.execution.task_builder import partition_tasks


def _panel() -> pd.DataFrame:
    dates = pd.date_range("2024-01-07", periods=20, freq="W")
    return pd.concat(
        [
            pd.DataFrame({UNIQUE_ID: "A", DS: dates, Y: [10.0, 20.0] * 10}),
            pd.DataFrame({UNIQUE_ID: "B", DS: dates, Y: [5.0, 15.0] * 10}),
        ],
        ignore_index=True,
    )


def _tasks(panel: pd.DataFrame) -> list[ForecastTask]:
    return [
        ForecastTask(
            history=group.reset_index(drop=True),
            horizon=2,
            model_config={
                "backend": "statsforecast",
                "model": "SeasonalNaive",
                "season_length": 2,
            },
        )
        for _, group in panel.groupby(UNIQUE_ID, sort=False)
    ]


def _global_task(panel: pd.DataFrame) -> ForecastTask:
    return ForecastTask(
        history=panel,
        horizon=2,
        model_config={
            "backend": "mlforecast",
            "scope": "global",
            "model": "lightgbm.LGBMRegressor",
            "lags": [1, 2, 3, 4],
            "verbosity": -1,
            "n_estimators": 10,
            "n_jobs": 1,
        },
    )


def _sorted(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values([UNIQUE_ID, FORECAST_ORIGIN, H]).reset_index(drop=True)


def _write_vn2_benchmark_fixture(root: Path) -> None:
    dates = pd.date_range("2023-01-02", periods=18, freq="W-MON")
    rows = []
    for store, product, base in [(1, 100, 10.0), (2, 200, 18.0)]:
        row: dict[str, object] = {"Store": store, "Product": product}
        for idx, ds in enumerate(dates):
            row[ds.strftime("%Y-%m-%d")] = base + float(idx % 4)
        rows.append(row)
    full = pd.DataFrame(rows)
    date_cols = [col for col in full.columns if col not in {"Store", "Product"}]
    full[["Store", "Product", *date_cols[:-1]]].to_csv(root / "week_0_sales.csv", index=False)
    full.to_csv(root / "week_1_sales.csv", index=False)
    pd.DataFrame(
        {
            "Store": [1, 2],
            "Product": [100, 200],
            "End Inventory": [50.0, 50.0],
            "In Transit W+1": [0.0, 0.0],
            "In Transit W+2": [0.0, 0.0],
        }
    ).to_csv(root / "week_0_initial_state.csv", index=False)


def _fast_benchmark_model_config() -> dict:
    return {
        "backend": "mlforecast",
        "scope": "global",
        "name": "test_global_lgbm_q0p52",
        "model": "lightgbm.LGBMRegressor",
        "objective": "quantile",
        "quantiles": [0.52],
        "strategy": "direct",
        "lags": [1, 2, 4],
        "n_estimators": 5,
        "learning_rate": 0.1,
        "num_leaves": 7,
        "min_child_samples": 2,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "reg_alpha": 0.0,
        "reg_lambda": 0.0,
        "verbosity": -1,
        "n_jobs": 1,
        "random_state": 42,
        "_quantile_alpha": 0.52,
    }


def test_ray_backend_matches_local_backend() -> None:
    pytest.importorskip("ray")

    panel = _panel()
    tasks = _tasks(panel)
    origins = [pd.Timestamp("2024-03-17"), pd.Timestamp("2024-03-24")]
    expected = BackendEngine().execute(partition_tasks(tasks), panel, origins).ledger.to_df()
    engine = BackendEngine(
        execution=ExecutionOptions(backend="ray", max_concurrency=1, ray_threshold=1)
    )
    try:
        actual = engine.execute(partition_tasks(tasks), panel, origins).ledger.to_df()
    finally:
        engine.close()

    pd.testing.assert_frame_equal(_sorted(actual), _sorted(expected))


def test_ray_remote_handle_rebuilds_after_owned_runtime_shutdown() -> None:
    ray = pytest.importorskip("ray")

    if ray.is_initialized():
        ray.shutdown()

    panel = _panel()
    tasks = _tasks(panel)
    origins = [pd.Timestamp("2024-03-17")]

    try:
        for _ in range(2):
            engine = BackendEngine(
                execution=ExecutionOptions(backend="ray", max_concurrency=1, ray_threshold=1)
            )
            try:
                result = engine.execute(partition_tasks(tasks), panel, origins)
            finally:
                engine.close()
            assert not result.ledger.to_df().empty
    finally:
        if ray.is_initialized():
            ray.shutdown()


def test_ray_quantile_columns_survive() -> None:
    ray = pytest.importorskip("ray")
    pytest.importorskip("mlforecast")

    owns_ray = not ray.is_initialized()
    if owns_ray:
        ray.init(include_dashboard=False, ignore_reinit_error=True, _skip_env_hook=True)

    all_series = _panel()
    model_config = {
        "backend": "mlforecast",
        "scope": "local",
        "model": "lightgbm.LGBMRegressor",
        "objective": "quantile",
        "quantiles": [0.5, 0.833],
        "strategy": "direct",
        "lags": [1, 2],
        "verbosity": -1,
        "n_estimators": 10,
        "n_jobs": 1,
    }
    tasks = [
        ForecastTask(
            history=group.reset_index(drop=True),
            horizon=2,
            model_config=model_config,
        )
        for _, group in all_series.groupby(UNIQUE_ID, sort=False)
    ]
    origins = [pd.Timestamp("2024-04-14")]
    engine = BackendEngine(
        execution=ExecutionOptions(backend="ray", max_concurrency=1, ray_threshold=1)
    )
    try:
        actual = engine.execute(partition_tasks(tasks), all_series, origins).ledger.to_df()
    finally:
        engine.close()
        if owns_ray:
            ray.shutdown()

    assert "q_0p5" in actual.columns
    assert "q_0p833" in actual.columns
    assert actual["q_0p5"].notna().all()
    assert actual["q_0p833"].notna().all()


def test_ray_worker_loads_mlforecast_adapter_without_mlforecast_importable() -> None:
    # NOTE: This monkeypatch is safe only in local_mode because workers are
    # not reused. On real clusters with reused workers, the patched state could
    # leak to subsequent tasks.
    ray = pytest.importorskip("ray")

    owns_ray = not ray.is_initialized()
    if not ray.is_initialized():
        ray.init(include_dashboard=False, ignore_reinit_error=True, _skip_env_hook=True)

    @ray.remote
    def _load_adapter_with_blocked_mlforecast_import() -> str:
        original_import = builtins.__import__
        blocked = {
            name: module
            for name, module in sys.modules.items()
            if name == "mlforecast" or name.startswith("mlforecast.")
        }
        for name in blocked:
            sys.modules.pop(name, None)
        sys.modules.pop("calibre.forecasting.mlforecast_adapter", None)

        def _blocked_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "mlforecast" or name.startswith("mlforecast."):
                raise ModuleNotFoundError("No module named 'mlforecast'")
            return original_import(name, globals, locals, fromlist, level)

        builtins.__import__ = _blocked_import
        try:
            module = importlib.import_module("calibre.forecasting.mlforecast_adapter")
            return module.MLForecastAdapter.__name__
        finally:
            builtins.__import__ = original_import
            sys.modules.update(blocked)

    try:
        assert ray.get(_load_adapter_with_blocked_mlforecast_import.remote()) == "MLForecastAdapter"
    finally:
        if owns_ray:
            ray.shutdown()


def _write_cli_config(tmp_path: Path, *, backend: str) -> Path:
    config_path = tmp_path / f"{backend}.yaml"
    ledger_path = tmp_path / f"{backend}.parquet"
    config_path.write_text(
        f"""
config_schema: "1.0"
dataset:
  adapter: vn2
  path: benchmarks/vn2/fixture
  period: 0
tasks:
  - model: SeasonalNaive
    horizon: 2
    config:
      backend: statsforecast
      season_length: 2
origins:
  start: 2024-01-29
  end: 2024-01-29
  freq: W-MON
output:
  ledger_path: {ledger_path.as_posix()}
  streaming: false
execution:
  backend: {backend}
  ray_threshold: 1
  max_concurrency: 1
  seed: 42
""",
        encoding="utf-8",
    )
    return config_path


def test_cli_ray_config_matches_cli_local(tmp_path: Path) -> None:
    pytest.importorskip("ray")

    local = run(_write_cli_config(tmp_path, backend="local"))
    ray_result = run(_write_cli_config(tmp_path, backend="ray"))

    assert isinstance(local, BackendResult)
    assert isinstance(ray_result, BackendResult)
    pd.testing.assert_frame_equal(
        _sorted(ray_result.ledger.to_df()),
        _sorted(local.ledger.to_df()),
    )


def test_vn2_ray_benchmark_cost_matches_local(tmp_path: Path) -> None:
    pytest.importorskip("ray")

    _write_vn2_benchmark_fixture(tmp_path)
    common_kwargs = {
        "data_dir": tmp_path,
        "horizon": 2,
        "lead_time": 1,
        "review_period": 1,
        "decision_rounds": 1,
        "delivery_weeks": 0,
        "results_dir": None,
        "verbose": False,
        "best_config": _fast_benchmark_model_config(),
        "order_conformal_config": None,
    }
    expected = run_benchmark(**common_kwargs).sort_values("unique_id").reset_index(drop=True)
    actual = (
        run_benchmark(
            **common_kwargs,
            execution_backend="ray",
            ray_threshold=1,
            max_concurrency=1,
        )
        .sort_values("unique_id")
        .reset_index(drop=True)
    )

    pd.testing.assert_frame_equal(actual, expected, check_exact=False, atol=1e-9)
