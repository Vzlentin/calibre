import pickle

import fsspec
import numpy as np
import pandas as pd
import pytest
from prometheus_client import REGISTRY

from calibre.conformal import (
    CumulativeConformalRiskConfig,
    CumulativeRiskRuntime,
    SymmetricIntervalConfig,
    SymmetricIntervalRuntime,
)
from calibre.conformal.runtime import to_json_safe_state
from calibre.core.forecast_frame import (
    CALIBRATION_STATE,
    CALIBRATION_STATE_REF,
    CONFORMAL_ALPHA,
    CONFORMAL_METHOD,
    CONFORMAL_MODE,
    CONFORMAL_PARTITION,
    DS,
    FORECAST_ORIGIN,
    MODEL_NAME,
    NONCONFORMITY_SCORE,
    UNIQUE_ID,
    Y_HAT,
    H,
    Y,
)
from calibre.core.forecast_task import ForecastTask
from calibre.core.metrics import conformal_coverage_ratio
from calibre.core.order_types import NewsvendorPolicyParameters, RsPolicyParameters
from calibre.execution.backend import (
    BackendEngine,
    BackendResult,
    ConformalOptions,
    ExecutionOptions,
    LedgerOutputOptions,
    _process_task_ref,
)
from calibre.execution.ledger import OrderLedger
from calibre.execution.task_builder import partition_tasks
from calibre.forecasting.adapter_base import ModelAdapter
from calibre.ordering.policy_config import NewsvendorConfig, RsConfig


@pytest.fixture
def single_series_setup(dates, repeating_pattern):
    """Single series, single model, two origins."""
    actuals = pd.DataFrame(
        {
            "unique_id": "SKU_001",
            "ds": dates,
            "y": repeating_pattern,
        }
    )

    task = ForecastTask(
        history=pd.DataFrame({"unique_id": "SKU_001", "ds": dates, "y": repeating_pattern}),
        horizon=4,
        model_config={"backend": "statsforecast", "model": "SeasonalNaive", "season_length": 4},
    )

    origins = [dates[7], dates[11]]

    return task, actuals, origins


def test_execute_returns_backend_result(single_series_setup):
    task, actuals, origins = single_series_setup
    engine = BackendEngine()
    result = engine.execute(partition_tasks([task]), actuals, origins)

    assert isinstance(result, BackendResult)
    df = result.ledger.to_df()
    assert len(df) == 8


def test_execute_accepts_grouped_constructor_options(single_series_setup, tmp_path):
    task, actuals, origins = single_series_setup
    path = tmp_path / "grouped-ledger.parquet"
    engine = BackendEngine(
        execution=ExecutionOptions(freq="W", seed=42),
        output=LedgerOutputOptions(forecast_path=path.as_posix(), streaming=True),
    )
    result = engine.execute(partition_tasks([task]), actuals, origins)

    assert path.exists()
    assert len(result.ledger.to_df()) == 8


def test_ray_worker_function_is_module_level_picklable():
    pickle.dumps(_process_task_ref)


def test_remote_ray_staging_uses_shared_uri_and_cleans_up(monkeypatch):
    dates = pd.date_range("2024-01-07", periods=8, freq="W")
    actuals = pd.DataFrame({UNIQUE_ID: "A", DS: dates, Y: [float(i) for i in range(8)]})
    task = ForecastTask(
        history=actuals,
        horizon=1,
        model_config={"backend": "stub", "model": "stub_model"},
    )
    staging_uri = "memory://calibre-backend-staging-test/tasks"
    fs = fsspec.filesystem("memory")
    if fs.exists("/calibre-backend-staging-test"):
        fs.rm("/calibre-backend-staging-test", recursive=True)

    class _StubAdapter(ModelAdapter):
        def fit(self, task: ForecastTask) -> None:
            pass

        def predict(self, task: ForecastTask) -> pd.DataFrame:
            return pd.DataFrame(
                {
                    UNIQUE_ID: [task.unique_id],
                    DS: [pd.Timestamp("2024-03-03")],
                    Y_HAT: [1.0],
                    H: [1],
                }
            )

    monkeypatch.setattr("calibre.execution.backend.resolve_adapter", lambda _: _StubAdapter())
    engine = BackendEngine(
        execution=ExecutionOptions(
            backend="local",
            ray_address="ray://scheduler:10001",
            staging_uri=staging_uri,
        )
    )

    result = engine.execute(partition_tasks([task]), actuals, origins=[dates[-1]])

    assert not result.ledger.to_df().empty
    staged_parquet = [
        path for path in fs.find("/calibre-backend-staging-test") if path.endswith(".parquet")
    ]
    assert not staged_parquet


def test_cpu_per_task_caps_threaded_model_configs(monkeypatch):
    dates = pd.date_range("2024-01-07", periods=8, freq="W")
    actuals = pd.DataFrame({UNIQUE_ID: "A", DS: dates, Y: [float(i) for i in range(8)]})
    task = ForecastTask(
        history=actuals,
        horizon=1,
        model_config={
            "backend": "stub",
            "model": "lightgbm.LGBMRegressor",
            "n_jobs": -1,
            "num_threads": 16,
        },
    )
    seen: dict[str, int] = {}

    class _StubAdapter(ModelAdapter):
        def fit(self, task: ForecastTask) -> None:
            seen["n_jobs"] = int(task.model_config["n_jobs"])
            seen["num_threads"] = int(task.model_config["num_threads"])

        def predict(self, task: ForecastTask) -> pd.DataFrame:
            return pd.DataFrame(
                {
                    UNIQUE_ID: [task.unique_id],
                    DS: [pd.Timestamp("2024-03-03")],
                    Y_HAT: [1.0],
                    H: [1],
                }
            )

    monkeypatch.setattr("calibre.execution.backend.resolve_adapter", lambda _: _StubAdapter())

    BackendEngine(execution=ExecutionOptions(backend="local", cpu_per_task=2)).execute(
        partition_tasks([task]), actuals, origins=[dates[-1]]
    )

    assert seen == {"n_jobs": 2, "num_threads": 2}


def test_forecast_frame_columns_present(single_series_setup):
    task, actuals, origins = single_series_setup
    engine = BackendEngine()
    result = engine.execute(partition_tasks([task]), actuals, origins)

    df = result.ledger.to_df().sort_values([FORECAST_ORIGIN, H]).reset_index(drop=True)
    for col in [UNIQUE_ID, DS, Y_HAT, H, FORECAST_ORIGIN, MODEL_NAME]:
        assert col in df.columns

    # SeasonalNaive (season_length=4) on the perfectly periodic [10,20,30,40]
    # history repeats the last full season for every origin → [40, 10, 20, 30].
    assert df[Y_HAT].tolist() == pytest.approx([40.0, 10.0, 20.0, 30.0] * 2)
    assert (df[UNIQUE_ID] == "SKU_001").all()

    first = df[df[FORECAST_ORIGIN] == origins[0]].sort_values(H)
    second = df[df[FORECAST_ORIGIN] == origins[1]].sort_values(H)
    # The first origin is fully in the past by the final origin, so all four
    # horizons resolve to the actuals (which equal the forecasts here).
    assert first[Y].tolist() == pytest.approx([40.0, 10.0, 20.0, 30.0])
    # The final origin only resolves h=1 (ds == origin); later horizons stay NaN.
    assert second[Y].iloc[0] == pytest.approx(40.0)
    assert second[Y].iloc[1:].isna().all()


def test_partial_resolution(single_series_setup):
    task, actuals, origins = single_series_setup
    engine = BackendEngine()
    result = engine.execute(partition_tasks([task]), actuals, origins)

    df = result.ledger.to_df()

    resolved = df[Y].notna().sum()
    unresolved = df[Y].isna().sum()
    assert resolved == 5
    assert unresolved == 3


def test_error_columns_on_resolved(single_series_setup):
    task, actuals, origins = single_series_setup
    engine = BackendEngine()
    result = engine.execute(partition_tasks([task]), actuals, origins)

    df = result.ledger.to_df()
    resolved = df[df[Y].notna()]

    assert "error" in resolved.columns
    assert "abs_error" in resolved.columns
    np.testing.assert_array_almost_equal(resolved["error"].dropna().values, 0.0)


def test_model_name_stamped(single_series_setup):
    task, actuals, origins = single_series_setup
    engine = BackendEngine()
    result = engine.execute(partition_tasks([task]), actuals, origins)

    df = result.ledger.to_df()
    assert (df[MODEL_NAME] == "SeasonalNaive").all()


def test_execute_with_conformal_config_enriches_ledger(single_series_setup):
    task, actuals, origins = single_series_setup
    conformal_config = SymmetricIntervalConfig(
        method="aci",
        coverage=0.9,
        calibration_window=4,
        gamma=0.05,
    )
    engine = BackendEngine(
        conformal=ConformalOptions(runtime=SymmetricIntervalRuntime(conformal_config))
    )
    result = engine.execute(partition_tasks([task]), actuals, origins)

    df = result.ledger.to_df()
    lower_col, upper_col = conformal_config.interval_columns
    assert lower_col in df.columns
    assert upper_col in df.columns
    assert CONFORMAL_METHOD in df.columns
    assert CALIBRATION_STATE not in df.columns
    assert CALIBRATION_STATE_REF in df.columns
    assert CONFORMAL_PARTITION in df.columns
    assert df[CONFORMAL_METHOD].eq("aci").all()
    assert df[CALIBRATION_STATE_REF].str.startswith("aci:perhorizon:").all()
    assert df.loc[df[Y].notna(), NONCONFORMITY_SCORE].notna().all()


def test_execute_with_mscp_config_enriches_ledger(single_series_setup):
    task, actuals, origins = single_series_setup
    conformal_config = SymmetricIntervalConfig(
        method="mscp",
        coverage=0.9,
        calibration_window=4,
    )
    engine = BackendEngine(
        conformal=ConformalOptions(runtime=SymmetricIntervalRuntime(conformal_config))
    )
    result = engine.execute(partition_tasks([task]), actuals, origins)

    df = result.ledger.to_df()
    lower_col, upper_col = conformal_config.interval_columns
    assert lower_col in df.columns
    assert upper_col in df.columns
    assert df[CONFORMAL_METHOD].eq("mscp").all()
    assert df[lower_col].isna().all()
    assert df[upper_col].isna().all()


def test_execute_accepts_injected_cumulative_risk_runtime(single_series_setup):
    task, actuals, origins = single_series_setup
    runtime = CumulativeRiskRuntime(
        CumulativeConformalRiskConfig(
            coverage=0.5,
            protection_period=2,
            calibration_window=4,
            weight_decay=None,
        )
    )
    engine = BackendEngine(conformal=ConformalOptions(runtime=runtime))
    result = engine.execute(partition_tasks([task]), actuals, origins)

    df = result.ledger.to_df()
    lower_col, upper_col = runtime.interval_columns
    terminal = df[df[H] == 2]
    assert lower_col in df.columns
    assert upper_col in df.columns
    assert terminal[upper_col].notna().all()
    assert df[CONFORMAL_METHOD].eq("weighted_crc").all()


def test_conformal_updates_before_next_origin():
    dates = pd.date_range("2024-01-07", periods=20, freq="W")
    pattern = [10.0, 20.0, 30.0, 40.0] * 5
    pattern[7] = 41.0
    actuals = pd.DataFrame({"unique_id": "SKU_001", "ds": dates, "y": pattern})
    task = ForecastTask(
        history=pd.DataFrame({"unique_id": "SKU_001", "ds": dates, "y": pattern}),
        horizon=2,
        model_config={"backend": "statsforecast", "model": "SeasonalNaive", "season_length": 4},
    )
    conformal_config = SymmetricIntervalConfig(
        method="aci",
        coverage=0.9,
        calibration_window=4,
        gamma=0.05,
    )
    engine = BackendEngine(
        conformal=ConformalOptions(runtime=SymmetricIntervalRuntime(conformal_config))
    )
    result = engine.execute(partition_tasks([task]), actuals, origins=[dates[7], dates[8]])

    df = result.ledger.to_df()
    lower_col, upper_col = conformal_config.interval_columns
    second_origin_h1 = df[(df[FORECAST_ORIGIN] == dates[8]) & (df[H] == 1)].iloc[0]

    # The first origin's h=1 residual (|41 - 40| = 1) is observed before the
    # second origin is forecast, so the second-origin band is finite and wide.
    lower = second_origin_h1[lower_col]
    upper = second_origin_h1[upper_col]
    point = second_origin_h1[Y_HAT]
    assert pd.notna(lower) and pd.notna(upper)
    assert upper - lower > 0.0
    # SeasonalNaive forecasts y_hat=10 for this row; the band is symmetric about it.
    assert point == pytest.approx(10.0)
    assert (lower + upper) / 2 == pytest.approx(point)


def test_multi_series():
    """Two series, single model, single origin."""
    dates = pd.date_range("2024-01-07", periods=20, freq="W")
    pattern_a = [10.0, 20.0, 30.0, 40.0] * 5
    pattern_b = [5.0, 15.0, 25.0, 35.0] * 5

    actuals = pd.concat(
        [
            pd.DataFrame({"unique_id": "A", "ds": dates, "y": pattern_a}),
            pd.DataFrame({"unique_id": "B", "ds": dates, "y": pattern_b}),
        ],
        ignore_index=True,
    )

    tasks = [
        ForecastTask(
            history=pd.DataFrame({"unique_id": "A", "ds": dates, "y": pattern_a}),
            horizon=4,
            model_config={"backend": "statsforecast", "model": "SeasonalNaive", "season_length": 4},
        ),
        ForecastTask(
            history=pd.DataFrame({"unique_id": "B", "ds": dates, "y": pattern_b}),
            horizon=4,
            model_config={"backend": "statsforecast", "model": "SeasonalNaive", "season_length": 4},
        ),
    ]

    engine = BackendEngine()
    result = engine.execute(partition_tasks(tasks), actuals, origins=[dates[11]])

    df = result.ledger.to_df()
    assert len(df) == 8
    assert set(df[UNIQUE_ID].unique()) == {"A", "B"}


def test_to_parquet_roundtrip(single_series_setup, tmp_path):
    task, actuals, origins = single_series_setup
    engine = BackendEngine()
    result = engine.execute(partition_tasks([task]), actuals, origins)

    path = str(tmp_path / "backtest.parquet")
    result.ledger.to_parquet(path)
    loaded = pd.read_parquet(path)

    # The round-trip must preserve the ledger contents, not merely the row count.
    original = result.ledger.to_df()
    assert len(loaded) == 8
    assert set(loaded.columns) == set(original.columns)
    loaded_sorted = loaded.sort_values([FORECAST_ORIGIN, H]).reset_index(drop=True)
    assert loaded_sorted[Y_HAT].tolist() == pytest.approx([40.0, 10.0, 20.0, 30.0] * 2)
    assert (loaded_sorted[MODEL_NAME] == "SeasonalNaive").all()


def test_engine_without_order_config_returns_none_order_ledger(single_series_setup):
    task, actuals, origins = single_series_setup
    engine = BackendEngine()
    result = engine.execute(partition_tasks([task]), actuals, origins)

    assert isinstance(result, BackendResult)
    assert result.order_ledger is None


def test_engine_with_rs_order_config_populates_order_ledger(single_series_setup):
    task, actuals, origins = single_series_setup
    conformal_config = SymmetricIntervalConfig(
        method="aci",
        coverage=0.9,
        calibration_window=4,
        gamma=0.05,
    )
    params = [
        RsPolicyParameters(
            unique_id="SKU_001",
            inventory_position=50.0,
            lead_time=1,
            review_period=1,
        )
    ]
    order_config = RsConfig(params=params, coverage=0.9)
    engine = BackendEngine(
        conformal=ConformalOptions(runtime=SymmetricIntervalRuntime(conformal_config)),
        order=order_config,
    )
    result = engine.execute(partition_tasks([task]), actuals, origins)

    assert isinstance(result.order_ledger, OrderLedger)
    order_df = result.order_ledger.to_df()
    # One R,S decision per origin for the single series.
    assert len(order_df) == 2
    # R,S arithmetic: order up to the target stock level, clipped at zero.
    expected_qty = (order_df["target_stock_level"] - 50.0).clip(lower=0.0)
    pd.testing.assert_series_equal(order_df["order_qty"], expected_qty, check_names=False)


def test_engine_with_newsvendor_config_populates_order_ledger(single_series_setup):
    task, actuals, origins = single_series_setup
    conformal_config = SymmetricIntervalConfig(
        method="aci",
        coverage=0.9,
        calibration_window=4,
        gamma=0.05,
    )
    params = [
        NewsvendorPolicyParameters(
            unique_id="SKU_001",
            underage_cost=3.0,
            overage_cost=1.0,
            inventory_position=50.0,
        )
    ]
    order_config = NewsvendorConfig(params=params, coverage=0.9)
    engine = BackendEngine(
        conformal=ConformalOptions(runtime=SymmetricIntervalRuntime(conformal_config)),
        order=order_config,
    )
    result = engine.execute(partition_tasks([task]), actuals, origins)

    assert isinstance(result.order_ledger, OrderLedger)
    order_df = result.order_ledger.to_df()
    # One newsvendor decision per origin for the single series.
    assert len(order_df) == 2
    # Critical ratio = underage / (underage + overage) = 3 / 4.
    assert (order_df["critical_ratio"] == 0.75).all()
    # Order up to the interpolated demand quantile, clipped at zero.
    expected_qty = (order_df["demand_quantile"] - 50.0).clip(lower=0.0)
    pd.testing.assert_series_equal(order_df["order_qty"], expected_qty, check_names=False)


def test_engine_records_conformal_coverage_metric(single_series_setup):
    task, actuals, origins = single_series_setup
    conformal_config = SymmetricIntervalConfig(
        method="aci",
        coverage=0.9,
        calibration_window=4,
        gamma=0.05,
    )
    engine = BackendEngine(
        conformal=ConformalOptions(runtime=SymmetricIntervalRuntime(conformal_config))
    )

    engine.execute(partition_tasks([task]), actuals, origins)

    labels = {
        (sample.labels["model"], sample.labels["mode"])
        for sample in conformal_coverage_ratio.collect()[0].samples
    }
    assert ("SeasonalNaive", "perhorizon") in labels
    coverage = REGISTRY.get_sample_value(
        "calibre_conformal_coverage_ratio",
        {"model": "SeasonalNaive", "mode": "perhorizon"},
    )
    # The fixture is perfectly periodic, so SeasonalNaive's resolved forecasts
    # equal the actuals (error 0) and every finite band brackets the truth.
    assert coverage == pytest.approx(1.0)


def test_global_scope_produces_forecasts_for_all_series():
    """scope='global' on mlforecast should produce forecasts for all series in history."""
    dates = pd.date_range("2024-01-07", periods=20, freq="W")
    pattern_a = [10.0, 20.0, 30.0, 40.0] * 5
    pattern_b = [5.0, 15.0, 25.0, 35.0] * 5
    all_series = pd.concat(
        [
            pd.DataFrame({"unique_id": "A", "ds": dates, "y": pattern_a}),
            pd.DataFrame({"unique_id": "B", "ds": dates, "y": pattern_b}),
        ],
        ignore_index=True,
    )

    global_task = ForecastTask(
        history=all_series,
        horizon=4,
        model_config={
            "backend": "mlforecast",
            "scope": "global",
            "model": "lightgbm.LGBMRegressor",
            "lags": [1, 2, 3, 4],
            "verbosity": -1,
            "n_estimators": 10,
        },
    )

    engine = BackendEngine()
    result = engine.execute(
        tasks=partition_tasks([global_task]), actuals=all_series, origins=[dates[11]]
    )

    df = result.ledger.to_df()
    # One global fit, single origin, horizon 4 → 4 rows per series for both.
    assert len(df) == 8
    assert df.groupby(UNIQUE_ID)[H].count().to_dict() == {"A": 4, "B": 4}
    assert df[Y_HAT].notna().all()
    assert all(col in df.columns for col in [UNIQUE_ID, DS, Y_HAT, H, FORECAST_ORIGIN, MODEL_NAME])
    assert sorted(df[H].unique().tolist()) == [1, 2, 3, 4]


def test_global_quantile_columns_survive_engine():
    """q_<p> columns from a quantile-producing global adapter must reach the ledger."""
    dates = pd.date_range("2024-01-07", periods=20, freq="W")
    pattern_a = [10.0, 20.0, 30.0, 40.0] * 5
    pattern_b = [5.0, 15.0, 25.0, 35.0] * 5
    all_series = pd.concat(
        [
            pd.DataFrame({"unique_id": "A", "ds": dates, "y": pattern_a}),
            pd.DataFrame({"unique_id": "B", "ds": dates, "y": pattern_b}),
        ],
        ignore_index=True,
    )

    global_task = ForecastTask(
        history=all_series,
        horizon=3,
        model_config={
            "backend": "mlforecast",
            "scope": "global",
            "model": "lightgbm.LGBMRegressor",
            "objective": "quantile",
            "quantiles": [0.5, 0.833],
            "strategy": "direct",
            "lags": [1, 2, 3, 4],
            "verbosity": -1,
            "n_estimators": 10,
        },
    )

    engine = BackendEngine()
    result = engine.execute(
        tasks=partition_tasks([global_task]), actuals=all_series, origins=[dates[11]]
    )

    df = result.ledger.to_df()
    # Two series × horizon 3 from one global quantile fit.
    assert len(df) == 6
    assert df.groupby(UNIQUE_ID)[H].count().to_dict() == {"A": 3, "B": 3}
    assert "q_0p5" in df.columns
    assert "q_0p833" in df.columns
    assert df["q_0p5"].notna().all()
    assert df["q_0p833"].notna().all()


def test_run_parallel_slices_future_x_per_uid(monkeypatch):
    """_run_parallel must filter future_x to the current uid before passing to the adapter."""
    dates = pd.date_range("2024-01-07", periods=12, freq="W")
    pattern = [10.0, 20.0, 30.0, 40.0] * 3
    future_x = pd.DataFrame(
        {
            "unique_id": ["A", "B"],
            "ds": [pd.Timestamp("2024-04-07")] * 2,
            "promo": [1.0, 0.0],
        }
    )
    actuals = pd.concat(
        [
            pd.DataFrame({"unique_id": "A", "ds": dates, "y": pattern}),
            pd.DataFrame({"unique_id": "B", "ds": dates, "y": pattern}),
        ],
        ignore_index=True,
    )
    tasks = [
        ForecastTask(
            history=pd.DataFrame({"unique_id": "A", "ds": dates, "y": pattern}),
            horizon=1,
            model_config={"backend": "stub", "model": "stub_model"},
            future_x=future_x,
        ),
        ForecastTask(
            history=pd.DataFrame({"unique_id": "B", "ds": dates, "y": pattern}),
            horizon=1,
            model_config={"backend": "stub", "model": "stub_model"},
            future_x=future_x,
        ),
    ]

    received: dict[str, pd.DataFrame | None] = {}

    class _StubAdapter(ModelAdapter):
        def fit(self, task: ForecastTask) -> None:
            received[task.unique_id] = task.future_x

        def predict(self, task: ForecastTask) -> pd.DataFrame:
            return pd.DataFrame(
                {
                    "unique_id": [task.unique_id],
                    "ds": [pd.Timestamp("2024-04-07")],
                    "y_hat": [10.0],
                    "h": [1],
                }
            )

    monkeypatch.setattr("calibre.execution.backend.resolve_adapter", lambda _: _StubAdapter())

    engine = BackendEngine()
    engine.execute(partition_tasks(tasks), actuals, origins=[dates[11]])

    assert set(received["A"][UNIQUE_ID].unique()) == {"A"}
    assert set(received["B"][UNIQUE_ID].unique()) == {"B"}


def test_run_direct_passes_full_future_x(monkeypatch):
    """_run_direct must pass the complete (un-sliced) future_x to the adapter."""
    dates = pd.date_range("2024-01-07", periods=12, freq="W")
    pattern = [10.0, 20.0, 30.0, 40.0] * 3
    future_x = pd.DataFrame(
        {
            "unique_id": ["A", "B"],
            "ds": [pd.Timestamp("2024-04-07")] * 2,
            "promo": [1.0, 0.0],
        }
    )
    all_series = pd.concat(
        [
            pd.DataFrame({"unique_id": "A", "ds": dates, "y": pattern}),
            pd.DataFrame({"unique_id": "B", "ds": dates, "y": pattern}),
        ],
        ignore_index=True,
    )
    task = ForecastTask(
        history=all_series,
        horizon=1,
        model_config={"backend": "stub", "model": "stub_model", "scope": "global"},
        future_x=future_x,
    )

    received: list[pd.DataFrame | None] = []

    class _StubAdapter(ModelAdapter):
        def fit(self, task: ForecastTask) -> None:
            received.append(task.future_x)

        def predict(self, task: ForecastTask) -> pd.DataFrame:
            return pd.DataFrame(
                {
                    "unique_id": ["A", "B"],
                    "ds": [pd.Timestamp("2024-04-07")] * 2,
                    "y_hat": [10.0, 10.0],
                    "h": [1, 1],
                }
            )

    monkeypatch.setattr("calibre.execution.backend.resolve_adapter", lambda _: _StubAdapter())

    engine = BackendEngine()
    engine.execute(partition_tasks([task]), all_series, origins=[dates[11]])

    assert len(received) == 1
    assert received[0] is not None
    assert set(received[0][UNIQUE_ID].unique()) == {"A", "B"}


def test_mixed_local_and_global_tasks():
    """Local and global tasks should both appear in the ledger."""
    dates = pd.date_range("2024-01-07", periods=20, freq="W")
    pattern_a = [10.0, 20.0, 30.0, 40.0] * 5
    pattern_b = [5.0, 15.0, 25.0, 35.0] * 5
    all_series = pd.concat(
        [
            pd.DataFrame({"unique_id": "A", "ds": dates, "y": pattern_a}),
            pd.DataFrame({"unique_id": "B", "ds": dates, "y": pattern_b}),
        ],
        ignore_index=True,
    )

    local_task = ForecastTask(
        history=pd.DataFrame({"unique_id": "A", "ds": dates, "y": pattern_a}),
        horizon=4,
        model_config={"backend": "statsforecast", "model": "SeasonalNaive", "season_length": 4},
    )
    global_task = ForecastTask(
        history=all_series,
        horizon=4,
        model_config={
            "backend": "mlforecast",
            "scope": "global",
            "model": "lightgbm.LGBMRegressor",
            "name": "global_lgbm",
            "lags": [1, 2, 3, 4],
            "verbosity": -1,
            "n_estimators": 10,
        },
    )

    engine = BackendEngine()
    result = engine.execute(
        tasks=partition_tasks([local_task, global_task]), actuals=all_series, origins=[dates[11]]
    )

    df = result.ledger.to_df()
    model_names = set(df[MODEL_NAME].unique())
    assert "SeasonalNaive" in model_names
    assert "global_lgbm" in model_names


def test_auto_backend_uses_ray_at_threshold():
    """backend='auto' with task_count == ray_threshold should use Ray."""
    pytest.importorskip("ray")
    dates = pd.date_range("2024-01-07", periods=8, freq="W")
    actuals = pd.concat(
        [
            pd.DataFrame({UNIQUE_ID: "A", DS: dates, Y: [float(i) for i in range(8)]}),
            pd.DataFrame({UNIQUE_ID: "B", DS: dates, Y: [float(i) + 1 for i in range(8)]}),
        ],
        ignore_index=True,
    )

    # Use a real adapter — monkeypatch.setattr cannot reach Ray workers.
    tasks = [
        ForecastTask(
            history=actuals[actuals[UNIQUE_ID] == uid].reset_index(drop=True),
            horizon=1,
            model_config={
                "backend": "statsforecast",
                "model": "SeasonalNaive",
                "season_length": 2,
            },
        )
        for uid in ("A", "B")
    ]

    engine = BackendEngine(
        execution=ExecutionOptions(backend="auto", ray_threshold=2, max_concurrency=1)
    )
    try:
        result = engine.execute(partition_tasks(tasks), actuals, origins=[dates[-1]])
    finally:
        engine.close()

    assert not result.ledger.to_df().empty


# ---------------------------------------------------------------------------
# Characterization lock for the per-origin pipeline (U4 #114, part U4a).
#
# These tests pin the CURRENT, unrefactored end-to-end outputs of
# ``BackendEngine``'s per-origin pipeline (the ``_execute_origin`` /
# ``_resolve_ledger`` path) over a multi-origin run with conformal calibration
# AND an order policy both active. They are a tripwire: the follow-up phase
# extraction (U4b) must keep every value below byte/value-identical. The pins
# cover the ordering-sensitive surfaces a U4b reorder would disturb: the
# cross-origin conformal feedback (interval bounds plus the per-row
# CONFORMAL_ALPHA trajectory and CALIBRATION_STATE_REF), the resolved
# error/score columns, the persisted conformal-state *content* (resume + per-
# partition snapshots), and the order ledger. Dropping or reordering the
# pre-predict ResolveOpen leaves the integer interval bounds unmoved but shifts
# the ACI alpha trajectory, so the CONFORMAL_ALPHA pin is what makes that phase
# boundary load-bearing. (That persist is *called* exactly once is asserted in
# U4b, where the Commit phase consolidates it.)
#
# The setup perturbs one history point (index 7, 40 -> 41) so the conformal
# nonconformity scores and error columns are non-zero and therefore sensitive to
# step ordering, rather than the all-zero residuals of the periodic fixtures.
# ---------------------------------------------------------------------------

# Pinned as literals on purpose: the lock must catch a rename of the interval
# columns, so it must not derive them from the same production helper
# (interval_column_names) that would rename in lockstep and hide the drift.
_LOCK_LOWER_COL = "lo_0p9"
_LOCK_UPPER_COL = "hi_0p9"


@pytest.fixture(scope="module")
def characterization_lock_run():
    """Deterministic 3-origin run with conformal (ACI) + R,S ordering active.

    Returns ``(result, runtime, origins)`` so the lock tests can pin the forecast
    ledger, the conformal resume/partition state, and the order ledger from a
    single run.
    """
    dates = pd.date_range("2024-01-07", periods=20, freq="W")
    pattern = [10.0, 20.0, 30.0, 40.0] * 5
    # Perturb one observed point so residuals (and thus scores) are non-zero.
    pattern[7] = 41.0
    actuals = pd.DataFrame({"unique_id": "SKU_001", "ds": dates, "y": pattern})
    task = ForecastTask(
        history=pd.DataFrame({"unique_id": "SKU_001", "ds": dates, "y": pattern}),
        horizon=2,
        model_config={"backend": "statsforecast", "model": "SeasonalNaive", "season_length": 4},
    )
    runtime = SymmetricIntervalRuntime(
        SymmetricIntervalConfig(method="aci", coverage=0.9, calibration_window=4, gamma=0.05)
    )
    order_config = RsConfig(
        params=[
            RsPolicyParameters(
                unique_id="SKU_001",
                inventory_position=50.0,
                lead_time=1,
                review_period=1,
            )
        ],
        coverage=0.9,
    )
    engine = BackendEngine(
        conformal=ConformalOptions(runtime=runtime),
        order=order_config,
    )
    origins = [dates[7], dates[8], dates[9]]
    result = engine.execute(partition_tasks([task]), actuals, origins)
    return result, runtime, origins


def test_characterization_lock_forecast_ledger(characterization_lock_run):
    """Pin the resolved/scored forecast ledger over the full multi-origin run."""
    result, _runtime, origins = characterization_lock_run

    df = result.ledger.to_df().sort_values([FORECAST_ORIGIN, H]).reset_index(drop=True)

    # Shape and schema are part of the lock: U4b must not add/drop columns.
    assert len(df) == 6
    assert list(df.columns) == [
        UNIQUE_ID,
        DS,
        Y,
        Y_HAT,
        H,
        FORECAST_ORIGIN,
        MODEL_NAME,
        _LOCK_LOWER_COL,
        _LOCK_UPPER_COL,
        CONFORMAL_METHOD,
        CONFORMAL_MODE,
        CONFORMAL_ALPHA,
        CALIBRATION_STATE_REF,
        CONFORMAL_PARTITION,
        NONCONFORMITY_SCORE,
        "error",
        "abs_error",
        "pct_error",
    ]

    # Key identity columns.
    assert (df[UNIQUE_ID] == "SKU_001").all()
    assert (df[MODEL_NAME] == "SeasonalNaive").all()
    assert df[FORECAST_ORIGIN].tolist() == [
        origins[0],
        origins[0],
        origins[1],
        origins[1],
        origins[2],
        origins[2],
    ]
    assert df[H].tolist() == [1, 2, 1, 2, 1, 2]

    # Point forecasts (SeasonalNaive, season_length=4) per (origin, h).
    np.testing.assert_array_equal(
        df[Y_HAT].to_numpy(), np.array([40.0, 10.0, 10.0, 20.0, 20.0, 30.0])
    )

    # Resolved actuals: the perturbed point (41.0) and the periodic values; the
    # final origin's h=2 forecast is still unresolved (NaN).
    np.testing.assert_array_equal(
        df[Y].to_numpy(),
        np.array([41.0, 10.0, 10.0, 20.0, 20.0, np.nan]),
    )

    # Resolved error / abs_error / pct_error columns — these are written in the
    # Commit/resolve step; their exact values lock the resolve sequencing.
    np.testing.assert_array_equal(
        df["error"].to_numpy(), np.array([1.0, 0.0, 0.0, 0.0, 0.0, np.nan])
    )
    np.testing.assert_array_equal(
        df["abs_error"].to_numpy(), np.array([1.0, 0.0, 0.0, 0.0, 0.0, np.nan])
    )
    np.testing.assert_allclose(
        df["pct_error"].to_numpy(),
        np.array([1.0 / 41.0, 0.0, 0.0, 0.0, 0.0, np.nan]),
        equal_nan=True,
    )

    # Nonconformity scores written by conformal observe() during resolve.
    np.testing.assert_array_equal(
        df[NONCONFORMITY_SCORE].to_numpy(),
        np.array([1.0, 0.0, 0.0, 0.0, 0.0, np.nan]),
    )

    # Conformal interval columns: the bands tighten/widen as observed residuals
    # feed back in before each subsequent origin. Pinning both endpoints locks
    # the observe-before-predict ordering across origins.
    np.testing.assert_array_equal(
        df[_LOCK_LOWER_COL].to_numpy(),
        np.array([40.0, 10.0, 9.0, 20.0, 19.0, 30.0]),
    )
    np.testing.assert_array_equal(
        df[_LOCK_UPPER_COL].to_numpy(),
        np.array([40.0, 10.0, 11.0, 20.0, 21.0, 30.0]),
    )

    # ACI per-row alpha trajectory: the adaptive controller advances as observed
    # residuals feed back. The integer interval bounds above do NOT move when the
    # pre-predict ResolveOpen is dropped, but this alpha trajectory does — so
    # pinning it is the load-bearing assertion for the ResolveOpen phase boundary
    # U4b extracts. atol tolerates last-bit cross-arch float noise while still
    # catching the ~5e-3 shift a reorder produces.
    np.testing.assert_allclose(
        df[CONFORMAL_ALPHA].to_numpy(),
        np.array([0.10, 0.10, 0.06, 0.06, 0.07, 0.07]),
        rtol=0.0,
        atol=1e-9,
    )

    # State-ref per row encodes the issued-origin count at apply() time; pinning
    # it locks the issued-count accounting relative to the per-origin sequence.
    assert df[CALIBRATION_STATE_REF].tolist() == [
        "aci:perhorizon:0:SeasonalNaive:h1:__global__",
        "aci:perhorizon:0:SeasonalNaive:h2:__global__",
        "aci:perhorizon:1:SeasonalNaive:h1:__global__",
        "aci:perhorizon:1:SeasonalNaive:h2:__global__",
        "aci:perhorizon:2:SeasonalNaive:h1:__global__",
        "aci:perhorizon:2:SeasonalNaive:h2:__global__",
    ]

    # Every resolved row carries a conformal score; the one unresolved row does not.
    assert df.loc[df[Y].notna(), NONCONFORMITY_SCORE].notna().all()
    assert df.loc[df[Y].isna(), NONCONFORMITY_SCORE].isna().all()


def test_characterization_lock_conformal_state(characterization_lock_run):
    """Pin the conformal runtime's resume + partition state after the run.

    The spec invariant: conformal-state snapshots over a multi-origin run are
    byte-equal pre/post the U4b refactor. We compare the JSON-safe state (exactly
    what ``_persist_conformal_state`` writes) field-by-field.
    """
    _result, runtime, _origins = characterization_lock_run

    resume_state = to_json_safe_state(runtime.get_resume_state())
    assert resume_state == {
        "method": "aci",
        "mode": "perhorizon",
        "coverage": 0.9,
        "calibration_window": 4,
        "gamma": 0.05,
        "quantile_rule": "conformal",
        "protection_period": None,
        "issued_count": 3,
        "controller": {
            "type": "adaptive",
            "target_alpha": 0.09999999999999998,
            "gamma": 0.05,
            "current_alpha": 0.07499999999999998,
            "alpha_bounds": [1e-06, 0.999999],
        },
        "calibrator": {
            "calibration_window": 4,
            "quantile_rule": "conformal",
            "ready_on_empty": True,
            "score_history": {
                "SeasonalNaive:h1:__global__": [1.0, 0.0, 0.0],
                "SeasonalNaive:h2:__global__": [0.0, 0.0],
            },
        },
    }

    # Per-partition snapshots are persisted one row per partition; pin both.
    partition_states = {
        partition: to_json_safe_state(state)
        for partition, state in runtime.get_partition_states().items()
    }
    assert set(partition_states) == {
        "SeasonalNaive:h1:__global__",
        "SeasonalNaive:h2:__global__",
    }
    assert partition_states["SeasonalNaive:h1:__global__"]["calibrator"]["score_history"] == {
        "SeasonalNaive:h1:__global__": [1.0, 0.0, 0.0],
    }
    assert partition_states["SeasonalNaive:h2:__global__"]["calibrator"]["score_history"] == {
        "SeasonalNaive:h2:__global__": [0.0, 0.0],
    }
    # Issued-origin accounting is shared across partitions and counts every
    # issued origin exactly once — the tripwire for a double-observe regression.
    for state in partition_states.values():
        assert state["issued_count"] == 3


def test_characterization_lock_order_ledger(characterization_lock_run):
    """Pin the order ledger produced alongside the multi-origin forecast run."""
    result, _runtime, origins = characterization_lock_run

    assert isinstance(result.order_ledger, OrderLedger)
    order_df = result.order_ledger.to_df().reset_index(drop=True)

    assert len(order_df) == 3
    assert list(order_df.columns) == [
        UNIQUE_ID,
        FORECAST_ORIGIN,
        MODEL_NAME,
        "inventory_position",
        "lead_time",
        "review_period",
        "protection_period",
        "target_stock_level",
        "order_qty",
    ]

    assert (order_df[UNIQUE_ID] == "SKU_001").all()
    assert (order_df[MODEL_NAME] == "SeasonalNaive").all()
    assert order_df[FORECAST_ORIGIN].tolist() == list(origins)
    np.testing.assert_array_equal(
        order_df["inventory_position"].to_numpy(), np.array([50.0, 50.0, 50.0])
    )
    assert order_df["lead_time"].tolist() == [1, 1, 1]
    assert order_df["review_period"].tolist() == [1, 1, 1]
    assert order_df["protection_period"].tolist() == [2, 2, 2]
    # One R,S decision per origin; the target stock level tracks the per-origin
    # forecast band and order_qty = max(target - inventory_position, 0).
    np.testing.assert_array_equal(
        order_df["target_stock_level"].to_numpy(), np.array([50.0, 31.0, 51.0])
    )
    np.testing.assert_array_equal(order_df["order_qty"].to_numpy(), np.array([0.0, 0.0, 1.0]))
    expected_qty = (order_df["target_stock_level"] - 50.0).clip(lower=0.0)
    pd.testing.assert_series_equal(order_df["order_qty"], expected_qty, check_names=False)
