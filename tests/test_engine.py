import pickle

import numpy as np
import pandas as pd
import pytest

from calibre.conformal import (
    CumulativeConformalRiskConfig,
    CumulativeRiskRuntime,
    SymmetricIntervalConfig,
    SymmetricIntervalRuntime,
)
from calibre.core.forecast_frame import (
    CALIBRATION_STATE,
    CONFORMAL_METHOD,
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
from calibre.execution.backend import BackendEngine, BackendResult, _process_task_ref_partition
from calibre.execution.ledger import OrderLedger
from calibre.ordering.policy_config import OrderPolicyConfig


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
    engine = BackendEngine(freq="W")
    result = engine.execute([task], actuals, origins)

    assert isinstance(result, BackendResult)
    df = result.ledger.to_df()
    assert len(df) == 8


def test_fugue_partition_worker_is_module_level_picklable():
    pickle.dumps(_process_task_ref_partition)


def test_forecast_frame_columns_present(single_series_setup):
    task, actuals, origins = single_series_setup
    engine = BackendEngine(freq="W")
    result = engine.execute([task], actuals, origins)

    df = result.ledger.to_df()
    for col in [UNIQUE_ID, DS, Y_HAT, H, FORECAST_ORIGIN, MODEL_NAME]:
        assert col in df.columns


def test_partial_resolution(single_series_setup):
    task, actuals, origins = single_series_setup
    engine = BackendEngine(freq="W")
    result = engine.execute([task], actuals, origins)

    df = result.ledger.to_df()

    resolved = df[Y].notna().sum()
    unresolved = df[Y].isna().sum()
    assert resolved == 5
    assert unresolved == 3


def test_error_columns_on_resolved(single_series_setup):
    task, actuals, origins = single_series_setup
    engine = BackendEngine(freq="W")
    result = engine.execute([task], actuals, origins)

    df = result.ledger.to_df()
    resolved = df[df[Y].notna()]

    assert "error" in resolved.columns
    assert "abs_error" in resolved.columns
    np.testing.assert_array_almost_equal(resolved["error"].dropna().values, 0.0)


def test_model_name_stamped(single_series_setup):
    task, actuals, origins = single_series_setup
    engine = BackendEngine(freq="W")
    result = engine.execute([task], actuals, origins)

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
    engine = BackendEngine(freq="W", conformal_runtime=SymmetricIntervalRuntime(conformal_config))
    result = engine.execute([task], actuals, origins)

    df = result.ledger.to_df()
    lower_col, upper_col = conformal_config.interval_columns
    assert lower_col in df.columns
    assert upper_col in df.columns
    assert CONFORMAL_METHOD in df.columns
    assert CALIBRATION_STATE in df.columns
    assert df[CONFORMAL_METHOD].eq("aci").all()
    assert df[CALIBRATION_STATE].str.startswith("{").all()
    assert df.loc[df[Y].notna(), NONCONFORMITY_SCORE].notna().all()


def test_execute_with_mscp_config_enriches_ledger(single_series_setup):
    task, actuals, origins = single_series_setup
    conformal_config = SymmetricIntervalConfig(
        method="mscp",
        coverage=0.9,
        calibration_window=4,
    )
    engine = BackendEngine(freq="W", conformal_runtime=SymmetricIntervalRuntime(conformal_config))
    result = engine.execute([task], actuals, origins)

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
    engine = BackendEngine(freq="W", conformal_runtime=runtime)
    result = engine.execute([task], actuals, origins)

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
    engine = BackendEngine(freq="W", conformal_runtime=SymmetricIntervalRuntime(conformal_config))
    result = engine.execute([task], actuals, origins=[dates[7], dates[8]])

    df = result.ledger.to_df()
    lower_col, upper_col = conformal_config.interval_columns
    widths = df[upper_col] - df[lower_col]
    second_origin_mask = df[FORECAST_ORIGIN] == dates[8]

    assert widths.loc[second_origin_mask & (df[H] == 1)].iloc[0] > 0.0


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

    engine = BackendEngine(freq="W")
    result = engine.execute(tasks, actuals, origins=[dates[11]])

    df = result.ledger.to_df()
    assert len(df) == 8
    assert set(df[UNIQUE_ID].unique()) == {"A", "B"}


def test_to_parquet_roundtrip(single_series_setup, tmp_path):
    task, actuals, origins = single_series_setup
    engine = BackendEngine(freq="W")
    result = engine.execute([task], actuals, origins)

    path = str(tmp_path / "backtest.parquet")
    result.ledger.to_parquet(path)
    loaded = pd.read_parquet(path)
    assert len(loaded) == 8


def test_engine_without_order_config_returns_none_order_ledger(single_series_setup):
    task, actuals, origins = single_series_setup
    engine = BackendEngine(freq="W")
    result = engine.execute([task], actuals, origins)

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
    order_config = OrderPolicyConfig(policy="rs", params=params, coverage=0.9)
    engine = BackendEngine(
        freq="W",
        conformal_runtime=SymmetricIntervalRuntime(conformal_config),
        order_config=order_config,
    )
    result = engine.execute([task], actuals, origins)

    assert isinstance(result.order_ledger, OrderLedger)
    order_df = result.order_ledger.to_df()
    assert not order_df.empty
    assert "order_qty" in order_df.columns


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
    order_config = OrderPolicyConfig(policy="newsvendor", params=params, coverage=0.9)
    engine = BackendEngine(
        freq="W",
        conformal_runtime=SymmetricIntervalRuntime(conformal_config),
        order_config=order_config,
    )
    result = engine.execute([task], actuals, origins)

    assert isinstance(result.order_ledger, OrderLedger)
    order_df = result.order_ledger.to_df()
    assert not order_df.empty
    assert "order_qty" in order_df.columns


def test_engine_records_conformal_coverage_metric(single_series_setup):
    task, actuals, origins = single_series_setup
    conformal_config = SymmetricIntervalConfig(
        method="aci",
        coverage=0.9,
        calibration_window=4,
        gamma=0.05,
    )
    engine = BackendEngine(
        freq="W",
        conformal_runtime=SymmetricIntervalRuntime(conformal_config),
    )

    engine.execute([task], actuals, origins)

    coverage = conformal_coverage_ratio.labels(
        model="SeasonalNaive", mode="perhorizon"
    )._value.get()
    assert 0.0 <= coverage <= 1.0


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

    engine = BackendEngine(freq="W")
    result = engine.execute(tasks=[global_task], actuals=all_series, origins=[dates[11]])

    df = result.ledger.to_df()
    assert not df.empty
    assert set(df[UNIQUE_ID].unique()) == {"A", "B"}
    assert all(col in df.columns for col in [UNIQUE_ID, DS, Y_HAT, H, FORECAST_ORIGIN, MODEL_NAME])


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

    engine = BackendEngine(freq="W")
    result = engine.execute(tasks=[global_task], actuals=all_series, origins=[dates[11]])

    df = result.ledger.to_df()
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

    class _StubAdapter:
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

    engine = BackendEngine(freq="W")
    engine.execute(tasks, actuals, origins=[dates[11]])

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

    class _StubAdapter:
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

    engine = BackendEngine(freq="W")
    engine.execute([task], all_series, origins=[dates[11]])

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

    engine = BackendEngine(freq="W")
    result = engine.execute(
        tasks=[local_task, global_task], actuals=all_series, origins=[dates[11]]
    )

    df = result.ledger.to_df()
    model_names = set(df[MODEL_NAME].unique())
    assert "SeasonalNaive" in model_names
    assert "global_lgbm" in model_names
