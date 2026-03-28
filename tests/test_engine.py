import numpy as np
import pandas as pd
import pytest

from calibre.conformal import ConformalPolicyConfig
from calibre.contracts.forecast_frame import (
    CALIBRATION_STATE,
    CONFORMAL_METHOD,
    UNIQUE_ID,
    DS,
    NONCONFORMITY_SCORE,
    Y,
    Y_HAT,
    H,
    FORECAST_ORIGIN,
    MODEL_NAME,
)
from calibre.engine.backend import BackendEngine, BackendResult
from calibre.engine.ledger import OrderLedger
from calibre.order.config import OrderPolicyConfig
from calibre.order.types import NewsvendorPolicyParameters, RsPolicyParameters
from calibre.tasks.forecast_task import ForecastTask


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
        unique_id="SKU_001",
        history=pd.DataFrame({"ds": dates, "y": repeating_pattern}),
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
    conformal_config = ConformalPolicyConfig(
        method="aci",
        coverage=0.9,
        calibration_window=4,
        gamma=0.05,
    )
    engine = BackendEngine(freq="W", conformal_config=conformal_config)
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
    conformal_config = ConformalPolicyConfig(
        method="mscp",
        coverage=0.9,
        calibration_window=4,
    )
    engine = BackendEngine(freq="W", conformal_config=conformal_config)
    result = engine.execute([task], actuals, origins)

    df = result.ledger.to_df()
    lower_col, upper_col = conformal_config.interval_columns
    assert lower_col in df.columns
    assert upper_col in df.columns
    assert df[CONFORMAL_METHOD].eq("mscp").all()
    assert df[lower_col].isna().all()
    assert df[upper_col].isna().all()


def test_conformal_updates_before_next_origin():
    dates = pd.date_range("2024-01-07", periods=20, freq="W")
    pattern = [10.0, 20.0, 30.0, 40.0] * 5
    pattern[7] = 41.0
    actuals = pd.DataFrame({"unique_id": "SKU_001", "ds": dates, "y": pattern})
    task = ForecastTask(
        unique_id="SKU_001",
        history=pd.DataFrame({"ds": dates, "y": pattern}),
        horizon=2,
        model_config={"backend": "statsforecast", "model": "SeasonalNaive", "season_length": 4},
    )
    conformal_config = ConformalPolicyConfig(
        method="aci",
        coverage=0.9,
        calibration_window=4,
        gamma=0.05,
    )
    engine = BackendEngine(freq="W", conformal_config=conformal_config)
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
            unique_id="A",
            history=pd.DataFrame({"ds": dates, "y": pattern_a}),
            horizon=4,
            model_config={"backend": "statsforecast", "model": "SeasonalNaive", "season_length": 4},
        ),
        ForecastTask(
            unique_id="B",
            history=pd.DataFrame({"ds": dates, "y": pattern_b}),
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
    conformal_config = ConformalPolicyConfig(
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
    engine = BackendEngine(freq="W", conformal_config=conformal_config, order_config=order_config)
    result = engine.execute([task], actuals, origins)

    assert isinstance(result.order_ledger, OrderLedger)
    order_df = result.order_ledger.to_df()
    assert not order_df.empty
    assert "order_qty" in order_df.columns


def test_engine_with_newsvendor_config_populates_order_ledger(single_series_setup):
    task, actuals, origins = single_series_setup
    conformal_config = ConformalPolicyConfig(
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
    engine = BackendEngine(freq="W", conformal_config=conformal_config, order_config=order_config)
    result = engine.execute([task], actuals, origins)

    assert isinstance(result.order_ledger, OrderLedger)
    order_df = result.order_ledger.to_df()
    assert not order_df.empty
    assert "order_qty" in order_df.columns
