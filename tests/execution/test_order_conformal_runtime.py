"""Engine-internal probe for the ``order_conformal`` decision runtime (S1, #242).

Drives the full production chain — ``order_conformal`` config -> ``prepare_run``
runtime resolution -> ``BackendEngine`` selection + single-shot apply -> ledger ->
``UpperBoundRule`` cumulative decision — on a tiny real-origin fixture. VN2 does
not cover this engine-internal cumulative path, so this is the load-bearing gate
that the decision bound lands on the production path.
"""

from __future__ import annotations

import pandas as pd
import pytest

from calibre.cli.config import load_config_from_mapping
from calibre.conformal.cumulative_risk import CumulativeRiskRuntime
from calibre.core.forecast_frame import (
    CONFORMAL_MODE,
    DS,
    UNIQUE_ID,
    H,
    Y,
    interval_column_names,
)
from calibre.core.order_types import CostStruct, RsPolicyParameters
from calibre.execution.backend import BackendEngine, ConformalOptions, ExecutionOptions
from calibre.execution.dataset import DatasetBundle
from calibre.execution.hierarchy_preparation import prepare_run
from calibre.ordering import apply_rs_policy

_COVERAGE = 0.74
_PROTECTION_PERIOD = 2


def _history() -> pd.DataFrame:
    dates = pd.date_range("2024-01-07", periods=20, freq="W")
    pattern = [10.0, 20.0, 30.0, 40.0] * 5
    return pd.DataFrame({UNIQUE_ID: "SKU_001", DS: dates, Y: pattern})


def _bundle() -> DatasetBundle:
    return DatasetBundle(
        history=_history(),
        future_x=None,
        costs=CostStruct(),
        hierarchy=None,
        censoring=None,
    )


def _config() -> object:
    history = _history()
    origins = history[DS].tolist()
    return load_config_from_mapping(
        {
            "config_schema": "1.0",
            "dataset": {"adapter": "unit", "path": "ignored"},
            "tasks": [
                {
                    "model": "SeasonalNaive",
                    "horizon": _PROTECTION_PERIOD,
                    "config": {"backend": "statsforecast", "season_length": 4},
                }
            ],
            "origins": {"start": origins[7], "end": origins[11], "freq": "W"},
            "output": {"streaming": False},
            "execution": {"backend": "local", "seed": 42},
            "order_conformal": {
                "coverage": _COVERAGE,
                "protection_period": _PROTECTION_PERIOD,
                "calibration_window": 8,
            },
        }
    )


def _run() -> tuple[CumulativeRiskRuntime, pd.DataFrame]:
    config = _config()
    preparation = prepare_run(config, _bundle())
    runtime = preparation.conformal_runtime
    assert isinstance(runtime, CumulativeRiskRuntime)

    engine = BackendEngine(
        execution=ExecutionOptions(freq=config.origins.freq),
        conformal=ConformalOptions(runtime=preparation.conformal_runtime),
    )
    result = engine.execute(preparation.tasks, preparation.actuals, preparation.origins)
    return runtime, result.ledger.to_df()


def test_order_conformal_config_selects_cumulative_risk_runtime() -> None:
    preparation = prepare_run(_config(), _bundle())
    assert isinstance(preparation.conformal_runtime, CumulativeRiskRuntime)
    assert preparation.conformal_config is None


def test_order_conformal_emits_terminal_bound_in_cumulative_mode() -> None:
    runtime, ledger = _run()
    _, upper_col = interval_column_names(_COVERAGE)

    assert upper_col == "hi_0p74"
    assert upper_col in ledger.columns
    assert (ledger[CONFORMAL_MODE] == "cumulative").all()

    terminal = ledger[ledger[H] == _PROTECTION_PERIOD]
    earlier = ledger[ledger[H] < _PROTECTION_PERIOD]
    # The bound lands only on the terminal-horizon row of each window.
    assert terminal[upper_col].notna().any()
    assert earlier[upper_col].isna().all()
    terminal_bound = float(terminal[upper_col].dropna().iloc[-1])
    assert pd.notna(terminal_bound)
    assert terminal_bound >= 0.0


def test_upper_bound_rule_reads_terminal_row_cumulative_bound() -> None:
    _, ledger = _run()
    _, upper_col = interval_column_names(_COVERAGE)

    # Use the latest origin whose terminal-row bound is populated, so the
    # production decision rule has a finite cumulative bound to read.
    populated = ledger[ledger[upper_col].notna()]
    assert not populated.empty
    latest_origin = populated["forecast_origin"].max()
    origin_frame = ledger[ledger["forecast_origin"] == latest_origin].copy()

    result = apply_rs_policy(
        origin_frame,
        [
            RsPolicyParameters(
                unique_id="SKU_001",
                inventory_position=0.0,
                lead_time=1,
                review_period=1,
            )
        ],
        coverage=_COVERAGE,
    )

    expected_bound = float(
        origin_frame.loc[origin_frame[H] == _PROTECTION_PERIOD, upper_col].dropna().iloc[-1]
    )
    target = float(result.iloc[0]["target_stock_level"])
    # The cumulative path reads the terminal-row bound itself, not a sum across H.
    assert target == pytest.approx(expected_bound)
    assert target >= 0.0


def test_no_order_conformal_block_is_behavior_neutral() -> None:
    history = _history()
    origins = history[DS].tolist()
    config = load_config_from_mapping(
        {
            "config_schema": "1.0",
            "dataset": {"adapter": "unit", "path": "ignored"},
            "tasks": [
                {
                    "model": "SeasonalNaive",
                    "horizon": _PROTECTION_PERIOD,
                    "config": {"backend": "statsforecast", "season_length": 4},
                }
            ],
            "origins": {"start": origins[7], "end": origins[11], "freq": "W"},
            "output": {"streaming": False},
            "execution": {"backend": "local", "seed": 42},
        }
    )
    preparation = prepare_run(config, _bundle())
    assert preparation.conformal_runtime is None
    assert preparation.conformal_config is None
