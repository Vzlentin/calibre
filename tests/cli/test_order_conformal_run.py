"""End-to-end `calibre run` coverage for the order_conformal decision runtime.

Exercises the full production path (config -> prepare_run -> BackendEngine) with
a real :class:`CumulativeRiskRuntime` (no mocks): an ``order_conformal`` block
must construct the decision runtime and emit a non-NaN ``hi_<coverage>`` bound at
the terminal-horizon row, while a diagnostic-only run (no block) is unaffected.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from pydantic import ValidationError

from calibre.cli.commands import run_config
from calibre.cli.config import load_config_from_mapping
from calibre.core.forecast_frame import FORECAST_ORIGIN, UNIQUE_ID, H

# Weekly wide-format VN2 sales with a roughly period-2 seasonal pattern that
# SeasonalNaive cannot fit perfectly (the level drifts up across periods), so
# resolved windows carry non-zero residuals and the calibration buffer is
# exercised. Long enough that the rolling origins below have in-window actuals.
_DATES = [
    "2024-01-01",
    "2024-01-08",
    "2024-01-15",
    "2024-01-22",
    "2024-01-29",
    "2024-02-05",
    "2024-02-12",
    "2024-02-19",
    "2024-02-26",
    "2024-03-04",
]
_WIDE_SALES = "\n".join(
    [
        "Store,Product," + ",".join(_DATES),
        "1,10," + ",".join(["10", "12", "11", "15", "9", "13", "12", "16", "10", "14"]),
        "2,20," + ",".join(["5", "7", "6", "9", "4", "8", "7", "10", "5", "9"]),
    ]
)


def _write_fixture(tmp_path: Path) -> str:
    (tmp_path / "week_0_sales.csv").write_text(_WIDE_SALES + "\n")
    return str(tmp_path)


def _config(data_dir: str, **overrides: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "config_schema": "1.0",
        "dataset": {"adapter": "vn2", "path": data_dir, "period": 0},
        "tasks": [
            {
                "model": "SeasonalNaive",
                "horizon": 2,
                "config": {"backend": "statsforecast", "season_length": 2},
            }
        ],
        "origins": {"start": "2024-01-29", "end": "2024-02-12", "freq": "W-MON"},
        "output": {"streaming": False},
        "execution": {"backend": "local", "seed": 42},
    }
    data.update(overrides)
    return data


def test_order_conformal_run_emits_decision_bound(tmp_path: Path) -> None:
    """A configured order_conformal block lands a non-NaN hi_0p74 decision bound.

    coverage=0.74 names the column ``hi_0p74`` (the interval_column_names form,
    ``.`` -> ``p``) — the exact column the R,S ordering policy reads — and the
    cumulative runtime populates it on the terminal-horizon (H == protection
    period) row of each (uid, origin) window via the real production path.
    """
    data_dir = _write_fixture(tmp_path)
    config = load_config_from_mapping(
        _config(
            data_dir,
            order_conformal={
                "coverage": 0.74,
                "protection_period": 2,
                "calibration_window": 5000,
                "buffer_max": 0.0,
            },
        )
    )

    result = run_config(config)
    ledger = result.ledger.to_df()

    assert "hi_0p74" in ledger.columns
    assert "lo_0p74" in ledger.columns

    terminal = ledger[ledger[H] == 2]
    assert not terminal.empty
    # Every terminal-horizon row carries the decision bound; no earlier-H row does.
    assert terminal["hi_0p74"].notna().all()
    assert ledger[ledger[H] < 2]["hi_0p74"].isna().all()


def test_order_conformal_bound_is_calibrated(tmp_path: Path) -> None:
    """The emitted hi_0p74 bound is calibrated: the engine drives the deferral.

    The cumulative decision runtime reaches calibration through the production
    (CLI) path: ``BackendEngine`` now defers incomplete cumulative windows for
    the one-sided runtime via the protection_period capability gate, so each
    window resolves whole, ``observe()`` records its residual, and the buffer is
    no longer structurally 0. This is the engine-path RED -> GREEN proof of the
    deferral generalization.

    Uses the default unbounded buffer (``buffer_max=None``) — this is the test
    fixture's buffer, NOT the VN2 winner config (which sets ``buffer_max=0.0`` so
    its ``hi`` can never exceed ``base_sum`` even when fully calibrated). The
    residual is ``actual_sum - base_sum`` and the 0.74-quantile base commonly
    over-forecasts, so the buffer is often negative and ``hi < base_sum`` when
    correctly calibrated; the proof is that calibration RAN and moved the bound
    off the bare base-sum, not the sign of the move.
    """
    data_dir = _write_fixture(tmp_path)
    config = load_config_from_mapping(
        _config(
            data_dir,
            order_conformal={
                "coverage": 0.74,
                "protection_period": 2,
                "calibration_window": 5000,
                # Default unbounded buffer: no clamp masks the calibrated buffer.
                "buffer_max": None,
            },
        )
    )

    result = run_config(config)
    ledger = result.ledger.to_df()

    # Calibration ran: at least one terminal window recorded a residual, so the
    # nonconformity score is non-NaN somewhere in the ledger.
    assert ledger["nonconformity_score"].notna().any()

    # The bound moved off the bare base-sum: on at least one terminal window the
    # calibrated buffer is non-zero, so hi != base_sum there. Direction is not
    # asserted — a negative buffer (hi < base_sum) is correct under the
    # over-forecasting 0.74-quantile base.
    window = ledger[ledger[H] <= 2]
    base_sums = window.groupby([UNIQUE_ID, "forecast_origin"])["y_hat"].sum()
    terminal = ledger[ledger[H] == 2]
    assert not terminal.empty
    moved = False
    for _, row in terminal.iterrows():
        base_sum = base_sums.loc[(row[UNIQUE_ID], row["forecast_origin"])]
        if row["hi_0p74"] != pytest.approx(base_sum):
            moved = True
            break
    assert moved, "calibration ran but the buffer was zero on every terminal window"


def test_diagnostic_only_run_has_no_decision_column(tmp_path: Path) -> None:
    """Without an order_conformal block, no decision column appears.

    This is the regression guard: the new block is inert unless configured, so a
    diagnostic-only run (here, no conformal at all) is byte-identical to before —
    it emits no ``hi_0p74`` column and its forecast values are untouched.
    """
    data_dir = _write_fixture(tmp_path)
    baseline = load_config_from_mapping(_config(data_dir))
    with_block = load_config_from_mapping(
        _config(
            data_dir,
            order_conformal={"coverage": 0.74, "protection_period": 2, "buffer_max": 0.0},
        )
    )

    baseline_ledger = run_config(baseline).ledger.to_df()
    decision_ledger = run_config(with_block).ledger.to_df()

    assert "hi_0p74" not in baseline_ledger.columns

    # The decision bound is purely additive: point forecasts are unchanged
    # between the diagnostic-only and decision runs.
    key = [UNIQUE_ID, "forecast_origin", H]
    merged = baseline_ledger.merge(decision_ledger, on=key, suffixes=("_base", "_dec"))
    pd.testing.assert_series_equal(merged["y_hat_base"], merged["y_hat_dec"], check_names=False)


def _rs_ordering() -> dict[str, Any]:
    # Params MUST be keyed on the vn2 adapter's composite unique_id
    # (f"{Store}_{Product}" -> "1_10"/"2_20"), not the raw Product id; otherwise
    # apply_rs_policy raises "Missing policy parameters". lead_time + review_period
    # = 2 matches order_conformal.protection_period. No coverage key: it is
    # back-filled from order_conformal.coverage (the no-drift guarantee).
    return {
        "policy": "rs",
        "params": [
            {"unique_id": "1_10", "inventory_position": 3.0, "lead_time": 1, "review_period": 1},
            {"unique_id": "2_20", "inventory_position": 2.0, "lead_time": 1, "review_period": 1},
        ],
    }


def test_order_conformal_run_emits_non_nan_orders_from_bound(tmp_path: Path) -> None:
    """The R,S policy drives non-NaN orders from the emitted decision bound, e2e.

    This is the production-path proof: a ``calibre run`` with BOTH an
    ``order_conformal`` block AND an ``rs`` ``ordering`` block routes the order
    ledger through ``_build_order_config`` -> ``OrderingConfig`` -> ``RsConfig`` ->
    ``UpperBoundRule``, reading back the exact ``hi_0p74`` column the cumulative
    runtime wrote. Coverage is back-filled from ``order_conformal`` (no drift).
    The VN2 regression paths hand-build ``RsConfig`` and bypass this surface, so
    they cannot stand in for this proof.
    """
    data_dir = _write_fixture(tmp_path)
    config = load_config_from_mapping(
        _config(
            data_dir,
            order_conformal={
                "coverage": 0.74,
                "protection_period": 2,
                "calibration_window": 5000,
                "buffer_max": 0.0,
            },
            ordering=_rs_ordering(),
        )
    )
    # The back-fill made ordering.coverage match the bound's namer.
    assert config.ordering is not None
    assert config.ordering.coverage == 0.74

    result = run_config(config)
    orders = result.order_ledger.to_df()

    # The ledger is non-empty and every order is finite (the headline acceptance).
    assert not orders.empty
    assert orders["order_qty"].notna().all()
    assert orders["target_stock_level"].notna().all()

    # The R,S arithmetic holds row-by-row: order_qty == max(TSL - ip, 0).
    expected_qty = (orders["target_stock_level"] - orders["inventory_position"]).clip(lower=0.0)
    pd.testing.assert_series_equal(orders["order_qty"], expected_qty, check_names=False)

    # target_stock_level equals the emitted hi_0p74 terminal bound per (uid, origin):
    # closes the loop from the decision column to the ordering policy.
    ledger = result.ledger.to_df()
    terminal = ledger[ledger[H] == 2].set_index([UNIQUE_ID, FORECAST_ORIGIN])["hi_0p74"]
    assert not terminal.empty
    for _, row in orders.iterrows():
        bound = terminal.loc[(row[UNIQUE_ID], row[FORECAST_ORIGIN])]
        assert row["target_stock_level"] == pytest.approx(bound)


def test_order_conformal_run_rejects_drifting_coverage(tmp_path: Path) -> None:
    """An explicit, mismatched ordering.coverage is rejected at parse time.

    The production-path guarantee that drift cannot reach the engine: the
    coverage-matching validator fires at ``load_config_from_mapping`` before any
    run executes.
    """
    data_dir = _write_fixture(tmp_path)
    ordering = _rs_ordering()
    ordering["coverage"] = 0.9  # explicit and != order_conformal.coverage (0.74)
    with pytest.raises(ValidationError, match="must equal order_conformal.coverage"):
        load_config_from_mapping(
            _config(
                data_dir,
                order_conformal={
                    "coverage": 0.74,
                    "protection_period": 2,
                    "calibration_window": 5000,
                    "buffer_max": 0.0,
                },
                ordering=ordering,
            )
        )
