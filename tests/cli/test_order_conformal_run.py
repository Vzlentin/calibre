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

from calibre.cli.commands import run_config
from calibre.cli.config import load_config_from_mapping
from calibre.core.forecast_frame import UNIQUE_ID, H

# Weekly wide-format VN2 sales with a period-2 seasonal pattern; long enough that
# the rolling origins below have in-window actuals to resolve against.
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
        "1,10," + ",".join(["10", "12"] * 5),
        "2,20," + ",".join(["5", "7"] * 5),
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
