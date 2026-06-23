"""Engine-internal deferral coverage for the one-sided ``order_conformal`` runtime.

VN2's external-conformal gate does not exercise the engine-internal
defer→complete→score path for the one-sided :class:`CumulativeRiskRuntime`, so
this is the load-bearing engine-internal gate that the capability gate actually
drives that runtime end-to-end:

* a cumulative window spanning more than one origin is DEFERRED while incomplete
  (its rows stay pending, ``y`` reverted to NaN in the open set), then COMPLETED
  and SCORED exactly once at its terminal origin (not stranded, not
  double-counted);
* a ``protection_period > min(horizon)`` misconfig fails loudly at start rather
  than deferring every in-window row forever.

Tiny single-series fixture with origins spaced tighter than the window span — no
full panel.
"""

from __future__ import annotations

import pandas as pd
import pytest

from calibre.conformal.cumulative_risk import (
    CumulativeConformalRiskConfig,
    CumulativeRiskRuntime,
)
from calibre.core.forecast_frame import (
    DS,
    FORECAST_ORIGIN,
    NONCONFORMITY_SCORE,
    UNIQUE_ID,
    H,
    Y,
)
from calibre.core.forecast_task import ForecastTask
from calibre.execution.backend import BackendEngine, ConformalOptions
from calibre.execution.task_builder import partition_tasks

_PROTECTION_PERIOD = 3


def _split_window_setup(protection_period: int = _PROTECTION_PERIOD):
    """Single-series inputs whose origins are spaced tighter than the window span.

    Weekly origins one step apart with ``protection_period`` horizons mean each
    window's horizons resolve across ``protection_period`` separate origins:
    SeasonalNaive maps h=1 to the origin date itself, so a window forecast at
    ``dates[k]`` resolves its terminal horizon (h=``protection_period``) only at
    ``dates[k + protection_period - 1]`` — the split-window case the engine
    deferral must hold open.
    """
    dates = pd.date_range("2024-01-07", periods=20, freq="W")
    pattern = [10.0, 20.0, 30.0, 40.0] * 5
    actuals = pd.DataFrame({UNIQUE_ID: "SKU_001", DS: dates, Y: pattern})
    task = ForecastTask(
        history=pd.DataFrame({UNIQUE_ID: "SKU_001", DS: dates, Y: pattern}),
        horizon=protection_period,
        model_config={"backend": "statsforecast", "model": "SeasonalNaive", "season_length": 4},
    )
    origins = [dates[7], dates[8], dates[9], dates[10], dates[11]]
    return task, actuals, origins, dates


def _runtime(protection_period: int = _PROTECTION_PERIOD) -> CumulativeRiskRuntime:
    return CumulativeRiskRuntime(
        CumulativeConformalRiskConfig(
            coverage=0.5,
            protection_period=protection_period,
            calibration_window=10,
            weight_decay=None,
        )
    )


def test_one_sided_runtime_defers_then_scores_split_window() -> None:
    """A window spanning 3 origins defers while incomplete, scores at the terminal.

    The capability gate drives the one-sided runtime through the engine deferral,
    so the split window is held open (rows pending, ``y`` NaN) until its terminal
    horizon lands, then scored once.
    """
    task, actuals, origins, _dates = _split_window_setup()
    runtime = _runtime()
    engine = BackendEngine(conformal=ConformalOptions(runtime=runtime))

    result = engine.execute(partition_tasks([task]), actuals, origins)
    ledger = result.ledger.to_df()

    # The one-sided calibrator scored at least one fully-completed window — the
    # window was NOT stranded by the engine.
    assert runtime.get_diagnostics()["n_scores"] >= 1

    # The terminal-horizon row of a completed window carries a real
    # nonconformity score in the ledger, not NaN.
    terminal_scored = ledger[(ledger[H] == _PROTECTION_PERIOD) & ledger[Y].notna()]
    assert not terminal_scored.empty
    assert terminal_scored[NONCONFORMITY_SCORE].notna().any()

    # Open-set invariant: deferral is all-or-nothing per window. Among in-window
    # rows due by the final origin, each window is either all resolved (completed
    # + scored) or all still y-NaN (deferred together) — never a mixed window
    # (which would be a resolve-then-strand bug).
    last_origin = origins[-1]
    due = ledger[(ledger[H] <= _PROTECTION_PERIOD) & (ledger[DS] <= last_origin)]
    per_window = due.groupby([UNIQUE_ID, FORECAST_ORIGIN])[Y].agg(["count", "size"])
    assert ((per_window["count"] == 0) | (per_window["count"] == per_window["size"])).all()
    # Non-vacuous: the fixture produces both completed and still-deferred windows.
    assert (per_window["count"] == per_window["size"]).any()
    assert (per_window["count"] == 0).any()


def test_one_sided_runtime_scores_each_window_exactly_once() -> None:
    """A completed split window scores once, not twice across ResolveOpen/Commit.

    ``_resolve_due`` runs twice per origin (ResolveOpen pre-Predict, Commit at
    end). A window completed in ResolveOpen must not reappear in Commit's due
    frame, so the calibrator records one residual per completed window. With
    SeasonalNaive mapping h=1 to the origin date, windows forecast at
    dates[7..9] complete at dates[9..11] within the run — exactly three.
    """
    task, actuals, origins, _dates = _split_window_setup()
    runtime = _runtime()
    engine = BackendEngine(conformal=ConformalOptions(runtime=runtime))

    engine.execute(partition_tasks([task]), actuals, origins)

    assert runtime.get_diagnostics()["n_scores"] == 3


def test_one_sided_runtime_protection_period_exceeding_horizon_raises() -> None:
    """``protection_period > min(horizon)`` fails loudly at start (no silent hang).

    Without the preflight on the one-sided path, every in-window row would defer
    forever (total silent data loss). The capability gate now raises before any
    ledger is allocated.
    """
    task, actuals, origins, _dates = _split_window_setup()
    runtime = _runtime(protection_period=task.horizon + 1)
    engine = BackendEngine(conformal=ConformalOptions(runtime=runtime))

    with pytest.raises(ValueError, match="exceeds available horizon"):
        engine.execute(partition_tasks([task]), actuals, origins)
