"""Shared-fixture parity for the one-sided cumulative completeness mirrors.

The one-sided ``CumulativeRiskRuntime`` window-completeness rule is mirrored in
three places that must agree on which ``(unique_id, model_name,
forecast_origin)`` windows score, or the engine deferral, the driver-side
observe gate, and the runtime would drift apart:

1. :meth:`calibre.conformal.cumulative_risk.CumulativeRiskRuntime.observe` — the
   runtime's own inline rule: a window's ``h <= protection_period`` rows must all
   be present (count ``>= protection_period``) with no NaN ``y``. (This is the
   one-sided runtime's rule, distinct from
   :meth:`SymmetricIntervalRuntime._observe_cumulative`, which carries an extra
   terminal-horizon check the one-sided rule lacks.)
2. :meth:`calibre.execution.backend.BackendEngine._defer_incomplete_cumulative_windows`
   — the streaming-ledger engine deferral.
3. :func:`calibre.execution.decision_loop.observe_cumulative` — the driver-side
   conservative outer gate (``count == size`` over every row in the group).

Under the contiguous ``h = 1..horizon`` invariant the task builder guarantees,
the three rules coincide for every window shape. The fixture below holds that
invariant and covers a complete window, a split (partial) window, and a
contiguous ``horizon > protection_period`` window — confirming the agreement
extends to the ``horizon > protection_period`` band. The rules could only
diverge if that contiguity invariant were violated, which the task builder
prevents.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from calibre.conformal import CumulativeConformalRiskConfig, CumulativeRiskRuntime
from calibre.core.forecast_frame import (
    DS,
    FORECAST_ORIGIN,
    MODEL_NAME,
    NONCONFORMITY_SCORE,
    UNIQUE_ID,
    Y_HAT,
    H,
    Y,
)
from calibre.execution.backend import BackendEngine
from calibre.execution.decision_loop import observe_cumulative

_PROTECTION_PERIOD = 2
_MODEL = "M"
_WINDOW_COLS = [UNIQUE_ID, MODEL_NAME, FORECAST_ORIGIN]


def _window(
    *,
    unique_id: str,
    origin: str,
    horizon: int,
    y_present: list[bool],
) -> pd.DataFrame:
    """Build one contiguous ``h = 1..horizon`` window.

    ``y_present[k]`` marks whether horizon ``k + 1`` has a resolved actual; absent
    horizons carry NaN ``y`` so a window is "split" when any in-window horizon is
    missing.
    """
    origin_ts = pd.Timestamp(origin)
    y_hats = [10.0 * (h + 1) for h in range(horizon)]
    actuals = [
        yh + 1.0 if present else np.nan for yh, present in zip(y_hats, y_present, strict=True)
    ]
    return pd.DataFrame(
        {
            UNIQUE_ID: [unique_id] * horizon,
            DS: pd.date_range(origin_ts + pd.Timedelta(weeks=1), periods=horizon, freq="W"),
            Y: actuals,
            Y_HAT: y_hats,
            H: list(range(1, horizon + 1)),
            FORECAST_ORIGIN: [origin_ts] * horizon,
            MODEL_NAME: [_MODEL] * horizon,
        }
    )


def _fixture() -> pd.DataFrame:
    """A frame with a complete window, a split window, and a horizon>pp window."""
    return pd.concat(
        [
            # Complete: both in-window horizons (h=1,2) resolved -> scores.
            _window(unique_id="A", origin="2024-01-01", horizon=2, y_present=[True, True]),
            # Split: h=1 resolved, h=2 missing -> deferred, does not score.
            _window(unique_id="B", origin="2024-01-01", horizon=2, y_present=[True, False]),
            # horizon > protection_period: h=1,2,3 with pp=2. All rows resolved so
            # the outer gate admits it; the h<=2 window completes and scores.
            _window(unique_id="C", origin="2024-01-01", horizon=3, y_present=[True, True, True]),
        ],
        ignore_index=True,
    )


def _window_key_set(frame: pd.DataFrame) -> set[tuple]:
    if frame.empty:
        return set()
    keys = frame[_WINDOW_COLS].drop_duplicates()
    return set(map(tuple, keys.itertuples(index=False, name=None)))


def _runtime_scored_windows(frame: pd.DataFrame) -> set[tuple]:
    """Mirror (i): windows the runtime's inline observe rule scores."""
    runtime = CumulativeRiskRuntime(
        CumulativeConformalRiskConfig(
            coverage=0.5,
            protection_period=_PROTECTION_PERIOD,
            weight_decay=None,
        )
    )
    observed = runtime.observe(frame.copy())
    scored = observed[observed[NONCONFORMITY_SCORE].notna()]
    return _window_key_set(scored)


def _engine_deferral_scored_windows(frame: pd.DataFrame) -> set[tuple]:
    """Mirror (ii): windows the engine deferral does NOT hold back.

    ``updated`` is the whole window frame (every present horizon); ``newly_resolved``
    is the in-window rows with a resolved actual. The deferral returns the rows it
    keeps; the windows still represented in the kept in-window rows are the ones
    that scored.
    """
    updated = frame.copy()
    newly_resolved = frame[frame[Y].notna()].copy()
    _, kept = BackendEngine._defer_incomplete_cumulative_windows(
        updated, newly_resolved, _PROTECTION_PERIOD
    )
    in_window_kept = kept[kept[H] <= _PROTECTION_PERIOD]
    return _window_key_set(in_window_kept)


def _observe_cumulative_scored_windows(frame: pd.DataFrame) -> set[tuple]:
    """Mirror (iii): windows the driver-side outer gate admits to observe."""
    empty_lookup = pd.Series(dtype=float)
    still_pending = observe_cumulative(
        CumulativeRiskRuntime(
            CumulativeConformalRiskConfig(
                coverage=0.5,
                protection_period=_PROTECTION_PERIOD,
                weight_decay=None,
            )
        ),
        [frame.copy()],
        empty_lookup,
    )
    pending_keys = set()
    for pending in still_pending:
        pending_keys |= _window_key_set(pending)
    all_keys = _window_key_set(frame)
    return all_keys - pending_keys


def test_one_sided_completeness_mirrors_agree_on_scored_windows():
    """The three one-sided completeness mirrors score the same window set.

    Over a fixture with a complete window, a split window, and a horizon>pp
    window, the runtime inline rule, the engine deferral, and the driver-side
    outer gate must agree on which windows score — no fourth implementation can
    drift them apart.
    """
    frame = _fixture()

    runtime_scored = _runtime_scored_windows(frame)
    engine_scored = _engine_deferral_scored_windows(frame)
    observe_scored = _observe_cumulative_scored_windows(frame)

    # The complete window (A) and the horizon>pp window (C) score; the split
    # window (B) does not.
    expected = {
        ("A", _MODEL, pd.Timestamp("2024-01-01")),
        ("C", _MODEL, pd.Timestamp("2024-01-01")),
    }
    assert runtime_scored == expected
    assert engine_scored == expected
    assert observe_scored == expected


def test_one_sided_split_window_is_held_back_by_all_mirrors():
    """A split window scores in none of the three mirrors.

    The split window (B: h=1 resolved, h=2 missing) is the regression-bearing
    case: every mirror must hold it back, or a partial window would score on one
    path and not another.
    """
    frame = _fixture()
    split_key = ("B", _MODEL, pd.Timestamp("2024-01-01"))

    assert split_key not in _runtime_scored_windows(frame)
    assert split_key not in _engine_deferral_scored_windows(frame)
    assert split_key not in _observe_cumulative_scored_windows(frame)
