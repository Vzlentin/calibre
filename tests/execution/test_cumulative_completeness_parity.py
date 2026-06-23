"""Shared-fixture parity for the three cumulative completeness mirrors (#191).

The cumulative defer→complete→score lifecycle is decided by the same completeness
rule expressed in three different *forms*:

* engine deferral — :meth:`BackendEngine._defer_incomplete_cumulative_windows`
  (``size >= protection_period and count == size`` over ``h <= protection_period``),
* runtime readiness — the ``observe`` skip rule shared by
  :meth:`SymmetricIntervalRuntime._observe_cumulative` and
  :meth:`CumulativeRiskRuntime.observe`
  (``len(window) >= protection_period and not window[y].isna().any()``),
* decision-loop observe — :func:`observe_cumulative`'s whole-group outer gate
  (``count == size``), a conservative gate that, for contiguous
  ``h = 1..protection_period`` windows, agrees with the inner rule.

A *stricter* engine gate would starve a cold runtime (the #157 deadlock), so the
engine gate must be EQUAL to runtime readiness, never stricter. This feeds ONE
fixture of mixed complete/incomplete windows through all three and asserts an
identical complete/incomplete partition, binding the three forms against drift
rather than adding a fourth implementation. A separate assertion pins the inner
rule's ``h > protection_period`` boundary, where the decision-loop's whole-group
outer gate intentionally diverges (it is documented as a strict subset only for
contiguous ``h = 1..protection_period``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from calibre.conformal.cumulative_risk import (
    CumulativeConformalRiskConfig,
    CumulativeRiskRuntime,
)
from calibre.conformal.runtime import (
    ConformalRuntime,
    SymmetricIntervalConfig,
    SymmetricIntervalRuntime,
)
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

_PROTECTION_PERIOD = 3
_MODEL = "SeasonalNaive"

WindowKey = tuple[str, str, pd.Timestamp]


def _window(
    uid: str,
    origin: pd.Timestamp,
    *,
    horizons: list[int],
    y: list[float | None],
) -> pd.DataFrame:
    """One contiguous ``(uid, model, origin)`` window with explicit per-h actuals.

    ``horizons``/``y`` are aligned positionally; a ``None`` ``y`` is an unresolved
    horizon (NaN actual). Forecast dates step weekly off the origin so the frame
    validates as a real forecast frame.
    """
    base_origin = pd.Timestamp(origin)
    return pd.DataFrame(
        {
            UNIQUE_ID: uid,
            MODEL_NAME: _MODEL,
            FORECAST_ORIGIN: base_origin,
            H: horizons,
            DS: [base_origin + pd.Timedelta(weeks=h - 1) for h in horizons],
            Y_HAT: [10.0 * h for h in horizons],
            Y: [np.nan if value is None else float(value) for value in y],
        }
    )


_ORIGIN = pd.Timestamp("2024-01-07")


@pytest.fixture
def equal_horizon_windows() -> tuple[pd.DataFrame, set[WindowKey]]:
    """Mixed windows with ``horizon == protection_period`` + expected-complete keys.

    The regime where all three mirrors provably agree: contiguous
    ``h = 1..protection_period`` with every horizon ROW present (the streaming
    ledger keeps unresolved horizons as NaN-``y`` rows, never absent — the
    decision-loop's whole-group outer gate relies on this). Covers two of the
    plan's scenarios as a true three-way agreement:

    * exactly-``protection_period`` rows, all ``y`` present → complete in all three;
    * ``protection_period`` rows but one ``y`` NaN → incomplete in all three.

    The "missing horizon row" and "more-than-``protection_period`` rows" scenarios
    are inner-rule concerns covered in
    :func:`test_engine_and_runtime_agree_on_over_horizon_boundary` — the outer gate
    is documented to agree only when all rows are present.
    """
    frames = [
        _window("complete_exact", _ORIGIN, horizons=[1, 2, 3], y=[5.0, 6.0, 7.0]),
        _window("incomplete_nan_y", _ORIGIN, horizons=[1, 2, 3], y=[5.0, None, 7.0]),
    ]
    frame = pd.concat(frames, ignore_index=True)
    expected_complete: set[WindowKey] = {("complete_exact", _MODEL, _ORIGIN)}
    return frame, expected_complete


def _all_window_keys(frame: pd.DataFrame) -> set[WindowKey]:
    group_cols = [UNIQUE_ID, MODEL_NAME, FORECAST_ORIGIN]
    return {
        (str(uid), str(model), pd.Timestamp(origin))
        for uid, model, origin in frame[group_cols].itertuples(index=False, name=None)
    }


def _engine_complete_keys(frame: pd.DataFrame) -> set[WindowKey]:
    """Windows the engine deferral DOES NOT defer (i.e. selects as complete).

    Drives the in-window slice as both the ``updated`` (open-set) and
    ``newly_resolved`` (this-batch) frame, mirroring the terminal origin where the
    whole window resolves together; the keys surviving in ``newly_resolved`` are
    the engine's complete set.
    """
    in_window = frame[frame[H] <= _PROTECTION_PERIOD].copy()
    _, newly_resolved = BackendEngine._defer_incomplete_cumulative_windows(
        in_window.copy(),
        in_window.copy(),
        _PROTECTION_PERIOD,
    )
    return _all_window_keys(newly_resolved)


def _runtime_complete_keys(frame: pd.DataFrame, runtime: ConformalRuntime) -> set[WindowKey]:
    """Windows ``runtime.observe`` scores (its readiness rule selected as ready).

    ``apply`` first populates the interval columns ``observe`` structurally
    requires; a window is then scored exactly when the runtime writes a non-NaN
    ``nonconformity_score`` on its terminal row, so the scored keys are the
    runtime's complete set under its own readiness rule.
    """
    applied = runtime.apply(frame.copy())
    observed = runtime.observe(applied)
    scored = observed[observed[NONCONFORMITY_SCORE].notna()]
    return _all_window_keys(scored)


class _RecordingRuntime:
    """Cumulative runtime stub that records the frames handed to ``observe``.

    Lets the parity test read the decision-loop's whole-group outer gate purely
    from which windows it forwards to ``observe`` — without coupling to any real
    calibrator's scoring side effects.
    """

    mode = "cumulative"

    def __init__(self) -> None:
        self.observed_keys: set[WindowKey] = set()

    @property
    def interval_columns(self) -> tuple[str, str]:
        return ("lo", "hi")

    def observe(self, resolved: pd.DataFrame) -> pd.DataFrame:
        self.observed_keys |= _all_window_keys(resolved)
        return resolved

    def apply(self, frame: pd.DataFrame) -> pd.DataFrame:  # pragma: no cover - unused
        return frame

    def adaptive_drift(self) -> float | None:  # pragma: no cover - unused
        return None

    def get_resume_state(self) -> dict:  # pragma: no cover - unused
        return {}


def _decision_loop_complete_keys(frame: pd.DataFrame) -> set[WindowKey]:
    """Windows :func:`observe_cumulative` forwards to ``observe`` (its complete set).

    The actuals are already on the frame, so the empty lookup is a no-op fill; the
    helper's whole-group ``count == size`` gate then decides what it forwards.
    """
    recorder = _RecordingRuntime()
    observe_cumulative(recorder, [frame.copy()], pd.Series(dtype=float))
    return recorder.observed_keys


def _symmetric_runtime() -> SymmetricIntervalRuntime:
    return SymmetricIntervalRuntime(
        SymmetricIntervalConfig(
            method="mscp",
            coverage=0.9,
            calibration_window=10,
            mode="cumulative",
            protection_period=_PROTECTION_PERIOD,
        )
    )


def _one_sided_runtime() -> CumulativeRiskRuntime:
    return CumulativeRiskRuntime(
        CumulativeConformalRiskConfig(
            coverage=0.5,
            protection_period=_PROTECTION_PERIOD,
            calibration_window=10,
            weight_decay=None,
        )
    )


def test_three_completeness_mirrors_partition_identically(equal_horizon_windows) -> None:
    frame, expected_complete = equal_horizon_windows
    all_keys = _all_window_keys(frame)

    engine_keys = _engine_complete_keys(frame)
    symmetric_keys = _runtime_complete_keys(frame, _symmetric_runtime())
    one_sided_keys = _runtime_complete_keys(frame, _one_sided_runtime())
    decision_loop_keys = _decision_loop_complete_keys(frame)

    # All three forms select the SAME complete set, matching the partition
    # derived by hand from the fixture.
    assert engine_keys == expected_complete
    assert symmetric_keys == expected_complete
    assert one_sided_keys == expected_complete
    assert decision_loop_keys == expected_complete

    # And therefore the SAME incomplete complement.
    incomplete = all_keys - expected_complete
    assert all_keys - engine_keys == incomplete
    assert all_keys - symmetric_keys == incomplete
    assert all_keys - one_sided_keys == incomplete
    assert all_keys - decision_loop_keys == incomplete


def test_engine_gate_is_not_stricter_than_runtime_readiness(equal_horizon_windows) -> None:
    """The deadlock direction (#157): the engine never defers a ready window.

    A stricter engine gate would hold back a window the runtime is ready to score,
    starving a cold calibrator forever. Equality is asserted above; this pins the
    one-directional safety invariant explicitly so a future drift that makes the
    engine stricter fails here with an unambiguous message.
    """
    frame, _ = equal_horizon_windows
    engine_keys = _engine_complete_keys(frame)
    symmetric_keys = _runtime_complete_keys(frame, _symmetric_runtime())
    # Every window the runtime is ready to score, the engine also lets through.
    assert symmetric_keys <= engine_keys


def test_engine_and_runtime_agree_on_over_horizon_boundary() -> None:
    """Inner-rule boundary: rows ``h > protection_period`` never gate a window.

    A window with MORE than ``protection_period`` horizons is complete once its
    ``h <= protection_period`` rows are all present, even if the out-of-window
    ``h > protection_period`` row is still unresolved. The engine deferral and the
    runtime readiness rule (the two inner-rule forms) agree on this; the
    decision-loop's whole-group outer gate is excluded here by construction (it is
    a strict subset only for contiguous ``h = 1..protection_period``).
    """
    over = _window("complete_over", _ORIGIN, horizons=[1, 2, 3, 4], y=[5.0, 6.0, 7.0, None])
    expected = {("complete_over", _MODEL, _ORIGIN)}

    assert _engine_complete_keys(over) == expected
    assert _runtime_complete_keys(over, _symmetric_runtime()) == expected
    assert _runtime_complete_keys(over, _one_sided_runtime()) == expected


def test_engine_and_runtime_agree_on_missing_in_window_horizon() -> None:
    """Inner-rule: a window missing a ``h <= protection_period`` ROW is incomplete.

    Fewer than ``protection_period`` in-window rows means the window can never be
    scored (the cumulative sum is undefined), so the engine deferral and the
    runtime readiness rule both treat it as incomplete. This is the inner rule's
    ``len(window) < protection_period`` arm, distinct from the NaN-``y`` arm.
    """
    missing = _window("incomplete_missing_h", _ORIGIN, horizons=[1, 2], y=[5.0, 6.0])
    empty: set[WindowKey] = set()

    assert _engine_complete_keys(missing) == empty
    assert _runtime_complete_keys(missing, _symmetric_runtime()) == empty
    assert _runtime_complete_keys(missing, _one_sided_runtime()) == empty
