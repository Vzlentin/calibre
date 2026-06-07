"""Phase-level tests for the per-origin seam (U4 #114, part U4b).

U4b extracts the per-origin path into ordered, individually testable phases
(ResolveOpen -> Predict -> Calibrate -> Order -> Commit) called in fixed
order by ``run_origin``. These tests drive ONE phase at a time with a
constructed input — the test surface the seam unlocks. The end-to-end
byte-identical behavior is locked separately by the U4a characterization
tests in ``test_engine.py``; here we pin the contract of each individual phase.
"""

from contextlib import contextmanager

import pandas as pd
import pytest

from calibre.conformal import (
    SymmetricIntervalConfig,
    SymmetricIntervalRuntime,
)
from calibre.core.forecast_frame import (
    CONFORMAL_METHOD,
    FORECAST_ORIGIN,
    MODEL_NAME,
    UNIQUE_ID,
    Y_HAT,
    Y,
)
from calibre.core.forecast_task import ForecastTask
from calibre.core.order_types import RsPolicyParameters
from calibre.execution.backend import (
    BackendEngine,
    ConformalOptions,
    _with_group_tag,
)
from calibre.execution.ledger import InMemoryLedger, InMemoryOrderLedger
from calibre.execution.task_builder import partition_tasks
from calibre.ordering.policy_config import RsConfig


@contextmanager
def _materialize_refs_for(engine, tasks):
    """Build the URI-backed task refs the way ``iter_origins`` does.

    Yields ``(parallel_refs, direct_refs)`` for a pre-partitioned task list so
    phase tests can drive ``_predict`` with the same input the engine threads.
    The staging temp dir is alive only for the duration of the ``with`` block.
    """
    groups = partition_tasks(tasks)
    parallel_tasks = [_with_group_tag(t) for t in groups.local]
    direct_tasks = [_with_group_tag(t) for t in groups.global_]
    with engine._task_staging_prefix() as staging_prefix:
        parallel_refs = engine._materialize_task_refs(parallel_tasks, f"{staging_prefix}/local")
        direct_refs = engine._materialize_task_refs(direct_tasks, f"{staging_prefix}/global")
        yield parallel_refs, direct_refs


def _predict_frame(engine, parallel_refs, direct_refs, origin):
    return engine._predict(parallel_refs, direct_refs, origin).forecast


def _periodic_task(horizon=2):
    dates = pd.date_range("2024-01-07", periods=20, freq="W")
    pattern = [10.0, 20.0, 30.0, 40.0] * 5
    return (
        ForecastTask(
            history=pd.DataFrame({"unique_id": "SKU_001", "ds": dates, "y": pattern}),
            horizon=horizon,
            model_config={
                "backend": "statsforecast",
                "model": "SeasonalNaive",
                "season_length": 4,
            },
        ),
        dates,
        pattern,
    )


# --- Predict ---------------------------------------------------------------


def test_predict_phase_concatenates_local_and_global():
    """Predict returns local + global preds concatenated for the origin."""
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
        horizon=2,
        model_config={"backend": "statsforecast", "model": "SeasonalNaive", "season_length": 4},
    )
    global_task = ForecastTask(
        history=all_series,
        horizon=2,
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
    with _materialize_refs_for(engine, [local_task, global_task]) as (parallel_refs, direct_refs):
        preds = _predict_frame(engine, parallel_refs, direct_refs, dates[11])

    assert not preds.empty
    models = set(preds[MODEL_NAME].unique())
    assert "SeasonalNaive" in models  # from the local scope
    assert "global_lgbm" in models  # from the global scope
    assert (preds[FORECAST_ORIGIN] == dates[11]).all()


def test_predict_phase_empty_when_no_tasks():
    """Predict returns an empty forecast frame when there are no refs."""
    engine = BackendEngine()
    preds = _predict_frame(engine, [], [], pd.Timestamp("2024-03-31"))
    assert preds.empty


# --- Calibrate -------------------------------------------------------------


def test_calibrate_phase_applies_intervals():
    """Calibrate runs the runtime's apply() and stamps interval columns."""
    task, dates, _pattern = _periodic_task()
    runtime = SymmetricIntervalRuntime(
        SymmetricIntervalConfig(method="aci", coverage=0.9, calibration_window=4, gamma=0.05)
    )
    engine = BackendEngine(conformal=ConformalOptions(runtime=runtime))
    with _materialize_refs_for(engine, [task]) as (parallel_refs, direct_refs):
        raw = _predict_frame(engine, parallel_refs, direct_refs, dates[11])

    lower_col, upper_col = runtime.interval_columns
    assert lower_col not in raw.columns

    calibrated = engine._calibrate(raw, runtime)
    assert lower_col in calibrated.columns
    assert upper_col in calibrated.columns
    assert CONFORMAL_METHOD in calibrated.columns


def test_calibrate_phase_noop_without_runtime():
    """Calibrate returns the frame unchanged when there is no runtime."""
    engine = BackendEngine()
    frame = pd.DataFrame({UNIQUE_ID: ["A"], Y_HAT: [1.0]})
    out = engine._calibrate(frame, None)
    pd.testing.assert_frame_equal(out, frame)


def test_calibrate_phase_noop_on_empty_preds():
    """Calibrate is a no-op (does not call apply) on empty predictions."""
    runtime = SymmetricIntervalRuntime(
        SymmetricIntervalConfig(method="aci", coverage=0.9, calibration_window=4, gamma=0.05)
    )
    engine = BackendEngine(conformal=ConformalOptions(runtime=runtime))
    empty = pd.DataFrame(columns=[UNIQUE_ID, Y_HAT])
    out = engine._calibrate(empty, runtime)
    assert out.empty


# --- Order -----------------------------------------------------------------


def _rs_engine():
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
    return BackendEngine(
        conformal=ConformalOptions(runtime=runtime),
        order=order_config,
    )


def test_order_phase_appends_to_order_ledger():
    """Order derives one decision and appends it to the order ledger."""
    task, dates, _pattern = _periodic_task()
    engine = _rs_engine()
    with _materialize_refs_for(engine, [task]) as (parallel_refs, direct_refs):
        preds = _predict_frame(engine, parallel_refs, direct_refs, dates[11])
    preds = engine._calibrate(preds, engine.conformal_runtime)

    order_ledger = InMemoryOrderLedger()
    engine._order(preds, order_ledger)

    order_df = order_ledger.to_df()
    assert len(order_df) == 1
    assert (order_df[UNIQUE_ID] == "SKU_001").all()


def test_order_phase_skipped_without_order_config():
    """Order leaves the order ledger untouched when no policy is configured."""
    task, dates, _pattern = _periodic_task()
    engine = BackendEngine()  # no order config
    with _materialize_refs_for(engine, [task]) as (parallel_refs, direct_refs):
        preds = _predict_frame(engine, parallel_refs, direct_refs, dates[11])

    order_ledger = InMemoryOrderLedger()
    engine._order(preds, order_ledger)
    assert order_ledger.to_df().empty


# --- Commit: persist-exactly-once -----------------------------------------


class _CountingStateStore:
    """In-memory conformal-state store that counts upsert calls.

    Mirrors the ``ConformalStateStore`` protocol surface the engine uses while
    recording every ``upsert`` so a test can assert persist-exactly-once.
    """

    def __init__(self):
        self.states: dict[str, dict] = {}
        self.upsert_calls = 0

    def get(self, run_id, partition="__runtime__"):
        return self.states.get(partition)

    def list_for_run(self, run_id):
        return {p: s for p, s in self.states.items() if p != "__runtime__"}

    def upsert(self, run_id, partition, state):
        self.upsert_calls += 1
        self.states[partition] = state


def _engine_with_state_store():
    from uuid import uuid4

    config = SymmetricIntervalConfig(method="aci", coverage=0.9, calibration_window=4, gamma=0.05)
    store = _CountingStateStore()
    engine = BackendEngine(
        conformal=ConformalOptions(
            runtime=SymmetricIntervalRuntime(config),
            config=None,
            run_id=uuid4(),
            state_store=store,
        )
    )
    return engine, store


def test_commit_phase_validates_appends_and_persists_once_per_call():
    """One ``_commit`` call validates + appends preds and persists ONCE.

    The mutate-and-persist lives only in Commit; ``_persist_conformal_state``
    fires exactly once per ``_commit`` invocation. The h=1 row of the current
    origin is immediately due, so this Commit also observes + scores it (the
    store records the persisted partition state from that single persist).
    """
    task, dates, _pattern = _periodic_task()
    engine, store = _engine_with_state_store()
    with _materialize_refs_for(engine, [task]) as (parallel_refs, direct_refs):
        preds = _predict_frame(engine, parallel_refs, direct_refs, dates[11])
    preds = engine._calibrate(preds, engine.conformal_runtime)

    persist_calls = {"n": 0}
    original_persist = engine._persist_conformal_state

    def _spy(runtime):
        persist_calls["n"] += 1
        return original_persist(runtime)

    engine._persist_conformal_state = _spy

    ledger = InMemoryLedger()
    engine._commit(ledger, preds, task.history, dates[11], engine.conformal_runtime)
    assert len(ledger.to_df()) == 2  # two horizons appended
    assert persist_calls["n"] == 1  # persist called exactly once for this origin
    # The single persist wrote the partition snapshot for the now-due h=1 row.
    assert store.upsert_calls >= 1


def test_persist_fires_exactly_once_per_origin_over_full_run():
    """Across a real multi-origin run, persist fires exactly once per origin.

    This is the consolidation U4b performs: the pre-refactor path persisted
    inside each of its two per-origin resolve calls (up to twice per origin);
    the Commit phase now owns the only persist. We drive ``engine.execute`` with a
    counting state store and a ``_persist_conformal_state`` spy and assert the
    call count equals the number of executed origins — never double.
    """
    task, dates, _pattern = _periodic_task()
    engine, store = _engine_with_state_store()
    actuals = pd.DataFrame({"unique_id": "SKU_001", "ds": dates, "y": _pattern})
    origins = [dates[7], dates[8], dates[9]]

    persist_calls = {"n": 0}
    original_persist = engine._persist_conformal_state

    def _spy(runtime):
        persist_calls["n"] += 1
        return original_persist(runtime)

    engine._persist_conformal_state = _spy

    engine.execute(partition_tasks([task]), actuals, origins)

    assert persist_calls["n"] == len(origins)
    # State actually landed in the store (the persist is not a silent no-op).
    assert store.upsert_calls >= 1


def test_commit_phase_appends_without_runtime_and_does_not_persist():
    """Commit with no runtime still appends; persist no-ops (no store writes)."""
    task, dates, _pattern = _periodic_task()
    engine = BackendEngine()
    with _materialize_refs_for(engine, [task]) as (parallel_refs, direct_refs):
        preds = _predict_frame(engine, parallel_refs, direct_refs, dates[11])

    ledger = InMemoryLedger()
    engine._commit(ledger, preds, task.history, dates[11], None)
    assert len(ledger.to_df()) == 2


# --- ResolveOpen vs Commit: before/after-origin effects --------------------


def test_resolve_open_carries_forward_prior_origin_before_predict():
    """ResolveOpen resolves prior-origin rows already in the ledger.

    The before-origin effect: pending rows from a PRIOR origin that are now due
    get observed + scored into the ledger, with no new predictions appended.
    """
    task, dates, _pattern = _periodic_task()
    runtime = SymmetricIntervalRuntime(
        SymmetricIntervalConfig(method="aci", coverage=0.9, calibration_window=4, gamma=0.05)
    )
    engine = BackendEngine(conformal=ConformalOptions(runtime=runtime))
    actuals = pd.DataFrame({"unique_id": "SKU_001", "ds": dates, "y": _pattern})

    # Seed the ledger with a prior origin's (unresolved) predictions.
    ledger = InMemoryLedger()
    with _materialize_refs_for(engine, [task]) as (parallel_refs, direct_refs):
        prior = _predict_frame(engine, parallel_refs, direct_refs, dates[7])
    prior = engine._calibrate(prior, runtime)
    ledger.append(prior)
    assert ledger.to_df()[Y].isna().all()  # unresolved before carry-forward
    rows_before = len(ledger.to_df())

    # ResolveOpen at the next origin must resolve the now-due h=1 row in place.
    engine._resolve_open(ledger, actuals, dates[8], runtime)

    df = ledger.to_df()
    assert len(df) == rows_before  # ResolveOpen appends nothing
    assert df[Y].notna().any()  # the due prior row is now resolved
    assert "error" in df.columns


def test_resolve_open_noop_without_runtime():
    """ResolveOpen is a no-op when there is no conformal runtime."""
    task, dates, _pattern = _periodic_task()
    engine = BackendEngine()
    actuals = pd.DataFrame({"unique_id": "SKU_001", "ds": dates, "y": _pattern})
    with _materialize_refs_for(engine, [task]) as (parallel_refs, direct_refs):
        prior = _predict_frame(engine, parallel_refs, direct_refs, dates[7])

    ledger = InMemoryLedger()
    ledger.append(prior)
    before = ledger.to_df().copy()
    engine._resolve_open(ledger, actuals, dates[8], None)
    pd.testing.assert_frame_equal(ledger.to_df(), before)


def test_commit_appends_then_resolves_after_origin():
    """Commit's after-origin effect: append THIS origin, then resolve due rows.

    Complements ResolveOpen: Commit appends the current origin's predictions and
    only then resolves anything now due — the inverse ordering from ResolveOpen.
    """
    task, dates, _pattern = _periodic_task()
    runtime = SymmetricIntervalRuntime(
        SymmetricIntervalConfig(method="aci", coverage=0.9, calibration_window=4, gamma=0.05)
    )
    engine = BackendEngine(conformal=ConformalOptions(runtime=runtime))
    actuals = pd.DataFrame({"unique_id": "SKU_001", "ds": dates, "y": _pattern})

    ledger = InMemoryLedger()
    with _materialize_refs_for(engine, [task]) as (parallel_refs, direct_refs):
        preds = _predict_frame(engine, parallel_refs, direct_refs, dates[7])
    preds = engine._calibrate(preds, runtime)

    engine._commit(ledger, preds, actuals, dates[7], runtime)
    df = ledger.to_df()
    assert len(df) == 2  # this origin's two horizons were appended
    # h=1 is due as of this origin and resolves within the same Commit.
    assert df[Y].notna().any()


# --- Phase failure ---------------------------------------------------------


def test_phase_failure_names_phase_and_origin():
    """A failure inside a phase surfaces an error naming the phase and origin."""
    task, dates, _pattern = _periodic_task()
    engine = BackendEngine()
    actuals = pd.DataFrame({"unique_id": "SKU_001", "ds": dates, "y": _pattern})
    with _materialize_refs_for(engine, [task]) as (parallel_refs, direct_refs):
        refs = (parallel_refs, direct_refs)

    origin = dates[11]

    def _boom(*_args, **_kwargs):
        raise RuntimeError("predict exploded")

    engine._predict = _boom
    with pytest.raises(RuntimeError, match=rf"Predict phase failed at origin {origin}"):
        engine.run_origin(
            ledger=InMemoryLedger(),
            order_ledger=None,
            actuals=actuals,
            origin=origin,
            conformal_runtime=None,
            parallel_refs=refs[0],
            direct_refs=refs[1],
        )


def test_commit_phase_failure_preserves_valueerror_type():
    """A validation failure in Commit stays a ``ValueError`` through ``_phase``.

    ``_commit`` no longer wraps ``validate_forecast_frame`` itself; the phase
    seam re-raises with the phase + origin context while preserving the original
    type, so a caller's ``except ValueError`` (e.g. the API run store) still
    matches. This locks both the type preservation and the inner-wrap removal.
    """
    task, dates, _pattern = _periodic_task()
    engine = BackendEngine()
    actuals = pd.DataFrame({"unique_id": "SKU_001", "ds": dates, "y": _pattern})
    origin = dates[11]

    # Non-empty but missing the required forecast columns: validate_forecast_frame
    # raises ValueError inside Commit.
    bad_frame = pd.DataFrame({UNIQUE_ID: ["SKU_001"], "garbage": [1.0]})
    engine._predict = lambda *_a, **_k: bad_frame

    with pytest.raises(ValueError, match=rf"Commit phase failed at origin {origin}"):
        engine.run_origin(
            ledger=InMemoryLedger(),
            order_ledger=None,
            actuals=actuals,
            origin=origin,
            conformal_runtime=None,
            parallel_refs=[],
            direct_refs=[],
        )
