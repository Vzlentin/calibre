"""Exercise conformal observation through revision-bound engine snapshots."""

from __future__ import annotations

import inspect
import math
from collections.abc import Mapping
from pathlib import Path

import pandas as pd
import pytest

from newcalibre.conformal import (
    METHOD_SCOPE_LABEL,
    CalibrationSeedBatch,
    derive_partition_label,
    resolve_method,
)
from newcalibre.conformal.state import JsonStateCodec
from newcalibre.domain import (
    ACTUAL_VALUE,
    HORIZON_STEP,
    MODEL_NAME,
    OBSERVED_VALUE,
    ORIGIN,
    POINT_FORECAST,
    SERIES_KEY,
    TARGET_TIMESTAMP,
    TIMESTAMP,
    ActualsSemantics,
    Calendar,
    EmissionScope,
    ForecastTask,
    HierarchyIndex,
    Panel,
    Scope,
    SessionIdentity,
    TargetSupport,
    interval_columns,
    target_timestamp,
)
from newcalibre.engine import (
    Engine,
    EngineError,
    InMemoryIndexedRunStore,
    InMemoryPanelSource,
    InProcessDispatch,
    OriginCommit,
    OriginIntent,
    OriginRequest,
    OriginSnapshot,
    Spine,
)
from newcalibre.forecasting import AdapterCapability, AdapterCapabilityError

_CALENDAR = Calendar("D", phase=pd.Timestamp("2026-01-01"))
_MODEL_CONFIG = {"backend": "observe-fixture"}


class _LastValueAdapter:
    """Emit the latest strict-history observation for every requested row."""

    def __init__(self) -> None:
        self._points: dict[str, float] = {}

    @property
    def capabilities(self) -> frozenset[AdapterCapability]:
        return frozenset()

    @property
    def requested_capabilities(self) -> frozenset[AdapterCapability]:
        return frozenset()

    def fit(self, task: ForecastTask) -> None:
        history = task.history.materialize()
        self._points = {
            key: float(history.loc[history[SERIES_KEY] == key, OBSERVED_VALUE].iloc[-1])
            for key in task.series_keys
        }

    def predict(self, task: ForecastTask) -> pd.DataFrame:
        frame = pd.DataFrame.from_records(
            [
                {
                    SERIES_KEY: key,
                    TARGET_TIMESTAMP: target_timestamp(
                        task.origin,
                        step,
                        calendar=task.calendar,
                    ),
                    ACTUAL_VALUE: math.nan,
                    POINT_FORECAST: self._points[key],
                    HORIZON_STEP: step,
                    ORIGIN: task.origin,
                    MODEL_NAME: "observe-fixture",
                }
                for key in task.series_keys
                for step in range(1, task.horizon + 1)
            ]
        )
        return frame.astype(
            {
                SERIES_KEY: "string",
                ACTUAL_VALUE: "float64",
                POINT_FORECAST: "float64",
                HORIZON_STEP: "int64",
                MODEL_NAME: "string",
            }
        )

    def fitted_values(self):
        raise AdapterCapabilityError("fixture has no fitted values")

    def dump_state(self) -> bytes:
        raise AdapterCapabilityError("fixture has no persistence")

    def load_state(self, state: bytes) -> None:
        raise AdapterCapabilityError("fixture has no persistence")

    def update(self, delta) -> None:
        del delta
        raise AdapterCapabilityError("fixture has no incremental update")


def _panel(
    *,
    series_keys: tuple[str, ...] = ("a",),
    overrides: Mapping[tuple[str, pd.Timestamp], float] | None = None,
) -> Panel:
    changed = {} if overrides is None else dict(overrides)
    timestamps = pd.date_range("2026-01-01", periods=7, freq="D")
    frame = pd.DataFrame.from_records(
        [
            {
                SERIES_KEY: key,
                TIMESTAMP: timestamp,
                OBSERVED_VALUE: changed.get((key, timestamp), float(index + position)),
            }
            for position, key in enumerate(series_keys)
            for index, timestamp in enumerate(timestamps, start=1)
        ]
    ).astype({SERIES_KEY: "string", OBSERVED_VALUE: "float64"})
    return Panel.from_frame(frame, calendar=_CALENDAR, target_support=TargetSupport.REAL)


def _session(
    *,
    series_keys: tuple[str, ...] = ("a",),
    conformal_config: Mapping[str, object] | None = None,
) -> SessionIdentity:
    return SessionIdentity.derive(
        tenant="observe-engine",
        series_keys=series_keys,
        calendar=_CALENDAR,
        horizon=1,
        model_config=_MODEL_CONFIG,
        conformal_config=conformal_config,
    )


def _engine(
    *,
    forecast_panel: Panel,
    actuals_panel: Panel,
    session: SessionIdentity,
    store: InMemoryIndexedRunStore | None = None,
) -> tuple[Engine, InMemoryIndexedRunStore]:
    run_store = store or InMemoryIndexedRunStore(
        session=session,
        calendar=_CALENDAR,
        actuals=actuals_panel,
        actuals_semantics=ActualsSemantics.DEMAND,
    )
    return (
        Engine(
            session=session,
            panel_source=InMemoryPanelSource(forecast_panel),
            run_store=run_store,
            dispatch_backend=InProcessDispatch(),
            hierarchy=HierarchyIndex.flat(forecast_panel.series_keys),
            adapter_resolver=lambda _configuration: _LastValueAdapter(),
        ),
        run_store,
    )


def _snapshot(
    store: InMemoryIndexedRunStore,
    session: SessionIdentity,
    origin: pd.Timestamp,
) -> OriginSnapshot:
    snapshot = store.open(OriginIntent(session, origin))
    assert isinstance(snapshot, OriginSnapshot)
    return snapshot


def _run_origin(
    engine: Engine,
    store: InMemoryIndexedRunStore,
    session: SessionIdentity,
    origin: str,
):
    timestamp = pd.Timestamp(origin)
    return Spine(engine).run_origin(
        OriginRequest(session=session, origin=timestamp, scope=Scope.GLOBAL),
        snapshot=_snapshot(store, session, timestamp),
    )


def _run_three_origins(
    *,
    forecast_panel: Panel,
    actuals_panel: Panel,
    configuration: Mapping[str, object] | None,
):
    session = _session(
        series_keys=forecast_panel.series_keys,
        conformal_config=configuration,
    )
    engine, store = _engine(
        forecast_panel=forecast_panel,
        actuals_panel=actuals_panel,
        session=session,
    )
    _run_origin(engine, store, session, "2026-01-02")
    first = _run_origin(engine, store, session, "2026-01-03")
    second = _run_origin(engine, store, session, "2026-01-04")
    return first, second, engine, store, session


def test_identity_calibration_records_resolutions_without_rewriting_forecasts() -> None:
    """Append resolution facts to their side stream and drain all pending rows."""
    panel = _panel()
    _first, _second, engine, store, session = _run_three_origins(
        forecast_panel=panel,
        actuals_panel=panel,
        configuration=None,
    )
    origin = pd.Timestamp("2026-01-05")
    snapshot = _snapshot(store, session, origin)
    final = engine.observe(origin, session=session, snapshot=snapshot)
    engine.commit(
        OriginCommit(
            session=session,
            origin=origin,
            expected_revision=snapshot.revision,
            observe_cycle=final.cycle,
            state_updates=final.cycle.state_updates,
        )
    )

    assert len(store.observed_history) == 4
    assert [row.actual_value for row in store.forecasts] == [None, None, None]
    assert [row.actual for row in store.observation_resolutions] == [2.0, 3.0, 4.0]
    assert store.pending_observations == ()
    assert store.observe_annotations == ()
    assert store.states == {}


@pytest.mark.parametrize(
    ("method", "codec"),
    [("split-per-step", "split-per-step"), ("weighted-per-step", "weighted-per-step")],
)
def test_registered_runtime_escapes_cold_start_and_publishes_dirty_rows(
    method: str,
    codec: str,
) -> None:
    """Observe before issuance and atomically publish method-owned state."""
    configuration = {"method": method, "coverage": 0.5, "calibration_window": 20}
    panel = _panel()
    first, second, _engine_value, store, _session_value = _run_three_origins(
        forecast_panel=panel,
        actuals_panel=panel,
        configuration=configuration,
    )
    _lower, upper = interval_columns(0.5)
    partition = derive_partition_label(
        "observe-fixture",
        "global",
        EmissionScope.PER_STEP,
    )
    payload = JsonStateCodec(codec, 1).decode(store.states[partition], expected_label=partition)

    assert math.isnan(float(first.forecasts.frame[upper].iloc[0]))
    assert float(second.forecasts.frame[upper].iloc[0]) == 4.0
    assert len(store.observe_annotations) == 2
    assert all(value.advanced_delivered_score for value in store.observe_annotations)
    assert isinstance(payload, dict)
    assert payload["delivered_score_count"] == 2
    assert set(store.states) == {METHOD_SCOPE_LABEL, partition}


def test_sequential_adaptive_runtime_closes_feedback_through_the_generic_loop() -> None:
    """Advance feedback only after the configured finite-score boundary."""
    configuration = {
        "method": "sequential-adaptive-per-step",
        "coverage": 0.5,
        "calibration_window": 20,
        "learning_rate": 0.25,
    }
    panel = _panel()
    first, second, engine, store, session = _run_three_origins(
        forecast_panel=panel,
        actuals_panel=panel,
        configuration=configuration,
    )
    third = _run_origin(engine, store, session, "2026-01-05")
    _lower, upper = interval_columns(0.5)
    partition = derive_partition_label(
        "observe-fixture",
        "global",
        EmissionScope.PER_STEP,
    )
    payload = JsonStateCodec("sequential-adaptive-per-step", 1).decode(
        store.states[partition],
        expected_label=partition,
    )

    assert math.isnan(float(first.forecasts.frame[upper].iloc[0]))
    assert float(second.forecasts.frame[upper].iloc[0]) == 4.0
    assert float(third.forecasts.frame[upper].iloc[0]) == 5.0
    assert isinstance(payload, dict)
    assert payload["delivered_score_count"] == 3
    assert payload["feedback_count"] == 1
    assert payload["raw_alpha"] == 0.625


def test_observe_before_issue_changes_only_the_perturbed_next_bound() -> None:
    """Use the current origin's actual delta before issuing its calibrated bound."""
    configuration = {"method": "split-per-step", "coverage": 0.5, "calibration_window": 20}
    forecast_panel = _panel()
    baseline = _run_three_origins(
        forecast_panel=forecast_panel,
        actuals_panel=_panel(),
        configuration=configuration,
    )[1]
    perturbed = _run_three_origins(
        forecast_panel=forecast_panel,
        actuals_panel=_panel(overrides={("a", pd.Timestamp("2026-01-03")): 8.0}),
        configuration=configuration,
    )[1]
    _lower, upper = interval_columns(0.5)

    assert float(perturbed.forecasts.frame[upper].iloc[0]) != float(
        baseline.forecasts.frame[upper].iloc[0]
    )


def test_one_snapshot_preserves_an_untouched_partition_state() -> None:
    """Merge dirty runtime rows without overwriting a foreign partition."""
    configuration = {"method": "split-per-step", "coverage": 0.5, "calibration_window": 20}
    panel = _panel(series_keys=("a", "b"))
    session = _session(series_keys=panel.series_keys, conformal_config=configuration)
    store = InMemoryIndexedRunStore(
        session=session,
        calendar=_CALENDAR,
        actuals=panel,
        actuals_semantics=ActualsSemantics.DEMAND,
    )
    foreign_label = derive_partition_label(
        "observe-fixture",
        "foreign",
        EmissionScope.PER_STEP,
    )
    foreign_state = resolve_method(configuration).calibrate(
        CalibrationSeedBatch({foreign_label: [9.0]})
    )[foreign_label]
    store.commit(
        OriginCommit(
            session=session,
            origin=pd.Timestamp("2026-01-01"),
            expected_revision=store.revision,
            state_updates={foreign_label: foreign_state},
        )
    )
    engine, _store = _engine(
        forecast_panel=panel,
        actuals_panel=panel,
        session=session,
        store=store,
    )
    result = _run_origin(engine, store, session, "2026-01-03")

    references = {
        facts.state_reference for facts in result.forecasts.observation_issuances.values()
    }
    assert len(references) == 1
    assert store.states[foreign_label] == foreign_state
    assert METHOD_SCOPE_LABEL in store.states


def test_cross_engine_staged_observation_is_rejected() -> None:
    """Reject a cycle token produced from another engine/store revision context."""
    panel = _panel()
    session = _session()
    first, first_store = _engine(
        forecast_panel=panel,
        actuals_panel=panel,
        session=session,
    )
    second, second_store = _engine(
        forecast_panel=panel,
        actuals_panel=panel,
        session=session,
    )
    first_store.commit(
        OriginCommit(
            session=session,
            origin=pd.Timestamp("2026-01-01"),
            expected_revision=first_store.revision,
        )
    )
    origin = pd.Timestamp("2026-01-03")
    staged = first.observe(
        origin,
        session=session,
        snapshot=_snapshot(first_store, session, origin),
    )
    second_observation = second.observe(
        origin,
        session=session,
        snapshot=_snapshot(second_store, session, origin),
    )
    request = OriginRequest(session=session, origin=origin, scope=Scope.GLOBAL)
    forecasts = second.predict(second.fit(request))
    assert second_observation.token is not None

    with pytest.raises(EngineError, match="cycle token"):
        second.calibrate(forecasts, session=session, observation=staged)


def test_removed_callback_and_method_branches_are_structurally_absent() -> None:
    """Keep engine and observe packages generic over registered conformal methods."""
    callback_terms = ("Calibrator", "Observer", "observer=", "calibrator=")
    method_terms = (
        "WeightedPerStep",
        "WeightedConformal",
        "SequentialAdaptive",
    )
    source_root = Path(__file__).parents[2] / "src" / "newcalibre"
    callback_violations = {
        str(path.relative_to(source_root)): value
        for path in source_root.rglob("*.py")
        for value in callback_terms
        if value in path.read_text()
    }
    method_violations = {
        str(path.relative_to(source_root)): value
        for package in ("engine", "observe")
        for path in (source_root / package).rglob("*.py")
        for value in method_terms
        if value in path.read_text()
    }
    assert callback_violations == {}
    assert method_violations == {}


def test_engine_constructor_exposes_only_the_three_remaining_ports() -> None:
    """Keep snapshot observation on the closed engine surface without callbacks."""
    parameters = inspect.signature(Engine).parameters
    assert {"session", "panel_source", "run_store", "dispatch_backend"} <= set(parameters)
    assert "observer" not in parameters
    assert "calibrator" not in parameters
