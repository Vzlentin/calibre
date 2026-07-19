"""Exercise the conformal observe loop through the public engine spine."""

from __future__ import annotations

import inspect
import math
from collections.abc import Mapping
from pathlib import Path

import pandas as pd
import pytest

from newcalibre.conformal import (
    METHOD_SCOPE_LABEL,
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
    interval_columns,
    target_timestamp,
)
from newcalibre.engine import (
    Engine,
    EngineError,
    InMemoryActualsSource,
    InMemoryArtifactStore,
    InMemoryCalibrationStateStore,
    InMemoryLedgerSink,
    InMemoryPanelSource,
    InProcessDispatch,
    OriginCommit,
    OriginRequest,
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

    def fit(self, task: ForecastTask, *, collect_fitted_values: bool = False) -> None:
        if collect_fitted_values:
            raise AdapterCapabilityError("fixture has no fitted values")
        self._points = {
            series_key: float(
                task.history.loc[
                    task.history[SERIES_KEY] == series_key,
                    OBSERVED_VALUE,
                ].iloc[-1]
            )
            for series_key in task.series_keys
        }

    def predict(self, task: ForecastTask) -> pd.DataFrame:
        frame = pd.DataFrame.from_records(
            [
                {
                    SERIES_KEY: series_key,
                    TARGET_TIMESTAMP: target_timestamp(task.origin, step, calendar=task.calendar),
                    ACTUAL_VALUE: math.nan,
                    POINT_FORECAST: self._points[series_key],
                    HORIZON_STEP: step,
                    ORIGIN: task.origin,
                    MODEL_NAME: "observe-fixture",
                }
                for series_key in task.series_keys
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

    def fitted_values(self, task: ForecastTask):
        raise AdapterCapabilityError("fixture has no fitted values")

    def dump_state(self) -> bytes:
        raise AdapterCapabilityError("fixture has no persistence")

    def load_state(self, state: bytes) -> None:
        raise AdapterCapabilityError("fixture has no persistence")

    def update(self, task: ForecastTask) -> None:
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
                SERIES_KEY: series_key,
                TIMESTAMP: timestamp,
                OBSERVED_VALUE: changed.get(
                    (series_key, timestamp),
                    float(index + series_position),
                ),
            }
            for series_position, series_key in enumerate(series_keys)
            for index, timestamp in enumerate(timestamps, start=1)
        ]
    ).astype({SERIES_KEY: "string", OBSERVED_VALUE: "float64"})
    return Panel.from_frame(frame, calendar=_CALENDAR)


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
    states: InMemoryCalibrationStateStore | None = None,
    sink: InMemoryLedgerSink | None = None,
) -> tuple[Engine, InMemoryCalibrationStateStore, InMemoryLedgerSink]:
    state_store = states or InMemoryCalibrationStateStore()
    ledger = sink or InMemoryLedgerSink(session=session, calendar=_CALENDAR)
    engine = Engine(
        panel_source=InMemoryPanelSource(forecast_panel),
        actuals_source=InMemoryActualsSource(
            actuals_panel,
            actuals_semantics=ActualsSemantics.DEMAND,
        ),
        artifact_store=InMemoryArtifactStore(),
        calibration_state_store=state_store,
        ledger_sink=ledger,
        dispatch_backend=InProcessDispatch(),
        hierarchy=HierarchyIndex.flat(forecast_panel.series_keys),
        adapter_resolver=lambda _configuration: _LastValueAdapter(),
    )
    return engine, state_store, ledger


def _run_two_origins(
    *,
    forecast_panel: Panel,
    actuals_panel: Panel,
    configuration: Mapping[str, object] | None,
):
    session = _session(
        series_keys=forecast_panel.series_keys,
        conformal_config=configuration,
    )
    engine, states, sink = _engine(
        forecast_panel=forecast_panel,
        actuals_panel=actuals_panel,
        session=session,
    )
    spine = Spine(engine)
    spine.run_origin(
        OriginRequest(
            session=session,
            origin=pd.Timestamp("2026-01-02"),
            scope=Scope.GLOBAL,
        )
    )
    first = spine.run_origin(
        OriginRequest(
            session=session,
            origin=pd.Timestamp("2026-01-03"),
            scope=Scope.GLOBAL,
        )
    )
    second = spine.run_origin(
        OriginRequest(
            session=session,
            origin=pd.Timestamp("2026-01-04"),
            scope=Scope.GLOBAL,
        )
    )
    return first, second, engine, states, sink, session


def test_identity_calibration_still_records_resolves_and_drains() -> None:
    panel = _panel()
    _first, _second, engine, states, sink, session = _run_two_origins(
        forecast_panel=panel,
        actuals_panel=panel,
        configuration=None,
    )

    final = engine.observe(pd.Timestamp("2026-01-05"), session=session)
    engine.commit(
        OriginCommit(
            session=session,
            origin=pd.Timestamp("2026-01-05"),
            observe_cycle=final.cycle,
            state_updates=final.cycle.state_updates,
        )
    )

    assert len(sink.observed_history) == 4
    assert [row.actual_value for row in sink.forecasts] == [2.0, 3.0, 4.0]
    assert sink.pending_observations == ()
    assert sink.observe_annotations == ()
    assert states.snapshot(session) == {}


def test_real_split_runtime_escapes_cold_start_at_declared_boundary() -> None:
    configuration = {
        "method": "split-per-step",
        "coverage": 0.5,
        "calibration_window": 20,
    }
    panel = _panel()
    first, second, _engine_value, states, sink, session = _run_two_origins(
        forecast_panel=panel,
        actuals_panel=panel,
        configuration=configuration,
    )
    _lower, upper = interval_columns(0.5)

    assert math.isnan(float(first.forecasts.frame[upper].iloc[0]))
    assert float(second.forecasts.frame[upper].iloc[0]) == 4.0
    assert first.forecasts.observation_issuances
    assert second.forecasts.observation_issuances
    assert len(sink.observe_annotations) == 2
    assert all(value.advanced_delivered_score for value in sink.observe_annotations)
    assert set(states.snapshot(session)) == {
        METHOD_SCOPE_LABEL,
        derive_partition_label(
            "observe-fixture",
            "global",
            EmissionScope.PER_STEP,
        ),
    }


def test_real_weighted_runtime_escapes_cold_start_at_declared_boundary() -> None:
    configuration = {
        "method": "weighted-per-step",
        "coverage": 0.5,
        "calibration_window": 20,
    }
    panel = _panel()
    first, second, _engine_value, states, sink, session = _run_two_origins(
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
    snapshot = states.snapshot(session)
    payload = JsonStateCodec("weighted-per-step", 1).decode(
        snapshot[partition],
        expected_label=partition,
    )

    assert math.isnan(float(first.forecasts.frame[upper].iloc[0]))
    assert float(second.forecasts.frame[upper].iloc[0]) == 4.0
    assert first.forecasts.observation_issuances
    assert second.forecasts.observation_issuances
    assert len(sink.observe_annotations) == 2
    assert all(value.advanced_delivered_score for value in sink.observe_annotations)
    assert isinstance(payload, dict)
    assert payload["delivered_score_count"] == 2
    assert set(snapshot) == {METHOD_SCOPE_LABEL, partition}
    assert all(isinstance(value, bytes) for value in snapshot.values())


def test_observe_before_issue_changes_only_the_perturbed_next_bound() -> None:
    configuration = {
        "method": "split-per-step",
        "coverage": 0.5,
        "calibration_window": 20,
    }
    forecast_panel = _panel()
    baseline = _run_two_origins(
        forecast_panel=forecast_panel,
        actuals_panel=_panel(),
        configuration=configuration,
    )[1]
    control = _run_two_origins(
        forecast_panel=forecast_panel,
        actuals_panel=_panel(),
        configuration=configuration,
    )[1]
    perturbed = _run_two_origins(
        forecast_panel=forecast_panel,
        actuals_panel=_panel(
            overrides={("a", pd.Timestamp("2026-01-03")): 8.0},
        ),
        configuration=configuration,
    )[1]
    _lower, upper = interval_columns(0.5)

    baseline_bound = float(baseline.forecasts.frame[upper].iloc[0])
    assert float(control.forecasts.frame[upper].iloc[0]) == baseline_bound
    assert float(perturbed.forecasts.frame[upper].iloc[0]) != baseline_bound


def test_one_origin_uses_one_snapshot_and_preserves_untouched_partition_state() -> None:
    configuration = {
        "method": "split-per-step",
        "coverage": 0.5,
        "calibration_window": 20,
    }
    panel = _panel(series_keys=("a", "b"))
    session = _session(series_keys=panel.series_keys, conformal_config=configuration)
    states = InMemoryCalibrationStateStore()
    foreign_label = derive_partition_label(
        "observe-fixture",
        "foreign",
        EmissionScope.PER_STEP,
    )
    foreign_state = resolve_method(configuration).calibrate({foreign_label: [9.0]})[foreign_label]
    states.save(
        session,
        foreign_label,
        foreign_state,
        origin=pd.Timestamp("2026-01-01"),
    )
    engine, _states, _sink = _engine(
        forecast_panel=panel,
        actuals_panel=panel,
        session=session,
        states=states,
    )
    result = Spine(engine).run_origin(
        OriginRequest(
            session=session,
            origin=pd.Timestamp("2026-01-03"),
            scope=Scope.GLOBAL,
        )
    )

    references = {
        facts.state_reference for facts in result.forecasts.observation_issuances.values()
    }
    assert len(references) == 1
    assert states.snapshot(session)[foreign_label] == foreign_state
    assert METHOD_SCOPE_LABEL in states.snapshot(session)


def test_cross_engine_staged_observation_is_rejected() -> None:
    panel = _panel()
    session = _session()
    first, _states, _sink = _engine(
        forecast_panel=panel,
        actuals_panel=panel,
        session=session,
    )
    second, _other_states, _other_sink = _engine(
        forecast_panel=panel,
        actuals_panel=panel,
        session=session,
    )
    origin = pd.Timestamp("2026-01-03")
    staged = first.observe(origin, session=session)
    request = OriginRequest(session=session, origin=origin, scope=Scope.GLOBAL)
    forecasts = second.predict(second.fit(request))

    with pytest.raises(EngineError, match="not produced by this engine"):
        second.calibrate(forecasts, session=session, observation=staged)


def test_removed_callback_and_partition_surfaces_are_structurally_absent() -> None:
    forbidden = (
        "Calibrator",
        "Observer",
        "observer=",
        "calibrator=",
        "calibration_partitions",
    )
    witness = "Calibrator Observer observer= calibrator= calibration_partitions"
    assert all(value in witness for value in forbidden)

    source_root = Path(__file__).parents[2] / "src" / "newcalibre"
    violations = {
        str(path.relative_to(source_root)): value
        for path in source_root.rglob("*.py")
        for value in forbidden
        if value in path.read_text()
    }
    assert violations == {}


def test_weighted_method_addition_has_no_engine_or_observe_branch() -> None:
    identifiers = ("weighted-per-step", "WeightedPerStep", "WeightedConformal")
    witness = "weighted-per-step WeightedPerStep WeightedConformal"
    assert all(value in witness for value in identifiers)

    source_root = Path(__file__).parents[2] / "src" / "newcalibre"
    violations = {
        str(path.relative_to(source_root)): value
        for package in ("engine", "observe")
        for path in (source_root / package).rglob("*.py")
        for value in identifiers
        if value in path.read_text()
    }
    assert violations == {}


def test_legacy_observe_construction_paths_refuse_before_panel_io() -> None:
    panel = _panel()
    session = _session()

    class _RecordingPanelSource(InMemoryPanelSource):
        def __init__(self, value: Panel) -> None:
            super().__init__(value)
            self.loads = 0

        def load(self) -> Panel:
            self.loads += 1
            return super().load()

    source = _RecordingPanelSource(panel)
    kwargs = {
        "panel_source": source,
        "actuals_source": InMemoryActualsSource(
            panel,
            actuals_semantics=ActualsSemantics.DEMAND,
        ),
        "artifact_store": InMemoryArtifactStore(),
        "calibration_state_store": InMemoryCalibrationStateStore(),
        "ledger_sink": InMemoryLedgerSink(session=session, calendar=_CALENDAR),
        "dispatch_backend": InProcessDispatch(),
        "hierarchy": HierarchyIndex.flat(panel.series_keys),
    }

    with pytest.raises(TypeError, match="unexpected keyword argument 'observer'"):
        Engine(**kwargs, observer=lambda *_args: None)  # type: ignore[call-arg]

    assert source.loads == 0
    signature = inspect.signature(Engine)
    assert "observer" not in signature.parameters
    assert "calibrator" not in signature.parameters
