"""Prove incremental checkpoint resume and public cycle provenance."""

from __future__ import annotations

import copy
import inspect
from dataclasses import fields, replace
from pathlib import Path

import driver_scenarios as scenarios
import pandas as pd
import pytest
from durable_state import project_durable_state

from newcalibre.domain import (
    CycleToken,
    ForecastTask,
    HistoryDelta,
    HistoryView,
    Scope,
)
from newcalibre.engine import (
    CommitRequest,
    EngineError,
    InMemoryActualsSource,
    OrderRequest,
    OriginRequest,
    SettlementRequest,
)
from newcalibre.forecasting import ForecastAdapter


def test_fresh_engine_resume_matches_uninterrupted_incremental_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_name = "split-per-step"
    expected = scenarios.run_event_world(runtime_name)
    expected_state = project_durable_state(
        expected.sink,
        expected.states,
        expected.artifacts,
    )
    events: list[str] = []

    class RecordingAdapter(scenarios.DeterministicArtifactAdapter):
        def fit(self, task: ForecastTask) -> None:
            events.append("fit")
            super().fit(task)

        def load_state(self, state: bytes) -> None:
            events.append("load")
            super().load_state(state)

        def update(self, delta: HistoryDelta) -> None:
            events.append("update")
            assert not delta.materialize().empty
            super().update(delta)

    monkeypatch.setattr(scenarios, "DeterministicArtifactAdapter", RecordingAdapter)
    resumed = scenarios.make_world(runtime_name)
    first_engine = scenarios.build_event_driver(resumed)
    scenarios.seed_event_history(resumed, first_engine)
    scenarios.drive_origins(resumed, first_engine, origins=scenarios.ORIGINS[:3])
    prefix_checkpoints = dict(resumed.artifacts.artifacts)

    fresh_engine = scenarios.build_event_driver(resumed)
    scenarios.drive_origins(resumed, fresh_engine, origins=scenarios.ORIGINS[3:])
    resumed_state = project_durable_state(
        resumed.sink,
        resumed.states,
        resumed.artifacts,
    )

    assert events.count("fit") == 1
    assert events.count("load") == len(scenarios.ORIGINS) - 1
    assert events.count("update") == len(scenarios.ORIGINS) - 1
    assert len(prefix_checkpoints) == 3
    assert len(resumed.artifacts.artifacts) == len(scenarios.ORIGINS)
    assert resumed.sink.forecasts == expected.sink.forecasts
    assert resumed.sink.orders == expected.sink.orders
    assert resumed.states.snapshot(resumed.session) == expected.states.snapshot(expected.session)
    assert resumed.artifacts.artifacts == expected.artifacts.artifacts
    assert resumed_state == expected_state


def test_engine_rejects_mutated_cycle_components_at_every_composition_seam() -> None:
    world = scenarios.make_world(None)
    actuals = InMemoryActualsSource(
        world.panel,
        actuals_semantics=scenarios.ActualsSemantics.DEMAND,
    )
    engine = scenarios.build_engine(world, actuals=actuals)
    origin = scenarios.ORIGINS[0]
    request = OriginRequest(
        session=world.session,
        origin=origin,
        scope=Scope.GLOBAL,
        inventory_positions=scenarios.INITIAL_INVENTORY,
    )
    observation = engine.observe(origin, session=world.session)
    fitted = engine.fit(request)
    assert observation.token is not None
    token = observation.token
    for forged in (
        CycleToken(scenarios.make_session("split-per-step"), origin, token.revision),
        CycleToken(world.session, scenarios.CALENDAR.advance(origin, 1), token.revision),
        CycleToken(world.session, origin, token.revision + 1),
    ):
        forged_fitted = copy.copy(fitted[0])
        object.__setattr__(forged_fitted, "token", forged)
        with pytest.raises(EngineError, match="session does not match|stale or foreign"):
            engine.predict((forged_fitted, *fitted[1:]))

    predicted = engine.predict(fitted)
    forged_token = CycleToken(world.session, origin, token.revision + 1)
    forged_forecasts = copy.copy(predicted)
    object.__setattr__(forged_forecasts, "_token", forged_token)
    with pytest.raises(EngineError, match="stale or foreign"):
        engine.reconcile(forged_forecasts)

    reconciled = engine.reconcile(predicted)
    forged_observation = replace(observation, token=forged_token)
    with pytest.raises(EngineError, match="share one cycle token"):
        engine.calibrate(
            reconciled,
            session=world.session,
            observation=forged_observation,
        )

    calibrated = engine.calibrate(
        reconciled,
        session=world.session,
        observation=observation,
    )
    forged_order_forecasts = copy.copy(calibrated.forecasts)
    object.__setattr__(forged_order_forecasts, "_token", forged_token)
    with pytest.raises(EngineError, match="stale or foreign"):
        engine.order(
            OrderRequest(
                session=world.session,
                origin=origin,
                forecasts=forged_order_forecasts,
                inventory_positions=scenarios.INITIAL_INVENTORY,
            )
        )

    decisions = engine.order(
        OrderRequest(
            session=world.session,
            origin=origin,
            forecasts=calibrated.forecasts,
            inventory_positions=scenarios.INITIAL_INVENTORY,
        )
    )
    assert decisions is not None
    snapshot = world.sink.settlement_snapshot((origin,))
    with pytest.raises(EngineError, match="stale or foreign"):
        engine.settle(
            SettlementRequest(
                session=world.session,
                snapshot=snapshot,
                actuals={
                    record.key: record.recorded_value
                    for record in scenarios.actual_records(world, timestamps=(origin,))
                },
                inventory_positions=scenarios.INITIAL_INVENTORY,
                actuals_semantics=scenarios.ActualsSemantics.DEMAND,
                token=forged_token,
            )
        )

    valid_commit = CommitRequest(
        session=world.session,
        origin=origin,
        token=token,
        observation=observation,
        calibration=calibrated,
        inventory_positions=scenarios.INITIAL_INVENTORY,
        decisions=decisions,
    )
    forged_commit = copy.copy(valid_commit)
    object.__setattr__(forged_commit, "token", forged_token)
    with pytest.raises(EngineError, match="stale or foreign"):
        engine.commit(forged_commit)


def test_removed_transport_and_adapter_surfaces_cannot_reappear() -> None:
    source_root = Path(__file__).parents[2] / "src" / "newcalibre"
    source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted(source_root.rglob("*.py"), key=lambda value: str(value).encode())
    )
    for removed in (
        "forecast_tasks",
        "_fit_one",
        "_predict_one",
        "_phase_token",
        "collect_fitted_values",
    ):
        assert removed not in source

    task_fields = {value.name: value.type for value in fields(ForecastTask)}
    assert task_fields["_history"] in (HistoryView, "HistoryView")
    assert all(
        annotation not in (pd.DataFrame, "pd.DataFrame") for annotation in task_fields.values()
    )
    expected = {
        "fit": ("self", "task"),
        "update": ("self", "delta"),
        "predict": ("self", "task"),
        "fitted_values": ("self",),
        "dump_state": ("self",),
        "load_state": ("self", "state"),
    }
    for method_name, parameter_names in expected.items():
        method = getattr(ForecastAdapter, method_name)
        assert tuple(inspect.signature(method).parameters) == parameter_names
