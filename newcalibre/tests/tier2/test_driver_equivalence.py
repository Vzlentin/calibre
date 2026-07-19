"""Prove canonical durable-state equivalence across both engine drivers."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import fields, replace

import pandas as pd
import pytest
from driver_scenarios import (
    CALENDAR,
    EXPECTED_BOOKED_COSTS,
    EXPECTED_FINAL_INVENTORY,
    EXPECTED_LATE_BOOKED_COSTS,
    EXPECTED_LATE_FINAL_INVENTORY,
    INITIAL_INVENTORY,
    ORIGINS,
    RUNTIME_CASES,
    RUNTIME_CONFIGURATIONS,
    SERIES_KEYS,
    DriverWorld,
    actual_records,
    actuals_event,
    build_event_driver,
    build_time_loop,
    drive_origins,
    make_world,
    origin_event,
    run_event_world,
    run_time_world,
    runtime_witness,
    seed_event_history,
    submit_actuals,
)
from durable_state import DurableState, project_durable_state

from newcalibre.conformal import available_methods
from newcalibre.domain import InventoryPosition, Scope
from newcalibre.engine import (
    CommitReceipt,
    EventDriverError,
    InMemoryLedgerSink,
    OriginCommit,
    OriginEvent,
    PhaseError,
)


class _InterruptingSink(InMemoryLedgerSink):
    """Fail once immediately before or after a selected journal publication."""

    def __init__(
        self,
        *,
        session,
        fail_after_journal: bool,
        selected: Callable[[OriginCommit], bool],
    ) -> None:
        super().__init__(session=session, calendar=CALENDAR)
        self._fail_after_journal = fail_after_journal
        self._selected = selected
        self._failed = False

    def commit(self, write: OriginCommit) -> CommitReceipt:
        """Interrupt one selected commit at the configured journal boundary."""
        if self._failed or not self._selected(write):
            return super().commit(write)
        self._failed = True
        if not self._fail_after_journal:
            raise RuntimeError("failure before equivalence journal")
        super().commit(write)
        raise RuntimeError("failure after equivalence journal")


class _ReorderedSnapshots:
    """Expose one sink's public row snapshots in reverse physical order."""

    def __init__(self, sink: InMemoryLedgerSink) -> None:
        self._sink = sink

    def __getattr__(self, name: str):
        return getattr(self._sink, name)

    @property
    def forecasts(self):
        return tuple(reversed(self._sink.forecasts))

    @property
    def orders(self):
        return tuple(reversed(self._sink.orders))

    @property
    def settlements(self):
        return tuple(reversed(self._sink.settlements))

    @property
    def observed_history(self):
        return tuple(reversed(self._sink.observed_history))

    @property
    def pending_observations(self):
        return tuple(reversed(self._sink.pending_observations))

    @property
    def observation_resolutions(self):
        return tuple(reversed(self._sink.observation_resolutions))

    @property
    def observe_annotations(self):
        return tuple(reversed(self._sink.observe_annotations))


def _state(world: DriverWorld) -> DurableState:
    return project_durable_state(world.sink, world.states, world.artifacts)


def _assert_complete(
    world: DriverWorld,
    *,
    expected_inventory=EXPECTED_FINAL_INVENTORY,
    expected_costs=EXPECTED_BOOKED_COSTS,
) -> None:
    assert len(world.sink.forecasts) == 48
    assert len(world.sink.orders) == 8
    assert len(world.sink.settlements) == 16
    assert len(world.sink.observed_history) == 24
    assert len(world.artifacts.artifacts) == len(ORIGINS)
    assert world.sink.pending_observations
    assert len({row.key for row in world.sink.settlements}) == len(world.sink.settlements)
    probe = CALENDAR.advance(max(row.period for row in world.sink.settlements), 1)
    snapshot = world.sink.settlement_snapshot((probe,))
    assert snapshot.current_positions == expected_inventory
    assert snapshot.open_order_quantities == {series_key: 0.0 for series_key in SERIES_KEYS}
    assert all(
        isinstance(value, InventoryPosition) for value in snapshot.current_positions.values()
    )
    holding = math.fsum(row.holding.amount for row in world.sink.settlements)
    shortage = math.fsum(row.shortage.amount for row in world.sink.settlements)
    total = math.fsum(row.realized_cost for row in world.sink.settlements)
    assert (holding, shortage, total) == expected_costs
    costs = _state(world).booked_costs
    assert costs[0][0] == "holding"
    assert costs[1][0] == "shortage"
    assert costs[2][0] == "total"


def _mutate_family(state: DurableState, field_name: str) -> DurableState:
    value = getattr(state, field_name)
    assert isinstance(value, tuple)
    return replace(state, **{field_name: (*value, ("mutated", field_name))})


def test_durable_state_ignores_physical_order_and_transaction_grouping() -> None:
    time_world = run_time_world(None)
    event_world = run_event_world(None)

    reordered = project_durable_state(
        _ReorderedSnapshots(event_world.sink),
        event_world.states,
        event_world.artifacts,
    )

    assert _state(event_world) == reordered
    assert _state(time_world) == _state(event_world)
    time_receipt = time_world.sink.receipt(ORIGINS[0])
    event_receipt = event_world.sink.receipt(ORIGINS[0])
    assert time_receipt is not None and time_receipt.settlement_periods == (ORIGINS[0],)
    assert event_receipt is not None and event_receipt.settlement_periods == ()


@pytest.mark.parametrize(
    "field_name",
    [field.name for field in fields(DurableState) if field.name not in {"session"}],
)
def test_durable_state_changes_when_an_included_fact_changes(field_name: str) -> None:
    state = _state(run_event_world("split-per-step"))
    assert _mutate_family(state, field_name) != state


def test_runtime_configuration_registry_is_complete() -> None:
    assert tuple(sorted(RUNTIME_CONFIGURATIONS, key=str.encode)) == available_methods()
    assert (None, *available_methods()) == RUNTIME_CASES


@pytest.mark.parametrize("runtime_name", RUNTIME_CASES)
def test_every_runtime_has_nonvacuous_durable_execution(runtime_name: str | None) -> None:
    world = run_time_world(runtime_name)
    witness = runtime_witness(world)
    _assert_complete(world)

    if runtime_name is None:
        assert witness.delivered_scores == ()
        assert witness.state_before == witness.state_after == ()
        assert witness.finite_issuances == ()
        assert all(row.observation_issuance is None for row in world.sink.forecasts)
        assert all(not row.issuances for row in world.sink.forecasts)
    else:
        assert witness.delivered_scores
        assert witness.state_after != witness.state_before
        assert witness.finite_issuances
        assert sum(
            annotation.advanced_delivered_score for annotation in world.sink.observe_annotations
        ) == len(witness.delivered_scores)


@pytest.mark.parametrize("runtime_name", RUNTIME_CASES)
def test_in_order_time_and_event_drivers_match_through_final_drain(
    runtime_name: str | None,
) -> None:
    time_world = run_time_world(runtime_name)
    event_world = run_event_world(runtime_name)

    _assert_complete(time_world)
    _assert_complete(event_world)
    assert _state(event_world) == _state(time_world)


@pytest.mark.parametrize("runtime_name", RUNTIME_CASES)
def test_atomic_record_order_and_sequence_preserving_rechunking_are_equivalent(
    runtime_name: str | None,
) -> None:
    reference = run_time_world(runtime_name)
    permuted = run_event_world(runtime_name, reverse=True)
    rechunk_reference = _run_rechunk_schedule(runtime_name, combined=False)
    rechunked = _run_rechunk_schedule(runtime_name, combined=True)

    assert _state(permuted) == _state(reference)
    assert _state(rechunked) == _state(rechunk_reference)


@pytest.mark.parametrize("runtime_name", RUNTIME_CASES)
def test_genuinely_late_actual_is_pending_then_delivered_once_on_same_schedule_replay(
    runtime_name: str | None,
) -> None:
    first = _run_late_schedule(runtime_name, reconstruct_before_late=False)
    replay = _run_late_schedule(runtime_name, reconstruct_before_late=True)

    _assert_complete(
        first,
        expected_inventory=EXPECTED_LATE_FINAL_INVENTORY,
        expected_costs=EXPECTED_LATE_BOOKED_COSTS,
    )
    assert _state(replay) == _state(first)


@pytest.mark.parametrize("runtime_name", RUNTIME_CASES)
def test_identical_retries_are_idempotent_and_changed_natural_facts_conflict(
    runtime_name: str | None,
) -> None:
    reference = run_event_world(runtime_name)
    world = make_world(runtime_name)
    driver = build_event_driver(world)
    seed_event_history(world, driver)

    origin = origin_event(world, ORIGINS[0], seed=True)
    driver.handle(origin)
    before_origin_retry = _state(world)
    driver.handle(origin)
    assert _state(world) == before_origin_retry
    with pytest.raises(EventDriverError, match="different input facts"):
        driver.handle(
            OriginEvent(
                session=world.session,
                origin=ORIGINS[0],
                scope=Scope.LOCAL,
                initial_inventory_positions=INITIAL_INVENTORY,
            )
        )
    assert _state(world) == before_origin_retry

    records = actual_records(world, timestamps=(ORIGINS[0],))
    event = actuals_event(world, records)
    driver.handle(event)
    before_actual_retry = _state(world)
    driver.handle(event)
    assert _state(world) == before_actual_retry
    changed = (
        replace(records[0], recorded_value=float(records[0].recorded_value) + 1.0),
        *records[1:],
    )
    with pytest.raises(EventDriverError, match="different input facts"):
        driver.handle(actuals_event(world, changed))
    assert _state(world) == before_actual_retry

    drive_origins(world, driver, origins=ORIGINS[1:])
    assert _state(world) == _state(reference)


@pytest.mark.parametrize("runtime_name", RUNTIME_CASES)
def test_journal_failures_repair_and_incomplete_windows_deliver_once(
    runtime_name: str | None,
) -> None:
    expected = run_event_world(runtime_name)
    target = tuple(
        record.key
        for record in actual_records(
            expected,
            timestamps=(ORIGINS[2],),
        )
    )

    for fail_after_journal in (False, True):
        world = make_world(
            runtime_name,
            sink_factory=lambda session, after=fail_after_journal: _InterruptingSink(
                session=session,
                fail_after_journal=after,
                selected=lambda write: write.actual_keys == target,
            ),
        )
        driver = build_event_driver(world)
        seed_event_history(world, driver)
        drive_origins(world, driver, origins=ORIGINS[:2])
        assert any(
            row.forecast_key.origin == ORIGINS[0]
            and row.target_timestamp == ORIGINS[2]
            and row.resolution is None
            for row in world.sink.pending_observations
        )

        driver = build_event_driver(world)
        driver.handle(origin_event(world, ORIGINS[2]))
        event = actuals_event(
            world,
            actual_records(world, timestamps=(ORIGINS[2],)),
        )
        before = _state(world)
        with pytest.raises(RuntimeError, match="failure (before|after) equivalence journal"):
            driver.handle(event)
        if not fail_after_journal:
            assert _state(world) == before
        else:
            assert _state(world) != before

        resumed = build_event_driver(world)
        resumed.handle(event)
        delivered_after_repair = tuple(
            annotation.forecast_key for annotation in world.sink.observe_annotations
        )
        resumed.handle(event)
        assert (
            tuple(annotation.forecast_key for annotation in world.sink.observe_annotations)
            == delivered_after_repair
        )
        drive_origins(world, resumed, origins=ORIGINS[3:])
        assert len(delivered_after_repair) == len(set(delivered_after_repair))
        assert _state(world) == _state(expected)


@pytest.mark.parametrize("runtime_name", RUNTIME_CASES)
@pytest.mark.parametrize("handoff", [ORIGINS[1], ORIGINS[5]], ids=["before-ready", "after-finite"])
def test_time_loop_to_event_continuation_matches_uninterrupted_time_loop(
    runtime_name: str | None,
    handoff: pd.Timestamp,
) -> None:
    expected = run_time_world(runtime_name)
    world = make_world(
        runtime_name,
        sink_factory=lambda session: _InterruptingSink(
            session=session,
            fail_after_journal=False,
            selected=lambda write: write.origin == handoff and bool(write.forecasts),
        ),
    )
    with pytest.raises(PhaseError, match="failure before equivalence journal"):
        build_time_loop(world).run()

    assert world.sink.pending_observations
    snapshot = world.sink.settlement_snapshot((handoff,))
    assert any(quantity > 0.0 for quantity in snapshot.open_order_quantities.values())
    if runtime_name is not None:
        finite = runtime_witness(world).finite_issuances
        assert bool(finite) is (handoff == ORIGINS[5])

    driver = build_event_driver(world)
    previous = ORIGINS[ORIGINS.index(handoff) - 1]
    submit_actuals(world, driver, actual_records(world, timestamps=(previous,)))
    remaining = ORIGINS[ORIGINS.index(handoff) :]
    drive_origins(world, driver, origins=remaining)
    assert _state(world) == _state(expected)


@pytest.mark.parametrize("runtime_name", RUNTIME_CASES)
@pytest.mark.parametrize("handoff", [ORIGINS[1], ORIGINS[5]], ids=["before-ready", "after-finite"])
def test_event_to_time_loop_continuation_matches_uninterrupted_event_driver(
    runtime_name: str | None,
    handoff: pd.Timestamp,
) -> None:
    expected = run_event_world(runtime_name)
    world = make_world(runtime_name)
    driver = build_event_driver(world)
    seed_event_history(world, driver)
    prefix = ORIGINS[: ORIGINS.index(handoff)]
    drive_origins(world, driver, origins=prefix)

    assert world.sink.pending_observations
    snapshot = world.sink.settlement_snapshot((handoff,))
    assert any(quantity > 0.0 for quantity in snapshot.open_order_quantities.values())
    if runtime_name is not None:
        finite = runtime_witness(world).finite_issuances
        assert bool(finite) is (handoff == ORIGINS[5])

    build_time_loop(world).run()
    assert _state(world) == _state(expected)


def _run_rechunk_schedule(
    runtime_name: str | None,
    *,
    combined: bool,
) -> DriverWorld:
    origins = (ORIGINS[0], ORIGINS[3], ORIGINS[4], ORIGINS[5])
    world = make_world(runtime_name)
    if not combined:
        build_time_loop(world, origins=origins, settlement_end=origins[-1]).run()
        return world

    driver = build_event_driver(world)
    seed_event_history(world, driver, reverse=True)
    driver.handle(origin_event(world, origins[0], seed=True))
    submit_actuals(
        world,
        driver,
        actual_records(world, timestamps=(origins[0],)),
        reverse=True,
    )
    between = actual_records(world, timestamps=(ORIGINS[1], ORIGINS[2]))
    submit_actuals(world, driver, between, reverse=True)
    drive_origins(world, driver, origins=origins[1:], reverse=True)
    return world


def _run_late_schedule(
    runtime_name: str | None,
    *,
    reconstruct_before_late: bool,
) -> DriverWorld:
    world = make_world(runtime_name)
    driver = build_event_driver(world)
    seed_event_history(world, driver)
    drive_origins(world, driver, origins=ORIGINS[:2])

    late_period = ORIGINS[2]
    driver.handle(origin_event(world, late_period))
    day_records = actual_records(world, timestamps=(late_period,))
    held = next(record for record in day_records if record.series_key == SERIES_KEYS[0])
    submit_actuals(
        world,
        driver,
        tuple(record for record in day_records if record != held),
    )
    drive_origins(world, driver, origins=ORIGINS[3:5])

    retained = tuple(
        row
        for row in world.sink.pending_observations
        if row.forecast_key.series_key == held.series_key and row.target_timestamp == held.timestamp
    )
    assert retained
    assert all(row.resolution is None for row in retained)
    committed = tuple(row for row in world.sink.forecasts if row.origin == ORIGINS[3])

    if reconstruct_before_late:
        driver = build_event_driver(world)
    event = actuals_event(world, (held,))
    annotations_before = len(world.sink.observe_annotations)
    driver.handle(event)
    pending_keys = {row.forecast_key for row in world.sink.pending_observations}
    assert all(row.forecast_key not in pending_keys for row in retained)
    assert tuple(row for row in world.sink.forecasts if row.origin == ORIGINS[3]) == committed
    annotations_after = len(world.sink.observe_annotations)
    driver.handle(event)
    assert len(world.sink.observe_annotations) == annotations_after
    if runtime_name is not None:
        assert annotations_after > annotations_before

    drive_origins(world, driver, origins=ORIGINS[5:])
    return world
