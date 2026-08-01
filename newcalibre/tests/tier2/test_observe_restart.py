"""Exercise late-actual observation across transactional restart."""

from __future__ import annotations

import pytest
from tests.tier2.driver_scenarios import (
    ORIGINS,
    RUNTIME_CASES,
    SERIES_KEYS,
    actual_records,
    actuals_event,
    build_event_driver,
    drive_origins,
    make_world,
    origin_event,
    seed_event_history,
    submit_actuals,
)
from tests.tier2.durable_state import DurableState, project_durable_state

pytestmark = pytest.mark.tier2


@pytest.mark.parametrize("runtime_name", RUNTIME_CASES)
def test_late_actual_resolves_once_after_driver_reconstruction(
    runtime_name: str | None,
) -> None:
    """Compare uninterrupted and reconstructed delivery of one held bottom actual."""
    uninterrupted = _run_late_schedule(runtime_name, reconstruct_before_late=False)
    reconstructed = _run_late_schedule(runtime_name, reconstruct_before_late=True)

    assert reconstructed == uninterrupted


def _run_late_schedule(
    runtime_name: str | None,
    *,
    reconstruct_before_late: bool,
) -> DurableState:
    world = make_world(runtime_name, source_actuals=False)
    driver = build_event_driver(world)
    seed_event_history(world, driver)
    drive_origins(world, driver, origins=ORIGINS[:2])

    late_period = ORIGINS[2]
    driver.handle(origin_event(world, late_period))
    day_records = actual_records(world, timestamps=(late_period,))
    held = next(record for record in day_records if record.series_key == SERIES_KEYS[0])
    submit_actuals(world, driver, tuple(record for record in day_records if record != held))
    drive_origins(world, driver, origins=ORIGINS[3:5])

    retained = tuple(
        row
        for row in world.store.pending_observations
        if row.forecast_key.series_key == held.series_key and row.target_timestamp == held.timestamp
    )
    assert retained
    assert all(row.resolution is None for row in retained)
    immutable_segment = tuple(row for row in world.store.forecasts if row.origin == ORIGINS[3])

    if reconstruct_before_late:
        driver = build_event_driver(world)
    event = actuals_event(world, (held,))
    annotations_before = len(world.store.observe_annotations)
    driver.handle(event)
    pending_keys = {row.forecast_key for row in world.store.pending_observations}
    assert all(row.forecast_key not in pending_keys for row in retained)
    assert (
        tuple(row for row in world.store.forecasts if row.origin == ORIGINS[3]) == immutable_segment
    )
    annotations_after = len(world.store.observe_annotations)
    revision = world.store.revision
    driver.handle(event)
    assert len(world.store.observe_annotations) == annotations_after
    assert world.store.revision == revision
    if runtime_name is not None:
        assert annotations_after > annotations_before

    drive_origins(world, driver, origins=ORIGINS[5:])
    return project_durable_state(world.store)
