"""Exercise checkpoint restoration through the transactional run store."""

from __future__ import annotations

import pytest
from tests.tier2 import driver_scenarios as scenarios
from tests.tier2.durable_state import project_durable_state

pytestmark = pytest.mark.tier2


@pytest.mark.parametrize("runtime_name", scenarios.RUNTIME_CASES)
def test_reconstructed_driver_restores_incremental_checkpoint_lineage(
    runtime_name: str | None,
) -> None:
    """Continue a prefix from its indexed checkpoint and match an uninterrupted run."""
    expected = scenarios.run_event_world(runtime_name)
    resumed = scenarios.make_world(runtime_name)
    first = scenarios.build_event_driver(resumed)
    scenarios.seed_event_history(resumed, first)
    scenarios.drive_origins(resumed, first, origins=scenarios.ORIGINS[:4])

    prefix_indexes = dict(resumed.store.checkpoint_indexes)
    prefix_checkpoints = dict(resumed.store.checkpoints)
    second = scenarios.build_event_driver(resumed)
    scenarios.drive_origins(resumed, second, origins=scenarios.ORIGINS[4:])

    assert prefix_indexes
    assert prefix_checkpoints
    assert len(resumed.store.checkpoint_indexes) == 1
    assert len(resumed.store.checkpoints) == len(scenarios.ORIGINS)
    assert set(prefix_checkpoints).issubset(resumed.store.checkpoints)
    assert dict(resumed.store.checkpoint_indexes) != prefix_indexes
    assert project_durable_state(resumed.store) == project_durable_state(expected.store)


def test_checkpoint_snapshots_are_defensive_and_revision_bound() -> None:
    """Keep published checkpoint bytes isolated from caller-owned mappings."""
    world = scenarios.run_event_world(None)
    checkpoint_snapshot = world.store.checkpoints
    index_snapshot = world.store.checkpoint_indexes
    latest_revision = world.store.revision

    assert checkpoint_snapshot == world.store.checkpoints
    assert index_snapshot == world.store.checkpoint_indexes
    assert world.store.revision == latest_revision
    with pytest.raises(TypeError):
        checkpoint_snapshot["changed"] = b"nope"  # type: ignore[index]
    with pytest.raises(TypeError):
        index_snapshot["changed"] = b"nope"  # type: ignore[index]
