"""Exercise the engine-owned checkpointed forecasting lifecycle."""

from __future__ import annotations

import pandas as pd
import pytest

from newcalibre.domain import (
    OBSERVED_VALUE,
    SERIES_KEY,
    TIMESTAMP,
    Calendar,
    CycleToken,
    Panel,
    Scope,
    SessionIdentity,
    TargetSupport,
)
from newcalibre.engine import (
    ForecastLifecycle,
    ForecastLifecycleError,
    IndexedPanel,
    InMemoryArtifactStore,
)
from newcalibre.forecasting import resolve_adapter

pytestmark = pytest.mark.tier1


def _world():
    history = pd.DataFrame(
        {
            SERIES_KEY: pd.Series(["sku"] * 16, dtype="string"),
            TIMESTAMP: pd.date_range("2026-01-01", periods=16, freq="D"),
            OBSERVED_VALUE: pd.Series(range(1, 17), dtype="float64"),
        }
    )
    panel = IndexedPanel.from_panel(
        Panel.from_frame(
            history,
            calendar=Calendar("D"),
            target_support=TargetSupport.NONNEGATIVE,
        )
    )
    config = {"backend": "seasonal-naive", "m": 7}
    session = SessionIdentity.derive(
        tenant="test",
        series_keys=panel.series_keys,
        calendar=panel.calendar,
        horizon=2,
        model_config=config,
    )
    return panel, config, session


def _task(panel: IndexedPanel, config: dict[str, object], origin: str):
    return panel.tasks(
        origin=pd.Timestamp(origin),
        horizon=2,
        scope=Scope.GLOBAL,
        model_config=config,
    )[0]


def test_first_fit_exact_load_and_incremental_resume_publish_deterministic_checkpoints() -> None:
    panel, config, session = _world()
    store = InMemoryArtifactStore()
    first = _task(panel, config, "2026-01-15")
    first_token = CycleToken(session, first.origin, 1)
    lifecycle = ForecastLifecycle(artifact_store=store, adapter_resolver=resolve_adapter)
    lifecycle.prepare(session=session, task=first, token=first_token)
    first_frame = lifecycle.predict(session=session, task=first, token=first_token)
    first_artifacts = dict(store.artifacts)

    exact = ForecastLifecycle(artifact_store=store, adapter_resolver=resolve_adapter)
    exact.prepare(session=session, task=first, token=CycleToken(session, first.origin, 2))
    exact_frame = exact.predict(
        session=session,
        task=first,
        token=CycleToken(session, first.origin, 2),
    )

    second = _task(panel, config, "2026-01-16")
    resumed = ForecastLifecycle(artifact_store=store, adapter_resolver=resolve_adapter)
    second_token = CycleToken(session, second.origin, 3)
    resumed.prepare(session=session, task=second, token=second_token)
    second_frame = resumed.predict(session=session, task=second, token=second_token)

    pd.testing.assert_frame_equal(first_frame, exact_frame)
    assert dict(store.artifacts).items() >= first_artifacts.items()
    assert len(store.artifacts) == 2
    assert len(store.artifact_indexes) == 1
    assert second_frame["point_forecast"].tolist() == [9.0, 10.0]


def test_incremental_resume_loads_one_index_and_one_prior_checkpoint() -> None:
    class CountingStore(InMemoryArtifactStore):
        def __init__(self) -> None:
            super().__init__()
            self.artifact_loads = 0
            self.index_loads = 0

        def load(self, key: str) -> bytes | None:
            self.artifact_loads += 1
            return super().load(key)

        def load_index(self, key: str) -> bytes | None:
            self.index_loads += 1
            return super().load_index(key)

    panel, config, session = _world()
    store = CountingStore()
    first = _task(panel, config, "2026-01-08")
    first_token = CycleToken(session, first.origin, 1)
    lifecycle = ForecastLifecycle(artifact_store=store, adapter_resolver=resolve_adapter)
    lifecycle.prepare(session=session, task=first, token=first_token)
    lifecycle.predict(session=session, task=first, token=first_token)

    store.artifact_loads = 0
    store.index_loads = 0
    later = _task(panel, config, "2026-01-16")
    later_token = CycleToken(session, later.origin, 2)
    resumed = ForecastLifecycle(artifact_store=store, adapter_resolver=resolve_adapter)
    resumed.prepare(session=session, task=later, token=later_token)

    assert store.index_loads == 1
    assert store.artifact_loads == 2  # exact miss plus the indexed predecessor


def test_malformed_exact_checkpoint_fails_closed() -> None:
    panel, config, session = _world()
    store = InMemoryArtifactStore()
    task = _task(panel, config, "2026-01-15")
    token = CycleToken(session, task.origin, 1)
    lifecycle = ForecastLifecycle(artifact_store=store, adapter_resolver=resolve_adapter)
    lifecycle.prepare(session=session, task=task, token=token)
    lifecycle.predict(session=session, task=task, token=token)
    key = next(iter(store._artifacts))
    store._artifacts[key] = b"malformed"

    retry = ForecastLifecycle(artifact_store=store, adapter_resolver=resolve_adapter)
    with pytest.raises(ForecastLifecycleError, match="malformed"):
        retry.prepare(
            session=session,
            task=task,
            token=CycleToken(session, task.origin, 2),
        )


def test_predict_requires_the_exact_prepared_cycle() -> None:
    panel, config, session = _world()
    task = _task(panel, config, "2026-01-15")
    lifecycle = ForecastLifecycle(
        artifact_store=InMemoryArtifactStore(),
        adapter_resolver=resolve_adapter,
    )
    prepared = CycleToken(session, task.origin, 1)
    lifecycle.prepare(session=session, task=task, token=prepared)

    with pytest.raises(ForecastLifecycleError, match="prepared"):
        lifecycle.predict(
            session=session,
            task=task,
            token=CycleToken(session, task.origin, 2),
        )
