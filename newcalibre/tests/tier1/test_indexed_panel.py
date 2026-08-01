"""Exercise opaque indexed history, deterministic task identity, and cycle tokens."""

from __future__ import annotations

import pandas as pd
import pytest

from newcalibre.domain import (
    OBSERVED_VALUE,
    SERIES_KEY,
    TIMESTAMP,
    Calendar,
    CycleToken,
    HistoryCursor,
    HistoryError,
    Panel,
    Scope,
    SessionIdentity,
    TargetSupport,
)
from newcalibre.engine import IndexedPanel, IndexedPanelError

pytestmark = pytest.mark.tier1


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            SERIES_KEY: pd.Series(
                ["sku-b", "sku-a", "sku-a", "sku-b", "sku-b"],
                dtype="string",
            ),
            TIMESTAMP: pd.to_datetime(
                ["2026-01-01", "2026-01-01", "2026-01-03", "2026-01-02", "2026-01-03"]
            ),
            OBSERVED_VALUE: pd.Series([10.0, 1.0, 3.0, 20.0, 30.0]),
        }
    )


def _indexed(frame: pd.DataFrame | None = None) -> IndexedPanel:
    panel = Panel.from_frame(
        _frame() if frame is None else frame,
        calendar=Calendar("D"),
        target_support=TargetSupport.NONNEGATIVE,
    )
    return IndexedPanel.from_panel(panel)


def _tasks(
    panel: IndexedPanel,
    origin: str,
    *,
    scope: Scope = Scope.GLOBAL,
    chunk_size: int | None = None,
    previous: dict[tuple[str, ...], HistoryCursor] | None = None,
):
    return panel.tasks(
        origin=pd.Timestamp(origin),
        horizon=2,
        scope=scope,
        series_chunk_size=chunk_size,
        model_config={"backend": "seasonal-naive", "m": 2},
        previous_cursors=previous,
    )


def test_views_share_staged_storage_and_materialize_isolated_history() -> None:
    panel = _indexed()
    first = _tasks(panel, "2026-01-03")[0]
    second = _tasks(panel, "2026-01-04")[0]

    assert first.history._storage is second.history._storage
    assert first.history.materialize()[TIMESTAMP].lt(first.origin).all()
    materialized = first.history.materialize()
    materialized.loc[:, OBSERVED_VALUE] = -1.0
    assert (first.history.materialize()[OBSERVED_VALUE] >= 0.0).all()


def test_delta_contains_exactly_newly_admissible_sparse_rows() -> None:
    panel = _indexed()
    first = _tasks(panel, "2026-01-03")[0]
    second = _tasks(
        panel,
        "2026-01-04",
        previous={first.series_keys: first.cursor},
    )[0]

    assert first.delta.materialize()[TIMESTAMP].tolist() == [
        pd.Timestamp("2026-01-01"),
        pd.Timestamp("2026-01-01"),
        pd.Timestamp("2026-01-02"),
    ]
    assert second.delta.materialize()[TIMESTAMP].tolist() == [
        pd.Timestamp("2026-01-03"),
        pd.Timestamp("2026-01-03"),
    ]
    assert second.delta.start_cursor == first.cursor
    assert second.cursor.time_bound == first.cursor.time_bound + 1


def test_local_chunks_are_canonical_and_identity_ignores_source_row_order() -> None:
    first_panel = _indexed()
    second_panel = _indexed(_frame().iloc[[1, 0, 3, 2, 4]].reset_index(drop=True))

    first = _tasks(first_panel, "2026-01-04", scope=Scope.LOCAL, chunk_size=1)
    second = _tasks(second_panel, "2026-01-04", scope=Scope.LOCAL, chunk_size=2)

    assert [task.series_keys for task in first] == [("sku-a",), ("sku-b",)]
    assert [task.identity for task in first] == [task.identity for task in second]
    for first_task, second_task in zip(first, second, strict=True):
        pd.testing.assert_frame_equal(
            first_task.history.materialize(),
            second_task.history.materialize(),
        )


def test_identity_changes_with_cursor_config_and_scope() -> None:
    panel = _indexed()
    global_task = _tasks(panel, "2026-01-03")[0]
    later_task = _tasks(panel, "2026-01-04")[0]
    local_task = _tasks(panel, "2026-01-03", scope=Scope.LOCAL, chunk_size=2)[0]
    configured = panel.tasks(
        origin=pd.Timestamp("2026-01-03"),
        horizon=2,
        scope=Scope.GLOBAL,
        model_config={"backend": "seasonal-naive", "m": 1},
    )[0]

    assert (
        len({global_task.identity, later_task.identity, local_task.identity, configured.identity})
        == 4
    )


def test_stale_and_foreign_cursors_fail_closed() -> None:
    panel = _indexed()
    other = _indexed(_frame().assign(value=lambda value: value[OBSERVED_VALUE] + 1))
    current = _tasks(panel, "2026-01-03")[0]
    stale = HistoryCursor(
        panel.identity,
        current.cursor.series_start,
        current.cursor.series_stop,
        current.cursor.time_bound + 1,
    )
    foreign = HistoryCursor(
        other.identity,
        current.cursor.series_start,
        current.cursor.series_stop,
        current.cursor.time_bound,
    )

    with pytest.raises(IndexedPanelError, match="newer"):
        _tasks(panel, "2026-01-03", previous={current.series_keys: stale})
    with pytest.raises(IndexedPanelError, match="another staged panel"):
        _tasks(panel, "2026-01-04", previous={current.series_keys: foreign})


def test_history_cursor_and_cycle_token_reject_invalid_identity() -> None:
    with pytest.raises(HistoryError, match="SHA-256"):
        HistoryCursor("bad", 0, 1, 0)
    with pytest.raises(HistoryError, match="non-empty"):
        HistoryCursor("0" * 64, 1, 1, 0)

    session = SessionIdentity.derive(
        tenant="test",
        series_keys=("sku-a", "sku-b"),
        calendar=Calendar("D"),
        horizon=2,
        model_config={"backend": "seasonal-naive", "m": 2},
    )
    token = CycleToken(session, pd.Timestamp("2026-01-03"), 1)
    assert token.revision == 1
    with pytest.raises(HistoryError, match="positive"):
        CycleToken(session, pd.Timestamp("2026-01-03"), 0)
