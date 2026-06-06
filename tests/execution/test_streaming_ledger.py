from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from calibre.conformal import SymmetricIntervalConfig, SymmetricIntervalRuntime
from calibre.core.forecast_frame import (
    DS,
    FORECAST_ORIGIN,
    MODEL_NAME,
    UNIQUE_ID,
    Y_HAT,
    H,
    Y,
)
from calibre.core.forecast_task import ForecastTask
from calibre.core.order_types import NewsvendorPolicyParameters
from calibre.execution.backend import BackendEngine, ConformalOptions, LedgerOutputOptions
from calibre.execution.ledger import (
    InMemoryLedger,
    StreamingLedger,
    StreamingOrderLedger,
    resolved_ledger_uri,
)
from calibre.execution.task_builder import partition_tasks
from calibre.forecasting.adapter_base import ModelAdapter
from calibre.ordering.policy_config import NewsvendorConfig


class _StubAdapter(ModelAdapter):
    def fit(self, task: ForecastTask) -> None:
        self._task = task

    def predict(self, task: ForecastTask) -> pd.DataFrame:
        last = float(task.history[Y].iloc[-1])
        return pd.DataFrame(
            {
                UNIQUE_ID: [task.unique_id] * task.horizon,
                DS: pd.date_range(
                    task.forecast_origin + pd.Timedelta(weeks=1), periods=task.horizon, freq="W"
                ),
                Y_HAT: [last + h for h in range(1, task.horizon + 1)],
                H: list(range(1, task.horizon + 1)),
            }
        )


def _pending_frame(n: int = 2, origin: str = "2024-01-01") -> pd.DataFrame:
    """A forecast frame with all-unresolved (NaN ``y``) rows."""
    return pd.DataFrame(
        {
            UNIQUE_ID: ["SKU_001"] * n,
            DS: pd.date_range("2024-01-07", periods=n, freq="W"),
            Y: [np.nan] * n,
            Y_HAT: [10.0 * (i + 1) for i in range(n)],
            H: list(range(1, n + 1)),
            FORECAST_ORIGIN: pd.Timestamp(origin),
            MODEL_NAME: ["SeasonalNaive"] * n,
        }
    )


# ---------------------------------------------------------------------------
# Engine-level characterization: streaming adapter == in-memory adapter.
# ---------------------------------------------------------------------------


def test_streaming_output_matches_in_memory_ledger(monkeypatch, tmp_path) -> None:
    dates = pd.date_range("2024-01-07", periods=8, freq="W")
    actuals = pd.DataFrame({UNIQUE_ID: "A", DS: dates, Y: [float(i) for i in range(8)]})
    task = ForecastTask(
        history=actuals,
        horizon=2,
        model_config={"backend": "stub", "model": "stub_model"},
    )
    origins = [dates[3], dates[4]]

    monkeypatch.setattr("calibre.execution.backend.resolve_adapter", lambda _: _StubAdapter())

    in_memory_result = BackendEngine().execute(partition_tasks([task]), actuals, origins)
    expected = in_memory_result.ledger.to_df()
    path = tmp_path / "ledger.parquet"
    streaming_result = BackendEngine(
        output=LedgerOutputOptions(forecast_path=str(path), streaming=True),
    ).execute(partition_tasks([task]), actuals, origins)

    actual = streaming_result.ledger.to_df()
    pd.testing.assert_frame_equal(actual, expected)
    # The mode is chosen once at construction: streaming_output unset → in-memory
    # adapter; set → streaming adapter (no in-memory frame buffer to leak into).
    assert isinstance(in_memory_result.ledger, InMemoryLedger)
    assert isinstance(streaming_result.ledger, StreamingLedger)
    assert path.exists()
    assert (tmp_path / "ledger.resolved.parquet").exists()


def test_streaming_output_accepts_fsspec_uri(monkeypatch) -> None:
    dates = pd.date_range("2024-01-07", periods=8, freq="W")
    actuals = pd.DataFrame({UNIQUE_ID: "A", DS: dates, Y: [float(i) for i in range(8)]})
    task = ForecastTask(
        history=actuals,
        horizon=1,
        model_config={"backend": "stub", "model": "stub_model"},
    )
    monkeypatch.setattr("calibre.execution.backend.resolve_adapter", lambda _: _StubAdapter())

    result = BackendEngine(
        output=LedgerOutputOptions(
            forecast_path="memory://calibre-tests/streaming/ledger.parquet",
            streaming=True,
        ),
    ).execute(partition_tasks([task]), actuals, [dates[3]])

    assert len(result.ledger.to_df()) == 1
    assert pd.read_parquet("memory://calibre-tests/streaming/ledger.parquet").shape[0] == 1


def test_streaming_resolution_keeps_only_pending_rows(monkeypatch, tmp_path) -> None:
    dates = pd.date_range("2024-01-07", periods=24, freq="W")
    actuals = pd.DataFrame({UNIQUE_ID: "A", DS: dates, Y: [float(i) for i in range(24)]})
    task = ForecastTask(
        history=actuals,
        horizon=4,
        model_config={"backend": "stub", "model": "stub_model"},
    )
    origins = list(dates[5:17])

    monkeypatch.setattr("calibre.execution.backend.resolve_adapter", lambda _: _StubAdapter())

    expected = BackendEngine().execute(partition_tasks([task]), actuals, origins).ledger.to_df()
    path = tmp_path / "bounded-ledger.parquet"
    result = BackendEngine(
        output=LedgerOutputOptions(forecast_path=str(path), streaming=True),
    ).execute(partition_tasks([task]), actuals, origins)

    actual = result.ledger.to_df()
    pd.testing.assert_frame_equal(actual, expected)
    # resolution_frame() is the public view of the still-pending rows; bounded
    # memory means it never grows to the full origins * horizon row count.
    assert len(result.ledger.resolution_frame()) <= 10
    assert len(pd.read_parquet(path)) == len(origins) * task.horizon


def test_origin_iterator_matches_batch_conformal_and_ordering(monkeypatch) -> None:
    dates = pd.date_range("2024-01-07", periods=12, freq="W")
    actuals = pd.DataFrame({UNIQUE_ID: "A", DS: dates, Y: [float(i) for i in range(12)]})
    task = ForecastTask(
        history=actuals,
        horizon=2,
        model_config={"backend": "stub", "model": "stub_model"},
    )
    origins = [dates[4], dates[5], dates[6]]
    conformal_config = SymmetricIntervalConfig(
        method="aci",
        coverage=0.9,
        calibration_window=4,
        gamma=0.05,
    )
    order_config = NewsvendorConfig(
        params=[
            NewsvendorPolicyParameters(
                unique_id="A",
                underage_cost=3.0,
                overage_cost=1.0,
                inventory_position=0.0,
            )
        ],
        coverage=0.9,
    )

    monkeypatch.setattr("calibre.execution.backend.resolve_adapter", lambda _: _StubAdapter())

    batch_runtime = SymmetricIntervalRuntime(conformal_config)
    batch_result = BackendEngine(
        conformal=ConformalOptions(runtime=batch_runtime),
        order=order_config,
    ).execute(partition_tasks([task]), actuals, origins)

    stream_runtime = SymmetricIntervalRuntime(conformal_config)
    engine = BackendEngine(
        conformal=ConformalOptions(runtime=stream_runtime),
        order=order_config,
    )
    yielded = list(engine.iter_origins(partition_tasks([task]), actuals, origins))

    assert len(yielded) == len(origins)
    pd.testing.assert_frame_equal(
        yielded[-1].ledger.to_df().reset_index(drop=True),
        batch_result.ledger.to_df().reset_index(drop=True),
    )
    pd.testing.assert_frame_equal(
        yielded[-1].order_ledger.to_df().reset_index(drop=True),
        batch_result.order_ledger.to_df().reset_index(drop=True),
    )
    assert (
        stream_runtime.get_diagnostics()["issued_count"]
        == batch_runtime.get_diagnostics()["issued_count"]
    )


# ---------------------------------------------------------------------------
# Direct StreamingLedger adapter tests — the merge is now reachable without a
# full backtest (the deepening's new test surface).
# ---------------------------------------------------------------------------


def test_streaming_append_keeps_only_pending_in_resolution_frame(tmp_path) -> None:
    ledger = StreamingLedger(tmp_path / "ledger.parquet")
    ledger.append(_pending_frame(3))
    pending = ledger.resolution_frame()
    assert len(pending) == 3
    assert pending[Y].isna().all()
    ledger.close()


def test_streaming_merge_prefers_resolved_on_key_collision(tmp_path) -> None:
    path = tmp_path / "ledger.parquet"
    ledger = StreamingLedger(path)
    ledger.append(_pending_frame(2))

    # Resolve only h=1; h=2 stays pending.
    resolved = _pending_frame(2)
    resolved.loc[resolved[H] == 1, Y] = 11.0
    ledger.update_resolved(resolved)
    # The streaming buffer now holds only the still-pending row.
    assert len(ledger.resolution_frame()) == 1
    ledger.close()

    final = pd.read_parquet(resolved_ledger_uri(path)).sort_values(H).reset_index(drop=True)
    assert final.loc[final[H] == 1, Y].iloc[0] == 11.0  # resolved value wins on collision
    assert pd.isna(final.loc[final[H] == 2, Y].iloc[0])  # unresolved row stays NaN


def test_streaming_close_finalizes_artifact_and_removes_updates_temp(tmp_path) -> None:
    path = tmp_path / "ledger.parquet"
    ledger = StreamingLedger(path)
    ledger.append(_pending_frame(2))
    resolved = _pending_frame(2)
    resolved[Y] = [5.0, 6.0]
    ledger.update_resolved(resolved)

    # The resolved-updates side file exists while the ledger is open.
    updates_path = tmp_path / "ledger.resolved-updates.parquet"
    assert updates_path.exists()

    ledger.close()

    # close() finalizes the merged artifact and removes the temp side file.
    assert (tmp_path / "ledger.resolved.parquet").exists()
    assert not updates_path.exists()


def test_streaming_merge_missing_key_columns_raises(tmp_path) -> None:
    path = tmp_path / "ledger.parquet"
    ledger = StreamingLedger(path)
    raw = _pending_frame(2)
    raw.to_parquet(ledger._stream_path)
    # A resolved-updates artifact missing a key column cannot be merged.
    updates = raw.iloc[[0]].drop(columns=[MODEL_NAME])
    updates.to_parquet(ledger._resolved_updates_path)

    with pytest.raises(ValueError, match="without key columns"):
        ledger._materialize_streaming_frame()


def test_partitioned_streaming_output_writes_hive_partitions(tmp_path) -> None:
    path = tmp_path / "partitioned-ledger"
    ledger = StreamingLedger(path, partition_cols=[UNIQUE_ID])
    first = pd.DataFrame(
        {
            UNIQUE_ID: ["A", "B"],
            DS: pd.date_range("2024-01-07", periods=2, freq="W"),
            Y: [float("nan"), float("nan")],
            Y_HAT: [1.0, 2.0],
            H: [1, 1],
            FORECAST_ORIGIN: [pd.Timestamp("2024-01-01")] * 2,
            MODEL_NAME: ["stub", "stub"],
        }
    )
    second = first.assign(**{Y_HAT: [3.0, 4.0]})

    ledger.append(first)
    ledger.append(second)
    # All rows are unresolved, so the streaming buffer carries them as pending.
    assert len(ledger.resolution_frame()) == 4
    ledger.close()

    written = pd.read_parquet(path).sort_values([UNIQUE_ID, Y_HAT]).reset_index(drop=True)
    expected = pd.concat([first, second], ignore_index=True).sort_values([UNIQUE_ID, Y_HAT])
    expected = expected.reset_index(drop=True)
    written[UNIQUE_ID] = written[UNIQUE_ID].astype(str)
    pd.testing.assert_frame_equal(written[expected.columns], expected, check_dtype=False)
    assert (path / "unique_id=A" / "part-0.parquet").exists()
    assert (path / "unique_id=B" / "part-0.parquet").exists()


def test_partitioned_streaming_requires_partition_columns(tmp_path) -> None:
    ledger = StreamingLedger(tmp_path / "partitioned-ledger", partition_cols=["missing"])
    frame = pd.DataFrame(
        {
            UNIQUE_ID: ["A"],
            DS: [pd.Timestamp("2024-01-07")],
            Y: [float("nan")],
            Y_HAT: [1.0],
            H: [1],
            FORECAST_ORIGIN: [pd.Timestamp("2024-01-01")],
            MODEL_NAME: ["stub"],
        }
    )

    try:
        with pytest.raises(ValueError, match="Missing partition columns"):
            ledger.append(frame)
    finally:
        ledger.close()


# ---------------------------------------------------------------------------
# Direct StreamingOrderLedger adapter tests — the forecast adapter got a direct
# suite above; its append-only order-ledger sibling had none.
# ---------------------------------------------------------------------------


def test_streaming_order_ledger_streams_nonempty_and_skips_empty(tmp_path) -> None:
    path = tmp_path / "orders.parquet"
    ledger = StreamingOrderLedger(path)
    orders = pd.DataFrame({UNIQUE_ID: ["SKU_001", "SKU_002"], "order_qty": [3.0, 5.0]})
    ledger.append(orders)
    ledger.append(pd.DataFrame(columns=[UNIQUE_ID, "order_qty"]))  # empty -> skipped
    ledger.close()

    written = pd.read_parquet(path).reset_index(drop=True)
    pd.testing.assert_frame_equal(written, orders, check_dtype=False)


def test_streaming_order_ledger_empty_when_nothing_appended(tmp_path) -> None:
    ledger = StreamingOrderLedger(tmp_path / "orders.parquet")
    ledger.close()
    assert ledger.to_df().empty
