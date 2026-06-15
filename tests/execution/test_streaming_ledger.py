"""Tests for streaming ledger writes during a run."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from calibre.conformal import SymmetricIntervalConfig, SymmetricIntervalRuntime
from calibre.core.forecast_frame import (
    DS,
    FORECAST_ORIGIN,
    MODEL_NAME,
    NONCONFORMITY_SCORE,
    UNIQUE_ID,
    Y_HAT,
    H,
    Y,
)
from calibre.core.forecast_task import ForecastTask
from calibre.core.order_types import NewsvendorPolicyParameters
from calibre.execution.backend import BackendEngine, ConformalOptions, LedgerOutputOptions
from calibre.execution.ledger import (
    _APPEND_SEQ,
    _FORECAST_KEY_COLUMNS,
    InMemoryLedger,
    StreamingLedger,
    StreamingOrderLedger,
    _key_index,
    resolved_ledger_uri,
)
from calibre.execution.task_builder import partition_tasks
from calibre.forecasting.adapter_base import ModelAdapter
from calibre.ordering.policy_config import NewsvendorConfig

_KEY = _FORECAST_KEY_COLUMNS
_KEY_INDEX = _key_index

# Committed pre-refactor due_frame(origin) pin captured at ada263c (scenario 3).
_PRE_REFACTOR_DUE_FRAME_FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "streaming_ledger_pre_refactor_due_frame.parquet"
)
# ORIGIN and the synthetic input below reproduce the capture exactly so the
# post-refactor due_frame output can be asserted byte-equal against the pin.
_PRE_REFACTOR_ORIGIN = pd.Timestamp("2024-02-15")


def _interval_batch(
    uid: str, ds_values: list[str], origin: str, *, start_seq: int = 0
) -> pd.DataFrame:
    """A pending (NaN-y) batch with genuinely distinct ds + appended interval cols.

    Unlike :func:`_pending_frame` (which hardcodes the same ds range for every
    batch), the ds values are caller-supplied so two batches can interleave
    across ds buckets — exercising the cross-bucket append-seq re-sort. The
    ``y_hat_lower``/``y_hat_upper`` columns sit beyond ``REQUIRED_COLUMNS`` so
    they also pin the full-appended-column-set contract.

    ``start_seq`` only shifts the y_hat values so distinct batches stay
    value-distinguishable; it is not the ledger's private append-seq.
    """
    n = len(ds_values)
    return pd.DataFrame(
        {
            UNIQUE_ID: [uid] * n,
            DS: [pd.Timestamp(d) for d in ds_values],
            Y: [np.nan] * n,
            Y_HAT: [10.0 * (start_seq + i + 1) for i in range(n)],
            H: list(range(1, n + 1)),
            FORECAST_ORIGIN: pd.Timestamp(origin),
            MODEL_NAME: ["SeasonalNaive"] * n,
            "y_hat_lower": [5.0 * (i + 1) for i in range(n)],
            "y_hat_upper": [15.0 * (i + 1) for i in range(n)],
        }
    )


class _StubAdapter(ModelAdapter):
    def fit(self, task: ForecastTask, *, collect_fitted_values: bool = False) -> None:
        del collect_fitted_values
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

    monkeypatch.setattr("calibre.execution.prediction.resolve_adapter", lambda _: _StubAdapter())

    in_memory_result = BackendEngine().execute(partition_tasks([task]), actuals, origins)
    expected = in_memory_result.ledger.to_df()
    path = tmp_path / "ledger.parquet"
    streaming_result = BackendEngine(
        output=LedgerOutputOptions(forecast_path=str(path), streaming=True),
    ).execute(partition_tasks([task]), actuals, origins)

    actual = streaming_result.ledger.to_df()
    # The streaming finalize reconstructs the artifact as unresolved-then-
    # resolved rows (membership join), so it does not preserve the in-memory
    # append order; the pinned semantic is row-for-row equality once both are
    # key-sorted (streaming-matches-in-memory).
    pd.testing.assert_frame_equal(
        actual.sort_values(_KEY).reset_index(drop=True),
        expected.sort_values(_KEY).reset_index(drop=True),
    )
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
    monkeypatch.setattr("calibre.execution.prediction.resolve_adapter", lambda _: _StubAdapter())

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

    monkeypatch.setattr("calibre.execution.prediction.resolve_adapter", lambda _: _StubAdapter())

    expected = BackendEngine().execute(partition_tasks([task]), actuals, origins).ledger.to_df()
    path = tmp_path / "bounded-ledger.parquet"
    result = BackendEngine(
        output=LedgerOutputOptions(forecast_path=str(path), streaming=True),
    ).execute(partition_tasks([task]), actuals, origins)

    actual = result.ledger.to_df()
    pd.testing.assert_frame_equal(
        actual.sort_values(_KEY).reset_index(drop=True),
        expected.sort_values(_KEY).reset_index(drop=True),
    )
    # due_frame(far future) returns every still-pending row; bounded memory
    # means it never grows to the full origins * horizon row count.
    all_pending = result.ledger.due_frame(pd.Timestamp("2999-01-01"))
    assert len(all_pending) <= 10
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

    monkeypatch.setattr("calibre.execution.prediction.resolve_adapter", lambda _: _StubAdapter())

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
    # Streaming finalize reorders to unresolved-then-resolved; compare key-sorted.
    pd.testing.assert_frame_equal(
        yielded[-1].ledger.to_df().sort_values(_KEY).reset_index(drop=True),
        batch_result.ledger.to_df().sort_values(_KEY).reset_index(drop=True),
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


def test_streaming_append_keeps_only_pending_in_due_frame(tmp_path) -> None:
    ledger = StreamingLedger(tmp_path / "ledger.parquet")
    ledger.append(_pending_frame(3))
    pending = ledger.due_frame(pd.Timestamp("2999-01-01"))
    assert len(pending) == 3
    assert pending[Y].isna().all()
    assert list(pending.index) == [0, 1, 2]  # RangeIndex contract
    ledger.close()


def test_streaming_apply_resolutions_prefers_resolved_on_key_collision(tmp_path) -> None:
    path = tmp_path / "ledger.parquet"
    ledger = StreamingLedger(path)
    ledger.append(_pending_frame(2))

    # Resolve only h=1; h=2 stays pending (keyed upsert, not wholesale replace).
    resolved = _pending_frame(2)
    resolved.loc[resolved[H] == 1, Y] = 11.0
    ledger.apply_resolutions(resolved)
    # The streaming buffer now holds only the still-pending row.
    assert len(ledger.due_frame(pd.Timestamp("2999-01-01"))) == 1
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
    ledger.apply_resolutions(resolved)

    # The resolved-updates side file exists while the ledger is open.
    updates_path = tmp_path / "ledger.resolved-updates.parquet"
    assert updates_path.exists()

    ledger.close()

    # close() finalizes the merged artifact and removes the temp side file.
    assert (tmp_path / "ledger.resolved.parquet").exists()
    assert not updates_path.exists()


def test_streaming_due_frame_boundary_and_empty(tmp_path) -> None:
    ledger = StreamingLedger(tmp_path / "ledger.parquet")
    # ds values: 2024-01-07 (h=1), 2024-01-14 (h=2).
    ledger.append(_pending_frame(2))
    on_boundary = ledger.due_frame(pd.Timestamp("2024-01-07"))
    assert on_boundary[H].tolist() == [1]  # ds == origin included, ds > origin excluded
    none_due = ledger.due_frame(pd.Timestamp("2024-01-01"))
    assert none_due.empty
    ledger.close()


def test_streaming_apply_resolutions_keyed_upsert_leaves_not_due_untouched(tmp_path) -> None:
    ledger = StreamingLedger(tmp_path / "ledger.parquet")
    ledger.append(_pending_frame(4))  # ds weekly from 2024-01-07
    # Only the first two are due; resolve h=1, leave h=2 pending.
    due = ledger.due_frame(pd.Timestamp("2024-01-14"))
    assert due[H].tolist() == [1, 2]
    due.loc[due[H] == 1, Y] = 99.0
    ledger.apply_resolutions(due)
    remaining = ledger.due_frame(pd.Timestamp("2999-01-01"))
    # h=1 left the open set; h=2,3,4 (incl. the never-due rows) remain.
    assert sorted(remaining[H].tolist()) == [2, 3, 4]
    ledger.close()


def test_streaming_apply_resolutions_unknown_key_raises(tmp_path) -> None:
    ledger = StreamingLedger(tmp_path / "ledger.parquet")
    ledger.append(_pending_frame(2))
    rogue = _pending_frame(1, origin="2099-01-01")  # never appended key
    rogue.loc[0, Y] = 1.0
    with pytest.raises(ValueError, match="not in the open set"):
        ledger.apply_resolutions(rogue)
    ledger.close()


def test_streaming_append_duplicate_key_raises(tmp_path) -> None:
    ledger = StreamingLedger(tmp_path / "ledger.parquet")
    ledger.append(_pending_frame(2))
    with pytest.raises(ValueError, match="duplicate forecast keys"):
        ledger.append(_pending_frame(2))  # identical 5-tuples
    ledger.close()


def test_streaming_apply_resolutions_empty_is_noop(tmp_path) -> None:
    ledger = StreamingLedger(tmp_path / "ledger.parquet")
    ledger.append(_pending_frame(2))
    before = len(ledger.due_frame(pd.Timestamp("2999-01-01")))
    ledger.apply_resolutions(_pending_frame(2).iloc[0:0])
    assert len(ledger.due_frame(pd.Timestamp("2999-01-01"))) == before
    assert not (tmp_path / "ledger.resolved-updates.parquet").exists()
    ledger.close()


# ---------------------------------------------------------------------------
# ds-bucketing refactor (#211): the pending store buckets by ds with a private
# append-seq stamp, so due_frame/apply_resolutions are O(due), not O(open set).
# These pin the load-bearing equivalences: append-order re-sort, no seq leak,
# cross-origin fragmentation, the empty-due column set, and bucket bookkeeping.
# ---------------------------------------------------------------------------


def test_streaming_due_frame_returns_append_order_across_interleaved_ds(tmp_path) -> None:
    # Scenario 1 (happy path). Two batches whose ds buckets interleave: batch B's
    # earliest ds (2024-01-14) < batch A's latest ds (2024-03-03). due_frame must
    # return APPEND order (all of A's due rows, then B's), NOT ds order.
    ledger = StreamingLedger(tmp_path / "ledger.parquet")
    batch_a = _interval_batch(
        "SKU_001", ["2024-01-07", "2024-02-04", "2024-03-03"], origin="2024-01-01"
    )
    batch_b = _interval_batch(
        "SKU_002", ["2024-01-14", "2024-02-11", "2024-02-25"], origin="2024-01-08", start_seq=10
    )
    ledger.append(batch_a)
    ledger.append(batch_b)

    due = ledger.due_frame(pd.Timestamp("2024-02-15"))
    # ds <= 2024-02-15: A@01-07, A@02-04, B@01-14, B@02-11 (A's 03-03 and B's
    # 02-25 are future-dated and excluded).
    assert due[DS].tolist() == [
        pd.Timestamp("2024-01-07"),
        pd.Timestamp("2024-02-04"),
        pd.Timestamp("2024-01-14"),
        pd.Timestamp("2024-02-11"),
    ]
    assert due[UNIQUE_ID].tolist() == ["SKU_001", "SKU_001", "SKU_002", "SKU_002"]
    assert list(due.index) == [0, 1, 2, 3]  # clean RangeIndex
    assert _APPEND_SEQ not in due.columns  # private seq never leaks into output

    # Liveness: the future-dated bucket's rows never appear and its bucket object
    # is not mutated by the due-path gather.
    future_bucket_before = ledger._buckets[pd.Timestamp("2024-03-03")].copy()
    _ = ledger.due_frame(pd.Timestamp("2024-02-15"))
    pd.testing.assert_frame_equal(ledger._buckets[pd.Timestamp("2024-03-03")], future_bucket_before)
    ledger.close()


def test_streaming_due_frame_matches_pre_refactor_fixture(tmp_path) -> None:
    # Scenario 3 (characterization anchor on the changed surface). The
    # post-refactor due_frame(origin) output must equal a pre-refactor pin
    # captured at ada263c — same synthetic input, same origin, no re-sort.
    ledger = StreamingLedger(tmp_path / "ledger.parquet")
    batch_a = _interval_batch(
        "SKU_001", ["2024-01-07", "2024-02-04", "2024-03-03"], origin="2024-01-01"
    )
    batch_b = _interval_batch(
        "SKU_002", ["2024-01-14", "2024-02-11", "2024-02-25"], origin="2024-01-08"
    )
    ledger.append(batch_a)
    ledger.append(batch_b)
    due = ledger.due_frame(_PRE_REFACTOR_ORIGIN)
    ledger.close()

    # The fixture was captured via to_parquet at ada263c, so round-trip the
    # post-refactor frame through the same serialization before comparing — this
    # pins the captured artifact byte-for-byte and neutralises parquet's
    # timestamp-resolution round-trip (datetime64[s] in-memory vs [ms] on read),
    # which is an I/O artifact, not a due_frame behaviour change.
    due_path = tmp_path / "post_refactor_due_frame.parquet"
    due.to_parquet(due_path)
    actual = pd.read_parquet(due_path)
    expected = pd.read_parquet(_PRE_REFACTOR_DUE_FRAME_FIXTURE)
    pd.testing.assert_frame_equal(actual, expected)


def test_streaming_due_frame_empty_due_preserves_appended_columns(tmp_path) -> None:
    # A populated store with zero due rows returns an empty frame that
    # preserves the FULL appended column set (interval columns beyond
    # REQUIRED_COLUMNS), not a REQUIRED_COLUMNS-only frame.
    ledger = StreamingLedger(tmp_path / "ledger.parquet")
    ledger.append(_interval_batch("SKU_001", ["2024-06-01", "2024-06-08"], origin="2024-05-01"))
    none_due = ledger.due_frame(pd.Timestamp("2024-01-01"))  # everything future
    assert none_due.empty
    assert "y_hat_lower" in none_due.columns
    assert "y_hat_upper" in none_due.columns
    assert _APPEND_SEQ not in none_due.columns
    ledger.close()


def test_streaming_due_frame_empty_due_unions_columns_across_buckets(tmp_path) -> None:
    # Edge case: buckets carry DIFFERING column sets (a base-only bucket and an
    # interval-bearing one). The old single-concat store exposed the column
    # UNION on an empty slice; the empty-due frame must too — taking one
    # arbitrary bucket's columns would drop the interval columns when the
    # base-only bucket is iterated first. Populate the bucket store directly:
    # the raw stream sink is schema-locked to the first batch, so appending
    # differing-column batches through append() is not the engine's contract;
    # the bucket-store column union is what is under test here.
    ledger = StreamingLedger(tmp_path / "ledger.parquet")
    base_only = _pending_frame(1, origin="2024-05-01")  # no interval columns
    base_only[DS] = [pd.Timestamp("2024-06-01")]
    base_only[_APPEND_SEQ] = [0]
    base_only.index = _KEY_INDEX(base_only)
    interval = _interval_batch("SKU_002", ["2024-06-08"], origin="2024-05-02")
    interval[_APPEND_SEQ] = [1]
    interval.index = _KEY_INDEX(interval)
    # base-only bucket inserted first → it would be the arbitrary sample.
    ledger._buckets = {
        pd.Timestamp("2024-06-01"): base_only,
        pd.Timestamp("2024-06-08"): interval,
    }

    none_due = ledger.due_frame(pd.Timestamp("2024-01-01"))  # everything future
    assert none_due.empty
    # The union must carry the interval columns even though the first-iterated
    # bucket lacks them.
    assert "y_hat_lower" in none_due.columns
    assert "y_hat_upper" in none_due.columns
    assert _APPEND_SEQ not in none_due.columns


def test_streaming_due_frame_nat_ds_row_stays_in_open_set(tmp_path) -> None:
    # NaT-ds rows must stay in the open set (their ds <= origin is always False),
    # exactly as the old single-frame store retained them. groupby's default
    # dropna=True would silently drop the NaT bucket; dropna=False keeps it.
    ledger = StreamingLedger(tmp_path / "ledger.parquet")
    batch = _pending_frame(2, origin="2024-01-01")
    batch[DS] = [pd.Timestamp("2024-01-07"), pd.NaT]  # one real ds, one NaT
    ledger.append(batch)
    # The NaT-ds row is retained as its own bucket, not dropped from the store.
    assert any(pd.isna(ds) for ds in ledger._buckets)
    # Far-future origin: the real-ds row is due, the NaT-ds row is never due.
    due = ledger.due_frame(pd.Timestamp("2999-01-01"))
    assert due[DS].notna().all()
    assert len(due) == 1  # only the real-ds row is ever due
    ledger.close()


def test_streaming_due_frame_resorts_noncontiguous_appends_to_one_bucket(tmp_path) -> None:
    # The append-seq re-sort is load-bearing only when a single ds bucket
    # accumulates rows from NON-CONTIGUOUS appends: bucket-dict insertion order
    # then no longer equals append order, so _gather_ordered must restore append
    # order (the byte-identity contract) rather than emit buckets in insertion
    # order. Append the SAME ds in two non-adjacent batches, straddling a second
    # bucket, so the middle seq lands in the other bucket.
    ledger = StreamingLedger(tmp_path / "ledger.parquet")
    ledger.append(_interval_batch("SKU_A", ["2024-01-07"], origin="2024-01-01"))  # seq 0
    ledger.append(_interval_batch("SKU_B", ["2024-01-14"], origin="2024-01-08"))  # seq 1
    ledger.append(_interval_batch("SKU_C", ["2024-01-07"], origin="2024-01-15"))  # seq 2
    # Bucket 2024-01-07 holds seqs [0, 2]; bucket 2024-01-14 holds seq [1].
    # Concatenating buckets in insertion order yields [A(0), C(2), B(1)] — only
    # the seq re-sort restores append order [A, B, C]. (Dropping the sort makes
    # this assert fail with [A, C, B].)
    due = ledger.due_frame(pd.Timestamp("2024-03-01"))  # everything due
    assert due[UNIQUE_ID].tolist() == ["SKU_A", "SKU_B", "SKU_C"]
    assert _APPEND_SEQ not in due.columns
    ledger.close()


def test_streaming_apply_resolutions_drops_emptied_bucket_keeps_others(tmp_path) -> None:
    # Scenario 5: keys spanning two ds buckets; resolving all keys in one bucket
    # del's that bucket key and leaves the other bucket's keys intact.
    ledger = StreamingLedger(tmp_path / "ledger.parquet")
    ledger.append(_interval_batch("SKU_001", ["2024-01-07", "2024-01-14"], origin="2024-01-01"))
    due = ledger.due_frame(pd.Timestamp("2024-01-07"))  # only the 01-07 bucket
    assert due[DS].tolist() == [pd.Timestamp("2024-01-07")]
    due[Y] = [99.0]
    ledger.apply_resolutions(due)

    assert pd.Timestamp("2024-01-07") not in ledger._buckets  # emptied bucket dropped
    assert pd.Timestamp("2024-01-14") in ledger._buckets  # other bucket kept
    remaining = ledger.due_frame(pd.Timestamp("2999-01-01"))
    assert remaining[DS].tolist() == [pd.Timestamp("2024-01-14")]
    ledger.close()


def test_streaming_apply_resolutions_multi_origin_straggler_reorders_by_seq(tmp_path) -> None:
    # Scenario 5 (primary fragmentation gate): a key appended at an early origin
    # resolves LATER than later-appended keys, so the append-seq re-sort — not ds
    # order or resolution order — drives due_frame's row order.
    ledger = StreamingLedger(tmp_path / "ledger.parquet")
    # Append order: early straggler first (origin 2024-01-01), then a later batch
    # (origin 2024-01-08). The straggler's ds is the LATEST, so ds order would put
    # it last; append (seq) order keeps it first.
    straggler = _interval_batch("SKU_001", ["2024-03-01"], origin="2024-01-01")
    later = _interval_batch("SKU_002", ["2024-02-01", "2024-02-08"], origin="2024-01-08")
    ledger.append(straggler)
    ledger.append(later)

    due = ledger.due_frame(pd.Timestamp("2024-03-15"))
    # Append/seq order: straggler (seq 0) first, then later's two rows — NOT ds
    # order (which would be 2024-02-01, 2024-02-08, 2024-03-01).
    assert due[DS].tolist() == [
        pd.Timestamp("2024-03-01"),
        pd.Timestamp("2024-02-01"),
        pd.Timestamp("2024-02-08"),
    ]
    assert due[UNIQUE_ID].tolist() == ["SKU_001", "SKU_002", "SKU_002"]

    # Resolve the later-appended keys first; the straggler stays pending. The
    # remaining due frame still leads with the straggler.
    resolve_later = due[due[UNIQUE_ID] == "SKU_002"].copy()
    resolve_later[Y] = [1.0, 2.0]
    ledger.apply_resolutions(resolve_later)
    remaining = ledger.due_frame(pd.Timestamp("2024-03-15"))
    assert remaining[DS].tolist() == [pd.Timestamp("2024-03-01")]
    assert remaining[UNIQUE_ID].tolist() == ["SKU_001"]
    ledger.close()


def test_streaming_apply_resolutions_unknown_key_across_buckets_raises(tmp_path) -> None:
    # Scenario 6 (bucket-spanning variant): an unknown resolution key whose ds
    # bucket is absent still raises with the exact open-set string.
    ledger = StreamingLedger(tmp_path / "ledger.parquet")
    ledger.append(_interval_batch("SKU_001", ["2024-01-07", "2024-01-14"], origin="2024-01-01"))
    rogue = _interval_batch("SKU_999", ["2099-01-01"], origin="2024-01-01")  # absent ds bucket
    rogue[Y] = [1.0]
    with pytest.raises(ValueError, match="not in the open set"):
        ledger.apply_resolutions(rogue)
    ledger.close()


def test_streaming_append_duplicate_key_across_buckets_raises(tmp_path) -> None:
    # Scenario 6 (bucket-spanning variant): a duplicate 5-tuple lands in the same
    # ds bucket and the per-bucket duplicate probe rejects it.
    ledger = StreamingLedger(tmp_path / "ledger.parquet")
    batch = _interval_batch("SKU_001", ["2024-01-07", "2024-02-04"], origin="2024-01-01")
    ledger.append(batch)
    with pytest.raises(ValueError, match="duplicate forecast keys"):
        ledger.append(batch)  # identical 5-tuples across two ds buckets
    ledger.close()


def test_streaming_finalize_missing_key_columns_raises(tmp_path) -> None:
    path = tmp_path / "ledger.parquet"
    ledger = StreamingLedger(path)
    raw = _pending_frame(2)
    raw.to_parquet(ledger._stream_path)
    # A resolved-updates artifact missing a key column cannot be resolved; the
    # finalize path must abort naming the missing key column.
    updates = raw.iloc[[0]].drop(columns=[MODEL_NAME])
    updates.to_parquet(ledger._resolved_updates_path)

    with pytest.raises(ValueError, match=r"without key columns.*model_name"):
        ledger.close()


# ---------------------------------------------------------------------------
# U1: streaming finalize rewrite — union-schema Arrow membership algorithm.
# The raw stream and resolved-updates side file are written directly so the
# finalize path is exercised at its own boundary (no full backtest needed).
# ---------------------------------------------------------------------------


def _raw_stream_row(uid: str, h: int, *, score: float = np.nan, origin: str = "2024-01-01") -> dict:
    """A raw (pending) ledger row as the engine appends it.

    y is NaN, the score column is present (NaN before resolution), and no error
    columns exist yet.
    """
    return {
        UNIQUE_ID: uid,
        DS: pd.Timestamp("2024-01-07") + pd.Timedelta(weeks=h - 1),
        Y: np.nan,
        Y_HAT: 10.0 * h,
        H: h,
        FORECAST_ORIGIN: pd.Timestamp(origin),
        MODEL_NAME: "SeasonalNaive",
        NONCONFORMITY_SCORE: score,
    }


def _update_row(raw: dict, *, y: float, score: float, error: float) -> dict:
    """A resolved update row derived from a raw row.

    y is filled, the score is made real, and error columns are added
    (updates-only). It never nulls a column that was non-null in raw.
    """
    row = dict(raw)
    row[Y] = y
    row[NONCONFORMITY_SCORE] = score
    row["error"] = error
    row["abs_error"] = abs(error)
    row["pct_error"] = error / y if y else np.nan
    return row


def _reference_merge(raw: pd.DataFrame, updates: pd.DataFrame) -> pd.DataFrame:
    """The old pandas combine_first merge, kept only as a reference.

    It is the byte-equivalence reference the new streamed finalize must reproduce.
    """
    updates = updates.drop_duplicates(_KEY, keep="last")
    update_cols = [col for col in updates.columns if col not in _KEY]
    merged = raw.merge(
        updates[_KEY + update_cols], on=_KEY, how="left", suffixes=("", "__resolved")
    )
    final = raw.copy()
    for col in update_cols:
        if col in raw.columns:
            resolved_col = f"{col}__resolved"
            if resolved_col in merged.columns:
                final[col] = merged[resolved_col].combine_first(final[col])
        else:
            final[col] = merged[col]
    return final


def _finalize(tmp_path, raw: pd.DataFrame, updates: pd.DataFrame | None) -> pd.DataFrame:
    """Write raw + updates side files, run close(), read the resolved artifact."""
    path = tmp_path / "ledger.parquet"
    ledger = StreamingLedger(path)
    raw.to_parquet(ledger._stream_path)
    if updates is not None:
        updates.to_parquet(ledger._resolved_updates_path)
    ledger.close()
    return pd.read_parquet(resolved_ledger_uri(path))


def test_streaming_finalize_matches_reference_merge_with_keep_last(tmp_path) -> None:
    # A key resolved at one origin and re-resolved later exercises keep-last:
    # the later update value must win.
    raw = pd.DataFrame(
        [
            _raw_stream_row("SKU_001", 1),
            _raw_stream_row("SKU_001", 2),
            _raw_stream_row("SKU_002", 1),
        ]
    )
    first = _update_row(raw.iloc[0].to_dict(), y=11.0, score=0.5, error=1.0)
    later = _update_row(raw.iloc[0].to_dict(), y=12.0, score=0.7, error=2.0)  # re-resolution
    other = _update_row(raw.iloc[2].to_dict(), y=21.0, score=0.9, error=3.0)
    updates = pd.DataFrame([first, later, other])  # ascending stream position

    actual = _finalize(tmp_path, raw, updates)
    expected = _reference_merge(raw, updates)
    # check_dtype=True: the union-schema cast must not drift dtypes vs the old
    # combine_first reference (this is what _cast_keys_to_canonical and
    # _align_to_schema exist to guarantee).
    pd.testing.assert_frame_equal(
        actual.sort_values(_KEY).reset_index(drop=True),
        expected.sort_values(_KEY).reset_index(drop=True),
    )
    # keep-last: the later re-resolution (y=12.0, score=0.7) wins, not y=11.0.
    won = actual.set_index([UNIQUE_ID, H]).loc[("SKU_001", 1)]
    assert won[Y] == 12.0
    assert won[NONCONFORMITY_SCORE] == 0.7
    assert won["error"] == 2.0


def test_streaming_finalize_union_schema_keeps_updates_only_columns(tmp_path) -> None:
    # Stream lacks error/abs_error/pct_error (the real engine shape); updates
    # carry them. Resolved rows keep all three; unresolved rows get nulls.
    raw = pd.DataFrame(
        [
            _raw_stream_row("SKU_001", 1),  # resolved below
            _raw_stream_row("SKU_001", 2),  # stays pending
        ]
    )
    assert "error" not in raw.columns
    updates = pd.DataFrame([_update_row(raw.iloc[0].to_dict(), y=11.0, score=0.5, error=1.0)])

    actual = _finalize(tmp_path, raw, updates).set_index([UNIQUE_ID, H])
    expected_values = {"error": 1.0, "abs_error": 1.0, "pct_error": 1.0 / 11.0}
    for col, expected in expected_values.items():
        assert col in actual.columns
        assert actual.loc[("SKU_001", 1), col] == pytest.approx(expected)
        assert pd.isna(actual.loc[("SKU_001", 2), col])  # unresolved -> null
    # nonconformity_score: NaN in raw, real in update -> the update value wins.
    assert actual.loc[("SKU_001", 1), NONCONFORMITY_SCORE] == 0.5
    assert pd.isna(actual.loc[("SKU_001", 2), NONCONFORMITY_SCORE])


def test_streaming_finalize_no_null_regression_invariant(tmp_path) -> None:
    # The precondition that makes wholesale-row writes equal old combine_first:
    # no update row carries NaN where its raw row was non-null.
    raw = pd.DataFrame([_raw_stream_row("SKU_001", 1, score=np.nan)])
    update = _update_row(raw.iloc[0].to_dict(), y=11.0, score=0.5, error=1.0)
    updates = pd.DataFrame([update])
    for col in raw.columns:
        raw_non_null = raw[col].notna()
        update_null = updates[col].isna() if col in updates.columns else pd.Series([False])
        assert not (raw_non_null & update_null).any(), f"update nulls non-null raw column {col}"
    # And the finalize reproduces the combine_first reference.
    actual = _finalize(tmp_path, raw, updates)
    expected = _reference_merge(raw, updates)
    pd.testing.assert_frame_equal(
        actual.sort_values(_KEY).reset_index(drop=True),
        expected.sort_values(_KEY).reset_index(drop=True),
        check_dtype=False,
    )


def test_streaming_finalize_zero_updates_equals_raw_stream(tmp_path) -> None:
    raw = pd.DataFrame([_raw_stream_row("SKU_001", 1), _raw_stream_row("SKU_001", 2)])
    actual = _finalize(tmp_path, raw, updates=None)
    pd.testing.assert_frame_equal(
        actual.sort_values(_KEY).reset_index(drop=True),
        raw.sort_values(_KEY).reset_index(drop=True),
        check_dtype=False,
    )


def test_apply_resolutions_before_any_append_raises_in_memory() -> None:
    resolved = pd.DataFrame(
        [_update_row(_raw_stream_row("SKU_001", 1), y=11.0, score=0.5, error=1.0)]
    )
    with pytest.raises(ValueError, match="before any"):
        InMemoryLedger().apply_resolutions(resolved)


def test_apply_resolutions_before_any_append_raises_streaming(tmp_path) -> None:
    ledger = StreamingLedger(str(tmp_path / "ledger.parquet"))
    resolved = pd.DataFrame(
        [_update_row(_raw_stream_row("SKU_001", 1), y=11.0, score=0.5, error=1.0)]
    )
    with pytest.raises(ValueError, match="before any"):
        ledger.apply_resolutions(resolved)


def test_streaming_finalize_unmatched_update_key_aborts(tmp_path) -> None:
    raw = pd.DataFrame([_raw_stream_row("SKU_001", 1)])
    # An update for a key never issued (h=2) is corruption — finalize must abort
    # and not write the resolved artifact.
    rogue = _update_row(_raw_stream_row("SKU_001", 2).copy(), y=99.0, score=0.5, error=1.0)
    updates = pd.DataFrame([rogue])

    path = tmp_path / "ledger.parquet"
    ledger = StreamingLedger(path)
    raw.to_parquet(ledger._stream_path)
    updates.to_parquet(ledger._resolved_updates_path)
    with pytest.raises(ValueError, match="finalize aborted"):
        ledger.close()
    assert not (tmp_path / "ledger.resolved.parquet").exists()


def test_streaming_finalize_passes_through_already_resolved_raw_rows(tmp_path) -> None:
    # Resume shape: the stream contains rows with y already set and no matching
    # update row — they pass through as kept rows; accounting holds.
    resolved_in_raw = _raw_stream_row("SKU_000", 1)
    resolved_in_raw[Y] = 5.0  # already resolved, no update for it
    raw = pd.DataFrame([resolved_in_raw, _raw_stream_row("SKU_001", 1)])
    updates = pd.DataFrame([_update_row(raw.iloc[1].to_dict(), y=11.0, score=0.5, error=1.0)])

    actual = _finalize(tmp_path, raw, updates).set_index([UNIQUE_ID, H])
    assert actual.loc[("SKU_000", 1), Y] == 5.0  # passed through unchanged
    assert actual.loc[("SKU_001", 1), Y] == 11.0  # resolved by update
    assert len(actual) == 2


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
    # second carries a distinct forecast_origin so its 5-tuple keys do not
    # collide with first (the keyed open set rejects duplicate appends).
    second = first.assign(**{Y_HAT: [3.0, 4.0], FORECAST_ORIGIN: [pd.Timestamp("2024-01-08")] * 2})

    ledger.append(first)
    ledger.append(second)
    # All rows are unresolved, so the streaming buffer carries them as pending.
    assert len(ledger.due_frame(pd.Timestamp("2999-01-01"))) == 4
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
