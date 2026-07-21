# PROTOTYPE — throw away after Gate C data-plane planning.
# ruff: noqa: T201
"""Compare scan-shaped and indexed data planes on synthetic M5-shaped loads."""

from __future__ import annotations

import argparse
import math
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa

M5_DAYS = 1_941
M5_ORIGINS = 64
M5_HORIZON = 28
CALIBRATION_CYCLES = 64


@dataclass(frozen=True)
class Timing:
    """Hold one measured duration and logical owned-byte count."""

    seconds: float
    owned_bytes: int


@dataclass(frozen=True)
class PanelResult:
    """Hold immutable-panel construction and history-view measurements."""

    build: Timing
    history: Timing
    history_rows: int
    update_rows: int


@dataclass(frozen=True)
class LedgerResult:
    """Hold append/resolve measurements and terminal ledger state."""

    seconds: float
    stream_rows: int
    resolved_rows: int
    pending_rows: int
    peak_hot_bytes: int
    stream_bytes: int
    checksum: float


@dataclass(frozen=True)
class ConformalResult:
    """Hold batched-state update measurements and terminal state checksums."""

    seconds: float
    partitions: int
    state_bytes: int
    mean_checksum: float
    level_checksum: float


@dataclass(frozen=True)
class ScaleResult:
    """Hold both candidate results for one panel size."""

    series: int
    scan_panel: PanelResult
    indexed_panel: PanelResult
    scan_ledger: LedgerResult
    indexed_ledger: LedgerResult
    scalar_conformal: ConformalResult
    batched_conformal: ConformalResult


@dataclass(slots=True)
class _PendingSpan:
    """Keep one target-contiguous pending slice in the hot ledger index."""

    origin: int
    horizon_step: int
    points: np.ndarray


class _HistoryView:
    """Describe a zero-copy two-axis view over a calendar-aligned panel."""

    __slots__ = ("_values", "series_start", "series_stop", "time_stop")

    def __init__(
        self,
        values: np.ndarray,
        *,
        time_stop: int,
        series_start: int,
        series_stop: int,
    ) -> None:
        self._values = values
        self.time_stop = time_stop
        self.series_start = series_start
        self.series_stop = series_stop

    @property
    def values(self) -> np.ndarray:
        """Return the zero-copy history view used by an adapter."""
        return self._values[: self.time_stop, self.series_start : self.series_stop]

    def newly_admissible(self, previous_stop: int) -> np.ndarray:
        """Return only rows added since the previous origin."""
        return self._values[
            previous_stop : self.time_stop,
            self.series_start : self.series_stop,
        ]


def _measure(action):
    started = time.perf_counter()
    value = action()
    return value, time.perf_counter() - started


def _scan_panel(series_count: int) -> PanelResult:
    def build() -> pd.DataFrame:
        series_ids = np.repeat(np.arange(series_count, dtype=np.int32), M5_DAYS)
        time_ids = np.tile(np.arange(M5_DAYS, dtype=np.int16), series_count)
        values = ((series_ids % 97) * 0.01 + (time_ids % 29)).astype(np.float32)
        return pd.DataFrame(
            {"series_id": series_ids, "time_id": time_ids, "value": values},
            copy=False,
        )

    frame, build_seconds = _measure(build)
    owned = int(frame.memory_usage(index=True, deep=True).sum())
    origin = M5_DAYS - M5_ORIGINS
    history, history_seconds = _measure(
        lambda: frame.loc[frame["time_id"] < origin].reset_index(drop=True)
    )
    history_bytes = int(history.memory_usage(index=True, deep=True).sum())
    result = PanelResult(
        build=Timing(build_seconds, owned),
        history=Timing(history_seconds, history_bytes),
        history_rows=len(history),
        update_rows=series_count,
    )
    return result


def _indexed_panel(series_count: int) -> PanelResult:
    def build() -> tuple[np.ndarray, np.ndarray]:
        values = np.empty((M5_DAYS, series_count), dtype=np.float32)
        values[:] = (np.arange(series_count, dtype=np.float32) % 97) * 0.01
        values += (np.arange(M5_DAYS, dtype=np.float32) % 29)[:, None]
        # A bit-packed presence plane distinguishes absent/missing observations
        # without repeating series and timestamp keys on every long-format row.
        presence = np.full((M5_DAYS, math.ceil(series_count / 8)), 255, dtype=np.uint8)
        return values, presence

    (values, presence), build_seconds = _measure(build)
    origin = M5_DAYS - M5_ORIGINS

    def make_views() -> tuple[_HistoryView, _HistoryView, np.ndarray]:
        global_view = _HistoryView(
            values,
            time_stop=origin,
            series_start=0,
            series_stop=series_count,
        )
        chunk_view = _HistoryView(
            values,
            time_stop=origin,
            series_start=0,
            series_stop=min(256, series_count),
        )
        update = chunk_view.newly_admissible(origin - 1)
        return global_view, chunk_view, update

    (global_view, chunk_view, update), history_seconds = _measure(make_views)
    assert np.shares_memory(global_view.values, values)
    assert np.shares_memory(chunk_view.values, values)
    assert np.shares_memory(update, values)
    result = PanelResult(
        build=Timing(build_seconds, int(values.nbytes + presence.nbytes)),
        history=Timing(history_seconds, sys.getsizeof(global_view) + sys.getsizeof(chunk_view)),
        history_rows=int(global_view.values.size),
        update_rows=int(update.size),
    )
    return result


def _forecast_arrays(
    series_count: int,
    origin: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    series_ids = np.tile(np.arange(series_count, dtype=np.int32), M5_HORIZON)
    horizon_steps = np.repeat(np.arange(1, M5_HORIZON + 1, dtype=np.uint8), series_count)
    origins = np.full(series_ids.size, origin, dtype=np.int16)
    targets = (origin + horizon_steps.astype(np.int16)).astype(np.int16, copy=False)
    points = (
        (series_ids % 101) * 0.01
        + horizon_steps.astype(np.float32) * 0.1
        + np.float32(origin) * 0.001
    ).astype(np.float32, copy=False)
    return series_ids, horizon_steps, origins, targets, points


def _scan_ledger(series_count: int) -> LedgerResult:
    pending = pd.DataFrame(
        {
            "series_id": pd.Series(dtype="int32"),
            "h": pd.Series(dtype="uint8"),
            "origin": pd.Series(dtype="int16"),
            "target": pd.Series(dtype="int16"),
            "point": pd.Series(dtype="float32"),
        }
    )
    stream_rows = 0
    resolved_rows = 0
    stream_bytes = 0
    peak_hot_bytes = 0
    checksum = 0.0

    started = time.perf_counter()
    for origin in range(M5_ORIGINS + M5_HORIZON + 1):
        if not pending.empty:
            due_mask = pending["target"].to_numpy() < origin
            if due_mask.any():
                due = pending.loc[due_mask]
                resolved_rows += len(due)
                checksum += float(due["point"].sum())
                pending = pending.loc[~due_mask].reset_index(drop=True)
        if origin < M5_ORIGINS:
            arrays = _forecast_arrays(series_count, origin)
            batch = pd.DataFrame(
                dict(zip(("series_id", "h", "origin", "target", "point"), arrays, strict=True)),
                copy=False,
            )
            stream_rows += len(batch)
            stream_bytes += int(batch.memory_usage(index=False, deep=True).sum())
            pending = pd.concat((pending, batch), ignore_index=True)
        peak_hot_bytes = max(
            peak_hot_bytes,
            int(pending.memory_usage(index=True, deep=True).sum()),
        )
    seconds = time.perf_counter() - started
    return LedgerResult(
        seconds=seconds,
        stream_rows=stream_rows,
        resolved_rows=resolved_rows,
        pending_rows=len(pending),
        peak_hot_bytes=peak_hot_bytes,
        stream_bytes=stream_bytes,
        checksum=checksum,
    )


def _indexed_ledger(series_count: int) -> LedgerResult:
    due_index: dict[int, list[_PendingSpan]] = defaultdict(list)
    stream_rows = 0
    resolved_rows = 0
    stream_bytes = 0
    pending_rows = 0
    peak_hot_bytes = 0
    checksum = 0.0

    started = time.perf_counter()
    for origin in range(M5_ORIGINS + M5_HORIZON + 1):
        due_targets = tuple(target for target in due_index if target < origin)
        for target in sorted(due_targets):
            spans = due_index.pop(target)
            for span in spans:
                resolved_rows += span.points.size
                pending_rows -= span.points.size
                checksum += float(span.points.sum())
        if origin < M5_ORIGINS:
            arrays = _forecast_arrays(series_count, origin)
            batch = pa.record_batch(
                [pa.array(values) for values in arrays],
                names=("series_id", "h", "origin", "target", "point"),
            )
            stream_rows += batch.num_rows
            stream_bytes += batch.nbytes
            points = arrays[-1].reshape(M5_HORIZON, series_count)
            for offset in range(M5_HORIZON):
                target = origin + offset + 1
                due_index[target].append(
                    _PendingSpan(
                        origin=origin,
                        horizon_step=offset + 1,
                        points=points[offset].copy(),
                    )
                )
                pending_rows += series_count
        hot_bytes = sum(
            span.points.nbytes + sys.getsizeof(span)
            for spans in due_index.values()
            for span in spans
        )
        peak_hot_bytes = max(peak_hot_bytes, hot_bytes)
    seconds = time.perf_counter() - started
    return LedgerResult(
        seconds=seconds,
        stream_rows=stream_rows,
        resolved_rows=resolved_rows,
        pending_rows=pending_rows,
        peak_hot_bytes=peak_hot_bytes,
        stream_bytes=stream_bytes,
        checksum=checksum,
    )


def _scalar_conformal(series_count: int) -> ConformalResult:
    partitions = series_count * M5_HORIZON
    states: dict[int, tuple[int, float, float]] = {}
    started = time.perf_counter()
    for cycle in range(CALIBRATION_CYCLES):
        for partition in range(partitions):
            count, mean, level = states.get(partition, (0, 0.0, 0.1))
            residual = ((partition % 17) + cycle) * 0.01
            count += 1
            mean += (residual - mean) / count
            covered = residual <= 0.5
            level = min(0.99, max(0.01, level + 0.01 * (float(covered) - 0.9)))
            states[partition] = count, mean, level
    seconds = time.perf_counter() - started
    sample_count = min(2_048, partitions)
    sample_bytes = sum(
        sys.getsizeof(key) + sys.getsizeof(value) + sum(sys.getsizeof(item) for item in value)
        for key, value in list(states.items())[:sample_count]
    )
    state_bytes = sys.getsizeof(states) + int(sample_bytes * partitions / sample_count)
    return ConformalResult(
        seconds=seconds,
        partitions=partitions,
        state_bytes=state_bytes,
        mean_checksum=sum(value[1] for value in states.values()),
        level_checksum=sum(value[2] for value in states.values()),
    )


def _batched_conformal(series_count: int) -> ConformalResult:
    partitions = series_count * M5_HORIZON
    partition_ids = np.arange(partitions, dtype=np.int32)
    counts = np.zeros(partitions, dtype=np.int32)
    means = np.zeros(partitions, dtype=np.float64)
    levels = np.full(partitions, 0.1, dtype=np.float64)
    started = time.perf_counter()
    for cycle in range(CALIBRATION_CYCLES):
        residuals = ((partition_ids % 17) + cycle).astype(np.float64) * 0.01
        counts += 1
        means += (residuals - means) / counts
        covered = residuals <= 0.5
        np.clip(levels + 0.01 * (covered.astype(np.float64) - 0.9), 0.01, 0.99, out=levels)
    seconds = time.perf_counter() - started
    return ConformalResult(
        seconds=seconds,
        partitions=partitions,
        state_bytes=int(counts.nbytes + means.nbytes + levels.nbytes),
        mean_checksum=float(means.sum()),
        level_checksum=float(levels.sum()),
    )


def _validate(result: ScaleResult) -> None:
    expected_rows = result.series * M5_ORIGINS * M5_HORIZON
    for ledger in (result.scan_ledger, result.indexed_ledger):
        assert ledger.stream_rows == expected_rows
        assert ledger.resolved_rows == expected_rows
        assert ledger.pending_rows == 0
    assert math.isclose(
        result.scan_ledger.checksum,
        result.indexed_ledger.checksum,
        rel_tol=2e-6,
    )
    assert result.scalar_conformal.partitions == result.batched_conformal.partitions
    assert math.isclose(
        result.scalar_conformal.mean_checksum,
        result.batched_conformal.mean_checksum,
        rel_tol=1e-12,
    )
    assert math.isclose(
        result.scalar_conformal.level_checksum,
        result.batched_conformal.level_checksum,
        rel_tol=1e-12,
    )


def _run_scale(series_count: int) -> ScaleResult:
    print(f"\n=== {series_count:,} series ===", flush=True)
    print("ACTION stage immutable history", flush=True)
    scan_panel = _scan_panel(series_count)
    indexed_panel = _indexed_panel(series_count)
    print(
        "STATE history "
        f"rows={scan_panel.history_rows:,} "
        f"scan_copy={_mib(scan_panel.history.owned_bytes):.1f}MiB "
        f"indexed_view={_mib(indexed_panel.history.owned_bytes):.4f}MiB",
        flush=True,
    )

    print("ACTION append forecast batches and resolve due rows", flush=True)
    scan_ledger = _scan_ledger(series_count)
    indexed_ledger = _indexed_ledger(series_count)
    print(
        "STATE ledger "
        f"stream_rows={indexed_ledger.stream_rows:,} "
        f"resolved={indexed_ledger.resolved_rows:,} pending={indexed_ledger.pending_rows:,} "
        f"scan_peak_hot={_mib(scan_ledger.peak_hot_bytes):.1f}MiB "
        f"indexed_peak_hot={_mib(indexed_ledger.peak_hot_bytes):.1f}MiB",
        flush=True,
    )

    print("ACTION update all conformal partitions in canonical order", flush=True)
    scalar_conformal = _scalar_conformal(series_count)
    batched_conformal = _batched_conformal(series_count)
    print(
        "STATE conformal "
        f"partitions={batched_conformal.partitions:,} cycles={CALIBRATION_CYCLES} "
        f"mean_checksum={batched_conformal.mean_checksum:.6f} "
        f"level_checksum={batched_conformal.level_checksum:.6f}",
        flush=True,
    )

    result = ScaleResult(
        series=series_count,
        scan_panel=scan_panel,
        indexed_panel=indexed_panel,
        scan_ledger=scan_ledger,
        indexed_ledger=indexed_ledger,
        scalar_conformal=scalar_conformal,
        batched_conformal=batched_conformal,
    )
    _validate(result)
    return result


def _ratio(numerator: float, denominator: float) -> float:
    return math.inf if denominator == 0 else numerator / denominator


def _mib(value: int) -> float:
    return value / (1024 * 1024)


def _print_summary(results: list[ScaleResult]) -> None:
    print("\n=== comparison ===")
    print(
        "series | history scan/indexed ms | panel scan/indexed MiB | "
        "ledger scan/indexed s | ledger hot scan/indexed MiB | "
        "conformal scalar/batched s | state scalar/batched MiB"
    )
    for result in results:
        print(
            f"{result.series:>6,} | "
            f"{result.scan_panel.history.seconds * 1_000:>8.2f}/"
            f"{result.indexed_panel.history.seconds * 1_000:<8.4f} | "
            f"{_mib(result.scan_panel.build.owned_bytes):>8.1f}/"
            f"{_mib(result.indexed_panel.build.owned_bytes):<8.1f} | "
            f"{result.scan_ledger.seconds:>8.3f}/"
            f"{result.indexed_ledger.seconds:<8.3f} | "
            f"{_mib(result.scan_ledger.peak_hot_bytes):>8.1f}/"
            f"{_mib(result.indexed_ledger.peak_hot_bytes):<8.1f} | "
            f"{result.scalar_conformal.seconds:>8.3f}/"
            f"{result.batched_conformal.seconds:<8.3f} | "
            f"{_mib(result.scalar_conformal.state_bytes):>8.1f}/"
            f"{_mib(result.batched_conformal.state_bytes):<8.1f}"
        )
    largest = results[-1]
    history_speedup = _ratio(
        largest.scan_panel.history.seconds,
        largest.indexed_panel.history.seconds,
    )
    ledger_speedup = _ratio(
        largest.scan_ledger.seconds,
        largest.indexed_ledger.seconds,
    )
    conformal_speedup = _ratio(
        largest.scalar_conformal.seconds,
        largest.batched_conformal.seconds,
    )
    print("\nVERDICT candidate B: indexed columnar batches")
    print(
        "- History is one calendar-aligned immutable value plane plus a presence bitmap; "
        "origin and series slices are zero-copy views."
    )
    print(
        "- Ledger forecast rows are immutable Arrow-shaped append batches; a target-time "
        "index owns only unresolved spans and resolutions append as a side stream."
    )
    print(
        "- Conformal routing uses integer partition IDs and one batch call per origin; "
        "methods update columnar state arrays without a Python call per partition."
    )
    print(
        "- Keep pandas and string labels at ingestion/export adapters, not in the hot "
        "per-origin path."
    )
    print(
        "Largest-scale speedups: "
        f"history={history_speedup:.1f}x, ledger={ledger_speedup:.1f}x, "
        f"conformal={conformal_speedup:.1f}x."
    )


def _jsonable(results: list[ScaleResult]) -> list[dict[str, Any]]:
    return [asdict(result) for result in results]


def main() -> None:
    """Run both candidates and print state after every prototype action."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--series",
        nargs="+",
        type=int,
        default=(1_000, 10_000),
        help="Synthetic panel sizes (default: 1000 10000).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print a final machine-readable Python-literal snapshot.",
    )
    args = parser.parse_args()
    print("PROTOTYPE — no production interfaces or persistence")
    print(
        f"SHAPE days={M5_DAYS} origins={M5_ORIGINS} horizon={M5_HORIZON} "
        f"conformal_cycles={CALIBRATION_CYCLES}"
    )
    results = [_run_scale(series_count) for series_count in args.series]
    _print_summary(results)
    if args.json:
        print(f"\nRESULTS={_jsonable(results)!r}")


if __name__ == "__main__":
    main()
