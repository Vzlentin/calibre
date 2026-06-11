from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, Protocol, runtime_checkable
from urllib.parse import quote

import fsspec
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as pds
import pyarrow.fs  # noqa: F401  (registers pa.fs)
import pyarrow.parquet as pq

from calibre.core.forecast_frame import (
    DS,
    FORECAST_ORIGIN,
    MODEL_NAME,
    REQUIRED_COLUMNS,
    UNIQUE_ID,
    H,
    Y,
    validate_forecast_frame,
)
from calibre.core.io import exists, is_local_fs, join_uri, open_fs, rm, write_parquet


class LedgerSink(Protocol):
    def append(self, df: pd.DataFrame) -> None: ...

    def close(self) -> None: ...


class _ParquetLedgerSink:
    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        fs, fs_path = open_fs(self.path)
        with contextlib.suppress(FileNotFoundError):
            fs.rm(fs_path)
        self._handle = None
        self._writer: pq.ParquetWriter | None = None

    def append(self, df: pd.DataFrame) -> None:
        table = pa.Table.from_pandas(df, preserve_index=False)
        if self._writer is None:
            self._handle = fsspec.open(self.path, "wb").open()
            self._writer = pq.ParquetWriter(self._handle, table.schema)
        self._writer.write_table(table)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None
        if self._handle is not None:
            self._handle.close()
            self._handle = None


def _partition_value(value: Any) -> str:
    if pd.isna(value):
        return "__HIVE_DEFAULT_PARTITION__"
    return quote(str(value), safe="")


class _PartitionedParquetLedgerSink:
    def __init__(self, path: str | Path, partition_cols: list[str]) -> None:
        self.path = str(path)
        self.partition_cols = list(partition_cols)
        self._writers: dict[tuple[str, ...], pq.ParquetWriter] = {}
        self._handles: dict[tuple[str, ...], Any] = {}
        fs, fs_path = open_fs(self.path)
        with contextlib.suppress(FileNotFoundError):
            fs.rm(fs_path, recursive=True)
        fs.mkdirs(fs_path, exist_ok=True)

    def append(self, df: pd.DataFrame) -> None:
        missing = [col for col in self.partition_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing partition columns: {missing}")

        grouped = df.groupby(self.partition_cols, sort=False, dropna=False)
        for raw_key, group in grouped:
            key_values = raw_key if isinstance(raw_key, tuple) else (raw_key,)
            key = tuple(_partition_value(value) for value in key_values)
            partition_parts = [
                f"{col}={value}" for col, value in zip(self.partition_cols, key, strict=True)
            ]
            partition_dir = join_uri(self.path, *partition_parts)
            table = pa.Table.from_pandas(
                group.drop(columns=self.partition_cols),
                preserve_index=False,
            )
            if key not in self._writers:
                fs, fs_path = open_fs(partition_dir)
                fs.mkdirs(fs_path, exist_ok=True)
                output_path = join_uri(partition_dir, "part-0.parquet")
                handle = fsspec.open(output_path, "wb").open()
                self._handles[key] = handle
                self._writers[key] = pq.ParquetWriter(handle, table.schema)
            self._writers[key].write_table(table)

    def close(self) -> None:
        for writer in self._writers.values():
            writer.close()
        self._writers.clear()
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()


def resolved_ledger_uri(path: str | Path) -> str:
    text = str(path)
    if "://" not in text:
        return str(Path(text).with_suffix(".resolved.parquet"))
    if text.endswith(".parquet"):
        return f"{text[: -len('.parquet')]}.resolved.parquet"
    return f"{text}.resolved.parquet"


def _resolved_updates_uri(path: str | Path) -> str:
    text = str(path)
    if "://" not in text:
        return str(Path(text).with_suffix(".resolved-updates.parquet"))
    if text.endswith(".parquet"):
        return f"{text[: -len('.parquet')]}.resolved-updates.parquet"
    return f"{text}.resolved-updates.parquet"


_FORECAST_KEY_COLUMNS = [UNIQUE_ID, DS, FORECAST_ORIGIN, MODEL_NAME, H]

# The Arrow membership join hard-raises on any key-dtype divergence (string vs
# large_string, ns vs us timestamps, int vs float, null-typed key columns), so
# both key tables are cast to this one canonical schema before the join. Keeping
# it the single source of truth means dtype drift fails with a named error
# instead of a cryptic Arrow join error.
_CANONICAL_KEY_SCHEMA = pa.schema(
    [
        pa.field(UNIQUE_ID, pa.string()),
        pa.field(DS, pa.timestamp("ns")),
        pa.field(FORECAST_ORIGIN, pa.timestamp("ns")),
        pa.field(MODEL_NAME, pa.string()),
        pa.field(H, pa.int64()),
    ]
)


def _cast_keys_to_canonical(table: pa.Table) -> pa.Table:
    """Cast the 5 key columns of ``table`` to the canonical key schema.

    Raises a named error if a key column is missing, null-typed, or otherwise
    cannot be cast — so dtype drift surfaces here, not as an opaque join error.
    """
    columns = []
    for field in _CANONICAL_KEY_SCHEMA:
        if field.name not in table.column_names:
            raise ValueError(
                f"Cannot resolve streaming ledger without key columns: [{field.name!r}]"
            )
        column = table.column(field.name)
        if pa.types.is_null(column.type):
            raise ValueError(f"streaming ledger key column {field.name!r} is null-typed")
        try:
            columns.append(column.cast(field.type))
        except (pa.ArrowInvalid, pa.ArrowNotImplementedError) as exc:
            raise ValueError(
                f"streaming ledger key column {field.name!r} has incompatible dtype "
                f"{column.type} (cannot cast to {field.type})"
            ) from exc
    return pa.table(columns, schema=_CANONICAL_KEY_SCHEMA)


def _winning_update_rows(update_keys: pa.Table) -> tuple[pa.Table, np.ndarray]:
    """Keep-last dedup of update keys by ascending stream position.

    Returns the distinct (winning) key table and a boolean mask over the update
    rows selecting the last occurrence of each key. Arrow group-bys give no
    stable-order guarantee, so the last occurrence is chosen explicitly by the
    maximum ``__upd_row`` position within each key group.
    """
    n_upd = update_keys.num_rows
    positions = pa.array(np.arange(n_upd, dtype=np.int64))
    with_pos = update_keys.append_column("__upd_row", positions)
    winners = (
        with_pos.group_by(_FORECAST_KEY_COLUMNS)
        .aggregate([("__upd_row", "max")])
        .column("__upd_row_max")
        .to_numpy()
    )
    winning_mask = np.zeros(n_upd, dtype=bool)
    winning_mask[winners] = True
    distinct_keys = with_pos.filter(pa.array(winning_mask)).drop_columns(["__upd_row"])
    return distinct_keys, winning_mask


def _pa_filesystem(uri: str) -> tuple[Any, str]:
    """Return a pyarrow-compatible filesystem and path for ``uri``.

    Local paths use pyarrow's native LocalFileSystem; fsspec protocols
    (``memory://``, ``s3://`` …) are wrapped with ``FSSpecHandler`` so the same
    dataset/ParquetWriter code path works regardless of backend.
    """
    fs, path = open_fs(uri)
    if is_local_fs(fs):
        return pa.fs.LocalFileSystem(), path
    return pa.fs.PyFileSystem(pa.fs.FSSpecHandler(fs)), path


def _iter_row_group_tables(
    dataset: pds.Dataset, columns: list[str] | None = None
) -> Iterator[pa.Table]:
    """Yield one table per row group across all fragments of ``dataset``.

    Streaming a row group at a time keeps finalize memory bounded — the full
    panel is never read into memory at once. ``columns`` restricts the read to
    the key columns for the membership join.
    """
    for fragment in dataset.get_fragments():
        for row_group in fragment.split_by_row_group():
            yield row_group.to_table(schema=dataset.schema, columns=columns)


def _union_schema(raw_schema: pa.Schema, updates_schema: pa.Schema) -> pa.Schema:
    """Union of the stream and updates schemas, with key columns pinned to the
    canonical key types.

    The writer schema must carry every column present on either side: the raw
    stream lacks the updates-only ``error``/``abs_error``/``pct_error`` columns,
    while both sides may carry ``nonconformity_score``. Raw column order is kept
    first (so the resolved artifact's columns line up with the stream), then any
    updates-only columns are appended in updates order. Key columns are taken
    from the canonical schema so both row-group sides cast to identical types.
    """
    canonical = {field.name: field.type for field in _CANONICAL_KEY_SCHEMA}
    fields: list[pa.Field] = []
    seen: set[str] = set()
    for field in raw_schema:
        field_type = canonical.get(field.name, field.type)
        fields.append(pa.field(field.name, field_type))
        seen.add(field.name)
    for field in updates_schema:
        if field.name in seen:
            continue
        field_type = canonical.get(field.name, field.type)
        fields.append(pa.field(field.name, field_type))
        seen.add(field.name)
    return pa.schema(fields)


def _align_to_schema(table: pa.Table, schema: pa.Schema) -> pa.Table:
    """Null-fill columns missing from ``table`` and cast it to ``schema``.

    Used per row group on both sides: raw row groups are null-filled for the
    updates-only columns; update row groups are column-aligned and cast. Done
    one row group at a time — never on a concatenated table.
    """
    columns = []
    for field in schema:
        if field.name in table.column_names:
            columns.append(table.column(field.name).cast(field.type))
        else:
            columns.append(pa.nulls(table.num_rows, type=field.type))
    return pa.table(columns, schema=schema)


def _make_sink(path: str, partition_cols: list[str] | None) -> LedgerSink:
    if partition_cols:
        return _PartitionedParquetLedgerSink(path, partition_cols)
    return _ParquetLedgerSink(path)


@runtime_checkable
class Ledger(Protocol):
    """Forecast ledger interface. The construction site picks an adapter once;
    no caller flips a mode after the fact.

    The ledger owns the resolve contract: ``due_frame(origin)`` hands the engine
    only the rows due for resolution as of ``origin`` (a RangeIndex copy), and
    ``apply_resolutions(df)`` performs a keyed upsert of those rows back into the
    open set — resolved rows leave it, still-pending due rows stay, and rows not
    in the due frame are untouched. The engine never pushes a mutated full
    pending frame.
    """

    def append(self, df: pd.DataFrame) -> None: ...

    def due_frame(self, origin: pd.Timestamp) -> pd.DataFrame: ...

    def apply_resolutions(self, df: pd.DataFrame) -> None: ...

    def to_df(self) -> pd.DataFrame: ...

    def to_parquet(self, path: str | Path) -> None: ...

    def close(self) -> None: ...


@runtime_checkable
class OrderLedger(Protocol):
    """Order ledger interface — append-only; no resolution merge."""

    def append(self, df: pd.DataFrame) -> None: ...

    def to_df(self) -> pd.DataFrame: ...

    def to_parquet(self, path: str | Path) -> None: ...

    def close(self) -> None: ...


def _key_index(df: pd.DataFrame) -> pd.MultiIndex:
    """Build the 5-tuple forecast key MultiIndex for ``df``."""
    return pd.MultiIndex.from_frame(df[_FORECAST_KEY_COLUMNS])


class InMemoryLedger:
    """Forecast ledger that accumulates rows in memory in append order.

    ``apply_resolutions`` updates matching keys *in place* (preserving each row's
    append position) rather than re-concatenating open and resolved rows — so
    ``to_df()`` stays byte-identical, in both values and row order, to the
    pre-keyed-contract path (the VN2 regression gate rides on this)."""

    def __init__(self) -> None:
        self._frame: pd.DataFrame | None = None
        self._positions: dict[tuple, int] = {}

    def append(self, df: pd.DataFrame) -> None:
        validate_forecast_frame(df)
        new_keys = list(_key_index(df))
        duplicates = [key for key in new_keys if key in self._positions]
        if duplicates:
            raise ValueError(f"duplicate forecast keys appended to ledger: {duplicates[:10]}")
        start = 0 if self._frame is None else len(self._frame)
        self._frame = df.copy() if self._frame is None else pd.concat([self._frame, df])
        self._frame = self._frame.reset_index(drop=True)
        for offset, key in enumerate(new_keys):
            self._positions[key] = start + offset

    def due_frame(self, origin: pd.Timestamp) -> pd.DataFrame:
        if self._frame is None:
            return pd.DataFrame(columns=REQUIRED_COLUMNS)
        mask = self._frame[Y].isna() & (self._frame[DS] <= origin)
        return self._frame.loc[mask].reset_index(drop=True)

    def apply_resolutions(self, df: pd.DataFrame) -> None:
        if df.empty:
            return
        if self._frame is None:
            raise ValueError("apply_resolutions called before any append")
        keys = list(_key_index(df))
        unknown = [key for key in keys if key not in self._positions]
        if unknown:
            raise ValueError(f"apply_resolutions for keys not in the open set: {unknown[:10]}")
        # The accumulated frame always carries a RangeIndex (append() resets it),
        # so each append position is also its index label.
        positions = [self._positions[key] for key in keys]
        # Update each resolved row in place at its append position, column by
        # column so each column keeps its own dtype (a single .to_numpy() block
        # assign would coerce a mixed frame to object and drift dtypes the VN2
        # gate depends on). New (updates-only) columns are created NaN-filled
        # first, then written on the resolved positions.
        for col in df.columns:
            if col not in self._frame.columns:
                self._frame[col] = np.nan
            self._frame.loc[positions, col] = df[col].to_numpy()

    def to_df(self) -> pd.DataFrame:
        if self._frame is None:
            return pd.DataFrame(columns=REQUIRED_COLUMNS)
        return self._frame.copy()

    def to_parquet(self, path: str | Path) -> None:
        write_parquet(self.to_df(), path)

    def close(self) -> None:
        return None


class StreamingLedger:
    """Forecast ledger that streams every append to a parquet sink and keeps
    only the still-pending (unresolved) rows in memory. Resolved updates land in
    a side file and are merged into the finalized artifact on ``close``."""

    def __init__(self, path: str | Path, *, partition_cols: list[str] | None = None) -> None:
        self._stream_path = str(path)
        self._resolved_path = resolved_ledger_uri(path)
        self._resolved_updates_path = _resolved_updates_uri(path)
        for stale in (self._resolved_path, self._resolved_updates_path):
            fs, fs_path = open_fs(stale)
            with contextlib.suppress(FileNotFoundError):
                fs.rm(fs_path)
        self._sink = _make_sink(self._stream_path, partition_cols)
        self._pending: pd.DataFrame | None = None
        self._resolved_update_sink: LedgerSink | None = None

    def append(self, df: pd.DataFrame) -> None:
        validate_forecast_frame(df)
        self._sink.append(df)
        pending = df[df[Y].isna()].copy()
        if pending.empty:
            return
        pending.index = _key_index(pending)
        if self._pending is None:
            self._pending = pending
            return
        duplicates = pending.index[pending.index.isin(self._pending.index)]
        if len(duplicates):
            raise ValueError(f"duplicate forecast keys appended to ledger: {list(duplicates)[:10]}")
        self._pending = pd.concat([self._pending, pending])

    def due_frame(self, origin: pd.Timestamp) -> pd.DataFrame:
        if self._pending is None:
            return pd.DataFrame(columns=REQUIRED_COLUMNS)
        # Pending rows all have y NaN by construction, so due = ds <= origin.
        due = self._pending.loc[self._pending[DS] <= origin]
        return due.reset_index(drop=True)

    def apply_resolutions(self, df: pd.DataFrame) -> None:
        if df.empty:
            return
        keys = _key_index(df)
        if self._pending is None:
            raise ValueError("apply_resolutions called before any pending append")
        unknown = keys[~keys.isin(self._pending.index)]
        if len(unknown):
            raise ValueError(
                f"apply_resolutions for keys not in the open set: {list(unknown)[:10]}"
            )
        resolved = df[df[Y].notna()].copy()
        if not resolved.empty:
            self._append_resolved_updates(resolved)
            # Keyed upsert: resolved rows leave the open set; still-pending due
            # rows (y still NaN, e.g. incomplete aggregates) stay, and rows that
            # were never in the due frame are untouched (not a wholesale replace).
            resolved_keys = _key_index(resolved)
            # isin keeps this O(pending) hash-probe instead of drop()'s
            # super-linear MultiIndex path — the open set is ~26M rows at M5.
            self._pending = self._pending[~self._pending.index.isin(resolved_keys)]
            if self._pending.empty:
                self._pending = None

    def to_df(self) -> pd.DataFrame:
        if exists(self._resolved_path):
            return pd.read_parquet(self._resolved_path)
        if exists(self._stream_path):
            return self._materialize_streaming_frame()
        if self._pending is not None:
            return self._pending.reset_index(drop=True)
        return pd.DataFrame(columns=REQUIRED_COLUMNS)

    def to_parquet(self, path: str | Path) -> None:
        write_parquet(self.to_df(), path)

    def close(self) -> None:
        self._sink.close()
        if self._resolved_update_sink is not None:
            self._resolved_update_sink.close()
            self._resolved_update_sink = None
        self._finalize_resolved_artifact()

    def _append_resolved_updates(self, df: pd.DataFrame) -> None:
        # No-null-regression invariant: each resolved update row is a full ledger
        # row derived from its raw row — resolution only *fills* columns (y plus
        # error/score columns), it never nulls a column that was non-null in the
        # raw stream. That is what makes finalize's wholesale-row write equal the
        # old per-column ``combine_first``: there is never a column where the raw
        # value is non-null and the update value is NaN, so "the update row wins"
        # and "the non-null value wins" coincide. A future change that nulled a
        # column on resolution would silently break that equivalence.
        if self._resolved_update_sink is None:
            self._resolved_update_sink = _ParquetLedgerSink(self._resolved_updates_path)
        self._resolved_update_sink.append(df)

    def _finalize_resolved_artifact(self) -> None:
        """Write ``.resolved.parquet`` via the union-schema Arrow membership
        algorithm, row-group at a time, and remove the updates side file.

        No full-frame pandas materialization happens here: the 60M-row stream is
        never read into memory at once. Resolved-row membership is decided by a
        key-only Arrow left-outer join; the raw and updates parquet files are
        then copied into a single ``ParquetWriter`` one row group at a time. The
        writer schema is the *union* of the stream and updates schemas (the raw
        stream lacks the updates-only ``error``/``abs_error``/``pct_error``
        columns), and the resolved-row value wins on any column the two sides
        share (e.g. ``nonconformity_score`` is NaN in the raw stream and
        real-valued in the update).
        """
        if not exists(self._stream_path):
            return
        writer: pq.ParquetWriter | None = None
        handle: Any = None

        def _write(table: pa.Table) -> None:
            # The output handle/writer are opened lazily on the first written row
            # group: an accounting abort fires before any write, so a failed
            # finalize leaves no partial .resolved.parquet on disk.
            nonlocal writer, handle
            if writer is None:
                handle = fsspec.open(self._resolved_path, "wb").open()
                writer = pq.ParquetWriter(handle, table.schema)
            writer.write_table(table)

        try:
            self._stream_resolved(_write)
        except Exception:
            if writer is not None:
                writer.close()
            if handle is not None:
                handle.close()
            rm(self._resolved_path, recursive=False)
            raise
        if writer is not None:
            writer.close()
        if handle is not None:
            handle.close()
        if exists(self._resolved_updates_path):
            rm(self._resolved_updates_path, recursive=False)

    def _materialize_streaming_frame(self) -> pd.DataFrame:
        """In-memory equivalent of the finalize artifact, for ``to_df()`` before
        ``close()`` has run. Applies the same union-schema membership semantics
        as :meth:`_finalize_resolved_artifact`; only used at small (test) scale,
        never on the M5 finalize path.
        """
        if not exists(self._stream_path):
            return pd.DataFrame(columns=REQUIRED_COLUMNS)
        tables: list[pa.Table] = []
        self._stream_resolved(tables.append)
        if not tables:
            return pd.read_parquet(self._stream_path)
        return pa.concat_tables(tables).to_pandas()

    def _stream_resolved(self, write_table: Callable[[pa.Table], None]) -> None:
        """Drive the union-schema membership algorithm, calling ``write_table``
        once per source row group (unresolved raw rows, then winning updates).

        The membership join is key-only (constant per-row-group memory); the row
        data itself is copied a row group at a time, so neither the raw stream
        nor the updates file is ever held in memory in full. ``pyarrow.dataset``
        reads the raw stream so a single-file or Hive-partitioned directory
        stream (and any fsspec backend) are handled the same way.
        """
        raw_fs, raw_path = _pa_filesystem(self._stream_path)
        raw_dataset = pds.dataset(raw_path, filesystem=raw_fs, partitioning="hive")
        raw_schema = raw_dataset.schema
        n_raw = raw_dataset.count_rows()

        # Zero-updates short-circuit: no resolution ever fired, so the resolved
        # artifact is a row-group-streamed copy of the raw stream (no join).
        if not exists(self._resolved_updates_path) or n_raw == 0:
            for table in _iter_row_group_tables(raw_dataset):
                if table.num_rows:
                    write_table(table)
            return

        upd_fs, upd_path = _pa_filesystem(self._resolved_updates_path)
        upd_dataset = pds.dataset(upd_path, filesystem=upd_fs)
        upd_schema = upd_dataset.schema
        n_upd = upd_dataset.count_rows()
        if n_upd == 0:
            for table in _iter_row_group_tables(raw_dataset):
                if table.num_rows:
                    write_table(table)
            return

        # Both sides must carry every key column before any key read — surface a
        # named error here rather than an opaque Arrow "No match for FieldRef".
        missing = [
            col
            for col in _FORECAST_KEY_COLUMNS
            if col not in raw_schema.names or col not in upd_schema.names
        ]
        if missing:
            raise ValueError(f"Cannot resolve streaming ledger without key columns: {missing}")

        # Build the key tables from the same row-group iterator used to write the
        # rows, so the boolean masks line up positionally with the streamed
        # row groups regardless of fragment ordering.
        update_keys = _cast_keys_to_canonical(
            pa.concat_tables(_iter_row_group_tables(upd_dataset, _FORECAST_KEY_COLUMNS))
        )
        distinct_keys, winning_mask = _winning_update_rows(update_keys)
        n_distinct = distinct_keys.num_rows

        raw_keys = _cast_keys_to_canonical(
            pa.concat_tables(_iter_row_group_tables(raw_dataset, _FORECAST_KEY_COLUMNS))
        )
        raw_keys = raw_keys.append_column("__row", pa.array(np.arange(n_raw, dtype=np.int64)))
        distinct_keys = distinct_keys.append_column(
            "__hit", pa.array(np.ones(n_distinct, dtype=np.int8))
        )

        joined = raw_keys.join(distinct_keys, keys=_FORECAST_KEY_COLUMNS, join_type="left outer")
        # __hit is null for raw rows with no matching update; valid rows resolved.
        hit_rows = joined.filter(joined.column("__hit").is_valid()).column("__row").to_numpy()
        # Accounting guard: every distinct update key must map onto exactly one
        # raw row. An unmatched update is a resolution for a row never issued —
        # corruption — so abort rather than silently drop it.
        if len(hit_rows) != n_distinct:
            raise ValueError(
                "streaming ledger finalize aborted: "
                f"{n_distinct} distinct update keys mapped onto {len(hit_rows)} raw rows "
                "(expected 1:1)"
            )
        resolved_mask = np.zeros(n_raw, dtype=bool)
        resolved_mask[hit_rows] = True
        del joined, raw_keys, distinct_keys, hit_rows

        union_schema = _union_schema(raw_schema, upd_schema)

        offset = 0
        kept = 0
        for table in _iter_row_group_tables(raw_dataset):
            block = ~resolved_mask[offset : offset + table.num_rows]
            offset += table.num_rows
            unresolved = table.filter(pa.array(block))
            if unresolved.num_rows:
                write_table(_align_to_schema(unresolved, union_schema))
                kept += unresolved.num_rows

        # Stream updates writing only winning rows (keep-last per key).
        written = 0
        offset = 0
        for table in _iter_row_group_tables(upd_dataset):
            block = winning_mask[offset : offset + table.num_rows]
            offset += table.num_rows
            winners = table.filter(pa.array(block))
            if winners.num_rows:
                write_table(_align_to_schema(winners, union_schema))
                written += winners.num_rows

        # Accounting guard: kept (unresolved raw) + written (winning updates)
        # must reconstruct the raw row count exactly.
        if kept + written != n_raw:
            raise ValueError(
                "streaming ledger finalize aborted: "
                f"{kept} unresolved + {written} resolved != {n_raw} raw rows"
            )


class InMemoryOrderLedger:
    """Order ledger that accumulates non-empty order frames in memory."""

    def __init__(self) -> None:
        self._frames: list[pd.DataFrame] = []

    def append(self, df: pd.DataFrame) -> None:
        if not df.empty:
            self._frames.append(df.copy())

    def to_df(self) -> pd.DataFrame:
        if not self._frames:
            return pd.DataFrame(columns=[])
        return pd.concat(self._frames, ignore_index=True)

    def to_parquet(self, path: str | Path) -> None:
        write_parquet(self.to_df(), path)

    def close(self) -> None:
        return None


class StreamingOrderLedger:
    """Order ledger that streams non-empty order frames straight to a parquet
    sink; nothing accumulates in memory."""

    def __init__(self, path: str | Path, *, partition_cols: list[str] | None = None) -> None:
        self._stream_path = str(path)
        self._sink = _make_sink(self._stream_path, partition_cols)

    def append(self, df: pd.DataFrame) -> None:
        if not df.empty:
            self._sink.append(df.copy())

    def to_df(self) -> pd.DataFrame:
        if exists(self._stream_path):
            return pd.read_parquet(self._stream_path)
        return pd.DataFrame(columns=[])

    def to_parquet(self, path: str | Path) -> None:
        write_parquet(self.to_df(), path)

    def close(self) -> None:
        self._sink.close()
