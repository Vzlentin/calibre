"""Forecast and order ledger protocols plus in-memory and streaming parquet implementations."""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, Protocol, cast, runtime_checkable
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
    """Append-and-close write target for ledger row batches."""

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
    """Derive the ``.resolved.parquet`` artifact URI alongside the stream ``path``."""
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

# Private monotonic append-order stamp on each ds-bucket frame. It reconstructs
# the single-``pd.concat`` append order once pending rows are fragmented across
# per-``ds`` buckets, and is dropped before any frame leaves the store — it never
# reaches the stream sink, the updates side-file, or the conformal observe path.
_APPEND_SEQ = "_append_seq"

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


def _key_dictionaries(datasets: list[pds.Dataset]) -> list[np.ndarray]:
    """Distinct canonical values per key column across ``datasets``.

    Key cardinalities are tiny relative to row counts (tens of thousands of
    ids, dozens of dates/origins/horizons), so the dictionaries fit trivially
    in memory; they are collected with a streamed per-row-group pass so the
    full key columns never do.
    """
    uniques: list[list[np.ndarray]] = [[] for _ in _CANONICAL_KEY_SCHEMA]
    for dataset in datasets:
        for table in _iter_row_group_tables(dataset, list(_FORECAST_KEY_COLUMNS)):
            keys = _cast_keys_to_canonical(table).to_pandas()
            for i, field in enumerate(_CANONICAL_KEY_SCHEMA):
                uniques[i].append(pd.unique(keys[field.name].to_numpy()))
    return [pd.unique(np.concatenate(per_column)) for per_column in uniques]


def _composite_key_codes(dataset: pds.Dataset, dictionaries: list[np.ndarray]) -> np.ndarray:
    """Collapse each row's 5-column key into one exact int64 composite code.

    Each key column is factorized against its (tiny) dictionary and the codes
    are combined with mixed-radix arithmetic, so membership between two key
    sets becomes vectorized int64 comparison instead of a multi-gigabyte
    string hash join — the Acero join on raw key columns is what OOMed the
    60M-row finalize on a 16 GB host. Codes are exact (no hashing), built one
    row group at a time; the only full-length allocations are the int64 code
    arrays themselves (~8 bytes/row).
    """
    radices = [max(len(dictionary), 1) for dictionary in dictionaries]
    capacity = 1
    for radix in radices:
        capacity *= radix
    if capacity >= 2**63:
        raise ValueError(
            "streaming ledger finalize aborted: key cardinality product "
            f"{capacity} overflows the int64 composite key space"
        )
    chunks: list[np.ndarray] = []
    for table in _iter_row_group_tables(dataset, list(_FORECAST_KEY_COLUMNS)):
        keys = _cast_keys_to_canonical(table).to_pandas()
        codes = np.zeros(len(keys), dtype=np.int64)
        for i, field in enumerate(_CANONICAL_KEY_SCHEMA):
            indices = pd.Categorical(keys[field.name], categories=dictionaries[i]).codes
            if (indices < 0).any():
                raise ValueError(f"streaming ledger key column {field.name!r} contains nulls")
            codes = codes * radices[i] + indices.astype(np.int64, copy=False)
        chunks.append(codes)
    if not chunks:
        return np.empty(0, dtype=np.int64)
    return np.concatenate(chunks)


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
    """Union the stream and updates schemas, pinning key columns to canonical types.

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
    """Forecast ledger interface.

    The construction site picks an adapter once; no caller flips a mode after
    the fact.

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
    """Forecast ledger that streams appends to parquet, keeping only pending rows in memory.

    Resolved updates land in a side file and are merged into the finalized
    artifact on ``close``.
    """

    def __init__(self, path: str | Path, *, partition_cols: list[str] | None = None) -> None:
        self._stream_path = str(path)
        self._resolved_path = resolved_ledger_uri(path)
        self._resolved_updates_path = _resolved_updates_uri(path)
        for stale in (self._resolved_path, self._resolved_updates_path):
            fs, fs_path = open_fs(stale)
            with contextlib.suppress(FileNotFoundError):
                fs.rm(fs_path)
        self._sink = _make_sink(self._stream_path, partition_cols)
        # Pending rows bucketed by ``ds`` (each bucket a key-indexed frame
        # carrying the private ``_APPEND_SEQ`` column). Due-ness is ``ds <=
        # origin``, so a per-origin resolve touches only the due buckets — never
        # the future-dated ones. The store is empty iff this dict is empty.
        self._buckets: dict[pd.Timestamp, pd.DataFrame] = {}
        self._seq: int = 0  # monotonic append-seq counter
        self._resolved_update_sink: LedgerSink | None = None

    def append(self, df: pd.DataFrame) -> None:
        validate_forecast_frame(df)
        # Stream the full RAW batch first — the seq is stamped only on the
        # bucketed copy below, so the sink (and thus the stream parquet) never
        # sees ``_APPEND_SEQ``.
        self._sink.append(df)
        pending = df[df[Y].isna()].copy()
        if pending.empty:
            return
        # Stamp the append-seq while ``ds`` is still a plain column and before the
        # key index is set. Positional assignment preserves the row order
        # ``df[df[Y].isna()]`` produced, which equals the single-``pd.concat``
        # order the old store materialised.
        pending[_APPEND_SEQ] = range(self._seq, self._seq + len(pending))
        self._seq += len(pending)
        # Split by ``ds`` on the COLUMN-indexed frame (``ds`` is still a plain
        # column here). Setting the key index first would make ``ds`` both an
        # index level and a column, and ``groupby(DS)`` would then raise the
        # "both an index level and a column label" ambiguity. Split first,
        # key-index each group second.
        # dropna=False keeps a NaT-ds group as its own bucket. The old single
        # frame retained NaT-ds rows in the open set (their ``ds <= origin`` was
        # simply always False); dropping the NaT group here would silently lose
        # them from the open set while still streaming them to the raw sink.
        for ds_raw, group in pending.groupby(DS, sort=False, dropna=False):
            # The DS column is datetime64[ns], so each group key is a Timestamp
            # (or NaT); cast narrows the broad groupby-key type without a runtime
            # coercion. A NaT key is dict-hashable and never satisfies
            # ``ds <= origin``, so its bucket is reachable but never due.
            ds_value = cast(pd.Timestamp, ds_raw)
            sub = group.copy()
            sub.index = _key_index(sub)
            existing = self._buckets.get(ds_value)
            if existing is None:
                self._buckets[ds_value] = sub
                continue
            # Duplicate keys share the full 5-tuple ⇒ share ``ds`` ⇒ land in this
            # bucket, so the duplicate probe is per-bucket. Hash the small new
            # batch against the existing bucket index.
            dup_mask = existing.index.isin(sub.index)
            if dup_mask.any():
                duplicates = existing.index[dup_mask]
                raise ValueError(
                    f"duplicate forecast keys appended to ledger: {list(duplicates)[:10]}"
                )
            self._buckets[ds_value] = pd.concat([existing, sub])

    def _gather_ordered(self, ds_keys: list[pd.Timestamp]) -> pd.DataFrame:
        """Concat the named buckets, restore append order, and drop the seq.

        Shared by :meth:`due_frame` (due subset) and :meth:`to_df` branch 3 (all
        buckets) so the re-sort logic cannot drift between them. The stable sort
        on the strictly-increasing ``_APPEND_SEQ`` reproduces the single-concat
        append order exactly; the seq column is dropped so it never leaves the
        store, and the index is reset to a clean ``RangeIndex``.
        """
        gathered = pd.concat([self._buckets[ds] for ds in ds_keys])
        gathered = gathered.sort_values(_APPEND_SEQ, kind="stable")
        return gathered.drop(columns=[_APPEND_SEQ]).reset_index(drop=True)

    def due_frame(self, origin: pd.Timestamp) -> pd.DataFrame:
        if not self._buckets:
            return pd.DataFrame(columns=REQUIRED_COLUMNS)
        # Pending rows all have y NaN by construction, so due = ds <= origin.
        # Iterate the bucket KEYS only (O(#buckets)); future-dated buckets are
        # never read or copied.
        due_keys = [ds for ds in self._buckets if ds <= origin]
        if not due_keys:
            # Populated store, nothing due: preserve the FULL appended column set
            # (interval/quantile columns beyond REQUIRED_COLUMNS), matching the
            # old ``self._pending.loc[mask]`` empty slice — NOT a
            # REQUIRED_COLUMNS-only frame. The old store was a single concat, so
            # the empty slice carried the UNION of columns across every append;
            # buckets may carry differing column sets (some appends added
            # interval columns, others did not), so take the ordered union across
            # all buckets, not one arbitrary bucket's columns.
            return pd.DataFrame(columns=self._appended_columns())
        return self._gather_ordered(due_keys)

    def _appended_columns(self) -> list[str]:
        """The ordered union of bucket columns (minus ``_APPEND_SEQ``).

        Reproduces the column set the old single-frame ``pd.concat`` store
        exposed: first-seen order across buckets, with the private seq excluded.
        """
        columns: list[str] = []
        for bucket in self._buckets.values():
            for col in bucket.columns:
                if col != _APPEND_SEQ and col not in columns:
                    columns.append(col)
        return columns

    def apply_resolutions(self, df: pd.DataFrame) -> None:
        if df.empty:
            return
        if not self._buckets:
            raise ValueError("apply_resolutions called before any pending append")
        # Known-key validation, bucket by bucket: probe only the buckets the
        # resolution ``df`` names (O(resolved buckets), never the open set). A
        # ``ds`` absent from ``self._buckets`` contributes zero matches, so its
        # keys count as unknown. Open-set keys are globally unique, so the summed
        # matched count equalling ``len(keys)`` proves every input key is known.
        matched = 0
        for ds_raw, group in df.groupby(DS, sort=False, dropna=False):
            bucket = self._buckets.get(cast(pd.Timestamp, ds_raw))
            if bucket is None:
                continue
            matched += int(bucket.index.isin(_key_index(group)).sum())
        # ``len(_key_index(df)) == len(df)`` (one key tuple per row), so the
        # success path needs only the count; build the full key index lazily in
        # the error branch where it is actually scanned.
        if matched != len(df):
            unknown = [key for key in _key_index(df) if not self._key_known(key)]
            raise ValueError(f"apply_resolutions for keys not in the open set: {unknown[:10]}")
        resolved = df[df[Y].notna()].copy()
        if not resolved.empty:
            self._append_resolved_updates(resolved)
            # Keyed upsert: resolved rows leave the open set; still-pending due
            # rows (y still NaN, e.g. incomplete aggregates) stay, never-due
            # buckets are untouched (not a wholesale replace). Filter only the
            # buckets the resolution touches; drop a bucket once it empties.
            for ds_raw, group in resolved.groupby(DS, sort=False, dropna=False):
                ds_value = cast(pd.Timestamp, ds_raw)
                bucket = self._buckets.get(ds_value)
                if bucket is None:
                    continue
                bucket = bucket[~bucket.index.isin(_key_index(group))]
                if bucket.empty:
                    del self._buckets[ds_value]
                else:
                    self._buckets[ds_value] = bucket

    def _key_known(self, key: tuple) -> bool:
        """Whether a 5-tuple forecast ``key`` is in the open set (its ds bucket)."""
        ds_value = pd.Timestamp(key[_FORECAST_KEY_COLUMNS.index(DS)])
        bucket = self._buckets.get(ds_value)
        return bucket is not None and key in bucket.index

    def to_df(self) -> pd.DataFrame:
        if exists(self._resolved_path):
            return pd.read_parquet(self._resolved_path)
        if exists(self._stream_path):
            return self._materialize_streaming_frame()
        # Synthetic-test-only branch: ``append`` always writes the stream sink
        # first, so in production the stream path exists after any append and
        # branch 2 above is taken. Reuse the shared gather helper so the bucket
        # re-sort cannot diverge from ``due_frame``.
        if self._buckets:
            return self._gather_ordered(list(self._buckets.keys()))
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
        """Write ``.resolved.parquet`` row-group at a time, then drop the updates side file.

        Uses the union-schema Arrow membership algorithm.
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
            # Cleanup must never mask the original failure: a raising
            # writer.close() (e.g. disk-full on the footer flush) would
            # otherwise skip the handle close and the partial-artifact removal
            # and replace the root-cause exception.
            if writer is not None:
                with contextlib.suppress(Exception):
                    writer.close()
            if handle is not None:
                with contextlib.suppress(Exception):
                    handle.close()
            with contextlib.suppress(Exception):
                rm(self._resolved_path, recursive=False)
            raise
        if writer is not None:
            writer.close()
        if handle is not None:
            handle.close()
        if exists(self._resolved_updates_path):
            rm(self._resolved_updates_path, recursive=False)

    def _materialize_streaming_frame(self) -> pd.DataFrame:
        """In-memory equivalent of the finalize artifact, for ``to_df()`` before ``close()``.

        Applies the same union-schema membership semantics as
        :meth:`_finalize_resolved_artifact`; only used at small (test) scale,
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
        """Drive the union-schema membership algorithm one source row group at a time.

        Calls ``write_table`` once per source row group (unresolved raw rows,
        then winning updates).
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

        # Factorize the 5-column keys into exact int64 composite codes using the
        # same row-group iterator order used to write the rows, so the boolean
        # masks line up positionally with the streamed row groups regardless of
        # fragment ordering. Membership is then vectorized int64 isin — the
        # earlier Acero string join OOMed at 60M rows on the 16 GB gate host.
        dictionaries = _key_dictionaries([upd_dataset, raw_dataset])
        upd_codes = _composite_key_codes(upd_dataset, dictionaries)

        # Keep-last dedup by ascending stream position: the first occurrence in
        # the reversed array is the last occurrence in stream order.
        first_in_reversed = np.unique(upd_codes[::-1], return_index=True)[1]
        winning_idx = (n_upd - 1) - first_in_reversed
        winning_mask = np.zeros(n_upd, dtype=bool)
        winning_mask[winning_idx] = True
        n_distinct = len(winning_idx)

        raw_codes = _composite_key_codes(raw_dataset, dictionaries)
        resolved_mask = np.isin(raw_codes, upd_codes[winning_mask])
        # Accounting guard: every distinct update key must map onto exactly one
        # raw row. An unmatched update is a resolution for a row never issued —
        # corruption — so abort rather than silently drop it.
        hits = int(resolved_mask.sum())
        if hits != n_distinct:
            raise ValueError(
                "streaming ledger finalize aborted: "
                f"{n_distinct} distinct update keys mapped onto {hits} raw rows "
                "(expected 1:1)"
            )
        del raw_codes, upd_codes, winning_idx, first_in_reversed

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
    """Order ledger that streams non-empty order frames straight to a parquet sink.

    Nothing accumulates in memory.
    """

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
