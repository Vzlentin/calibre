from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import quote

import fsspec  # type: ignore[import-untyped]
import pandas as pd
import pyarrow as pa
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
from calibre.execution.io import exists, join_uri, open_fs, write_parquet


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


class _BaseLedger:
    _empty_columns: list[str] = []

    def __init__(self) -> None:
        self._frames: list[pd.DataFrame] = []
        self._stream_sink: LedgerSink | None = None
        self._stream_path: str | None = None
        self._resolved_path: str | None = None

    def stream_to(
        self,
        path: str | Path,
        *,
        partition_cols: list[str] | None = None,
    ) -> None:
        self._stream_path = str(path)
        self._resolved_path = resolved_ledger_uri(path)
        fs, fs_path = open_fs(self._resolved_path)
        with contextlib.suppress(FileNotFoundError):
            fs.rm(fs_path)
        self._stream_sink = (
            _PartitionedParquetLedgerSink(self._stream_path, partition_cols)
            if partition_cols
            else _ParquetLedgerSink(self._stream_path)
        )

    @property
    def streaming(self) -> bool:
        return self._stream_sink is not None

    def append_streaming(self, df: pd.DataFrame) -> None:
        if self._stream_sink is None:
            raise RuntimeError("Call stream_to(path) before append_streaming(df)")
        self._stream_sink.append(df)

    def close(self) -> None:
        if self._stream_sink is not None:
            self._stream_sink.close()

    def to_df(self) -> pd.DataFrame:
        if self.streaming and self._stream_path is not None and exists(self._stream_path):
            return pd.read_parquet(self._stream_path)
        if not self._frames:
            return pd.DataFrame(columns=self._empty_columns)
        return pd.concat(self._frames, ignore_index=True)

    def to_parquet(self, path: str | Path) -> None:
        write_parquet(self.to_df(), path)


class ForecastLedger(_BaseLedger):
    _empty_columns = REQUIRED_COLUMNS

    def __init__(self) -> None:
        super().__init__()
        self._pending: pd.DataFrame | None = None
        self._resolved_updates_path: str | None = None
        self._resolved_update_sink: LedgerSink | None = None

    def stream_to(
        self,
        path: str | Path,
        *,
        partition_cols: list[str] | None = None,
    ) -> None:
        super().stream_to(path, partition_cols=partition_cols)
        self._resolved_updates_path = _resolved_updates_uri(path)
        fs, fs_path = open_fs(self._resolved_updates_path)
        with contextlib.suppress(FileNotFoundError):
            fs.rm(fs_path)

    def append(self, df: pd.DataFrame) -> None:
        validate_forecast_frame(df)
        if self.streaming:
            self.append_streaming(df)
            pending = df[df[Y].isna()].copy()
            if not pending.empty:
                self._pending = (
                    pending
                    if self._pending is None or self._pending.empty
                    else pd.concat([self._pending, pending], ignore_index=True)
                )
            return
        self._frames.append(df)

    def resolution_frame(self) -> pd.DataFrame:
        if self.streaming:
            if self._pending is None:
                return pd.DataFrame(columns=self._empty_columns)
            return self._pending.copy()
        return self.to_df()

    def update_resolved(self, df: pd.DataFrame) -> None:
        if self.streaming:
            resolved = df[df[Y].notna()].copy()
            if not resolved.empty:
                self._append_resolved_updates(resolved)
            pending = df[df[Y].isna()].copy()
            self._pending = pending if not pending.empty else None
            return
        self._frames = [df]

    def close(self) -> None:
        super().close()
        if self._resolved_update_sink is not None:
            self._resolved_update_sink.close()
            self._resolved_update_sink = None
        self._finalize_resolved_artifact()

    def to_df(self) -> pd.DataFrame:
        if self.streaming:
            if self._resolved_path is not None and exists(self._resolved_path):
                return pd.read_parquet(self._resolved_path)
            if self._stream_path is not None and exists(self._stream_path):
                return self._materialize_streaming_frame()
            if self._pending is not None:
                return self._pending.copy()
            return pd.DataFrame(columns=self._empty_columns)
        return super().to_df()

    def _append_resolved_updates(self, df: pd.DataFrame) -> None:
        if self._resolved_updates_path is None:
            return
        if self._resolved_update_sink is None:
            self._resolved_update_sink = _ParquetLedgerSink(self._resolved_updates_path)
        self._resolved_update_sink.append(df)

    def _finalize_resolved_artifact(self) -> None:
        if (
            self._stream_path is None
            or self._resolved_path is None
            or not exists(self._stream_path)
        ):
            return
        write_parquet(self._materialize_streaming_frame(), self._resolved_path)
        if self._resolved_updates_path is not None and exists(self._resolved_updates_path):
            fs, fs_path = open_fs(self._resolved_updates_path)
            with contextlib.suppress(FileNotFoundError):
                fs.rm(fs_path)

    def _materialize_streaming_frame(self) -> pd.DataFrame:
        if self._stream_path is None or not exists(self._stream_path):
            return pd.DataFrame(columns=self._empty_columns)
        raw = pd.read_parquet(self._stream_path)
        if (
            self._resolved_updates_path is None
            or not exists(self._resolved_updates_path)
            or raw.empty
        ):
            return raw
        updates = pd.read_parquet(self._resolved_updates_path)
        if updates.empty:
            return raw
        missing = [col for col in _FORECAST_KEY_COLUMNS if col not in raw or col not in updates]
        if missing:
            raise ValueError(f"Cannot resolve streaming ledger without key columns: {missing}")
        updates = updates.drop_duplicates(_FORECAST_KEY_COLUMNS, keep="last")
        update_cols = [col for col in updates.columns if col not in _FORECAST_KEY_COLUMNS]
        merged = raw.merge(
            updates[_FORECAST_KEY_COLUMNS + update_cols],
            on=_FORECAST_KEY_COLUMNS,
            how="left",
            suffixes=("", "__resolved"),
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


class OrderLedger(_BaseLedger):
    def append(self, df: pd.DataFrame) -> None:
        if not df.empty:
            if self.streaming:
                self.append_streaming(df.copy())
                return
            self._frames.append(df.copy())
