from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import quote

import fsspec  # type: ignore[import-untyped]
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from calibre.core.forecast_frame import REQUIRED_COLUMNS, validate_forecast_frame
from calibre.execution.io import join_uri, open_fs, write_parquet


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


class _BaseLedger:
    _empty_columns: list[str] = []

    def __init__(self) -> None:
        self._frames: list[pd.DataFrame] = []
        self._stream_sink: LedgerSink | None = None
        self._stream_path: str | None = None
        self._resolved_path: str | None = None
        self._stream_current: pd.DataFrame | None = None

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
        if self._stream_current is None or self._stream_current.empty:
            self._stream_current = df.copy()
        else:
            self._stream_current = pd.concat(
                [self._stream_current, df.copy()],
                ignore_index=True,
            )

    def close(self) -> None:
        if self._stream_sink is not None:
            self._stream_sink.close()

    def to_df(self) -> pd.DataFrame:
        if self._stream_current is not None:
            return self._stream_current.copy()
        if not self._frames:
            return pd.DataFrame(columns=self._empty_columns)
        return pd.concat(self._frames, ignore_index=True)

    def to_parquet(self, path: str | Path) -> None:
        write_parquet(self.to_df(), path)


class ForecastLedger(_BaseLedger):
    _empty_columns = REQUIRED_COLUMNS

    def append(self, df: pd.DataFrame) -> None:
        validate_forecast_frame(df)
        if self.streaming:
            self.append_streaming(df)
            return
        self._frames.append(df)

    def update_resolved(self, df: pd.DataFrame) -> None:
        if self.streaming:
            self._stream_current = df.copy()
            if self._resolved_path is not None:
                write_parquet(df, self._resolved_path)
            return
        self._frames = [df]


class OrderLedger(_BaseLedger):
    def append(self, df: pd.DataFrame) -> None:
        if not df.empty:
            if self.streaming:
                self.append_streaming(df.copy())
                return
            self._frames.append(df.copy())
