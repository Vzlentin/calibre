from __future__ import annotations

from pathlib import Path
from typing import Protocol

import fsspec  # type: ignore[import-untyped]
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from calibre.core.forecast_frame import REQUIRED_COLUMNS, validate_forecast_frame


class LedgerSink(Protocol):
    def append(self, df: pd.DataFrame) -> None: ...

    def close(self) -> None: ...


class _ParquetLedgerSink:
    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        fs, fs_path = fsspec.core.url_to_fs(self.path)
        parent = str(Path(fs_path).parent).replace("\\", "/")
        if parent and parent != ".":
            fs.mkdirs(parent, exist_ok=True)
        if fs.exists(fs_path):
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


def _resolved_uri(path: str | Path) -> str:
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
        if partition_cols:
            raise NotImplementedError("partitioned streaming ledgers are not implemented yet")
        self._stream_path = str(path)
        self._resolved_path = _resolved_uri(path)
        fs, fs_path = fsspec.core.url_to_fs(self._resolved_path)
        if fs.exists(fs_path):
            fs.rm(fs_path)
        self._stream_sink = _ParquetLedgerSink(self._stream_path)

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
        self.to_df().to_parquet(str(path), index=False)


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
                df.to_parquet(self._resolved_path, index=False)
            return
        self._frames = [df]


class OrderLedger(_BaseLedger):
    """Append-only store for order policy outputs.

    Accumulates per-origin order recommendations produced by an OrderPolicyConfig
    during a BackendEngine walk-forward run.
    """

    def append(self, df: pd.DataFrame) -> None:
        if not df.empty:
            if self.streaming:
                self.append_streaming(df.copy())
                return
            self._frames.append(df.copy())
