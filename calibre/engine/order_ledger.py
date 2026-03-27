from __future__ import annotations

from pathlib import Path

import pandas as pd


class OrderLedger:
    """Append-only store for order policy outputs.

    Accumulates per-origin order recommendations produced by an OrderPolicyConfig
    during a BackendEngine walk-forward run.
    """

    def __init__(self) -> None:
        self._frames: list[pd.DataFrame] = []

    def append(self, df: pd.DataFrame) -> None:
        if not df.empty:
            self._frames.append(df.copy())

    def to_df(self) -> pd.DataFrame:
        if not self._frames:
            return pd.DataFrame()
        return pd.concat(self._frames, ignore_index=True)

    def to_parquet(self, path: str | Path) -> None:
        self.to_df().to_parquet(str(path), index=False)
