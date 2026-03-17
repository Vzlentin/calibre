from __future__ import annotations

import pandas as pd

from calibre.contracts.forecast_frame import REQUIRED_COLUMNS, validate_forecast_frame


class Ledger:
    def __init__(self) -> None:
        self._frames: list[pd.DataFrame] = []

    def append(self, df: pd.DataFrame) -> None:
        validate_forecast_frame(df)
        self._frames.append(df)

    def to_df(self) -> pd.DataFrame:
        if not self._frames:
            return pd.DataFrame(columns=REQUIRED_COLUMNS)
        return pd.concat(self._frames, ignore_index=True)

    def update_resolved(self, df: pd.DataFrame) -> None:
        self._frames = [df]

    def to_parquet(self, path: str) -> None:
        self.to_df().to_parquet(path, index=False)
