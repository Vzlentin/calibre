"""UCI PRSA2017: hourly PM2.5 (micrograms/m3) for twelve Beijing stations."""

from dataclasses import dataclass
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

from calibre.data import Hierarchy, Panel
from imputation import forward_fill
from sources import Source

SOURCES = (
    Source(
        "beijing.zip",
        "https://archive.ics.uci.edu/ml/machine-learning-databases/00501/"
        "PRSA2017_Data_20130301-20170228.zip",
        "d1b9261c54132f04c374f762f1e5e512af19f95c95fd6bfa1e8ac7e927e3b0b8",
    ),
)


@dataclass(frozen=True)
class Beijing:
    """Station panel and flat hierarchy; observed [B, T] excludes imputed targets."""

    panel: Panel
    hierarchy: Hierarchy
    observed: np.ndarray
    warmup_periods_dropped: int


def load(path: str | Path) -> Beijing:
    """Verify the archive and causally forward-fill missing PM2.5/calendar cells.

    Trim leading periods until every station has an observation. Never back-fill;
    exclude targets marked False in observed from calibration and evaluation.
    """
    path = Path(path)
    (source,) = SOURCES
    source.verify(path / source.name)
    frames = []
    with ZipFile(path / source.name) as archive:
        for member in sorted(name for name in archive.namelist() if name.endswith(".csv")):
            with archive.open(member) as stream:
                frame = pd.read_csv(stream)
            frame["period"] = pd.to_datetime(frame[["year", "month", "day", "hour"]])
            frames.append(frame.rename(columns={"station": "series", "PM2.5": "value"}))
    wide = pd.concat(frames, ignore_index=True).pivot(
        index="period", columns="series", values="value"
    )
    wide, observed, dropped = forward_fill(wide, "h")
    panel = Panel(
        wide.columns.to_numpy(), pd.DatetimeIndex(wide.index), wide.to_numpy(np.float32).T, "h"
    )
    return Beijing(panel, Hierarchy.flat(panel.series), observed, dropped)
