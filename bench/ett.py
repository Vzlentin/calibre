"""ETT-small ETTh1: hourly published oil temperature (OT), no covariates."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from calibre.data import Hierarchy, Panel
from sources import Source

SOURCES = (
    Source(
        "ETTh1.csv",
        "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/"
        "1d16c8f4f943005d613b5bc962e9eeb06058cf07/ETT-small/ETTh1.csv",
        "f18de3ad269cef59bb07b5438d79bb3042d3be49bdeecf01c1cd6d29695ee066",
    ),
)


@dataclass(frozen=True)
class ETT:
    """Temperature panel, flat hierarchy, and original-target mask [B, T]."""

    panel: Panel
    hierarchy: Hierarchy
    observed: np.ndarray


def load(path: str | Path) -> ETT:
    """Read the hash-verified ETTh1 release from a directory, without imputation."""
    path = Path(path)
    (source,) = SOURCES
    source.verify(path / source.name)
    frame = pd.read_csv(path / source.name, parse_dates=["date"])
    wide = frame.set_index("date")[["OT"]].sort_index()
    panel = Panel(
        wide.columns.to_numpy(), pd.DatetimeIndex(wide.index), wide.to_numpy(np.float32).T, "h"
    )
    return ETT(panel, Hierarchy.flat(panel.series), np.ones(panel.values.shape, dtype=bool))
