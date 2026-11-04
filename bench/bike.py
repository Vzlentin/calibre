"""UCI Bike Sharing: hourly recorded rentals (cnt), not unconstrained demand.

No covariates are supplied; casual + registered reveals the target itself.
"""

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
        "bike.zip",
        "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip",
        "b70182d0d0508e9abbb79306ce5c0cec34869000f8220175ac83d11dbe845401",
    ),
)


@dataclass(frozen=True)
class Bike:
    """Rental panel and flat hierarchy; observed [B, T] excludes imputed targets."""

    panel: Panel
    hierarchy: Hierarchy
    observed: np.ndarray
    warmup_periods_dropped: int


def load(path: str | Path) -> Bike:
    """Verify the archive and causally forward-fill absent calendar hours.

    Missing hours are not zero rentals. Trim unfillable leading periods and exclude
    targets marked False in observed from calibration and evaluation.
    """
    path = Path(path)
    (source,) = SOURCES
    source.verify(path / source.name)
    with ZipFile(path / source.name) as archive, archive.open("hour.csv") as stream:
        frame = pd.read_csv(stream)
    periods = pd.to_datetime(frame["dteday"]) + pd.to_timedelta(frame["hr"], unit="h")
    wide, observed, dropped = forward_fill(frame[["cnt"]].set_axis(periods), "h")
    panel = Panel(
        wide.columns.to_numpy(), pd.DatetimeIndex(wide.index), wide.to_numpy(np.float32).T, "h"
    )
    return Bike(panel, Hierarchy.flat(panel.series), observed, dropped)
