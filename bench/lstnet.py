"""The four published LSTNet matrices, without timestamps or covariates.

Electricity contains processed client load (units not asserted), Traffic sensor
occupancy proportions (not vehicle counts), Solar-AL published PV production
(physical units not asserted), and Exchange exchange-rate levels in source order.

Dates starting at 2000-01-01 are positional labels required by Panel, not recovered
source dates. Never use them as calendar features or to join external covariates.
Exchange steps count observations, not calendar days.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from calibre.data import Hierarchy, Panel
from sources import Source

BASE_URL = (
    "https://raw.githubusercontent.com/laiguokun/multivariate-time-series-data/"
    "7f402f185cc2435b5e66aed13a3b560ed142e023"
)


@dataclass(frozen=True)
class MatrixSource(Source):
    """A pinned matrix and the frequency of its positional labels."""

    freq: str


SOURCES = {
    "electricity": MatrixSource(
        "electricity.txt.gz",
        f"{BASE_URL}/electricity/electricity.txt.gz",
        "3c4c069588198c1fcc95cace7bb69c99922129edfd673b7286661dad20badefa",
        "h",
    ),
    "traffic": MatrixSource(
        "traffic.txt.gz",
        f"{BASE_URL}/traffic/traffic.txt.gz",
        "c7be5a00519d344a5ec0eabdbfec5ea0c7dd1eed5f9b1a3843a93bb88086a56d",
        "h",
    ),
    "solar": MatrixSource(
        "solar_AL.txt.gz",
        f"{BASE_URL}/solar-energy/solar_AL.txt.gz",
        "1ae9329c3e1032acbb8dd080bfc397c44507006812a715d2788f172820489a1c",
        "10min",
    ),
    "exchange": MatrixSource(
        "exchange_rate.txt.gz",
        f"{BASE_URL}/exchange_rate/exchange_rate.txt.gz",
        "6a4fbb561dcdee43f071a83badbeec9223a335552278348656738b556f1c2e9d",
        "D",
    ),
}


@dataclass(frozen=True)
class LSTNet:
    """Positional panel, flat hierarchy, and original-target mask [B, T]."""

    panel: Panel
    hierarchy: Hierarchy
    observed: np.ndarray


def load(path: str | Path) -> LSTNet:
    """Read a directory containing exactly one of the four pinned matrix files."""
    path = Path(path)
    matches = [source for source in SOURCES.values() if (path / source.name).is_file()]
    if len(matches) != 1:
        raise ValueError("expected exactly one LSTNet matrix file in the directory")
    (source,) = matches
    source.verify(path / source.name)
    values = np.loadtxt(path / source.name, delimiter=",", dtype=np.float32, ndmin=2)
    periods = pd.date_range("2000-01-01", periods=values.shape[0], freq=source.freq)
    series = np.array([f"column_{i + 1:03d}" for i in range(values.shape[1])])
    panel = Panel(series, periods, values.T, source.freq)
    return LSTNet(panel, Hierarchy.flat(panel.series), np.ones(panel.values.shape, dtype=bool))
