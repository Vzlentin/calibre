"""Synthetic panels for the test suite."""

import numpy as np
import pandas as pd

from calibre.hierarchy import Hierarchy
from calibre.panel import Panel


def make_panel(n_series: int, n_periods: int, seed: int = 0) -> tuple[Panel, Hierarchy]:
    """Non-negative integer counts with a weekly pattern, dept and store attributes."""
    rng = np.random.default_rng(seed)
    series = np.array([f"s{i:05d}" for i in range(n_series)])
    periods = pd.date_range("2020-01-01", periods=n_periods, freq="D")
    weekly = 1.0 + 0.5 * np.sin(2 * np.pi * np.arange(n_periods) / 7)
    values = rng.poisson(3.0 * weekly, size=(n_series, n_periods)).astype(np.float32)
    panel = Panel(series=series, periods=periods, values=values, freq="D")
    attributes = pd.DataFrame(
        {
            "dept": [f"d{i % 7}" for i in range(n_series)],
            "store": [f"st{i % 10}" for i in range(n_series)],
        },
        index=series,
    )
    return panel, Hierarchy.from_attributes(series, attributes)
