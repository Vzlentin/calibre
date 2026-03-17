import pandas as pd
import pytest


@pytest.fixture
def weekly_dates():
    """20 weeks of weekly dates starting 2024-01-07."""
    return pd.date_range("2024-01-07", periods=20, freq="W")


@pytest.fixture
def repeating_pattern():
    """Repeating [10, 20, 30, 40] for 20 weeks."""
    return [10.0, 20.0, 30.0, 40.0] * 5
