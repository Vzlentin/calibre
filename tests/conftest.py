import subprocess
from pathlib import Path

import pandas as pd
import pytest


def _find_data_dir() -> Path:
    """Locate the data/ directory, handling git worktrees."""
    # First try the standard location relative to this file
    candidate = Path(__file__).parent.parent / "data"
    if candidate.is_dir():
        return candidate
    # In a git worktree, data/ lives in the main project root
    try:
        common_dir = subprocess.check_output(
            ["git", "rev-parse", "--git-common-dir"],
            cwd=Path(__file__).parent,
            text=True,
        ).strip()
        # common_dir is e.g. C:/path/to/repo/.git — go up one level for project root
        project_root = Path(common_dir).parent
        candidate = project_root / "data"
        if candidate.is_dir():
            return candidate
    except subprocess.CalledProcessError:
        pass
    raise FileNotFoundError(f"Cannot find data/ directory. Tried {Path(__file__).parent.parent / 'data'}")


@pytest.fixture(scope="session")
def data_dir() -> Path:
    """Session-scoped fixture to locate the data/ directory."""
    return _find_data_dir()


@pytest.fixture
def weekly_dates():
    """20 weeks of weekly dates starting 2024-01-07."""
    return pd.date_range("2024-01-07", periods=20, freq="W")


@pytest.fixture
def repeating_pattern():
    """Repeating [10, 20, 30, 40] for 20 weeks."""
    return [10.0, 20.0, 30.0, 40.0] * 5
