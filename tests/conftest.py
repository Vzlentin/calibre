import subprocess
from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture(autouse=True)
def _isolate_mlflow(tmp_path, monkeypatch):
    """Redirect MLflow tracking to a per-test tmp directory."""
    monkeypatch.setenv("MLFLOW_TRACKING_URI", str(tmp_path / "mlruns"))


def _find_data_dir() -> Path:
    """Locate the data/vn2/ directory, handling git worktrees."""
    # First try the standard location relative to this file
    candidate = Path(__file__).parent.parent / "data" / "vn2"
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
        candidate = project_root / "data" / "vn2"
        if candidate.is_dir():
            return candidate
    except subprocess.CalledProcessError:
        pass
    raise FileNotFoundError(
        f"Cannot find data/vn2/ directory. Tried {Path(__file__).parent.parent / 'data' / 'vn2'}"
    )


@pytest.fixture(scope="session")
def data_dir() -> Path:
    """Session-scoped fixture to locate the data/ directory."""
    return _find_data_dir()


@pytest.fixture
def period0_sales_path(data_dir: Path) -> Path:
    return data_dir / "week_0_sales.csv"


@pytest.fixture
def master_path(data_dir: Path) -> Path:
    return data_dir / "week_0_master.csv"


@pytest.fixture
def instock_path(data_dir: Path) -> Path:
    return data_dir / "week_0_in_stock.csv"


@pytest.fixture
def dates():
    """20 date points starting 2024-01-07, weekly frequency."""
    return pd.date_range("2024-01-07", periods=20, freq="W")


@pytest.fixture
def repeating_pattern():
    """Repeating [10, 20, 30, 40] for 20 periods."""
    return [10.0, 20.0, 30.0, 40.0] * 5
