"""Tests for calibre.pipeline.loading."""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pandas as pd
import pytest

from calibre.pipeline.loading import load_master, load_week, melt_wide_instock, melt_wide_sales

STORE_PRODUCT_PAIRS = 599
WEEK0_DATE_COLS = 157
WEEK1_DATE_COLS = 158


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


@pytest.fixture
def data_dir() -> Path:
    return _find_data_dir()


@pytest.fixture
def week0_sales_path(data_dir: Path) -> Path:
    return data_dir / "Week 0 - 2024-04-08 - Sales.csv"


@pytest.fixture
def master_path(data_dir: Path) -> Path:
    return data_dir / "Week 0 - Master.csv"


@pytest.fixture
def instock_path(data_dir: Path) -> Path:
    return data_dir / "Week 0 - In Stock.csv"


class TestMeltWideSales:
    def test_shape(self, week0_sales_path: Path) -> None:
        df = melt_wide_sales(week0_sales_path)
        assert df.shape == (STORE_PRODUCT_PAIRS * WEEK0_DATE_COLS, 3)

    def test_columns(self, week0_sales_path: Path) -> None:
        df = melt_wide_sales(week0_sales_path)
        assert list(df.columns) == ["unique_id", "ds", "y"]

    def test_ds_dtype(self, week0_sales_path: Path) -> None:
        df = melt_wide_sales(week0_sales_path)
        assert pd.api.types.is_datetime64_any_dtype(df["ds"])

    def test_y_dtype(self, week0_sales_path: Path) -> None:
        df = melt_wide_sales(week0_sales_path)
        assert df["y"].dtype == "float64"

    def test_unique_id_dtype(self, week0_sales_path: Path) -> None:
        df = melt_wide_sales(week0_sales_path)
        assert df["unique_id"].dtype == object

    def test_unique_id_format(self, week0_sales_path: Path) -> None:
        df = melt_wide_sales(week0_sales_path)
        pattern = re.compile(r"^\d+_\d+$")
        sample = df["unique_id"].drop_duplicates().head(20)
        assert all(pattern.match(uid) for uid in sample)

    def test_sorted_by_unique_id_then_ds(self, week0_sales_path: Path) -> None:
        df = melt_wide_sales(week0_sales_path)
        expected = df.sort_values(["unique_id", "ds"]).reset_index(drop=True)
        pd.testing.assert_frame_equal(df.reset_index(drop=True), expected)

    def test_accepts_string_path(self, week0_sales_path: Path) -> None:
        df = melt_wide_sales(str(week0_sales_path))
        assert df.shape == (STORE_PRODUCT_PAIRS * WEEK0_DATE_COLS, 3)


class TestLoadMaster:
    def test_columns(self, master_path: Path) -> None:
        df = load_master(master_path)
        expected_original = ["Store", "Product", "ProductGroup", "Division", "Department", "DepartmentGroup", "StoreFormat", "Format"]
        for col in expected_original:
            assert col in df.columns
        assert "unique_id" in df.columns

    def test_row_count(self, master_path: Path) -> None:
        df = load_master(master_path)
        assert len(df) == STORE_PRODUCT_PAIRS

    def test_unique_id_format(self, master_path: Path) -> None:
        df = load_master(master_path)
        pattern = re.compile(r"^\d+_\d+$")
        assert all(pattern.match(uid) for uid in df["unique_id"])

    def test_total_column_count(self, master_path: Path) -> None:
        df = load_master(master_path)
        # 8 original columns + unique_id
        assert len(df.columns) == 9


class TestMeltWideInstock:
    def test_columns(self, instock_path: Path) -> None:
        df = melt_wide_instock(instock_path)
        assert list(df.columns) == ["unique_id", "ds", "in_stock"]

    def test_in_stock_dtype(self, instock_path: Path) -> None:
        df = melt_wide_instock(instock_path)
        assert df["in_stock"].dtype == bool

    def test_ds_dtype(self, instock_path: Path) -> None:
        df = melt_wide_instock(instock_path)
        assert pd.api.types.is_datetime64_any_dtype(df["ds"])

    def test_row_count(self, instock_path: Path) -> None:
        df = melt_wide_instock(instock_path)
        # 599 pairs * 165 date cols (167 total cols - 2 id cols)
        assert df.shape[0] == STORE_PRODUCT_PAIRS * (167 - 2)

    def test_unique_id_format(self, instock_path: Path) -> None:
        df = melt_wide_instock(instock_path)
        pattern = re.compile(r"^\d+_\d+$")
        sample = df["unique_id"].drop_duplicates().head(20)
        assert all(pattern.match(uid) for uid in sample)


class TestLoadWeek:
    def test_week0_matches_melt_wide_sales(self, data_dir: Path, week0_sales_path: Path) -> None:
        from_load_week = load_week(data_dir, 0)
        from_melt = melt_wide_sales(week0_sales_path)
        pd.testing.assert_frame_equal(
            from_load_week.reset_index(drop=True),
            from_melt.reset_index(drop=True),
        )

    def test_week1_has_one_more_date(self, data_dir: Path) -> None:
        df = load_week(data_dir, 1)
        assert df.shape == (STORE_PRODUCT_PAIRS * WEEK1_DATE_COLS, 3)

    def test_accepts_string_path(self, data_dir: Path) -> None:
        df = load_week(str(data_dir), 0)
        assert df.shape == (STORE_PRODUCT_PAIRS * WEEK0_DATE_COLS, 3)
