"""Tests for the M5 dataset adapter."""

from __future__ import annotations

from pathlib import Path

import pytest

from calibre.core.order_types import CostStruct
from calibre.execution.m5_adapter import M5DatasetAdapter
from calibre.execution.validation import validate_dataset_bundle

_FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "m5"


def test_m5_adapter_loads_bundle() -> None:
    bundle = M5DatasetAdapter().load(_FIXTURE)
    assert bundle.hierarchy is not None
    assert bundle.future_x is None
    assert bundle.censoring is None
    assert isinstance(bundle.costs, CostStruct)

    history_uids = set(bundle.history["unique_id"].astype(str))
    hierarchy_uids = set(bundle.hierarchy["unique_id"].astype(str))
    assert history_uids <= hierarchy_uids
    for col in ("item_id", "dept_id", "cat_id", "store_id", "state_id"):
        assert col in bundle.hierarchy.columns

    validate_dataset_bundle(bundle)


def test_m5_adapter_missing_calendar_raises(tmp_path: Path) -> None:
    sales = _FIXTURE / "sales_train_evaluation.csv"
    (tmp_path / "sales_train_evaluation.csv").write_text(sales.read_text(), encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="calendar.csv"):
        M5DatasetAdapter().load(tmp_path)
