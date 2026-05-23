from __future__ import annotations

from pathlib import Path

from benchmarks.vn2.dataset import VN2DatasetAdapter
from calibre.core.forecast_frame import DS, UNIQUE_ID, Y

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "vn2"


def test_data_loading_round_trip() -> None:
    bundle = VN2DatasetAdapter().load(DATA_DIR, period=0)

    assert not bundle.history.empty
    assert {UNIQUE_ID, DS, Y}.issubset(bundle.history.columns)
    assert bundle.costs.holding_cost is not None
    assert bundle.censoring is not None and not bundle.censoring.empty
