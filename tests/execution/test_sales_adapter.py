from __future__ import annotations

from datetime import datetime

import pandas as pd

from calibre.execution.dataset import (
    SnapshotSalesAdapter,
    SyntheticSalesAdapter,
)
from calibre.storage.models import Base, Sales
from calibre.storage.postgres import make_engine, make_session_factory, session_scope
from calibre.storage.sales_repo import SqlSalesAdapter


def test_snapshot_returns_source_frame_for_sku_set(tmp_path) -> None:
    frame = pd.DataFrame(
        {
            "unique_id": ["A", "A", "B"],
            "ds": ["2024-01-07", "2024-01-14", "2024-01-07"],
            "y": [1.0, 2.0, 3.0],
        }
    )
    path = tmp_path / "sales.parquet"
    frame.to_parquet(path)

    out = SnapshotSalesAdapter(path).load_sales(["A"], None)

    assert out["unique_id"].tolist() == ["A", "A"]
    assert out["y"].tolist() == [1.0, 2.0]
    assert pd.api.types.is_datetime64_any_dtype(out["ds"])


def test_snapshot_applies_as_of_point_in_time(tmp_path) -> None:
    # (A, 2024-01-07) has two revisions; (A, 2024-01-14) was recorded later.
    frame = pd.DataFrame(
        {
            "unique_id": ["A", "A", "A"],
            "ds": pd.to_datetime(["2024-01-07", "2024-01-07", "2024-01-14"]),
            "y": [1.0, 1.5, 2.0],
            "as_of": pd.to_datetime(["2024-01-01", "2024-01-10", "2024-01-10"]),
        }
    )
    path = tmp_path / "sales.parquet"
    frame.to_parquet(path)
    adapter = SnapshotSalesAdapter(path)

    early = adapter.load_sales(["A"], pd.Timestamp("2024-01-05"))
    assert "as_of" not in early.columns  # revision marker never reaches the forecaster
    assert early["ds"].tolist() == [pd.Timestamp("2024-01-07")]
    assert early["y"].tolist() == [1.0]  # only the first revision is visible yet

    late = adapter.load_sales(["A"], pd.Timestamp("2024-01-10"))
    assert late.set_index("ds")["y"].to_dict() == {
        pd.Timestamp("2024-01-07"): 1.5,  # latest revision at/under the cutoff
        pd.Timestamp("2024-01-14"): 2.0,
    }


def test_snapshot_null_as_of_survives_cutoff(tmp_path) -> None:
    # (A, 01-07) has a revision under the cutoff plus a later one above it;
    # (B, 01-07) carries a NULL as_of that must stay visible regardless of the
    # cutoff. This is the M1 regression: the old ``as_of <= cutoff`` filter
    # silently dropped NULL rows because ``NaT <= cutoff`` is False.
    frame = pd.DataFrame(
        {
            "unique_id": ["A", "A", "B"],
            "ds": pd.to_datetime(["2024-01-07", "2024-01-07", "2024-01-07"]),
            "y": [1.0, 9.0, 3.0],
            "as_of": pd.to_datetime(["2024-01-01", "2024-01-20", None]),
        }
    )
    path = tmp_path / "sales.parquet"
    frame.to_parquet(path)
    adapter = SnapshotSalesAdapter(path)

    out = adapter.load_sales(["A", "B"], pd.Timestamp("2024-01-10"))

    assert "as_of" not in out.columns  # revision marker never reaches the forecaster
    by_uid = out.set_index("unique_id")["y"].to_dict()
    # A: the 01-20 revision is above the cutoff, so the 01-01 revision wins.
    assert by_uid["A"] == 1.0
    # B: NULL as_of is always visible (treated as the latest revision).
    assert by_uid["B"] == 3.0


def test_synthetic_filters_to_sku_set() -> None:
    frame = pd.DataFrame(
        {
            "unique_id": ["A", "B"],
            "ds": pd.to_datetime(["2024-01-07", "2024-01-07"]),
            "y": [1.0, 2.0],
        }
    )
    out = SyntheticSalesAdapter(frame).load_sales(["A"], None)
    assert out["unique_id"].tolist() == ["A"]
    assert out["y"].tolist() == [1.0]


def test_sql_sales_adapter_point_in_time(tmp_path) -> None:
    db_url = f"sqlite+pysqlite:///{(tmp_path / 'sales.db').as_posix()}"
    engine = make_engine(db_url)
    Base.metadata.create_all(engine)
    factory = make_session_factory(engine)
    with session_scope(factory) as session:
        session.add_all(
            [
                Sales(unique_id="A", ds=datetime(2024, 1, 7), y=1.0, as_of=datetime(2024, 1, 1)),
                Sales(unique_id="A", ds=datetime(2024, 1, 14), y=2.0, as_of=datetime(2024, 1, 20)),
                Sales(unique_id="B", ds=datetime(2024, 1, 7), y=3.0, as_of=None),
            ]
        )
    adapter = SqlSalesAdapter(factory)

    # (A, 01-14) was recorded as_of 01-20, after the cutoff -> excluded.
    out = adapter.load_sales(["A"], pd.Timestamp("2024-01-10"))
    assert out["ds"].tolist() == [pd.Timestamp("2024-01-07")]
    assert out["y"].tolist() == [1.0]

    # NULL as_of rows are always visible regardless of the cutoff.
    out_b = adapter.load_sales(["B"], pd.Timestamp("2024-01-05"))
    assert out_b["y"].tolist() == [3.0]
