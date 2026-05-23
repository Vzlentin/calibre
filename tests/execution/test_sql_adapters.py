from __future__ import annotations

import pandas as pd

from calibre.core.forecast_frame import DS, FORECAST_ORIGIN, UNIQUE_ID, Y
from calibre.storage.adapters import OrderRepo, SqlInventoryAdapter, SqlSalesAdapter
from calibre.storage.models import Base, InventorySnapshot, SalesRecord
from calibre.storage.postgres import make_engine, make_session_factory, session_scope


def _factory(tmp_path):
    engine = make_engine(f"sqlite+pysqlite:///{(tmp_path / 'data-plane.db').as_posix()}")
    Base.metadata.create_all(engine)
    return make_session_factory(engine)


def test_sql_inventory_adapter_round_trip(tmp_path) -> None:
    factory = _factory(tmp_path)
    with session_scope(factory) as session:
        session.add_all(
            [
                InventorySnapshot(
                    tenant="acme",
                    unique_id="A",
                    as_of=pd.Timestamp("2024-01-01").to_pydatetime(),
                    end_inventory=10.0,
                    pipeline=[1.0, 2.0],
                    lead_time_depth=2,
                    cumulative_costs={"holding": 3.0},
                ),
                InventorySnapshot(
                    tenant="acme",
                    unique_id="A",
                    as_of=pd.Timestamp("2024-01-08").to_pydatetime(),
                    end_inventory=12.0,
                    pipeline=[3.0, 4.0],
                    lead_time_depth=2,
                    cumulative_costs={"holding": 5.0, "shortage": 1.0},
                ),
            ]
        )
    adapter = SqlInventoryAdapter(factory, tenant="acme")

    state = adapter.load_state("A", pd.Timestamp("2024-01-09"))

    assert state.unique_id == "A"
    assert state.end_inventory == 12.0
    assert list(state.pipeline) == [3.0, 4.0]
    assert state.cumulative_costs == {"holding": 5.0, "shortage": 1.0}
    assert adapter.load_lead_times() == {"A": 2}


def test_sql_sales_adapter_loads_parquet(tmp_path) -> None:
    sales_path = tmp_path / "sales.parquet"
    pd.DataFrame(
        {
            UNIQUE_ID: ["B", "A"],
            DS: ["2024-01-14", "2024-01-07"],
            Y: [2, 1],
        }
    ).to_parquet(sales_path)
    adapter = SqlSalesAdapter()

    history = adapter.load_history(sales_path)

    assert history[UNIQUE_ID].tolist() == ["A", "B"]
    assert history[Y].tolist() == [1.0, 2.0]
    assert pd.api.types.is_datetime64_any_dtype(history[DS])


def test_sql_sales_adapter_loads_sql_rows(tmp_path) -> None:
    factory = _factory(tmp_path)
    with session_scope(factory) as session:
        session.add_all(
            [
                SalesRecord(
                    tenant="acme",
                    unique_id="B",
                    ds=pd.Timestamp("2024-01-14").to_pydatetime(),
                    y=2.0,
                    payload={"channel": "retail"},
                ),
                SalesRecord(
                    tenant="acme",
                    unique_id="A",
                    ds=pd.Timestamp("2024-01-07").to_pydatetime(),
                    y=1.0,
                    payload={},
                ),
            ]
        )
    adapter = SqlSalesAdapter(factory, tenant="acme")

    history = adapter.load_history()

    assert history[[UNIQUE_ID, Y]].to_dict(orient="records") == [
        {UNIQUE_ID: "A", Y: 1.0},
        {UNIQUE_ID: "B", Y: 2.0},
    ]
    assert "channel" in history.columns


def test_order_repo_persists_order_frame(tmp_path) -> None:
    factory = _factory(tmp_path)
    frame = pd.DataFrame(
        {
            UNIQUE_ID: ["A"],
            FORECAST_ORIGIN: [pd.Timestamp("2024-01-07")],
            "order_qty": [5.0],
        }
    )
    repo = OrderRepo(factory)

    repo.append_frame(tenant="acme", session_id="session-1", frame=frame)
    loaded = repo.list_for_session("session-1", tenant="acme", unique_id="A")

    assert loaded[UNIQUE_ID].tolist() == ["A"]
    assert loaded["order_qty"].tolist() == [5.0]
