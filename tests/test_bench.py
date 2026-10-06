import numpy as np
import pandas as pd
import pytest

from calibre import AbsoluteResidual, SplitQuantile, WindowSum
from calibre.forecast.origins import Forecasts
from calibre_bench import m5, vn2
from calibre_bench.compare import compare
from calibre_bench.inventory import order_up_to, settle


def test_settle_receives_sells_then_adds_the_order_at_the_end_of_the_pipeline():
    on_hand, pipeline, week = settle(
        np.array([2.0, 0.0]),
        np.array([[3.0, 1.0], [0.0, 4.0]]),
        demand=np.array([4.0, 2.0]),
        order=np.array([5.0, 6.0]),
        holding=0.2,
        shortage=1.0,
    )
    # Series 0: 2 + 3 in stock, sells 4, keeps 1. Series 1: nothing in stock, misses 2.
    assert week.sales.tolist() == [4, 0] and week.missed.tolist() == [0, 2]
    assert on_hand.tolist() == [1, 0]
    assert pipeline.tolist() == [[1, 5], [4, 6]]
    assert np.allclose(week.holding_cost, [0.2, 0]) and week.shortage_cost.tolist() == [0, 2]
    # Position is stock plus pipeline: 1 + 6 = 7, so a bound of 10 orders 3.
    assert order_up_to(np.array([10.0, 5.0]), on_hand, pipeline).tolist() == [3, 0]


def write_vn2(path, weekly: np.ndarray, in_stock: np.ndarray) -> None:
    weeks = pd.date_range(
        end=vn2.FIRST_SETTLED_WEEK + pd.Timedelta(weeks=7), periods=weekly.shape[1], freq="W-MON"
    )
    keys = pd.DataFrame({"Store": [0, 0], "Product": [1, 2]})
    names = weeks.strftime("%Y-%m-%d")
    pd.concat([keys, pd.DataFrame(weekly, columns=names)], axis=1).to_csv(
        path / "week_8_sales.csv", index=False
    )
    history = names[: in_stock.shape[1]]
    pd.concat([keys, pd.DataFrame(in_stock, columns=history)], axis=1).to_csv(
        path / "week_0_in_stock.csv", index=False
    )
    master = keys.assign(
        ProductGroup=[10, 11], Division=[1, 1], Department=[1, 1], DepartmentGroup=[1, 1]
    )
    master.assign(StoreFormat=1, Format=1).to_csv(path / "week_0_master.csv", index=False)
    state = keys.assign(
        **{"End Inventory": [3, 0], "In Transit W+1": [0, 1], "In Transit W+2": [2, 0]}
    )
    state.to_csv(path / "week_0_initial_state.csv", index=False)


def test_vn2_loads_the_release_and_plays_six_orders(tmp_path):
    weekly = np.full((2, 12), 2.0)
    weekly[1, :2] = np.nan  # not listed yet
    in_stock = np.ones((2, 4), dtype=bool)
    in_stock[0, 1] = False
    write_vn2(tmp_path, weekly, in_stock)
    data = vn2.load(tmp_path)
    assert data.panel.values[1, :2].tolist() == [0, 0]
    assert data.panel.censored[0].tolist() == [False, True] + [False] * 10
    assert data.decisions.tolist() == [3, 4, 5, 6, 7, 8]
    # One store and one division equal the total, and each group has one series.
    assert list(data.hierarchy.nodes) == ["0_1", "0_2", "total"]
    assert data.demand().shape == (2, 8)
    # Order up to 6 each time against a steady demand of 2 per week.
    played = vn2.play(data, np.full((6, 2), 6.0))
    assert played.orders[0].tolist() == [1, 5]
    assert played.total_cost == pytest.approx(played.holding_cost + played.shortage_cost)
    with pytest.raises(ValueError, match="finite"):
        vn2.play(data, np.full((6, 2), np.inf))


def test_compare_reports_ready_cells_only_per_method_and_label():
    rng = np.random.default_rng(0)
    actuals = rng.normal(size=(2, 80)).astype(np.float32)
    origins = np.arange(10, 70)
    forecasts = Forecasts(origins, np.zeros((len(origins), 2, 3), dtype=np.float32))
    methods = {"small": SplitQuantile(window=10), "all": SplitQuantile()}
    table, runs = compare(forecasts, actuals, methods, AbsoluteResidual(), 0.8, by=np.array([0, 1]))
    assert set(table.columns) == {"ready", "coverage", "width", "interval_score"}
    assert table.index.tolist() == [("small", 0), ("small", 1), ("all", 0), ("all", 1)]
    assert np.isfinite(table["width"]).all() and (table["ready"] < 1).all()
    one_sided, _ = compare(forecasts, actuals, methods, WindowSum(2), 0.8)
    assert set(one_sided.columns) == {"ready", "coverage", "pinball"}
    assert set(runs) == {"small", "all"}


def write_m5(path) -> None:
    days = [f"d_{day}" for day in range(1, 15)]
    sales = pd.DataFrame(
        {
            "item_id": ["A_1_001", "A_1_001", "B_1_001"],
            "dept_id": ["A_1", "A_1", "B_1"],
            "cat_id": ["A", "A", "B"],
            "store_id": ["CA_1", "TX_1", "CA_1"],
            "state_id": ["CA", "TX", "CA"],
        }
    )
    sales = pd.concat([sales, pd.DataFrame(np.ones((3, 14)), columns=days)], axis=1)
    sales.to_csv(path / "sales_train_evaluation.csv", index=False)
    calendar = pd.DataFrame(
        {
            "date": pd.date_range("2011-01-29", periods=21).strftime("%Y-%m-%d"),
            "wm_yr_wk": np.repeat([1, 2, 3], 7),
        }
    )
    calendar.to_csv(path / "calendar.csv", index=False)
    prices = pd.DataFrame(
        {
            "store_id": ["CA_1", "CA_1", "TX_1", "CA_1"],
            "item_id": ["A_1_001", "A_1_001", "A_1_001", "B_1_001"],
            "wm_yr_wk": [2, 3, 1, 1],
            "sell_price": [2.0, 3.0, 5.0, 7.0],
        }
    )
    prices.to_csv(path / "sell_prices.csv", index=False)


def test_m5_loads_sales_hierarchy_and_filled_prices(tmp_path):
    write_m5(tmp_path)
    data = m5.load(tmp_path)
    assert list(data.panel.series) == ["A_1_001_CA_1", "A_1_001_TX_1", "B_1_001_CA_1"]
    assert "state_id=CA" in data.hierarchy.nodes and "total" in data.hierarchy.nodes
    price = data.prices(horizon=7).values
    assert price.shape == (3, 21)
    # Week 1 takes the first listed price, 2.0. TX and B have one price each.
    assert price[0, :7].tolist() == [2.0] * 7 and price[0, -1] == 3.0
    assert (price[1] == 5.0).all() and (price[2] == 7.0).all()
    with pytest.raises(ValueError, match="calendar"):
        data.prices(horizon=8)
