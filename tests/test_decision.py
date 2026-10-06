import numpy as np
import pytest

from calibre.decision import critical_ratio, order_up_to, settle


def test_settle_receives_sells_then_adds_the_order_at_the_end_of_the_pipeline():
    on_hand, pipeline, settled = settle(
        np.array([2.0, 0.0]),
        np.array([[3.0, 1.0], [0.0, 4.0]]),
        demand=np.array([4.0, 2.0]),
        order=np.array([5.0, 6.0]),
        holding=0.2,
        shortage=1.0,
    )
    # Series 0: 2 + 3 in stock, sells 4, keeps 1. Series 1: nothing in stock, misses 2.
    assert settled.sales.tolist() == [4, 0] and settled.missed.tolist() == [0, 2]
    assert on_hand.tolist() == [1, 0]
    assert pipeline.tolist() == [[1, 5], [4, 6]]
    assert np.allclose(settled.holding_cost, [0.2, 0])
    assert settled.shortage_cost.tolist() == [0, 2]


def test_settle_rejects_unknown_demand():
    with pytest.raises(ValueError, match="finite"):
        settle(np.zeros(1), np.zeros((1, 1)), np.array([np.nan]), np.zeros(1), 0.2, 1.0)


def test_order_up_to_fills_the_gap_to_the_bound_and_never_orders_negative():
    on_hand, in_transit = np.array([1.0, 0.0]), np.array([[1.0, 5.0], [4.0, 6.0]])
    # Positions are 1 + 6 = 7 and 0 + 10 = 10.
    assert order_up_to(np.array([10.0, 5.0]), on_hand, in_transit).tolist() == [3, 0]


def test_critical_ratio_is_the_newsvendor_level_per_node():
    assert critical_ratio(0.2, 1.0) == pytest.approx(1 / 1.2)
    np.testing.assert_allclose(critical_ratio(np.array([1.0, 3.0]), 1.0), [0.5, 0.25])
    with pytest.raises(ValueError, match="positive"):
        critical_ratio(0.0, 1.0)
