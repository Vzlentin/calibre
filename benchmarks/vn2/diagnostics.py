"""VN2 cost-attribution and oracle diagnostics."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from functools import cache
from pathlib import Path
from typing import Any

import pandas as pd

from benchmarks.vn2.config import (
    CONFORMAL_ORDER_CONFIG,
    DATA_DIR,
    DECISION_ROUNDS,
    DELIVERY_WEEKS,
    HORIZON,
    LEAD_TIME,
    REVIEW_PERIOD,
)
from benchmarks.vn2.replay import (
    ReplayResult,
    build_replay_cache,
    replay_cached_cost,
    summary_from_simulator,
)
from benchmarks.vn2.simulator import ProductState, VN2Simulator
from calibre.conformal.cumulative_risk import CumulativeConformalRiskConfig
from calibre.core.forecast_frame import UNIQUE_ID


def _optimal_order_path_for_sku(
    initial_state: ProductState,
    demand_by_week: Mapping[int, float],
    *,
    decision_rounds: int,
    total_weeks: int,
    order_step: float = 1.0,
    lead_time: int = LEAD_TIME,
) -> dict[int, float]:
    """Exact finite-horizon oracle for one SKU under VN2 lead-time mechanics."""
    if order_step <= 0:
        raise ValueError("order_step must be positive")
    if lead_time != LEAD_TIME:
        raise ValueError(f"VN2 oracle currently supports lead_time={LEAD_TIME}, got {lead_time}")

    if total_weeks - decision_rounds <= lead_time:
        return _just_in_time_order_path_for_sku(
            initial_state,
            demand_by_week,
            decision_rounds=decision_rounds,
            total_weeks=total_weeks,
            order_step=order_step,
            lead_time=lead_time,
        )

    demands = tuple(float(demand_by_week.get(week, 0.0)) for week in range(1, total_weeks + 1))
    choices: dict[tuple[int, float, float, float], float] = {}

    def _key(
        week: int, end_inventory: float, p1: float, p2: float
    ) -> tuple[int, float, float, float]:
        return (week, round(end_inventory, 6), round(p1, 6), round(p2, 6))

    @cache
    def _dp(week: int, end_inventory: float, p1: float, p2: float) -> float:
        if week > total_weeks:
            return 0.0

        demand = demands[week - 1]
        if week <= decision_rounds:
            arrival_week = week + 2
            remaining_after_arrival = (
                sum(demands[arrival_week - 1 :]) if arrival_week <= total_weeks else 0.0
            )
            max_units = int(math.ceil(max(0.0, remaining_after_arrival) / order_step))
            order_values: Iterable[float] = (unit * order_step for unit in range(max_units + 1))
        else:
            order_values = (0.0,)

        best_cost = float("inf")
        best_order = 0.0
        for order in order_values:
            arrivals = p1
            start_inventory = end_inventory + arrivals
            sales = min(start_inventory, demand)
            missed_sales = demand - sales
            next_end_inventory = start_inventory - sales
            period_cost = (
                VN2Simulator.HOLDING_COST_RATE * next_end_inventory
                + VN2Simulator.SHORTAGE_COST_RATE * missed_sales
            )
            future_cost = _dp(
                week + 1,
                round(next_end_inventory, 6),
                round(p2, 6),
                round(order, 6),
            )
            total_cost = period_cost + future_cost
            if total_cost < best_cost:
                best_cost = total_cost
                best_order = float(order)

        choices[_key(week, end_inventory, p1, p2)] = best_order
        return best_cost

    week = 1
    end_inventory = float(initial_state.end_inventory)
    p1 = float(initial_state.in_transit_w1)
    p2 = float(initial_state.in_transit_w2)
    _dp(week, round(end_inventory, 6), round(p1, 6), round(p2, 6))

    orders: dict[int, float] = {}
    while week <= total_weeks:
        order = choices.get(_key(week, end_inventory, p1, p2), 0.0)
        if week <= decision_rounds:
            orders[week] = order

        demand = demands[week - 1]
        arrivals = p1
        start_inventory = end_inventory + arrivals
        sales = min(start_inventory, demand)
        end_inventory = round(start_inventory - sales, 6)
        next_p2 = round(order, 6) if week <= decision_rounds else 0.0
        p1, p2 = round(p2, 6), next_p2
        week += 1

    return orders


def _advance_state_values(
    end_inventory: float,
    p1: float,
    p2: float,
    demand: float,
    order: float,
) -> tuple[float, float, float]:
    start_inventory = float(end_inventory) + float(p1)
    sales = min(start_inventory, float(demand))
    next_end_inventory = start_inventory - sales
    return next_end_inventory, float(p2), float(order)


def _round_order_to_step(quantity: float, order_step: float) -> float:
    if quantity <= 0.0:
        return 0.0
    return float(math.ceil(quantity / order_step) * order_step)


def _just_in_time_order_path_for_sku(
    initial_state: ProductState,
    demand_by_week: Mapping[int, float],
    *,
    decision_rounds: int,
    total_weeks: int,
    order_step: float,
    lead_time: int,
) -> dict[int, float]:
    """Fast exact VN2 oracle when delivery horizon is no longer than lead time.

    With weekly review, no order cost, and ``delivery_weeks <= lead_time``, an
    order placed in round ``t`` only needs to cover demand in the first week it
    can arrive, ``t + lead_time``. Later demand can be served by later orders
    just in time, so carrying extra inventory is never better than waiting.
    """
    end_inventory = float(initial_state.end_inventory)
    p1 = float(initial_state.in_transit_w1)
    p2 = float(initial_state.in_transit_w2)
    orders: dict[int, float] = {}

    for week in range(1, total_weeks + 1):
        if week <= decision_rounds:
            arrival_week = week + lead_time
            if arrival_week <= total_weeks:
                projected_end = end_inventory
                projected_p1 = p1
                projected_p2 = p2
                for projected_week in range(week, arrival_week):
                    projected_end, projected_p1, projected_p2 = _advance_state_values(
                        projected_end,
                        projected_p1,
                        projected_p2,
                        demand_by_week.get(projected_week, 0.0),
                        order=0.0,
                    )
                available_at_arrival = projected_end + projected_p1
                target = float(demand_by_week.get(arrival_week, 0.0))
                order = _round_order_to_step(target - available_at_arrival, order_step)
            else:
                order = 0.0
            orders[week] = order
        else:
            order = 0.0

        end_inventory, p1, p2 = _advance_state_values(
            end_inventory,
            p1,
            p2,
            demand_by_week.get(week, 0.0),
            order=order,
        )

    return orders


def _simulate_orders(
    initial_states: dict[str, ProductState],
    actuals_by_round: Mapping[int, dict[str, float]],
    orders_by_round: Mapping[int, dict[str, float]],
    *,
    decision_rounds: int,
    delivery_weeks: int,
) -> VN2Simulator:
    simulator = VN2Simulator(initial_states)
    for week in range(1, decision_rounds + delivery_weeks + 1):
        actuals = actuals_by_round.get(week, dict.fromkeys(initial_states, 0.0))
        orders = orders_by_round.get(week, dict.fromkeys(initial_states, 0.0))
        if week > decision_rounds:
            orders = {uid: 0.0 for uid in actuals}
        simulator.step(week, orders=orders, actual_demand=actuals)
    return simulator


def _cost_diagnostic_tables(
    actual: ReplayResult,
    oracle_summary: pd.DataFrame,
    oracle_history: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    actual_history = actual.history.rename(
        columns={
            "holding_cost": "actual_holding_cost",
            "shortage_cost": "actual_shortage_cost",
            "end_inventory": "actual_end_inventory",
        }
    )
    oracle_history = oracle_history.rename(
        columns={
            "holding_cost": "oracle_holding_cost",
            "shortage_cost": "oracle_shortage_cost",
            "end_inventory": "oracle_end_inventory",
        }
    )
    merged = actual_history.merge(
        oracle_history[
            [
                UNIQUE_ID,
                "week",
                "oracle_holding_cost",
                "oracle_shortage_cost",
                "oracle_end_inventory",
            ]
        ],
        on=[UNIQUE_ID, "week"],
        how="left",
    )
    merged["avoidable_overstock_cost"] = (
        merged["actual_holding_cost"] - merged["oracle_holding_cost"]
    ).clip(lower=0.0)
    merged["avoidable_shortage_cost"] = (
        merged["actual_shortage_cost"] - merged["oracle_shortage_cost"]
    ).clip(lower=0.0)
    actual_total = merged["actual_holding_cost"] + merged["actual_shortage_cost"]
    oracle_total = merged["oracle_holding_cost"] + merged["oracle_shortage_cost"]
    merged["avoidable_total_cost"] = actual_total - oracle_total
    merged["timing_residual_cost"] = (
        merged["avoidable_total_cost"]
        - merged["avoidable_overstock_cost"]
        - merged["avoidable_shortage_cost"]
    )

    by_sku = (
        actual.summary.merge(
            oracle_summary,
            on=UNIQUE_ID,
            suffixes=("_actual", "_oracle"),
        )
        .assign(
            avoidable_total_cost=lambda df: df["total_cost_actual"] - df["total_cost_oracle"],
            avoidable_holding_cost=lambda df: df["holding_cost_actual"] - df["holding_cost_oracle"],
            avoidable_shortage_cost=lambda df: (
                df["shortage_cost_actual"] - df["shortage_cost_oracle"]
            ),
        )
        .sort_values("avoidable_total_cost", ascending=False)
        .reset_index(drop=True)
    )
    by_round = (
        merged.groupby("week", as_index=False)[
            [
                "avoidable_overstock_cost",
                "avoidable_shortage_cost",
                "timing_residual_cost",
                "avoidable_total_cost",
            ]
        ]
        .sum()
        .rename(columns={"week": "round"})
    )
    totals = {
        "actual_total_cost": float(actual.summary["total_cost"].sum()),
        "oracle_total_cost": float(oracle_summary["total_cost"].sum()),
        "avoidable_total_cost": float(
            actual.summary["total_cost"].sum() - oracle_summary["total_cost"].sum()
        ),
        "avoidable_overstock_cost": float(merged["avoidable_overstock_cost"].sum()),
        "avoidable_shortage_cost": float(merged["avoidable_shortage_cost"].sum()),
        "timing_residual_cost": float(merged["timing_residual_cost"].sum()),
    }
    return by_sku, by_round, totals


def oracle_diagnostic(
    *,
    data_dir: Path = DATA_DIR,
    model_config: dict[str, Any] | None = None,
    order_conformal_config: CumulativeConformalRiskConfig | None = CONFORMAL_ORDER_CONFIG,
    horizon: int = HORIZON,
    lead_time: int = LEAD_TIME,
    review_period: int = REVIEW_PERIOD,
    decision_rounds: int = DECISION_ROUNDS,
    delivery_weeks: int = DELIVERY_WEEKS,
    series_filter: list[str] | None = None,
    order_base_scale: float = 1.0,
    order_step: float = 1.0,
) -> dict[str, Any]:
    """Compare current benchmark orders with finite-horizon oracle orders.

    The oracle uses realised future demand and exact VN2 lead-time/cost
    mechanics. It is diagnostic only; no oracle information is used by the
    deployed benchmark policy.
    """
    cache = build_replay_cache(
        data_dir=data_dir,
        model_config=model_config,
        horizon=horizon,
        lead_time=lead_time,
        review_period=review_period,
        decision_rounds=decision_rounds,
        delivery_weeks=delivery_weeks,
        series_filter=series_filter,
    )
    actual = replay_cached_cost(
        cache,
        order_conformal_config=order_conformal_config,
        order_base_scale=order_base_scale,
    )

    total_weeks = decision_rounds + delivery_weeks
    oracle_orders_by_round: dict[int, dict[str, float]] = {
        week: {} for week in range(1, decision_rounds + 1)
    }
    for uid, state in cache.initial_states.items():
        sku_demands = {
            week: cache.actuals_by_round.get(week, {}).get(uid, 0.0)
            for week in range(1, total_weeks + 1)
        }
        sku_orders = _optimal_order_path_for_sku(
            state,
            sku_demands,
            decision_rounds=decision_rounds,
            total_weeks=total_weeks,
            order_step=order_step,
            lead_time=lead_time,
        )
        for week, order in sku_orders.items():
            oracle_orders_by_round[week][uid] = order

    oracle_simulator = _simulate_orders(
        cache.initial_states,
        cache.actuals_by_round,
        oracle_orders_by_round,
        decision_rounds=decision_rounds,
        delivery_weeks=delivery_weeks,
    )
    oracle_summary = summary_from_simulator(oracle_simulator)
    oracle_history = oracle_simulator.to_dataframe()

    order_rows = []
    for week in range(1, decision_rounds + 1):
        actual_orders = actual.orders_by_round.get(week, {})
        oracle_orders = oracle_orders_by_round.get(week, {})
        for uid in cache.initial_states:
            actual_order = float(actual_orders.get(uid, 0.0))
            oracle_order = float(oracle_orders.get(uid, 0.0))
            order_rows.append(
                {
                    "round": week,
                    UNIQUE_ID: uid,
                    "actual_order_qty": actual_order,
                    "oracle_order_qty": oracle_order,
                    "order_gap": actual_order - oracle_order,
                    "order_bias": "over"
                    if actual_order > oracle_order
                    else "under"
                    if actual_order < oracle_order
                    else "match",
                }
            )
    orders = pd.DataFrame(order_rows)
    by_sku, by_round, totals = _cost_diagnostic_tables(actual, oracle_summary, oracle_history)

    return {
        "totals": totals,
        "orders": orders,
        "by_sku": by_sku,
        "by_round": by_round,
        "actual_summary": actual.summary,
        "oracle_summary": oracle_summary,
        "actual_history": actual.history,
        "oracle_history": oracle_history,
    }


optimal_order_path_for_sku = _optimal_order_path_for_sku
__all__ = [
    "optimal_order_path_for_sku",
    "oracle_diagnostic",
]
