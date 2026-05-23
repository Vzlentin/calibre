"""Replay utilities: build forecast caches and simulate cached cost replays."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from benchmarks.vn2.config import (
    BEST_CONFIG,
    CONFORMAL_ORDER_CONFIG,
    DATA_DIR,
    DECISION_ROUNDS,
    DELIVERY_WEEKS,
    HORIZON,
    HPO_N_ORIGINS,
    LEAD_TIME,
    REVIEW_PERIOD,
)
from benchmarks.vn2.data import (
    _actuals_for_replay_round,
    _model_uses_cumulative_target,
    _prepare_model_history,
    _prepare_policy_forecast_frame,
    _strip_private,
)
from benchmarks.vn2.simulator import ProductState, VN2Simulator, load_initial_states
from calibre.conformal.cumulative_risk import CumulativeConformalRiskConfig, CumulativeRiskRuntime
from calibre.core.forecast_frame import (
    DS,
    FORECAST_ORIGIN,
    UNIQUE_ID,
    Y_HAT,
    Y,
    is_quantile_column,
    quantile_column,
)
from calibre.core.forecast_task import ForecastTask
from calibre.core.order_types import RsPolicyParameters
from calibre.execution import observe_cumulative
from calibre.execution.backend import BackendEngine, ExecutionOptions
from calibre.execution.data_loading import load_period
from calibre.execution.io import join_uri
from calibre.ordering.policy_config import OrderPolicyConfig, apply_order_policy


def _build_rs_params(
    simulator: VN2Simulator,
    lead_time: int,
    review_period: int,
) -> list[RsPolicyParameters]:
    return [
        RsPolicyParameters(
            unique_id=uid,
            inventory_position=s.end_inventory + s.in_transit_w1 + s.in_transit_w2,
            lead_time=lead_time,
            review_period=review_period,
        )
        for uid, s in simulator.states.items()
    ]


def _run_order_conformal_warmup(
    *,
    sales: pd.DataFrame,
    instock: pd.DataFrame | None,
    model_config: dict[str, Any],
    horizon: int,
    warmup_origins: int,
    runtime: CumulativeRiskRuntime,
    series_filter: list[str] | None,
    cumulative_target: bool = False,
    execution_backend: Literal["local", "ray", "auto"] = "auto",
    ray_address: str | None = None,
    staging_uri: str | None = None,
    ray_threshold: int = 10,
    max_concurrency: int | None = None,
    cpu_per_task: float | None = None,
) -> None:
    """Calibrate the cumulative order conformal runtime on resolved origins."""
    for frame in _order_conformal_warmup_frames(
        sales=sales,
        instock=instock,
        model_config=model_config,
        horizon=horizon,
        warmup_origins=warmup_origins,
        series_filter=series_filter,
        cumulative_target=cumulative_target,
        execution_backend=execution_backend,
        ray_address=ray_address,
        staging_uri=staging_uri,
        ray_threshold=ray_threshold,
        max_concurrency=max_concurrency,
        cpu_per_task=cpu_per_task,
    ):
        runtime.observe(runtime.apply(frame))


def _order_conformal_warmup_frames(
    *,
    sales: pd.DataFrame,
    instock: pd.DataFrame | None,
    model_config: dict[str, Any],
    horizon: int,
    warmup_origins: int,
    series_filter: list[str] | None,
    cumulative_target: bool = False,
    execution_backend: Literal["local", "ray", "auto"] = "auto",
    ray_address: str | None = None,
    staging_uri: str | None = None,
    ray_threshold: int = 10,
    max_concurrency: int | None = None,
    cpu_per_task: float | None = None,
) -> list[pd.DataFrame]:
    """Return resolved warmup forecast frames for CRC calibration."""
    if warmup_origins <= 0:
        return []

    history = _prepare_model_history(
        sales,
        instock,
        protection_period=horizon,
        cumulative_target=cumulative_target,
    )
    all_dates = sorted(history[DS].unique())
    if len(all_dates) < warmup_origins + horizon:
        warmup_origins = max(1, len(all_dates) - horizon)
    origin_dates = [pd.Timestamp(d) for d in all_dates[-(warmup_origins + horizon) : -horizon]]
    if not origin_dates:
        return []

    engine = BackendEngine(
        execution=ExecutionOptions(
            freq="W-MON",
            backend=execution_backend,
            ray_address=ray_address,
            staging_uri=staging_uri,
            ray_threshold=ray_threshold,
            max_concurrency=max_concurrency,
            cpu_per_task=cpu_per_task,
        )
    )
    task = ForecastTask(history=history, horizon=horizon, model_config=model_config)
    try:
        ledger_df = _prepare_policy_forecast_frame(
            engine.execute([task], actuals=sales, origins=origin_dates).ledger.to_df(),
            protection_period=horizon,
            cumulative_target=cumulative_target,
        )
    finally:
        engine.close()
    if ledger_df.empty:
        return []

    actuals_lookup = sales.drop_duplicates(subset=[UNIQUE_ID, DS]).set_index([UNIQUE_ID, DS])[Y]
    unresolved = ledger_df[Y].isna()
    if unresolved.any():
        keys = pd.MultiIndex.from_arrays(
            [ledger_df.loc[unresolved, UNIQUE_ID].values, ledger_df.loc[unresolved, DS].values]
        )
        ledger_df.loc[unresolved, Y] = actuals_lookup.reindex(keys).to_numpy()

    if series_filter is not None:
        ledger_df = ledger_df[ledger_df[UNIQUE_ID].isin(series_filter)]

    frames: list[pd.DataFrame] = []
    for origin in origin_dates:
        origin_rows = ledger_df[ledger_df[FORECAST_ORIGIN] == origin]
        if not origin_rows.empty:
            frames.append(origin_rows)
    return frames


def _summary_from_simulator(simulator: VN2Simulator) -> pd.DataFrame:
    rows = []
    for uid, state in simulator.states.items():
        rows.append(
            {
                "unique_id": uid,
                "holding_cost": state.cumulative_holding_cost,
                "shortage_cost": state.cumulative_shortage_cost,
                "total_cost": state.cumulative_holding_cost + state.cumulative_shortage_cost,
            }
        )
    return pd.DataFrame(rows).sort_values("unique_id").reset_index(drop=True)


@dataclass
class CachedRound:
    round_num: int
    origin: pd.Timestamp
    frame: pd.DataFrame


@dataclass
class VN2ReplayCache:
    """Forecast and actual-demand cache for simulator-cost experiments."""

    initial_states: dict[str, ProductState]
    warmup_frames: list[pd.DataFrame]
    rounds: dict[int, CachedRound]
    actuals_by_round: dict[int, dict[str, float]]
    model_config: dict[str, Any]
    quantile_alpha: float
    horizon: int
    lead_time: int
    review_period: int
    decision_rounds: int
    delivery_weeks: int
    cumulative_target: bool


@dataclass
class ReplayResult:
    summary: pd.DataFrame
    orders_by_round: dict[int, dict[str, float]]
    history: pd.DataFrame

    @property
    def total_cost(self) -> float:
        return float(self.summary["total_cost"].sum())


def _scale_base_forecasts(frame: pd.DataFrame, scale: float) -> pd.DataFrame:
    """Apply an explicit base-forecast calibration scale for ablation searches."""
    if scale == 1.0 or frame.empty:
        return frame.copy()
    result = frame.copy()
    for col in [Y_HAT, *(c for c in result.columns if is_quantile_column(c))]:
        result[col] = result[col].astype(float) * float(scale)
    return result


def _orders_from_policy_result(
    order_result: pd.DataFrame,
    state_keys: Mapping[str, object],
    reorder_point_scale: float | None = None,
) -> dict[str, float]:
    adjusted = order_result.copy()
    if reorder_point_scale is not None:
        reorder_point = adjusted["target_stock_level"].astype(float) * float(reorder_point_scale)
        inventory_position = adjusted["inventory_position"].astype(float)
        adjusted.loc[inventory_position >= reorder_point, "order_qty"] = 0.0

    orders: dict[str, float] = dict.fromkeys(state_keys, 0.0)
    for uid, qty in zip(
        adjusted[UNIQUE_ID].astype(str),
        adjusted["order_qty"].astype(float),
        strict=False,
    ):
        orders[uid] = float(max(math.ceil(qty), 0))
    return orders


def build_replay_cache(
    *,
    data_dir: str | Path = DATA_DIR,
    model_config: dict[str, Any] | None = None,
    horizon: int = HORIZON,
    lead_time: int = LEAD_TIME,
    review_period: int = REVIEW_PERIOD,
    decision_rounds: int = DECISION_ROUNDS,
    delivery_weeks: int = DELIVERY_WEEKS,
    series_filter: list[str] | None = None,
    order_conformal_warmup_origins: int = HPO_N_ORIGINS,
) -> VN2ReplayCache:
    """Fit/cache benchmark forecasts and realised demand for cost replay."""
    model_config = deepcopy(model_config if model_config is not None else BEST_CONFIG)
    quantile_alpha = float(model_config.get("_quantile_alpha", model_config["quantiles"][0]))
    cumulative_target = _model_uses_cumulative_target(model_config)
    engine_config = _strip_private(model_config)

    initial_states = load_initial_states(join_uri(data_dir, "week_0_initial_state.csv"))
    if series_filter is not None:
        initial_states = {uid: s for uid, s in initial_states.items() if uid in series_filter}

    from benchmarks.vn2.data import _load_instock

    instock = _load_instock(data_dir, series_filter)
    week0_sales = load_period(data_dir, 0)
    if series_filter is not None:
        week0_sales = week0_sales[week0_sales[UNIQUE_ID].isin(initial_states)]

    warmup_frames = _order_conformal_warmup_frames(
        sales=week0_sales,
        instock=instock,
        model_config=engine_config,
        horizon=horizon,
        warmup_origins=order_conformal_warmup_origins,
        series_filter=list(initial_states),
        cumulative_target=cumulative_target,
    )

    engine = BackendEngine(execution=ExecutionOptions(freq="W-MON"))
    rounds: dict[int, CachedRound] = {}
    try:
        for rn in range(1, decision_rounds + 1):
            round_sales = load_period(data_dir, rn - 1)
            if series_filter is not None:
                round_sales = round_sales[round_sales[UNIQUE_ID].isin(initial_states)]
            history = _prepare_model_history(
                round_sales,
                instock,
                protection_period=horizon,
                cumulative_target=cumulative_target,
            )
            origin = pd.Timestamp(round_sales[DS].max()) + pd.Timedelta(weeks=1)
            task = ForecastTask(history=history, horizon=horizon, model_config=engine_config)
            frame = _prepare_policy_forecast_frame(
                engine.execute([task], actuals=round_sales, origins=[origin]).ledger.to_df(),
                protection_period=horizon,
                cumulative_target=cumulative_target,
            )
            rounds[rn] = CachedRound(round_num=rn, origin=origin, frame=frame)
    finally:
        engine.close()

    actuals_by_round = {
        rn: _actuals_for_replay_round(data_dir, rn, decision_rounds, initial_states)
        for rn in range(1, decision_rounds + delivery_weeks + 1)
    }

    return VN2ReplayCache(
        initial_states=initial_states,
        warmup_frames=warmup_frames,
        rounds=rounds,
        actuals_by_round=actuals_by_round,
        model_config=model_config,
        quantile_alpha=quantile_alpha,
        horizon=horizon,
        lead_time=lead_time,
        review_period=review_period,
        decision_rounds=decision_rounds,
        delivery_weeks=delivery_weeks,
        cumulative_target=cumulative_target,
    )


def replay_cached_cost(
    cache: VN2ReplayCache,
    *,
    order_conformal_config: CumulativeConformalRiskConfig | None = CONFORMAL_ORDER_CONFIG,
    order_base_scale: float = 1.0,
    reorder_point_scale: float | None = None,
    on_policy_error: Callable[[int, Exception], None] | None = None,
    on_progress: Callable[[int, float], None] | None = None,
    raise_on_policy_error: bool = False,
) -> ReplayResult:
    """Replay cached forecasts through the exact VN2 simulator.

    Pass ``raise_on_policy_error=True`` in the HPO cost-search path to fail
    the trial immediately on policy failure rather than silently falling back
    to zero orders (which would make a broken config look cheap).

    In degraded-mode replay (the default), a policy failure falls back to
    zero orders so cost comparisons remain consistent across rounds.
    """
    simulator = VN2Simulator(cache.initial_states)
    target_quantile_col = quantile_column(cache.quantile_alpha)

    runtime: CumulativeRiskRuntime | None = None
    if order_conformal_config is not None:
        from dataclasses import replace

        resolved_config = replace(
            order_conformal_config,
            base_column=target_quantile_col,
            protection_period=cache.lead_time + cache.review_period,
        )
        runtime = CumulativeRiskRuntime(resolved_config)
        for frame in cache.warmup_frames:
            runtime.observe(runtime.apply(_scale_base_forecasts(frame, order_base_scale)))

    orders_by_round: dict[int, dict[str, float]] = {}
    pending: list[pd.DataFrame] = []
    actuals_cache: dict[tuple[str, pd.Timestamp], float] = {}
    freq_offset = pd.tseries.frequencies.to_offset("W-MON")
    progress_step = 0

    for rn in range(1, cache.decision_rounds + 1):
        cached_round = cache.rounds[rn]
        frame = _scale_base_forecasts(cached_round.frame, order_base_scale)
        try:
            if runtime is not None:
                policy_frame = runtime.apply(frame)
                pending.append(policy_frame.copy())
                order_config = OrderPolicyConfig(
                    policy="rs",
                    params=_build_rs_params(simulator, cache.lead_time, cache.review_period),
                    coverage=runtime.config.coverage,
                )
            else:
                policy_frame = frame
                order_config = OrderPolicyConfig(
                    policy="rs",
                    params=_build_rs_params(simulator, cache.lead_time, cache.review_period),
                    quantile=cache.quantile_alpha,
                )
            order_result = apply_order_policy(policy_frame, order_config)
            orders = _orders_from_policy_result(
                order_result,
                cache.initial_states,
                reorder_point_scale=reorder_point_scale,
            )
        except (ValueError, KeyError) as exc:
            if raise_on_policy_error:
                raise
            if on_policy_error is not None:
                on_policy_error(rn, exc)
            # zero orders in degraded-mode replay (not in HPO path)
            orders = dict.fromkeys(cache.initial_states, 0.0)

        actual_demand = cache.actuals_by_round.get(rn, dict.fromkeys(cache.initial_states, 0.0))
        simulator.step(rn, orders=orders, actual_demand=actual_demand)
        orders_by_round[rn] = orders
        progress_step += 1
        if on_progress is not None:
            on_progress(progress_step, simulator.total_cost())

        if runtime is not None and actual_demand:
            actuals_ds = cached_round.origin + freq_offset
            for uid, demand in actual_demand.items():
                actuals_cache[(uid, actuals_ds)] = demand
            lookup = pd.Series(actuals_cache, dtype=float)
            if not lookup.empty:
                lookup.index = pd.MultiIndex.from_tuples(lookup.index)
            pending = observe_cumulative(runtime, pending, lookup)

    for week_offset in range(1, cache.delivery_weeks + 1):
        delivery_num = cache.decision_rounds + week_offset
        actual_demand = cache.actuals_by_round.get(
            delivery_num,
            dict.fromkeys(cache.initial_states, 0.0),
        )
        simulator.step(
            delivery_num,
            orders={uid: 0.0 for uid in actual_demand},
            actual_demand=actual_demand,
        )
        progress_step += 1
        if on_progress is not None:
            on_progress(progress_step, simulator.total_cost())

    return ReplayResult(
        summary=_summary_from_simulator(simulator),
        orders_by_round=orders_by_round,
        history=simulator.to_dataframe(),
    )
