"""VN2 simulator replay and cached-cost helpers."""

from __future__ import annotations

import math
import tempfile
from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from benchmarks.common.tracking import (
    log_costs_dataframe,
    log_mlflow_params,
    mlflow,
    start_benchmark_run,
)
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
    load_instock,
    model_uses_cumulative_target,
    prepare_model_history,
    prepare_policy_forecast_frame,
    strip_private,
)
from benchmarks.vn2.simulator import (
    ProductState,
    VN2Simulator,
    extract_new_actuals,
    load_initial_states,
)
from calibre.conformal.cumulative_risk import (
    CumulativeConformalRiskConfig,
    CumulativeRiskRuntime,
)
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
from calibre.core.io import join_uri
from calibre.core.order_types import RsPolicyParameters
from calibre.execution import actuals_lookup_from_cache, build_actuals_lookup, observe_pending
from calibre.execution.backend import BackendEngine, ExecutionOptions
from calibre.execution.data_loading import load_period
from calibre.execution.task_builder import partition_tasks
from calibre.ordering.policy_config import RsConfig, apply_order_policy


def build_rs_params(
    simulator: VN2Simulator,
    lead_time: int,
    review_period: int,
) -> list[RsPolicyParameters]:
    """Build per-series (R,S) policy params from the simulator's state."""
    return [
        RsPolicyParameters(
            unique_id=uid,
            inventory_position=s.end_inventory + s.in_transit_w1 + s.in_transit_w2,
            lead_time=lead_time,
            review_period=review_period,
        )
        for uid, s in simulator.states.items()
    ]


def round_actuals(
    data_dir: str | Path,
    round_num: int,
    state_keys: Mapping[str, object],
) -> dict[str, float]:
    """Return per-series resolved actuals for a replay round."""
    # round_num indexes the resolved-actuals week directly: round 1's demand
    # is week_1_sales' last column. Earlier revisions used round_num + 1.
    try:
        actuals = extract_new_actuals(data_dir, round_num)
    except (FileNotFoundError, ValueError):
        # Fall back to the last date column of the current round's sales file.
        round_raw = pd.read_csv(join_uri(data_dir, f"week_{round_num}_sales.csv"))
        date_cols = [c for c in round_raw.columns if c not in ("Store", "Product")]
        last_col = date_cols[-1]
        unique_ids = (
            round_raw["Store"].astype(int).astype(str)
            + "_"
            + round_raw["Product"].astype(int).astype(str)
        )
        actuals = dict(zip(unique_ids, round_raw[last_col].fillna(0.0).astype(float), strict=False))
    return {uid: actuals.get(uid, 0.0) for uid in state_keys}


def run_order_conformal_warmup(
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
    frames = order_conformal_warmup_frames(
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
    )
    if not frames:
        return
    # Reach calibration through the package primitive instead of forking an
    # observe loop: the frames are already resolved and origin-ordered, so the
    # actuals lookup is a pass-through and observe_pending presents them in the
    # same order, preserving the runtime's per-origin recency weighting.
    actuals_lookup = build_actuals_lookup(pd.concat(frames, ignore_index=True))
    observe_pending(runtime, frames, actuals_lookup)


def order_conformal_warmup_frames(
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

    history = prepare_model_history(
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
        ledger_df = prepare_policy_forecast_frame(
            engine.execute(
                partition_tasks([task]), actuals=sales, origins=origin_dates
            ).ledger.to_df(),
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


def summary_from_simulator(simulator: VN2Simulator) -> pd.DataFrame:
    """Build a per-series summary frame from the simulator's end state."""
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
    """A cached replay round: its number, origin, and computed results."""

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
    """The outcome of a full replay: summary and per-round orders."""

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


def orders_from_policy_result(
    order_result: pd.DataFrame,
    state_keys: Mapping[str, object],
    reorder_point_scale: float | None = None,
) -> dict[str, float]:
    """Extract per-series order quantities from a policy result frame."""
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


def _actuals_for_replay_round(
    data_dir: str | Path,
    round_num: int,
    decision_rounds: int,
    state_keys: Mapping[str, object],
) -> dict[str, float]:
    if round_num <= decision_rounds:
        return round_actuals(data_dir, round_num, state_keys)
    try:
        actuals = extract_new_actuals(data_dir, round_num)
        return {uid: actuals.get(uid, 0.0) for uid in state_keys}
    except (FileNotFoundError, ValueError):
        return dict.fromkeys(state_keys, 0.0)


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
    cumulative_target = model_uses_cumulative_target(model_config)
    engine_config = strip_private(model_config)

    initial_states = load_initial_states(join_uri(data_dir, "week_0_initial_state.csv"))
    if series_filter is not None:
        initial_states = {uid: s for uid, s in initial_states.items() if uid in series_filter}

    instock = load_instock(data_dir, series_filter)
    week0_sales = load_period(data_dir, 0)
    if series_filter is not None:
        week0_sales = week0_sales[week0_sales[UNIQUE_ID].isin(initial_states)]

    warmup_frames = order_conformal_warmup_frames(
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
            history = prepare_model_history(
                round_sales,
                instock,
                protection_period=horizon,
                cumulative_target=cumulative_target,
            )
            origin = pd.Timestamp(round_sales[DS].max()) + pd.Timedelta(weeks=1)
            task = ForecastTask(history=history, horizon=horizon, model_config=engine_config)
            frame = prepare_policy_forecast_frame(
                engine.execute(
                    partition_tasks([task]), actuals=round_sales, origins=[origin]
                ).ledger.to_df(),
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
    on_progress: Callable[[int, float], None] | None = None,
) -> ReplayResult:
    """Replay cached forecasts through the exact VN2 simulator.

    A failure in ``apply_order_policy`` for a single round propagates: a broken
    policy must surface as a hard error, never as a silently cheap trajectory the
    HPO cost search can converge on.
    """
    simulator = VN2Simulator(cache.initial_states)
    target_quantile_col = quantile_column(cache.quantile_alpha)

    runtime: CumulativeRiskRuntime | None = None
    if order_conformal_config is not None:
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
        if runtime is not None:
            policy_frame = runtime.apply(frame)
            pending.append(policy_frame.copy())
            order_config = RsConfig(
                params=build_rs_params(simulator, cache.lead_time, cache.review_period),
                coverage=runtime.config.coverage,
            )
        else:
            policy_frame = frame
            order_config = RsConfig(
                params=build_rs_params(simulator, cache.lead_time, cache.review_period),
                quantile=cache.quantile_alpha,
            )
        order_result = apply_order_policy(policy_frame, order_config)
        orders = orders_from_policy_result(
            order_result,
            cache.initial_states,
            reorder_point_scale=reorder_point_scale,
        )

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
            lookup = actuals_lookup_from_cache(actuals_cache)
            pending = observe_pending(runtime, pending, lookup)

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
        summary=summary_from_simulator(simulator),
        orders_by_round=orders_by_round,
        history=simulator.to_dataframe(),
    )


def log_cached_replay_run(
    result: ReplayResult,
    *,
    run_name: str = "cached_replay",
    experiment_name: str = "vn2",
    params: Mapping[str, Any] | None = None,
    tags: Mapping[str, str] | None = None,
) -> None:
    """Log a cached simulator replay to MLflow."""
    resolved_tags = {"dataset": "vn2", "source": "cached_replay"}
    if tags:
        resolved_tags.update({str(key): str(value) for key, value in tags.items()})

    with start_benchmark_run(experiment_name, run_name, tags=resolved_tags):
        if params:
            log_mlflow_params(params)
        log_costs_dataframe(result.summary)
        mlflow.log_dict(
            {str(round_num): orders for round_num, orders in result.orders_by_round.items()},
            "replay/orders_by_round.json",
        )
        with tempfile.TemporaryDirectory() as tmp:
            history_path = Path(tmp) / "history.csv"
            result.history.to_csv(history_path, index=False)
            mlflow.log_artifact(str(history_path), artifact_path="replay")
