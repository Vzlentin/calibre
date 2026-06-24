"""Typed ordering-policy configs and the policy dispatcher."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, assert_never

import pandas as pd

from calibre.core.forecast_frame import DS, UNIQUE_ID
from calibre.core.order_types import RsPolicyParameters
from calibre.ordering.newsvendor import apply_newsvendor_policy
from calibre.ordering.periodic_review import apply_rs_policy
from calibre.ordering.reorder_point import apply_rss_policy

if TYPE_CHECKING:
    from calibre.ordering.simulation.simulator import Simulator


class _InventoryStates(Protocol):
    """Structural view of a simulator's per-series inventory position.

    The elevated :func:`build_rs_params` reads only ``inventory_position`` off
    each state, so any object exposing ``states`` mapping ``unique_id`` to a
    state with that property satisfies it — the generic
    :class:`~calibre.ordering.simulation.simulator.Simulator` and the VN2
    benchmark's wrapped generic simulator alike.
    """

    @property
    def states(self) -> Mapping[str, Any]: ...


def _validate_coverage(coverage: float) -> None:
    if coverage <= 0 or coverage >= 1:
        raise ValueError(f"coverage must be in (0, 1), got {coverage}")


@dataclass(frozen=True, slots=True)
class RsConfig:
    """Configuration for the periodic-review (R,S) order-up-to policy.

    Args:
        params: Policy parameters as a DataFrame or list of ``RsPolicyParameters``.
        coverage: Conformal interval coverage level. Default 0.9. Ignored when
            ``quantile`` is set.
        quantile: Optional direct quantile (in (0, 1)). When set, the target
            stock level uses the per-horizon ``q_<p>`` forecast column instead
            of the conformal upper bound and ``coverage`` is ignored.
    """

    params: pd.DataFrame | list
    coverage: float = 0.9
    quantile: float | None = None

    def __post_init__(self) -> None:
        _validate_coverage(self.coverage)
        if self.quantile is not None and not 0.0 < self.quantile < 1.0:
            raise ValueError(f"quantile must be in (0, 1), got {self.quantile}")


@dataclass(frozen=True, slots=True)
class RssConfig:
    """Configuration for the periodic-review (R,s,S) order-up-to policy.

    Args:
        params: Policy parameters as a DataFrame or list of ``RssPolicyParameters``.
        coverage: Conformal interval coverage level. Default 0.9.
    """

    params: pd.DataFrame | list
    coverage: float = 0.9

    def __post_init__(self) -> None:
        _validate_coverage(self.coverage)


@dataclass(frozen=True, slots=True)
class NewsvendorConfig:
    """Configuration for the newsvendor critical-ratio policy.

    Args:
        params: Policy parameters as a DataFrame or list of
            ``NewsvendorPolicyParameters``.
        coverage: Conformal interval coverage level. Default 0.9.
        period: Horizon step the demand quantile is drawn from. Default 1.
    """

    params: pd.DataFrame | list
    coverage: float = 0.9
    period: int = 1

    def __post_init__(self) -> None:
        _validate_coverage(self.coverage)


OrderPolicy = RsConfig | RssConfig | NewsvendorConfig


def build_order_policy(ordering: Mapping[str, Any]) -> OrderPolicy:
    """Build the per-policy order config from an untyped ordering mapping.

    Single owner of untyped-input -> typed-config construction for both
    callers: the API passes the ``/order`` request's ordering dict through
    unchanged; the CLI passes ``OrderingConfig.model_dump()``. Recognized
    keys: ``policy``, ``params``, ``coverage``, ``quantile``, ``period``.

    ``params`` accepts a DataFrame (passed through), a list of dicts, or a
    single dict (wrapped to a one-element list); missing or None ``params``
    raises ``ValueError``. ``quantile`` is rejected for the rss and
    newsvendor policies (where it does not apply); unrecognized knobs are
    otherwise ignored.

    Optional knobs are keyed on ``value is not None``, never key presence:
    ``model_dump()`` always emits ``quantile``/``period`` (None when unset)
    while a request dict may omit them entirely.
    """
    policy = ordering["policy"]
    params = ordering.get("params")
    if params is None:
        raise ValueError("ordering.params is required")
    if isinstance(params, pd.DataFrame):
        params_frame = params
    else:
        params_frame = pd.DataFrame([params] if isinstance(params, dict) else params)
    coverage = float(ordering.get("coverage", 0.9))
    if policy == "rs":
        quantile = ordering.get("quantile")
        return RsConfig(
            params=params_frame,
            coverage=coverage,
            quantile=None if quantile is None else float(quantile),
        )
    if policy == "rss":
        if ordering.get("quantile") is not None:
            raise ValueError("ordering.quantile is not a valid knob for the rss policy")
        return RssConfig(params=params_frame, coverage=coverage)
    if policy == "newsvendor":
        if ordering.get("quantile") is not None:
            raise ValueError("ordering.quantile is not a valid knob for the newsvendor policy")
        period = ordering.get("period")
        if period is None:
            return NewsvendorConfig(params=params_frame, coverage=coverage)
        return NewsvendorConfig(params=params_frame, coverage=coverage, period=int(period))
    raise ValueError(f"unknown order policy: {policy!r}")


def apply_order_policy(frame: pd.DataFrame, config: OrderPolicy) -> pd.DataFrame:
    """Dispatch to the appropriate order policy based on the config type."""
    if isinstance(config, RsConfig):
        return apply_rs_policy(frame, config.params, config.coverage, quantile=config.quantile)
    if isinstance(config, RssConfig):
        return apply_rss_policy(frame, config.params, config.coverage)
    if isinstance(config, NewsvendorConfig):
        return apply_newsvendor_policy(frame, config.params, config.coverage, config.period)
    assert_never(config)


def orders_from_policy_result(
    order_result: pd.DataFrame,
    state_keys: Mapping[str, Any] | list[str],
    reorder_point_scale: float | None = None,
) -> dict[str, float]:
    """Extract per-series integer order quantities from a policy result frame.

    Single owner of the order-units rule shared by the engine settle loop and the
    VN2 benchmark replay/seasonal paths: clamp each ``order_qty`` to whole units
    via ``ceil`` (so a fractional order-up-to gap never under-orders) then floor at
    zero. Series with no row in ``order_result`` default to a zero order.

    Args:
        order_result: A policy result frame carrying ``unique_id``/``order_qty``
            (and ``target_stock_level``/``inventory_position`` when scaling).
        state_keys: The series whose orders to return; every key gets an entry.
        reorder_point_scale: Optional (R,s,S)-style reorder-point gate — zero out
            an order when the inventory position already sits at or above
            ``target_stock_level * reorder_point_scale``.
    """
    orders: dict[str, float] = dict.fromkeys(state_keys, 0.0)
    if order_result.empty:
        return orders
    adjusted = order_result
    if reorder_point_scale is not None:
        adjusted = order_result.copy()
        reorder_point = adjusted["target_stock_level"].astype(float) * float(reorder_point_scale)
        inventory_position = adjusted["inventory_position"].astype(float)
        adjusted.loc[inventory_position >= reorder_point, "order_qty"] = 0.0
    for uid, qty in zip(
        adjusted[UNIQUE_ID].astype(str),
        adjusted["order_qty"].astype(float),
        strict=False,
    ):
        orders[uid] = float(max(math.ceil(qty), 0))
    return orders


def build_rs_params(
    simulator: Simulator | _InventoryStates,
    lead_time: int,
    review_period: int,
) -> list[RsPolicyParameters]:
    """Build per-series (R,S) policy params from a simulator's live state.

    Reads each series' ``inventory_position`` (end inventory plus all in-transit
    pipeline quantities) off the generic
    :class:`~calibre.ordering.simulation.state.ProductState`, so the order policy
    targets the position the simulator actually holds at this round. For the VN2
    two-slot pipeline this equals the manual sum
    ``end_inventory + in_transit_w1 + in_transit_w2`` the benchmark previously
    read off its VN2-shaped state.
    """
    return [
        RsPolicyParameters(
            unique_id=uid,
            inventory_position=float(state.inventory_position),
            lead_time=lead_time,
            review_period=review_period,
        )
        for uid, state in simulator.states.items()
    ]


def derive_warmup_origins(
    history: pd.DataFrame,
    horizon: int,
    warmup_origins: int,
) -> list[pd.Timestamp]:
    """Derive the calibration-warmup origin timestamps from a model history.

    Returns the last ``warmup_origins`` decision dates that still leave a full
    ``horizon`` of resolvable future, clamping the count down when the history is
    too short and returning an empty list when no origin resolves. Shared with
    the VN2 benchmark's warmup-frame derivation so the engine's loop-path
    warmup walk and the benchmark's stay one implementation.
    """
    if warmup_origins <= 0:
        return []
    all_dates = sorted(history[DS].unique())
    if len(all_dates) < warmup_origins + horizon:
        warmup_origins = max(1, len(all_dates) - horizon)
    origin_dates = [pd.Timestamp(d) for d in all_dates[-(warmup_origins + horizon) : -horizon]]
    if not origin_dates:
        return []
    return origin_dates
