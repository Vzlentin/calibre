from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, assert_never

import pandas as pd

from calibre.ordering.newsvendor import apply_newsvendor_policy
from calibre.ordering.periodic_review import apply_rs_policy
from calibre.ordering.reorder_point import apply_rss_policy


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
