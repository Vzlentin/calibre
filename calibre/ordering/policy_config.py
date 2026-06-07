from __future__ import annotations

from dataclasses import dataclass
from typing import assert_never

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


def apply_order_policy(frame: pd.DataFrame, config: OrderPolicy) -> pd.DataFrame:
    """Dispatch to the appropriate order policy based on the config type."""
    if isinstance(config, RsConfig):
        return apply_rs_policy(frame, config.params, config.coverage, quantile=config.quantile)
    if isinstance(config, RssConfig):
        return apply_rss_policy(frame, config.params, config.coverage)
    if isinstance(config, NewsvendorConfig):
        return apply_newsvendor_policy(frame, config.params, config.coverage, config.period)
    assert_never(config)
