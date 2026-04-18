from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

from calibre.order.newsvendor import apply_newsvendor_policy
from calibre.order.rs import apply_rs_policy
from calibre.order.rss import apply_rss_policy

OrderPolicyType = Literal["rs", "rss", "newsvendor"]


@dataclass(frozen=True, slots=True)
class OrderPolicyConfig:
    """Configuration for an order policy.

    Args:
        policy: Policy type — "rs" (R,S), "rss" (R,s,S), or "newsvendor".
        params: Policy parameters as a DataFrame or list of parameter dataclasses.
        coverage: Conformal interval coverage level to use. Default 0.9.
        period: Horizon step for newsvendor policy. Ignored for rs/rss. Default 1.
        quantile: Optional direct quantile (in (0, 1)). When set on the "rs"
            policy, the target stock level uses the per-horizon ``q_<p>``
            forecast column instead of the conformal upper bound and
            ``coverage`` is ignored.
    """

    policy: OrderPolicyType
    params: pd.DataFrame | list
    coverage: float = 0.9
    period: int = 1
    quantile: float | None = None

    def __post_init__(self) -> None:
        if self.coverage <= 0 or self.coverage >= 1:
            raise ValueError(f"coverage must be in (0, 1), got {self.coverage}")
        if self.quantile is not None and not 0.0 < self.quantile < 1.0:
            raise ValueError(f"quantile must be in (0, 1), got {self.quantile}")


def apply_order_policy(frame: pd.DataFrame, config: OrderPolicyConfig) -> pd.DataFrame:
    """Dispatch to the appropriate order policy function based on config.policy."""
    if config.policy == "rs":
        return apply_rs_policy(frame, config.params, config.coverage, quantile=config.quantile)
    if config.policy == "rss":
        return apply_rss_policy(frame, config.params, config.coverage)
    if config.policy == "newsvendor":
        return apply_newsvendor_policy(frame, config.params, config.coverage, config.period)
    raise ValueError(f"Unknown order policy: {config.policy!r}")
