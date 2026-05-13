"""Pure order-policy helpers for forecast-frame recommendations."""

from calibre.order.config import OrderPolicyConfig, apply_order_policy
from calibre.order.newsvendor import apply_newsvendor_policy
from calibre.order.protocols import DecisionRule, OrderingArithmetic
from calibre.order.rs import apply_rs_policy
from calibre.order.rss import apply_rss_policy
from calibre.order.rules import (
    CumulativeBoundRule,
    QuantileInterpolationRule,
    RSArithmetic,
    RSSArithmetic,
    UpperBoundRule,
)
from calibre.order.types import (
    CostStruct,
    NewsvendorPolicyParameters,
    RsPolicyParameters,
    RssPolicyParameters,
    normalize_newsvendor_policy_parameters,
    normalize_rs_policy_parameters,
    normalize_rss_policy_parameters,
)

__all__ = [
    "RsPolicyParameters",
    "CostStruct",
    "apply_rs_policy",
    "RssPolicyParameters",
    "apply_rss_policy",
    "NewsvendorPolicyParameters",
    "apply_newsvendor_policy",
    "DecisionRule",
    "OrderingArithmetic",
    "UpperBoundRule",
    "QuantileInterpolationRule",
    "CumulativeBoundRule",
    "RSArithmetic",
    "RSSArithmetic",
    "OrderPolicyConfig",
    "apply_order_policy",
    "normalize_rs_policy_parameters",
    "normalize_rss_policy_parameters",
    "normalize_newsvendor_policy_parameters",
]
