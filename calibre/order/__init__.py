"""Pure order-policy helpers for forecast-frame recommendations.

This package intentionally stops at policy computation. It does not wire order
recommendations into the engine/ledger flow yet, and it does not implement
`(R,s,S)` or `Newsvendor` policies in this first slice.
"""

from calibre.order.rs import apply_rs_policy
from calibre.order.types import RsPolicyParameters

__all__ = ["RsPolicyParameters", "apply_rs_policy"]
