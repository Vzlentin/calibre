"""Order service: build a policy spec, apply it, and persist orders per session."""

from __future__ import annotations

import pandas as pd
from sqlalchemy.orm import sessionmaker

from calibre.api.lifecycle import LifecycleStore
from calibre.ordering.policy_config import OrderPolicyConfig, apply_order_policy
from calibre.storage.adapters import OrderRepo


def build_policy_config(ordering: dict) -> OrderPolicyConfig:
    """Construct an OrderPolicyConfig from the request payload.

    Raises :class:`KeyError`, :class:`TypeError`, or :class:`ValueError` on
    malformed input — the HTTP layer translates those into 400s.
    """
    params = ordering["params"]
    params_frame = params if isinstance(params, pd.DataFrame) else pd.DataFrame(params)
    return OrderPolicyConfig(
        policy=ordering["policy"],
        params=params_frame,
        coverage=float(ordering.get("coverage", 0.9)),
        period=int(ordering.get("period", 1)),
        quantile=ordering.get("quantile"),
    )


def persist_orders(
    store: LifecycleStore,
    factory: sessionmaker | None,
    session_id: str,
    orders_frame: pd.DataFrame,
) -> None:
    """Attach orders to the session's first fit and append them to the orders table."""
    record = store.first_fit_for_session(session_id)
    if record is None:
        return
    store.update_fit(record.fit_id, last_orders=orders_frame)
    if factory is not None:
        OrderRepo(factory).append_frame(
            tenant=record.tenant,
            session_id=session_id,
            frame=orders_frame,
        )


__all__ = ["apply_order_policy", "build_policy_config", "persist_orders"]
