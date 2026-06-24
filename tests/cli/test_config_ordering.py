"""Schema coverage for the loop-path ordering/order-conformal config knobs.

Proves the schema additions: ``OrderingConfig`` gains
``lead_time``/``review_period`` and validates on the loop path with no static
``params``; ``OrderConformalConfig`` gains ``weight_decay``/``warmup_origins``/
``method_name`` and round-trips them into the runtime config, with an explicit
``weight_decay: null`` reaching the unweighted ("capped-CRC") branch rather than
inheriting the weighted ``0.85`` default. Unknown keys stay rejected
(``extra="forbid"``).
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from calibre.cli.config import OrderConformalConfig, OrderingConfig


def test_ordering_config_accepts_lead_time_and_review_period_without_params() -> None:
    """The loop path validates ``lead_time``/``review_period`` and no static params."""
    config = OrderingConfig.model_validate({"policy": "rs", "lead_time": 1, "review_period": 1})
    assert config.lead_time == 1
    assert config.review_period == 1
    # params stays optional; the loop builds them live from the simulator state.
    assert config.params is None


def test_ordering_config_defaults_lead_time_and_review_period_to_none() -> None:
    config = OrderingConfig.model_validate({"policy": "rs"})
    assert config.lead_time is None
    assert config.review_period is None


def test_ordering_config_rejects_unknown_key() -> None:
    with pytest.raises(ValidationError):
        OrderingConfig.model_validate({"policy": "rs", "lead_tim": 1})


def test_order_conformal_capped_crc_round_trips() -> None:
    """An explicit null ``weight_decay`` reaches the capped (unweighted) branch.

    ``weight_decay: null`` + ``buffer_max: 0.0`` + ``warmup_origins: 3`` +
    ``method_name: capped_crc`` round-trips into the runtime config with
    ``weight_decay is None`` — NOT the runtime default ``0.85`` — so the
    calibrator takes its unweighted split-conformal path.
    """
    config = OrderConformalConfig.model_validate(
        {
            "coverage": 0.5,
            "protection_period": 2,
            "weight_decay": None,
            "buffer_max": 0.0,
            "warmup_origins": 3,
            "method_name": "capped_crc",
        }
    )
    assert config.weight_decay is None
    assert config.warmup_origins == 3
    assert config.method_name == "capped_crc"

    runtime = config.to_runtime_config()
    assert runtime.weight_decay is None
    assert runtime.method_name == "capped_crc"
    assert runtime.buffer_max == 0.0


def test_order_conformal_omitted_weight_decay_inherits_runtime_default() -> None:
    """An omitted ``weight_decay`` inherits the runtime ``0.85`` weighted default.

    Omitting the field (vs. setting it to null) must not collapse to the capped
    branch: the runtime config's own default fills it, keeping the weighted path
    available for configs that don't opt into capped-CRC.
    """
    config = OrderConformalConfig.model_validate({"coverage": 0.5, "protection_period": 2})
    assert config.weight_decay is None
    runtime = config.to_runtime_config()
    assert runtime.weight_decay == 0.85
    assert runtime.method_name == "weighted_crc"


def test_order_conformal_rejects_unknown_key() -> None:
    with pytest.raises(ValidationError):
        OrderConformalConfig.model_validate(
            {"coverage": 0.5, "protection_period": 2, "weihgt_decay": 0.5}
        )
