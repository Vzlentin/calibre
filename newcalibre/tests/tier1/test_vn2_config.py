"""Exercise VN2's pure, strict protocol configuration.

All assertions are exact tolerance-class-1 configuration or refusal facts.
"""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
import yaml
from tests.vn2_fixtures import synthetic_config_payload, write_config

from newcalibre.domain import ActualsSemantics, StockoutRule
from newcalibre.protocols.vn2 import VN2ConfigError, VN2ProtocolConfig, load_vn2_config

pytestmark = pytest.mark.tier1

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_CONFIG = PROJECT_ROOT / "benchmarks" / "vn2" / "protocol.yaml"


def test_protocol_configuration_cannot_bypass_validation() -> None:
    with pytest.raises(TypeError, match="load_vn2_config"):
        VN2ProtocolConfig()


def test_committed_protocol_configuration_carries_every_gate_a_fact() -> None:
    config = load_vn2_config(PROTOCOL_CONFIG)

    assert config.dataset == "vn2"
    assert config.series_count == 599
    assert config.calendar.frequency == "W-MON"
    assert config.round_count == 6
    assert config.timing.lead_time == 2
    assert config.timing.review_period == 1
    assert config.timing.protection_period == 3
    assert config.task_horizon == 3
    assert config.drain_periods == 2
    assert config.holding_rate == 0.2
    assert config.shortage_rate == 1.0
    assert config.cost_structure.underage == 1.0
    assert config.cost_structure.overage == 0.2
    assert config.actuals_semantics is ActualsSemantics.CENSORED_SALES_SURROGATE
    assert config.stockout_rule is StockoutRule.LOST_SALES
    assert config.decision_origins == tuple(
        config.calendar.advance(config.decision_origins[0], offset) for offset in range(6)
    )
    assert config.files.sales_reveals == tuple(f"week_{reveal}_sales.csv" for reveal in range(9))
    assert config.columns.series_keys == ("Store", "Product")
    assert config.columns.initial_on_hand == "End Inventory"
    assert config.columns.initial_pipeline == ("In Transit W+1", "In Transit W+2")
    assert config.model_config == {
        "backend": "vn2-seasonal-naive-native-median",
        "censoring_aware": False,
        "m": 52,
        "model_name": "vn2-seasonal-naive-native-median",
        "quantile_levels": [0.5],
    }
    assert config.conformal_config is None
    assert config.ordering_policy == {
        "coverage": None,
        "explicit_decision_fractile": None,
        "name": "rs",
        "quantile": 0.5,
        "reorder_point": None,
        "reorder_point_scale": None,
        "target_cap": None,
        "target_floor": None,
        "target_scale": None,
    }


def test_protocol_constants_are_data_and_can_change_as_one_consistent_configuration(
    tmp_path: Path,
) -> None:
    payload = synthetic_config_payload()
    payload["series_count"] = 4
    payload["decision"] = {
        "round_count": 2,
        "lead_time": 1,
        "review_period": 1,
        "protection_period": 2,
        "task_horizon": 2,
        "drain_periods": 1,
        "origins": ["2024-04-15", "2024-04-22"],
    }
    payload["cost"] = {
        "currency": "USD",
        "underage_rate": 2.0,
        "overage_rate": 0.3,
        "holding_rate": 0.3,
        "shortage_rate": 2.0,
    }
    payload["files"]["sales_reveals"] = [  # type: ignore[index]
        f"week_{reveal}_sales.csv" for reveal in range(4)
    ]
    payload["columns"]["initial_pipeline"] = ["In Transit W+1"]  # type: ignore[index]

    config = load_vn2_config(write_config(tmp_path / "variant.yaml", payload))

    assert config.series_count == 4
    assert config.round_count == 2
    assert config.timing.protection_period == 2
    assert config.drain_periods == 1
    assert config.holding_rate == 0.3
    assert config.shortage_rate == 2.0
    assert config.currency == "USD"


def _mutated_config(
    tmp_path: Path,
    mutation: Callable[[dict[str, Any]], object],
) -> Path:
    payload = synthetic_config_payload()
    mutation(payload)
    return write_config(tmp_path / "invalid.yaml", payload)


def _drop(payload: dict[str, Any], section: str, key: str) -> None:
    nested = payload[section]
    assert isinstance(nested, dict)
    nested.pop(key)


def _put(payload: dict[str, Any], section: str, key: str, value: object) -> None:
    nested = payload[section]
    assert isinstance(nested, dict)
    nested[key] = value


@pytest.mark.parametrize(
    ("mutation", "pattern"),
    [
        (lambda payload: payload.update(extra=True), "top-level.*exact keys"),
        (lambda payload: payload.pop("actuals_semantics"), "top-level.*exact keys"),
        (lambda payload: _drop(payload, "decision", "drain_periods"), "decision.*exact keys"),
        (lambda payload: _put(payload, "decision", "lead_time", 0), "lead_time.*positive"),
        (lambda payload: _put(payload, "decision", "protection_period", 4), "protection"),
        (lambda payload: _put(payload, "decision", "task_horizon", 2), "task_horizon"),
        (lambda payload: _put(payload, "decision", "round_count", 7), "origins"),
        (
            lambda payload: _put(
                payload,
                "decision",
                "origins",
                ["2024-04-16", *payload["decision"]["origins"][1:]],
            ),
            "calendar",
        ),
        (
            lambda payload: _put(
                payload,
                "decision",
                "origins",
                [
                    payload["decision"]["origins"][0],
                    payload["decision"]["origins"][2],
                    *payload["decision"]["origins"][2:],
                ],
            ),
            "cadence",
        ),
        (lambda payload: _put(payload, "cost", "holding_rate", -0.1), "holding_rate"),
        (lambda payload: _put(payload, "cost", "shortage_rate", float("inf")), "finite"),
        (lambda payload: payload.update(actuals_semantics="demand"), "censored_sales_surrogate"),
        (lambda payload: payload.update(stockout_rule="backorder"), "lost-sales"),
        (
            lambda payload: _put(payload, "columns", "initial_pipeline", ["In Transit W+1"]),
            "pipeline.*lead_time",
        ),
        (
            lambda payload: _put(payload, "model_config", "quantile_levels", [0.4, 0.6]),
            "single 0.5",
        ),
        (
            lambda payload: _put(payload, "ordering_policy", "quantile", None),
            "quantile.*0.5",
        ),
        (
            lambda payload: payload.update(conformal_config={"coverage": 0.9}),
            "conformal_config.*null",
        ),
    ],
    ids=[
        "extra-top-level",
        "missing-top-level",
        "missing-decision-field",
        "zero-lead-time",
        "protection-mismatch",
        "short-horizon",
        "origin-count",
        "non-monday",
        "origin-cadence",
        "negative-rate",
        "non-finite-rate",
        "wrong-semantics",
        "backorder",
        "pipeline-width",
        "quantile-levels",
        "ordering-quantile",
        "non-null-conformal",
    ],
)
def test_protocol_configuration_refuses_inconsistent_or_ambiguous_facts(
    tmp_path: Path,
    mutation: Callable[[dict[str, Any]], object],
    pattern: str,
) -> None:
    with pytest.raises(VN2ConfigError, match=pattern):
        load_vn2_config(_mutated_config(tmp_path, mutation))


def test_loaded_configuration_does_not_retain_mutable_yaml_values(tmp_path: Path) -> None:
    payload = synthetic_config_payload()
    path = write_config(tmp_path / "config.yaml", payload)
    config = load_vn2_config(path)
    snapshot = deepcopy(config.model_config)
    payload["model_config"]["m"] = 1

    assert config.model_config == snapshot


@pytest.mark.parametrize(
    ("duplicate", "pattern"),
    [
        ("dataset: other\n", r"duplicate.*dataset"),
        ("cost:\n  currency: USD\n", r"duplicate.*currency"),
    ],
    ids=["top-level", "nested"],
)
def test_protocol_configuration_refuses_duplicate_yaml_keys_at_every_depth(
    tmp_path: Path,
    duplicate: str,
    pattern: str,
) -> None:
    rendered = yaml.safe_dump(synthetic_config_payload(), sort_keys=False)
    if duplicate.startswith("cost:"):
        rendered = rendered.replace("cost:\n", duplicate, 1)
    else:
        rendered = duplicate + rendered
    path = tmp_path / "duplicate.yaml"
    path.write_text(rendered, encoding="utf-8")

    with pytest.raises(VN2ConfigError, match=pattern):
        load_vn2_config(path)


@pytest.mark.parametrize(
    ("history", "decision", "pattern"),
    [
        (
            {
                "first_week": "2024-04-08",
                "initial_last_week": "2024-04-08",
                "initial_periods": 10**12,
            },
            None,
            r"history initial_periods.*calendar bounds",
        ),
        (
            {
                "first_week": "2024-04-08",
                "initial_last_week": "2024-04-08",
                "initial_periods": 1,
            },
            {
                "round_count": 1,
                "lead_time": 1,
                "review_period": 10**12,
                "protection_period": 10**12 + 1,
                "task_horizon": 10**12 + 1,
                "drain_periods": 1,
                "origins": ["2024-04-15"],
            },
            r"decision review_period.*calendar bounds",
        ),
    ],
    ids=["history", "decision-review-period"],
)
def test_calendar_arithmetic_failures_are_attributed_to_configuration_fields(
    tmp_path: Path,
    history: dict[str, object],
    decision: dict[str, object] | None,
    pattern: str,
) -> None:
    payload = synthetic_config_payload()
    payload["history"] = history
    if decision is not None:
        payload["decision"] = decision

    with pytest.raises(VN2ConfigError, match=pattern):
        load_vn2_config(write_config(tmp_path / "calendar-boundary.yaml", payload))
