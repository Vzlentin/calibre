"""Exercise deterministic session identity across processes and input orderings."""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from collections.abc import Mapping

import pandas as pd
import pytest

from newcalibre.domain import (
    Calendar,
    CostStructure,
    DecisionTiming,
    SessionIdentity,
    SessionIdentityError,
    StockoutRule,
)

TIMING = DecisionTiming(lead_time=2, review_period=3)


def _costs(**overrides: float) -> CostStructure:
    values = {"underage": 4.0, "overage": 1.0, "holding": 0.5, "shortage": 2.0}
    values.update(overrides)
    return CostStructure(**values)


def _derive(**overrides: object) -> SessionIdentity:
    values: dict[str, object] = {
        "tenant": "tenant-a",
        "series_keys": ("sku-b", "sku-a"),
        "calendar": Calendar("D"),
        "horizon": 4,
        "model_config": {"backend": "seasonal", "params": {"season": 7}},
        "conformal_config": {"method": "split", "levels": [0.8, 0.9]},
        "ordering_policy": {"name": "order-up-to", "params": {"mode": "window"}},
        "cost_structure": _costs(),
        "decision_timing": TIMING,
        "stockout_rule": StockoutRule.LOST_SALES,
    }
    values.update(overrides)
    return SessionIdentity.derive(**values)  # type: ignore[arg-type]


def test_identity_is_full_sha256_of_a_versioned_canonical_payload() -> None:
    identity = _derive()

    assert len(identity.value) == 64
    assert identity.value == identity.value.lower()
    assert int(identity.value, 16) >= 0
    assert identity.value == hashlib.sha256(identity.to_bytes()).hexdigest()
    assert str(identity) == identity.value
    assert identity.to_bytes() == (
        b'{"calendar_frequency":"D","conformal_config":{"levels":[0.8,0.9],'
        b'"method":"split"},"decision":{"cost_structure":{"holding_cost":0.5,'
        b'"overage_cost":1.0,"shortage_cost":2.0,"underage_cost":4.0},'
        b'"ordering_policy":{"name":"order-up-to","params":{"mode":"window"}},'
        b'"stockout_rule":"lost-sales","timing":{"lead_time":2,"review_period":3}},'
        b'"horizon":4,"model_config":{"backend":"seasonal","params":{"season":7}},'
        b'"schema":"newcalibre.session-identity","series_set":["sku-a","sku-b"],'
        b'"tenant":"tenant-a","version":2}'
    )


def test_identity_can_only_be_created_from_defining_inputs() -> None:
    with pytest.raises(TypeError, match="derive"):
        SessionIdentity()


def test_absent_optional_configurations_are_encoded_distinctly() -> None:
    absent_conformal = _derive(conformal_config=None)
    present_conformal = _derive(conformal_config={})
    absent_decision = _derive(
        ordering_policy=None,
        cost_structure=None,
        decision_timing=None,
        stockout_rule=None,
    )
    present_decision = _derive()

    assert absent_conformal != present_conformal
    assert b'"conformal_config":null' in absent_conformal.to_bytes()
    assert absent_decision != present_decision
    assert b'"decision":null' in absent_decision.to_bytes()


@pytest.mark.parametrize(
    "missing",
    ["ordering_policy", "cost_structure", "decision_timing", "stockout_rule"],
)
def test_decision_configuration_is_present_or_absent_as_a_unit(
    missing: str,
) -> None:
    with pytest.raises(SessionIdentityError, match="all be supplied"):
        _derive(**{missing: None})


def test_series_permutation_and_mapping_order_do_not_change_identity() -> None:
    first = _derive()
    second = _derive(
        series_keys=("sku-a", "sku-b"),
        model_config={"params": {"season": 7}, "backend": "seasonal"},
        conformal_config={"levels": [0.8, 0.9], "method": "split"},
        ordering_policy={"params": {"mode": "window"}, "name": "order-up-to"},
    )

    assert first == second
    assert first.value == second.value
    assert first.to_bytes() == second.to_bytes()


def test_calendar_phase_is_not_a_session_defining_input() -> None:
    first = _derive(calendar=Calendar("D", phase=pd.Timestamp("2026-01-01")))
    second = _derive(calendar=Calendar("D", phase=pd.Timestamp("2026-01-02")))

    assert first == second


@pytest.mark.parametrize(
    "override",
    [
        {"tenant": "tenant-b"},
        {"series_keys": ("sku-a", "sku-c")},
        {"calendar": Calendar("W-MON")},
        {"horizon": 5},
        {"model_config": {"backend": "other"}},
        {"conformal_config": {"method": "adaptive"}},
        {"ordering_policy": {"name": "newsvendor"}},
        {"decision_timing": DecisionTiming(lead_time=3, review_period=3)},
        {"decision_timing": DecisionTiming(lead_time=2, review_period=4)},
        {"stockout_rule": StockoutRule.BACKORDER},
        {"cost_structure": _costs(underage=5.0)},
        {"cost_structure": _costs(overage=2.0)},
        {"cost_structure": _costs(holding=1.5)},
        {"cost_structure": _costs(shortage=3.0)},
    ],
)
def test_every_defining_input_changes_identity(override: dict[str, object]) -> None:
    assert _derive(**override) != _derive()


def test_identity_does_not_retain_mutable_configuration_callers() -> None:
    model: dict[str, object] = {"backend": "seasonal", "params": {"lags": [1, 7]}}
    conformal: dict[str, object] = {"method": "split", "levels": [0.9]}
    ordering: dict[str, object] = {"name": "order-up-to", "params": {"mode": "window"}}
    identity = _derive(
        model_config=model,
        conformal_config=conformal,
        ordering_policy=ordering,
    )
    original_value = identity.value
    original_payload = identity.to_bytes()

    model["backend"] = "mutated"
    conformal["levels"] = [0.5]
    ordering["name"] = "mutated"

    assert identity.value == original_value
    assert identity.to_bytes() == original_payload


def test_identity_is_stable_across_process_hash_seeds() -> None:
    script = """
from newcalibre.domain.calendar import Calendar
from newcalibre.domain.cost import CostStructure
from newcalibre.domain.decision import DecisionTiming, StockoutRule
from newcalibre.domain.session import SessionIdentity

identity = SessionIdentity.derive(
    tenant="tenant-a",
    series_keys=("sku-b", "sku-a"),
    calendar=Calendar("D"),
    horizon=4,
    model_config={"params": {"season": 7}, "backend": "seasonal"},
    conformal_config={"levels": [0.8, 0.9], "method": "split"},
    ordering_policy={"params": {"mode": "window"}, "name": "order-up-to"},
    cost_structure=CostStructure(underage=4, overage=1, holding=0.5, shortage=2),
    decision_timing=DecisionTiming(lead_time=2, review_period=3),
    stockout_rule=StockoutRule.LOST_SALES,
)
print(identity.value)
"""

    values: list[str] = []
    for seed in ("1", "8675309"):
        environment = os.environ.copy()
        environment["PYTHONHASHSEED"] = seed
        result = subprocess.run(
            [sys.executable, "-c", script],
            check=True,
            capture_output=True,
            text=True,
            env=environment,
        )
        values.append(result.stdout.strip())

    assert values == [_derive().value, _derive().value]


@pytest.mark.parametrize(
    ("override", "pattern"),
    [
        ({"tenant": ""}, "tenant"),
        ({"tenant": "\ud800"}, "UTF-8"),
        ({"series_keys": ()}, "must not be empty"),
        ({"series_keys": ("sku", "sku")}, "duplicates"),
        ({"series_keys": ("sku", 1)}, "series key"),
        ({"calendar": "D"}, "Calendar"),
        ({"horizon": 0}, "positive integer"),
        ({"horizon": True}, "positive integer"),
        ({"model_config": []}, "mapping"),
        ({"model_config": {"scope": "global"}}, "scope is engine"),
        ({"conformal_config": {"alpha": float("nan")}}, "non-finite"),
        ({"ordering_policy": {1: "bad"}}, "non-string"),
        ({"ordering_policy": {"params": (1, 2)}}, "non-JSON"),
        ({"cost_structure": object()}, "CostStructure"),
        ({"decision_timing": object()}, "DecisionTiming"),
        ({"stockout_rule": "lost-sales"}, "StockoutRule"),
    ],
)
def test_rejects_invalid_defining_inputs(
    override: dict[str, object],
    pattern: str,
) -> None:
    with pytest.raises(SessionIdentityError, match=pattern):
        _derive(**override)


def test_rejects_cyclic_configuration() -> None:
    cycle: list[object] = []
    cycle.append(cycle)
    config: Mapping[str, object] = {"cycle": cycle}

    with pytest.raises(SessionIdentityError, match="cyclic"):
        _derive(conformal_config=config)


def test_rejects_excessive_configuration_nesting_with_the_public_error() -> None:
    nested: object = None
    for _ in range(sys.getrecursionlimit() + 10):
        nested = [nested]

    with pytest.raises(SessionIdentityError, match="nesting depth"):
        _derive(model_config={"nested": nested})
