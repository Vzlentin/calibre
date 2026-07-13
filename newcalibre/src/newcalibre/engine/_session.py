"""Bind engine inputs to the canonical definition carried by a session."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import cast

from newcalibre.domain import (
    Calendar,
    CostStructure,
    DecisionTiming,
    ForecastTask,
    Panel,
    Scope,
    SessionIdentity,
    StockoutRule,
)
from newcalibre.domain._canonical_json import canonical_json_bytes
from newcalibre.engine.errors import EngineError
from newcalibre.ordering import (
    OrderingConfigError,
    OrderingConfiguration,
    OrderingSetup,
    compile_ordering,
)

_ORDERING_POLICY_FIELDS = frozenset(
    {
        "coverage",
        "explicit_decision_fractile",
        "name",
        "quantile",
        "reorder_point",
        "reorder_point_scale",
        "target_cap",
        "target_floor",
        "target_scale",
    }
)
_COST_FIELDS = frozenset(
    {
        "holding_cost",
        "overage_cost",
        "shortage_cost",
        "underage_cost",
    }
)

type SessionCosts = Mapping[str, CostStructure]


@dataclass(frozen=True, slots=True)
class SessionDecision:
    """Expose one immutable decision scope and its session-owned configuration."""

    series_keys: tuple[str, ...]
    costs_by_series: SessionCosts
    timing: DecisionTiming
    stockout_rule: StockoutRule


def session_definition(session: SessionIdentity) -> dict[str, object]:
    """Materialize the immutable canonical preimage owned by a session."""
    try:
        definition = json.loads(session.to_bytes())
    except (TypeError, ValueError) as error:
        raise EngineError("session identity has an invalid canonical definition") from error
    if not isinstance(definition, dict):
        raise EngineError("session identity definition must be an object")
    return definition


def session_origin_inputs(
    session: SessionIdentity,
) -> tuple[int, bytes]:
    """Return origin facts that callers may not redefine outside the session."""
    definition = session_definition(session)
    horizon = definition.get("horizon")
    model_config = definition.get("model_config")
    if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon < 1:
        raise EngineError("session identity has an invalid horizon")
    if not isinstance(model_config, Mapping):
        raise EngineError("session identity has an invalid model configuration")
    encoded_config = canonical_json_bytes(
        dict(model_config),
        path="session model configuration",
    )
    return horizon, encoded_config


def session_model_config(session: SessionIdentity) -> dict[str, object]:
    """Return an isolated model configuration for pre-execution validation."""
    definition = session_definition(session)
    model_config = definition.get("model_config")
    if not isinstance(model_config, Mapping):
        raise EngineError("session identity has an invalid model configuration")
    return dict(cast(Mapping[str, object], model_config))


def session_ordering_configuration(
    session: SessionIdentity,
) -> OrderingConfiguration | None:
    """Compile session-owned ordering facts without loading execution data."""
    definition = session_definition(session)
    decision = definition.get("decision")
    if decision is None:
        return None
    if not isinstance(decision, Mapping):
        raise EngineError("session identity has an invalid decision configuration")
    raw_policy = decision.get("ordering_policy")
    if not isinstance(raw_policy, Mapping):
        raise EngineError("session identity has an invalid ordering policy")
    policy = dict(raw_policy)
    unsupported = sorted(set(policy) - _ORDERING_POLICY_FIELDS)
    if unsupported:
        fields = ", ".join(unsupported)
        raise OrderingConfigError(f"ordering policy has unsupported fields: {fields}")

    horizon = definition.get("horizon")
    if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon < 1:
        raise EngineError("session identity has an invalid horizon")
    decision_inputs = decision_from_definition(definition)
    if decision_inputs is None:
        raise EngineError("session identity has an incomplete decision configuration")

    conformal = definition.get("conformal_config")
    if conformal is not None and not isinstance(conformal, Mapping):
        raise EngineError("session identity has an invalid conformal configuration")
    calibration_coverage = None if conformal is None else conformal.get("coverage")

    return compile_ordering(
        OrderingSetup(
            policy=cast(str, policy.get("name")),
            series_keys=decision_inputs.series_keys,
            cost_structure=decision_inputs.costs_by_series,
            decision_timing=decision_inputs.timing,
            task_horizon=horizon,
            calibration_coverage=cast(float | None, calibration_coverage),
            calibration_protection_period=cast(
                int | None,
                None if conformal is None else conformal.get("protection_period"),
            ),
            policy_coverage=cast(float | None, policy.get("coverage")),
            explicit_quantile=cast(float | None, policy.get("quantile")),
            explicit_decision_fractile=cast(
                float | None,
                policy.get("explicit_decision_fractile"),
            ),
            reorder_point=cast(float | None, policy.get("reorder_point")),
            reorder_point_scale=cast(
                float | None,
                policy.get("reorder_point_scale"),
            ),
            target_cap=cast(float | None, policy.get("target_cap")),
            target_floor=cast(float | None, policy.get("target_floor")),
            target_scale=cast(float | None, policy.get("target_scale")),
        )
    )


def session_decision_inputs(
    session: SessionIdentity,
) -> SessionDecision | None:
    """Return the decision facts that callers may not redefine outside a session."""
    return decision_from_definition(session_definition(session))


def decision_from_definition(
    definition: Mapping[str, object],
) -> SessionDecision | None:
    """Rebuild the complete typed decision configuration from a session definition."""
    decision = definition.get("decision")
    if decision is None:
        return None
    if not isinstance(decision, Mapping):
        raise EngineError("session identity has an invalid decision configuration")
    cost = decision.get("cost_structure")
    if not isinstance(cost, Mapping):
        raise EngineError("session identity has an invalid cost structure")
    timing = decision.get("timing")
    if not isinstance(timing, Mapping):
        raise EngineError("session identity has invalid decision timing")
    rule = decision.get("stockout_rule")
    timing_values = dict(timing)
    try:
        series_keys = session_decision_series(definition)
        cost_structure = _cost_structure_from_definition(
            cast(Mapping[object, object], cost),
            series_keys=series_keys,
        )
        decision_timing = DecisionTiming(
            lead_time=cast(int, timing_values["lead_time"]),
            review_period=cast(int, timing_values["review_period"]),
        )
        stockout_rule = StockoutRule(rule)
    except (KeyError, TypeError, ValueError) as error:
        raise EngineError("session identity has invalid decision configuration") from error
    return SessionDecision(
        series_keys=series_keys,
        costs_by_series=cost_structure,
        timing=decision_timing,
        stockout_rule=stockout_rule,
    )


def _cost_structure_from_definition(
    value: Mapping[object, object],
    *,
    series_keys: tuple[str, ...],
) -> SessionCosts:
    snapshot = dict(value)
    if set(snapshot) == _COST_FIELDS:
        cost = _cost_from_definition(snapshot)
        return MappingProxyType({series_key: cost for series_key in series_keys})
    if set(snapshot) != {"per_series"}:
        raise EngineError("session identity has an invalid cost structure")
    per_series = snapshot["per_series"]
    if not isinstance(per_series, Mapping):
        raise EngineError("session identity has an invalid per-series cost structure")
    cost_payloads = dict(per_series)
    if set(cost_payloads) != set(series_keys):
        raise EngineError("session per-series cost structure must exactly match its series set")
    return MappingProxyType(
        {series_key: _cost_from_definition(cost_payloads[series_key]) for series_key in series_keys}
    )


def _cost_from_definition(value: object) -> CostStructure:
    if not isinstance(value, Mapping):
        raise EngineError("session identity has an invalid cost structure")
    values = dict(value)
    if set(values) != _COST_FIELDS:
        raise EngineError("session identity has an invalid cost structure")
    return CostStructure(
        underage=cast(float, values["underage_cost"]),
        overage=cast(float, values["overage_cost"]),
        holding=cast(float, values["holding_cost"]),
        shortage=cast(float, values["shortage_cost"]),
    )


def session_series_and_frequency(
    definition: Mapping[str, object],
) -> tuple[tuple[str, ...], str]:
    """Return the canonical series order and calendar frequency from a session."""
    series_set = definition.get("series_set")
    frequency = definition.get("calendar_frequency")
    if not isinstance(series_set, list) or not series_set or not isinstance(frequency, str):
        raise EngineError("session identity has an invalid series/calendar definition")
    series_keys: list[str] = []
    for series_key in series_set:
        if not isinstance(series_key, str):
            raise EngineError("session identity has an invalid series/calendar definition")
        series_keys.append(series_key)
    return tuple(series_keys), frequency


def session_decision_series(definition: Mapping[str, object]) -> tuple[str, ...]:
    """Return the canonical series subset owned by decision configuration."""
    session_series, _frequency = session_series_and_frequency(definition)
    decision = definition.get("decision")
    if not isinstance(decision, Mapping):
        raise EngineError("session identity has an invalid decision configuration")
    raw = decision.get("series_set")
    if raw is None:
        return session_series
    if not isinstance(raw, list) or not raw:
        raise EngineError("session identity has an invalid decision series set")
    normalized: list[str] = []
    for series_key in raw:
        if not isinstance(series_key, str) or not series_key:
            raise EngineError("session identity has an invalid decision series set")
        try:
            series_key.encode("utf-8")
        except UnicodeError as error:
            raise EngineError("session identity has an invalid decision series set") from error
        normalized.append(series_key)
    series_keys = tuple(normalized)
    if len(set(series_keys)) != len(series_keys):
        raise EngineError("session decision series set contains duplicates")
    if tuple(sorted(series_keys, key=str.encode)) != series_keys:
        raise EngineError("session decision series set is not canonical")
    if not set(series_keys) <= set(session_series):
        raise EngineError("session decision series set is not a subset of its series set")
    return series_keys


def require_panel_session_binding(
    panel: Panel,
    *,
    session: SessionIdentity,
    ledger_calendar: Calendar,
) -> None:
    """Reject a panel or ledger calendar outside the engine session."""
    definition = session_definition(session)
    series_keys, frequency = session_series_and_frequency(definition)
    if series_keys != panel.series_keys:
        raise EngineError("panel series set does not match the engine session")
    if frequency != ledger_calendar.frequency:
        raise EngineError("ledger calendar does not match the engine session")
    if panel.calendar != ledger_calendar:
        raise EngineError("panel calendar does not match the ledger calendar")


def require_task_session_binding(
    task: ForecastTask,
    *,
    session: SessionIdentity,
) -> None:
    """Reject a declarative fitted task whose defining facts drift from its session."""
    definition = session_definition(session)
    expected_horizon = definition.get("horizon")
    expected_model = definition.get("model_config")
    expected_series, expected_frequency = session_series_and_frequency(definition)
    if task.horizon != expected_horizon or task.model_config != expected_model:
        raise EngineError("fitted task configuration does not match its session")
    if not set(task.series_keys) <= set(expected_series):
        raise EngineError("fitted task series do not belong to its session")
    if task.scope is Scope.GLOBAL and tuple(expected_series) != task.series_keys:
        raise EngineError("global fitted task must cover its session series set")
    if task.calendar.frequency != expected_frequency:
        raise EngineError("fitted task calendar does not match its session")


__all__ = [
    "SessionDecision",
    "SessionCosts",
    "decision_from_definition",
    "require_panel_session_binding",
    "require_task_session_binding",
    "session_decision_inputs",
    "session_definition",
    "session_decision_series",
    "session_model_config",
    "session_ordering_configuration",
    "session_origin_inputs",
    "session_series_and_frequency",
]
