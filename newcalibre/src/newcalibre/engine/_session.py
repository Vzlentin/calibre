"""Bind engine inputs to the canonical definition carried by a session."""

from __future__ import annotations

import json
from collections.abc import Mapping
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
    }
)


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
) -> tuple[int, bytes, CostStructure | None]:
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
    decision = decision_from_definition(definition)
    return horizon, encoded_config, None if decision is None else decision[0]


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
    series_keys, _frequency = session_series_and_frequency(definition)
    decision_inputs = decision_from_definition(definition)
    if decision_inputs is None:
        raise EngineError("session identity has an incomplete decision configuration")
    cost_structure, timing, _stockout_rule = decision_inputs

    conformal = definition.get("conformal_config")
    if conformal is not None and not isinstance(conformal, Mapping):
        raise EngineError("session identity has an invalid conformal configuration")
    calibration_coverage = None if conformal is None else conformal.get("coverage")

    return compile_ordering(
        OrderingSetup(
            policy=cast(str, policy.get("name")),
            series_keys=series_keys,
            cost_structure=cost_structure,
            decision_timing=timing,
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
        )
    )


def session_decision_inputs(
    session: SessionIdentity,
) -> tuple[CostStructure, DecisionTiming, StockoutRule] | None:
    """Return the decision facts that callers may not redefine outside a session."""
    return decision_from_definition(session_definition(session))


def decision_from_definition(
    definition: Mapping[str, object],
) -> tuple[CostStructure, DecisionTiming, StockoutRule] | None:
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
    cost_values = dict(cost)
    timing_values = dict(timing)
    try:
        cost_structure = CostStructure(
            underage=cast(float, cost_values["underage_cost"]),
            overage=cast(float, cost_values["overage_cost"]),
            holding=cast(float, cost_values["holding_cost"]),
            shortage=cast(float, cost_values["shortage_cost"]),
        )
        decision_timing = DecisionTiming(
            lead_time=cast(int, timing_values["lead_time"]),
            review_period=cast(int, timing_values["review_period"]),
        )
        stockout_rule = StockoutRule(rule)
    except (KeyError, TypeError, ValueError) as error:
        raise EngineError("session identity has invalid decision configuration") from error
    return cost_structure, decision_timing, stockout_rule


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
    "decision_from_definition",
    "require_panel_session_binding",
    "require_task_session_binding",
    "session_decision_inputs",
    "session_definition",
    "session_model_config",
    "session_ordering_configuration",
    "session_origin_inputs",
    "session_series_and_frequency",
]
