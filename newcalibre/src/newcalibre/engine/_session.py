"""Bind engine inputs to the canonical definition carried by a session."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import cast

from newcalibre.domain import (
    Calendar,
    CostStructure,
    ForecastTask,
    Panel,
    Scope,
    SessionIdentity,
)
from newcalibre.domain._canonical_json import canonical_json_bytes
from newcalibre.engine.errors import EngineError


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
    return horizon, encoded_config, cost_from_definition(definition)


def session_cost_structure(session: SessionIdentity) -> CostStructure | None:
    """Return the decision cost fixed by a session, when ordering is configured."""
    return cost_from_definition(session_definition(session))


def cost_from_definition(definition: Mapping[str, object]) -> CostStructure | None:
    """Rebuild the typed cost value from a canonical session definition."""
    decision = definition.get("decision")
    if decision is None:
        return None
    if not isinstance(decision, Mapping):
        raise EngineError("session identity has an invalid decision configuration")
    cost = decision.get("cost_structure")
    if not isinstance(cost, Mapping):
        raise EngineError("session identity has an invalid cost structure")
    cost_values = dict(cost)
    try:
        return CostStructure(
            underage=cast(float, cost_values["underage_cost"]),
            overage=cast(float, cost_values["overage_cost"]),
            holding=cast(float, cost_values["holding_cost"]),
            shortage=cast(float, cost_values["shortage_cost"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise EngineError("session identity has an invalid cost structure") from error


def require_panel_session_binding(
    panel: Panel,
    *,
    session: SessionIdentity,
    ledger_calendar: Calendar,
) -> None:
    """Reject a panel or ledger calendar outside the engine session."""
    definition = session_definition(session)
    series_set = definition.get("series_set")
    frequency = definition.get("calendar_frequency")
    if not isinstance(series_set, list) or any(
        not isinstance(series_key, str) for series_key in series_set
    ):
        raise EngineError("session identity has an invalid series set")
    if tuple(series_set) != panel.series_keys:
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
    expected_series = definition.get("series_set")
    expected_frequency = definition.get("calendar_frequency")
    if task.horizon != expected_horizon or task.model_config != expected_model:
        raise EngineError("fitted task configuration does not match its session")
    if not isinstance(expected_series, list) or not set(task.series_keys) <= set(expected_series):
        raise EngineError("fitted task series do not belong to its session")
    if task.scope is Scope.GLOBAL and tuple(expected_series) != task.series_keys:
        raise EngineError("global fitted task must cover its session series set")
    if task.calendar.frequency != expected_frequency:
        raise EngineError("fitted task calendar does not match its session")


__all__ = [
    "cost_from_definition",
    "require_panel_session_binding",
    "require_task_session_binding",
    "session_cost_structure",
    "session_definition",
    "session_origin_inputs",
]
