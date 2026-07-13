"""Compile ordering configuration and expose pure ordering arithmetic."""

from newcalibre.domain import AppliedBinding
from newcalibre.ordering._core import (
    OrderingConfigError,
    OrderingConfiguration,
    OrderingInputError,
    OrderingSetup,
    compile_ordering,
    order_up_to,
)
from newcalibre.ordering._objective import (
    DEFAULT_OBJECTIVE,
    CandidateInfeasible,
    CostComponents,
    CostValue,
    DecisionCostKey,
    DiagnosticObjective,
    DiagnosticWindow,
    ObjectiveError,
    OriginPartial,
    RegretObjective,
    SettlementObjective,
    diagnostic_cost,
    evaluate_settlement_candidate,
    key_aligned_regret,
    realized_cost,
    settle_path_cost,
)
from newcalibre.ordering._policies import PolicyDecision, PolicyRequest, dispatch_policy

__all__ = [
    "DEFAULT_OBJECTIVE",
    "AppliedBinding",
    "CandidateInfeasible",
    "CostComponents",
    "CostValue",
    "DecisionCostKey",
    "DiagnosticObjective",
    "DiagnosticWindow",
    "ObjectiveError",
    "OriginPartial",
    "OrderingConfiguration",
    "OrderingConfigError",
    "OrderingInputError",
    "OrderingSetup",
    "PolicyDecision",
    "PolicyRequest",
    "RegretObjective",
    "SettlementObjective",
    "compile_ordering",
    "dispatch_policy",
    "diagnostic_cost",
    "evaluate_settlement_candidate",
    "key_aligned_regret",
    "order_up_to",
    "realized_cost",
    "settle_path_cost",
]
