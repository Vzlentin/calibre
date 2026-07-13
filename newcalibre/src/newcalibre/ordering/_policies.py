"""Apply pure ordering policy families to descriptor-backed forecast evidence."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from numbers import Integral, Real
from types import MappingProxyType
from typing import cast

import pandas as pd

from newcalibre.domain import (
    FRAME_KEY_COLUMNS,
    HORIZON_STEP,
    MODEL_NAME,
    ORIGIN,
    SERIES_KEY,
    AppliedBinding,
    DecisionEvidence,
    DecisionScope,
    DecisionScopeKind,
    EmissionScope,
    GuaranteeClaim,
    GuaranteeDescriptor,
    GuaranteeType,
    InventoryPosition,
    ScoredSeries,
    interval_columns,
    quantile_column,
)
from newcalibre.ledger import (
    BoundKey,
    ForecastIssuance,
    ForecastKey,
    GuaranteedSide,
)
from newcalibre.ordering._core import (
    OrderingConfiguration,
    OrderingInputError,
    _finite_real,
    order_up_to,
)


@dataclass(frozen=True, slots=True, init=False)
class PolicyRequest:
    """Snapshot every input consumed by one atomic pure policy dispatch."""

    _frame: pd.DataFrame = field(repr=False)
    issuances: Mapping[ForecastKey, Mapping[BoundKey, ForecastIssuance]]
    inventory_positions: Mapping[str, InventoryPosition]
    configuration: OrderingConfiguration

    def __init__(
        self,
        *,
        frame: pd.DataFrame,
        issuances: Mapping[ForecastKey, Mapping[BoundKey, ForecastIssuance]],
        inventory_positions: Mapping[str, InventoryPosition],
        configuration: OrderingConfiguration,
    ) -> None:
        if not isinstance(frame, pd.DataFrame):
            raise OrderingInputError("policy frame must be a pandas DataFrame")
        if not isinstance(issuances, Mapping):
            raise OrderingInputError("policy issuances must be a mapping")
        if not isinstance(inventory_positions, Mapping):
            raise OrderingInputError("inventory positions must be a mapping")
        if not isinstance(configuration, OrderingConfiguration):
            raise OrderingInputError("configuration must be an OrderingConfiguration")

        try:
            frozen_issuances = {
                key: MappingProxyType(dict(row_issuances))
                for key, row_issuances in issuances.items()
            }
        except (TypeError, ValueError) as error:
            raise OrderingInputError("policy issuances must contain row mappings") from error
        positions = dict(inventory_positions)
        object.__setattr__(self, "_frame", pd.DataFrame(frame, copy=True))
        object.__setattr__(self, "issuances", MappingProxyType(frozen_issuances))
        object.__setattr__(self, "inventory_positions", MappingProxyType(positions))
        object.__setattr__(self, "configuration", configuration)

    @property
    def frame(self) -> pd.DataFrame:
        """Return an isolated copy of the request's forecast evidence."""
        return self._frame.copy(deep=True)


@dataclass(frozen=True, slots=True)
class PolicyDecision:
    """Return one real-valued proposal with immutable decision evidence."""

    series_key: str
    origin: pd.Timestamp
    model_name: str
    quantity: float
    evidence: DecisionEvidence

    def __post_init__(self) -> None:
        _identifier(self.series_key, name="decision series key")
        _identifier(self.model_name, name="decision model name")
        if not isinstance(self.origin, pd.Timestamp) or pd.isna(self.origin):
            raise OrderingInputError("decision origin must be a non-missing pandas Timestamp")
        if self.origin.tz is not None:
            raise OrderingInputError("decision origin must be timezone-naive")
        object.__setattr__(self, "quantity", _finite_value(self.quantity, name="quantity"))
        if self.quantity < 0.0:
            raise OrderingInputError("decision quantity must be nonnegative")
        if not isinstance(self.evidence, DecisionEvidence):
            raise OrderingInputError("decision evidence must be DecisionEvidence")
        object.__setattr__(self, "evidence", DecisionEvidence.snapshot(self.evidence))


@dataclass(frozen=True, slots=True)
class _PolicyRow:
    values: Mapping[str, object]
    issuances: Mapping[BoundKey, ForecastIssuance]


@dataclass(frozen=True, slots=True)
class _DecisionGroup:
    series_key: str
    origin: pd.Timestamp
    model_name: str
    rows: Mapping[int, _PolicyRow]


@dataclass(frozen=True, slots=True)
class _Source:
    columns: tuple[str, ...]
    descriptor: GuaranteeDescriptor


def dispatch_policy(request: PolicyRequest) -> tuple[PolicyDecision, ...]:
    """Dispatch all configured decision groups atomically through one policy seam."""
    if not isinstance(request, PolicyRequest):
        raise OrderingInputError("dispatch requires a PolicyRequest")
    groups = _decision_groups(request)
    configuration = request.configuration
    explicit_quantile_descriptor = (
        _nonengine_quantile_descriptor(configuration.explicit_quantile)
        if configuration.policy in {"rs", "rss"} and configuration.explicit_quantile is not None
        else None
    )

    staged: list[PolicyDecision] = []
    for group in groups:
        if configuration.policy == "newsvendor":
            raw_target, source = _newsvendor_target(group, configuration)
        elif configuration.policy in {"rs", "rss"}:
            raw_target, source = _window_target(
                group,
                configuration,
                explicit_quantile_descriptor=explicit_quantile_descriptor,
            )
        else:  # pragma: no cover - compile_ordering owns the closed policy set.
            raise OrderingInputError("compiled policy is not supported")

        target, modifier_bindings = _modified_target(raw_target, configuration)
        bindings = configuration.applied_bindings + modifier_bindings
        effective_descriptor = configuration.descriptor_for_decision(
            source.descriptor,
            bindings=modifier_bindings,
        )
        position = request.inventory_positions[group.series_key]
        reorder_point: float | None = None
        if configuration.policy == "rss":
            reorder_point = _rss_reorder_point(target, configuration)
            quantity = (
                0.0
                if _inventory_value(position) >= reorder_point
                else order_up_to(target, position)
            )
        else:
            quantity = order_up_to(target, position)
        staged.append(
            PolicyDecision(
                series_key=group.series_key,
                origin=group.origin,
                model_name=group.model_name,
                quantity=quantity,
                evidence=DecisionEvidence(
                    raw_target=raw_target,
                    target=target,
                    source_columns=source.columns,
                    source_descriptor=source.descriptor,
                    effective_descriptor=effective_descriptor,
                    bindings=bindings,
                    reorder_point=reorder_point,
                ),
            )
        )
    return tuple(staged)


def _decision_groups(request: PolicyRequest) -> tuple[_DecisionGroup, ...]:
    frame = request._frame
    if frame.empty:
        raise OrderingInputError("policy frame must contain at least one decision group")
    if frame.columns.has_duplicates:
        raise OrderingInputError("policy frame column names must be unique")
    required = FRAME_KEY_COLUMNS
    missing_columns = [column for column in required if column not in frame.columns]
    if missing_columns:
        raise OrderingInputError(f"policy frame is missing required columns: {missing_columns!r}")

    row_records: list[tuple[ForecastKey, dict[str, object]]] = []
    columns = tuple(frame.columns)
    origins: set[pd.Timestamp] = set()
    for values in frame.itertuples(index=False, name=None):
        row = dict(zip(columns, values, strict=True))
        series_key = _identifier(row[SERIES_KEY], name="decision series key")
        model_name = _identifier(row[MODEL_NAME], name="decision model name")
        origin = row[ORIGIN]
        if not isinstance(origin, pd.Timestamp) or pd.isna(origin) or origin.tz is not None:
            raise OrderingInputError("policy rows require one timezone-naive pandas origin")
        step = row[HORIZON_STEP]
        if isinstance(step, bool) or not isinstance(step, Integral) or int(step) < 1:
            raise OrderingInputError("policy horizon steps must be positive integers")
        normalized_step = int(step)
        row[HORIZON_STEP] = normalized_step
        key = cast(ForecastKey, tuple(row[column] for column in FRAME_KEY_COLUMNS))
        row_records.append((key, row))
        origins.add(origin)
    if len(origins) != 1:
        raise OrderingInputError("one policy request must contain exactly one origin")

    frame_keys = tuple(key for key, _row in row_records)
    if len(set(frame_keys)) != len(frame_keys):
        raise OrderingInputError("duplicate horizon steps are forbidden within a decision group")
    if set(request.issuances) != set(frame_keys):
        raise OrderingInputError("issuance row keys must exactly match policy frame rows")

    configured_series = set(request.configuration.series_keys)
    supplied_series = {key[0] for key in frame_keys}
    if supplied_series != configured_series:
        missing = sorted(configured_series - supplied_series, key=str.encode)
        unexpected = sorted(supplied_series - configured_series, key=str.encode)
        raise OrderingInputError(
            "policy input must exactly cover configured decision series; "
            f"missing={missing!r}, unexpected={unexpected!r}"
        )
    supplied_positions = set(request.inventory_positions)
    if supplied_positions != configured_series:
        missing = sorted(configured_series - supplied_positions, key=str.encode)
        unexpected = sorted(supplied_positions - configured_series, key=str.encode)
        raise OrderingInputError(
            "inventory positions must exactly cover configured decision series; "
            f"missing={missing!r}, unexpected={unexpected!r}"
        )
    if any(
        not isinstance(position, InventoryPosition)
        for position in request.inventory_positions.values()
    ):
        raise OrderingInputError("inventory positions must contain InventoryPosition values")

    staged: dict[tuple[str, pd.Timestamp, str], dict[int, _PolicyRow]] = {}
    for key, values in row_records:
        series_key, origin, step, model_name = key
        group_key = (series_key, origin, model_name)
        rows = staged.setdefault(group_key, {})
        rows[step] = _PolicyRow(
            values=MappingProxyType(values),
            issuances=request.issuances[key],
        )
    expected_steps = set(range(1, request.configuration.task_horizon + 1))
    for (series_key, _origin, model_name), rows in staged.items():
        if set(rows) != expected_steps:
            missing = sorted(expected_steps - set(rows))
            unexpected = sorted(set(rows) - expected_steps)
            raise OrderingInputError(
                "every decision group must cover the complete task horizon; "
                f"series_key={series_key!r}, model_name={model_name!r}, "
                f"missing={missing!r}, unexpected={unexpected!r}"
            )
    return tuple(
        _DecisionGroup(
            series_key=series_key,
            origin=origin,
            model_name=model_name,
            rows=MappingProxyType(dict(sorted(rows.items()))),
        )
        for (series_key, origin, model_name), rows in sorted(
            staged.items(),
            key=lambda item: (item[0][0].encode(), item[0][2].encode()),
        )
    )


def _newsvendor_target(
    group: _DecisionGroup,
    configuration: OrderingConfiguration,
) -> tuple[float, _Source]:
    row = group.rows[1]
    fractile = configuration.decision_fractile
    if fractile is None:
        raise OrderingInputError("newsvendor requires a compiled decision fractile")
    dense_column = quantile_column(fractile)
    if dense_column in row.values:
        issued = _matching_issuance(row, (dense_column,), exact=True, missing_ok=True)
        if issued is not None:
            _validate_issuance(
                issued,
                expected_level=fractile,
                expected_window=EmissionScope.PER_STEP,
                values=row.values,
                consumed_columns=(dense_column,),
            )
            target = _finite_value(row.values[dense_column], name="newsvendor quantile")
            return target, _Source((dense_column,), issued.descriptor)

    coverage = configuration.coverage
    if coverage is None:
        raise OrderingInputError(
            "newsvendor requires an issued critical-ratio quantile or interval coverage"
        )
    lower, upper = interval_columns(coverage)
    _require_columns(group, (lower, upper), family="interval")
    descriptor = _newsvendor_interval_descriptor(
        row,
        lower=lower,
        upper=upper,
        coverage=coverage,
    )
    lower_value = _finite_value(row.values[lower], name="newsvendor lower bound")
    upper_value = _finite_value(row.values[upper], name="newsvendor upper bound")
    if lower_value > upper_value:
        raise OrderingInputError("newsvendor interval requires lower bound <= upper bound")
    target = lower_value + fractile * (upper_value - lower_value)
    return _finite_value(target, name="newsvendor interpolated target"), _Source(
        (lower, upper),
        descriptor,
    )


def _newsvendor_interval_descriptor(
    row: _PolicyRow,
    *,
    lower: str,
    upper: str,
    coverage: float,
) -> GuaranteeDescriptor:
    joint = _matching_issuance(row, (lower, upper), exact=True, missing_ok=True)
    support = _matching_issuance(row, (lower,), exact=True, missing_ok=True)
    guaranteed = _matching_issuance(row, (upper,), exact=True, missing_ok=True)

    if joint is not None:
        if support is not None or guaranteed is not None:
            raise OrderingInputError(
                "newsvendor interval provenance cannot mix joint and split issuances"
            )
        if joint.descriptor.type.claim is not GuaranteeClaim.TWO_SIDED_COVERAGE:
            raise OrderingInputError(
                "newsvendor joint interval requires two-sided coverage provenance"
            )
        _validate_issuance(
            joint,
            expected_level=coverage,
            expected_window=EmissionScope.PER_STEP,
            values=row.values,
            consumed_columns=(lower, upper),
        )
        return joint.descriptor

    if support is None or guaranteed is None:
        raise OrderingInputError(
            "newsvendor requires either joint two-sided or canonical split interval provenance"
        )
    if support.descriptor.type.claim is not GuaranteeClaim.NONE:
        raise OrderingInputError(
            "newsvendor split interval requires claim-none lower support provenance"
        )
    if (
        guaranteed.descriptor.type.claim is not GuaranteeClaim.ONE_SIDED_COVERAGE
        or guaranteed.guaranteed_side is not GuaranteedSide.UPPER
    ):
        raise OrderingInputError(
            "newsvendor split interval requires one-sided upper guarantee provenance"
        )
    _validate_issuance(
        support,
        expected_level=coverage,
        expected_window=EmissionScope.PER_STEP,
        values=row.values,
        consumed_columns=(lower,),
    )
    _validate_issuance(
        guaranteed,
        expected_level=coverage,
        expected_window=EmissionScope.PER_STEP,
        values=row.values,
        consumed_columns=(upper,),
    )
    expected_support_descriptor = replace(
        guaranteed.descriptor,
        type=GuaranteeType(
            claim=GuaranteeClaim.NONE,
            currency=None,
            declared_slack=None,
        ),
    )
    if support.descriptor != expected_support_descriptor:
        raise OrderingInputError(
            "newsvendor split interval descriptors must share one emission context"
        )
    return guaranteed.descriptor


def _window_target(
    group: _DecisionGroup,
    configuration: OrderingConfiguration,
    *,
    explicit_quantile_descriptor: GuaranteeDescriptor | None,
) -> tuple[float, _Source]:
    required_steps = tuple(configuration.decision_timing.protection_window)
    _require_steps(group, required_steps)
    if configuration.explicit_quantile is not None:
        assert explicit_quantile_descriptor is not None
        return _explicit_quantile_target(
            group,
            configuration,
            required_steps,
            fallback_descriptor=explicit_quantile_descriptor,
        )

    coverage = configuration.coverage
    if coverage is None:
        raise OrderingInputError("window policy requires a configured coverage source")
    upper = interval_columns(coverage)[1]
    _require_columns(group, (upper,), family="interval")
    issued_by_step: list[tuple[int, BoundKey, ForecastIssuance]] = []
    for step in required_steps:
        row = group.rows[step]
        match = _matching_issuance_with_key(row, (upper,), exact=False)
        assert match is not None
        bound_key, issuance = match
        issued_by_step.append((step, bound_key, issuance))
    descriptor = _consistent_issuance(issued_by_step, expected_level=coverage)
    window = descriptor.window
    if window is EmissionScope.PER_STEP:
        values: list[float] = []
        for step, _bound_key, issuance in issued_by_step:
            row = group.rows[step]
            _validate_issuance(
                issuance,
                expected_level=coverage,
                expected_window=EmissionScope.PER_STEP,
                values=row.values,
                consumed_columns=(upper,),
            )
            values.append(_finite_value(row.values[upper], name=f"upper bound h={step}"))
        return _finite_sum(values, name="per-step target"), _Source((upper,), descriptor)
    if window is EmissionScope.WINDOW_SUM:
        terminal = required_steps[-1]
        for step, _bound_key, issuance in issued_by_step:
            value = group.rows[step].values[upper]
            value_is_finite = _is_finite_real(value)
            if step == terminal:
                if not issuance.bounds_finite or not value_is_finite:
                    raise OrderingInputError("terminal window bound must be finite")
                _validate_issuance(
                    issuance,
                    expected_level=coverage,
                    expected_window=EmissionScope.WINDOW_SUM,
                    values=group.rows[step].values,
                    consumed_columns=(upper,),
                )
            elif issuance.bounds_finite or not _is_missing_scalar(value):
                raise OrderingInputError(
                    "window-sum mode requires intentionally null non-terminal bounds"
                )
        target = _finite_value(
            group.rows[terminal].values[upper],
            name="terminal window bound must be finite",
        )
        return target, _Source((upper,), descriptor)
    raise OrderingInputError("issued descriptor has an unsupported emission scope")


def _explicit_quantile_target(
    group: _DecisionGroup,
    configuration: OrderingConfiguration,
    required_steps: tuple[int, ...],
    *,
    fallback_descriptor: GuaranteeDescriptor,
) -> tuple[float, _Source]:
    level = cast(float, configuration.explicit_quantile)
    column = quantile_column(level)
    _require_columns(group, (column,), family="quantile")
    matches = [
        _matching_issuance_with_key(
            group.rows[step],
            (column,),
            exact=True,
            missing_ok=True,
        )
        for step in required_steps
    ]
    present = [match is not None for match in matches]
    if any(present) and not all(present):
        raise OrderingInputError("explicit quantile issuance provenance is mixed or incomplete")

    values = [
        _finite_value(group.rows[step].values[column], name=f"explicit quantile h={step}")
        for step in required_steps
    ]
    if all(present):
        issued_by_step: list[tuple[int, BoundKey, ForecastIssuance]] = []
        for step, match in zip(required_steps, matches, strict=True):
            assert match is not None
            issued_by_step.append((step, *match))
        descriptor = _consistent_issuance(issued_by_step, expected_level=level)
        if descriptor.window is not EmissionScope.PER_STEP:
            raise OrderingInputError(
                "an issued explicit quantile must declare per-step emission scope"
            )
        for step, _bound_key, issuance in issued_by_step:
            _validate_issuance(
                issuance,
                expected_level=level,
                expected_window=EmissionScope.PER_STEP,
                values=group.rows[step].values,
                consumed_columns=(column,),
            )
    else:
        descriptor = fallback_descriptor
    return _finite_sum(values, name="explicit quantile target"), _Source(
        (column,),
        descriptor,
    )


def _matching_issuance(
    row: _PolicyRow,
    columns: tuple[str, ...],
    *,
    exact: bool,
    missing_ok: bool = False,
) -> ForecastIssuance | None:
    match = _matching_issuance_with_key(
        row,
        columns,
        exact=exact,
        missing_ok=missing_ok,
    )
    return None if match is None else match[1]


def _matching_issuance_with_key(
    row: _PolicyRow,
    columns: tuple[str, ...],
    *,
    exact: bool,
    missing_ok: bool = False,
) -> tuple[BoundKey, ForecastIssuance] | None:
    matches: list[tuple[BoundKey, ForecastIssuance]] = []
    for bound_key, issuance in row.issuances.items():
        if not isinstance(bound_key, tuple) or any(
            not isinstance(column, str) for column in bound_key
        ):
            raise OrderingInputError("issuance bound keys must be tuples of column names")
        matched = bound_key == columns if exact else all(column in bound_key for column in columns)
        if matched:
            if not isinstance(issuance, ForecastIssuance):
                raise OrderingInputError("consumed bound requires readable ForecastIssuance")
            matches.append((bound_key, issuance))
    if not matches:
        if missing_ok:
            return None
        raise OrderingInputError(
            f"consumed columns require readable issuance provenance: {columns!r}"
        )
    if len(matches) != 1:
        raise OrderingInputError(
            f"consumed columns have duplicate issuance provenance: {columns!r}"
        )
    return matches[0]


def _consistent_issuance(
    values: list[tuple[int, BoundKey, ForecastIssuance]],
    *,
    expected_level: float,
) -> GuaranteeDescriptor:
    first_key = values[0][1]
    first = values[0][2]
    if first.descriptor.level != expected_level:
        raise OrderingInputError("issued descriptor level does not match requested coverage")
    for _step, bound_key, issuance in values[1:]:
        if issuance.descriptor.level != expected_level:
            raise OrderingInputError("issued descriptor level does not match requested coverage")
        if (
            bound_key != first_key
            or issuance.descriptor != first.descriptor
            or issuance.guaranteed_side != first.guaranteed_side
        ):
            raise OrderingInputError("decision group carries mixed or inconsistent issuances")
    return first.descriptor


def _validate_issuance(
    issuance: ForecastIssuance,
    *,
    expected_level: float,
    expected_window: EmissionScope,
    values: Mapping[str, object],
    consumed_columns: tuple[str, ...],
) -> None:
    descriptor = issuance.descriptor
    if descriptor.level != expected_level:
        raise OrderingInputError("issued descriptor level does not match requested coverage")
    if descriptor.window is not expected_window:
        raise OrderingInputError("issued descriptor has the wrong emission scope")
    if (
        descriptor.type.claim is GuaranteeClaim.ONE_SIDED_COVERAGE
        and issuance.guaranteed_side is not GuaranteedSide.UPPER
    ):
        raise OrderingInputError("ordering requires the issued upper guaranteed side")
    payload_is_finite = all(_is_finite_real(values[column]) for column in consumed_columns)
    if not payload_is_finite or not issuance.bounds_finite:
        raise OrderingInputError("every consumed issued bound must be finite")


def _require_steps(group: _DecisionGroup, required_steps: tuple[int, ...]) -> None:
    missing = sorted(set(required_steps) - set(group.rows))
    if missing:
        raise OrderingInputError(
            f"decision group is missing required protection-window horizons: {missing!r}"
        )


def _require_columns(
    group: _DecisionGroup,
    columns: tuple[str, ...],
    *,
    family: str,
) -> None:
    for column in columns:
        if any(column not in row.values for row in group.rows.values()):
            raise OrderingInputError(f"requested {family} column is absent: {column}")


def _modified_target(
    raw_target: float,
    configuration: OrderingConfiguration,
) -> tuple[float, tuple[AppliedBinding, ...]]:
    if configuration.target_cap is not None:
        target = min(raw_target, configuration.target_cap)
        binding = AppliedBinding(
            name="target_cap",
            value=configuration.target_cap,
            bound=target != raw_target,
        )
    elif configuration.target_floor is not None:
        target = max(raw_target, configuration.target_floor)
        binding = AppliedBinding(
            name="target_floor",
            value=configuration.target_floor,
            bound=target != raw_target,
        )
    elif configuration.target_scale is not None:
        target = raw_target * configuration.target_scale
        binding = AppliedBinding(
            name="target_scale",
            value=configuration.target_scale,
            bound=target != raw_target,
        )
    else:
        return raw_target, ()
    return _finite_value(target, name="modified target"), (binding,)


def _rss_reorder_point(
    target: float,
    configuration: OrderingConfiguration,
) -> float:
    if configuration.reorder_point is not None:
        reorder_point = configuration.reorder_point
    else:
        scale = configuration.reorder_point_scale
        if scale is None:  # pragma: no cover - compile_ordering owns this invariant.
            raise OrderingInputError("rss configuration requires a reorder gate")
        reorder_point = _finite_value(scale * target, name="scaled reorder point")
    if not 0.0 <= reorder_point <= target:
        raise OrderingInputError("rss reorder point must satisfy 0 <= s <= target")
    return reorder_point


def _nonengine_quantile_descriptor(level: float) -> GuaranteeDescriptor:
    return GuaranteeDescriptor(
        type=GuaranteeType(
            claim=GuaranteeClaim.NONE,
            currency=None,
            declared_slack=None,
        ),
        level=level,
        scored_series=ScoredSeries.DEMAND_HONEST,
        window=EmissionScope.PER_STEP,
        scope=DecisionScope(
            kind=DecisionScopeKind.PER_DECISION_NODE,
            class_system_name=None,
        ),
    )


def _inventory_value(position: InventoryPosition) -> float:
    try:
        return position.value
    except ValueError as error:
        raise OrderingInputError("inventory position exceeds finite float range") from error


def _finite_sum(values: list[float], *, name: str) -> float:
    try:
        result = math.fsum(values)
    except OverflowError as error:
        raise OrderingInputError(f"{name} exceeds finite float range") from error
    return _finite_value(result, name=name)


def _finite_value(value: object, *, name: str) -> float:
    return _finite_real(value, name=name, error_type=OrderingInputError)


def _is_finite_real(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, Real):
        return False
    try:
        return math.isfinite(float(value))
    except (OverflowError, TypeError, ValueError):
        return False


def _is_missing_scalar(value: object) -> bool:
    if value is None or value is pd.NA or value is pd.NaT:
        return True
    if isinstance(value, bool) or not isinstance(value, Real):
        return False
    try:
        return math.isnan(float(value))
    except (OverflowError, TypeError, ValueError):
        return False


def _identifier(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise OrderingInputError(f"{name} must be a non-empty string")
    try:
        value.encode("utf-8")
    except UnicodeError as error:
        raise OrderingInputError(f"{name} must be valid UTF-8") from error
    return value


__all__ = ["PolicyDecision", "PolicyRequest", "dispatch_policy"]
