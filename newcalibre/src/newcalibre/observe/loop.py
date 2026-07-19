"""Resolve accepted actuals into deterministic staged conformal deliveries."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from numbers import Integral
from types import MappingProxyType

import pandas as pd

from newcalibre.conformal import (
    METHOD_SCOPE_LABEL,
    CalibrationContext,
    ConformalRuntime,
    Delivery,
    ObserveEffect,
    ResolvedObservation,
)
from newcalibre.domain import (
    CensoringAssertion,
    EmissionScope,
    HierarchyIndex,
    HierarchyNode,
)
from newcalibre.observe.state import (
    Acceptance,
    ObservationResolution,
    ObserveCycle,
    ObservedActual,
    PendingObservation,
)
from newcalibre.observe.submission import (
    ActualKey,
    ActualsSubmission,
    ObserveError,
    _require_timestamp,
    _snapshot_iterable,
)


class ObserveLoop:
    """Stage one actual-acceptance and conformal-observation cycle in memory."""

    __slots__ = (
        "_committed_history",
        "_conformal_states",
        "_hierarchy",
        "_node_by_label",
        "_pending",
        "_runtime",
        "_staged_history",
    )

    def __init__(
        self,
        *,
        hierarchy: HierarchyIndex,
        observed_history: Iterable[ObservedActual] | Mapping[ActualKey, ObservedActual] = (),
        pending_observations: Iterable[PendingObservation] = (),
        conformal_states: Mapping[str, bytes | None] | None = None,
        runtime: ConformalRuntime | None = None,
    ) -> None:
        if not isinstance(hierarchy, HierarchyIndex):
            raise ObserveError("observe hierarchy must be a HierarchyIndex")
        history = _history_snapshot(observed_history)
        unknown_history = sorted(
            {value.series_key for value in history} - set(hierarchy.bottom_series),
            key=str.encode,
        )
        if unknown_history:
            raise ObserveError(
                f"observed history contains unknown bottom series: {unknown_history}"
            )
        pending = _pending_snapshot(pending_observations, hierarchy=hierarchy)
        states = _state_snapshot(conformal_states)
        if runtime is not None and not isinstance(runtime, ConformalRuntime):
            raise ObserveError("observe runtime must implement ConformalRuntime")

        self._hierarchy = hierarchy
        self._node_by_label = {node.label: node for node in hierarchy.nodes}
        self._committed_history = {value.key: value for value in history}
        self._pending = pending
        self._conformal_states = states
        self._runtime = runtime
        self._staged_history: dict[ActualKey, ObservedActual] = {}

    @property
    def committed_history(self) -> tuple[ObservedActual, ...]:
        """Return the caller-owned history snapshot in supplied storage order."""
        return tuple(self._committed_history.values())

    @property
    def staged_history(self) -> tuple[ObservedActual, ...]:
        """Return accepted uncommitted history appends in acceptance order."""
        return tuple(self._staged_history.values())

    @property
    def pending_observations(self) -> tuple[PendingObservation, ...]:
        """Return the immutable pending-row snapshot in durable append order."""
        return self._pending

    @property
    def conformal_states(self) -> Mapping[str, bytes | None]:
        """Return an isolated read-only snapshot of prior opaque state."""
        return MappingProxyType(dict(self._conformal_states))

    def accept(self, submission: ActualsSubmission) -> Acceptance:
        """Validate a whole actual submission before staging any new history fact."""
        if not isinstance(submission, ActualsSubmission):
            raise ObserveError("observe acceptance requires an ActualsSubmission")
        allowed = set(self._hierarchy.bottom_series)
        invalid = sorted(
            {
                record.series_key
                for record in submission.records
                if record.series_key not in allowed
            },
            key=str.encode,
        )
        if invalid:
            raise ObserveError(
                "actuals submissions may address declared bottom series only; "
                f"invalid keys: {invalid}"
            )

        staged: list[ObservedActual] = []
        idempotent: list[ActualKey] = []
        known = {**self._committed_history, **self._staged_history}
        for record in submission.records:
            value = ObservedActual.from_record(record)
            previous = known.get(value.key)
            if previous is None:
                staged.append(value)
                known[value.key] = value
                continue
            if previous.recorded_fact != value.recorded_fact:
                raise ObserveError(
                    f"conflicting actual for ({value.series_key!r}, {value.timestamp.isoformat()})"
                )
            idempotent.append(value.key)

        for value in staged:
            self._staged_history[value.key] = value
        return Acceptance(staged, idempotent)

    def cycle(self, origin: pd.Timestamp) -> ObserveCycle:
        """Resolve due rows and return one deterministic unpersisted cycle delta."""
        _require_timestamp(origin, name="observe-cycle origin")
        history = {**self._committed_history, **self._staged_history}
        candidates = tuple(
            self._resolve_pending(row, history=history, origin=origin) for row in self._pending
        )
        ready_keys = self._ready_keys(candidates, origin=origin)
        ready = tuple(row for row in candidates if row.forecast_key in ready_keys)
        retained = tuple(row for row in candidates if row.forecast_key not in ready_keys)

        deliveries: tuple[Delivery, ...] = ()
        annotations = ()
        updates: dict[str, bytes] = {}
        if self._runtime is not None and ready:
            deliveries = self._deliveries(ready)
            annotations, updates = self._observe_deliveries(deliveries)

        resolutions = tuple(row.resolution for row in ready if row.resolution is not None)
        return ObserveCycle(
            history_appends=self._staged_history.values(),
            resolutions=resolutions,
            pending_removals=(row.forecast_key for row in ready),
            pending_retentions=retained,
            deliveries=deliveries,
            annotations=annotations,
            state_updates=updates,
        )

    def _resolve_pending(
        self,
        row: PendingObservation,
        *,
        history: Mapping[ActualKey, ObservedActual],
        origin: pd.Timestamp,
    ) -> PendingObservation:
        if row.resolution is not None or row.target_timestamp >= origin:
            return row
        node = self._node_by_label[row.forecast_key.series_key]
        resolution = (
            self._bottom_resolution(row, history=history)
            if len(node.members) == 1 and node.label == node.members[0]
            else self._aggregate_resolution(row, node=node, history=history)
        )
        if resolution is None:
            return row
        return PendingObservation(
            forecast_key=row.forecast_key,
            target_timestamp=row.target_timestamp,
            point_forecast=row.point_forecast,
            issued=row.issued,
            resolution=resolution,
        )

    def _bottom_resolution(
        self,
        row: PendingObservation,
        *,
        history: Mapping[ActualKey, ObservedActual],
    ) -> ObservationResolution | None:
        actual = history.get((row.forecast_key.series_key, row.target_timestamp))
        if actual is None:
            return None
        return ObservationResolution(
            forecast_key=row.forecast_key,
            target_timestamp=row.target_timestamp,
            actual=actual.recorded_value,
            censoring_assertion=actual.censoring_assertion,
            availability_bound=actual.availability_bound,
        )

    def _aggregate_resolution(
        self,
        row: PendingObservation,
        *,
        node: HierarchyNode,
        history: Mapping[ActualKey, ObservedActual],
    ) -> ObservationResolution | None:
        members: list[ObservedActual] = []
        for member in node.members:
            actual = history.get((member, row.target_timestamp))
            if actual is None:
                return None
            members.append(actual)
        aggregated = self._hierarchy.aggregate(
            {value.series_key: value.recorded_value for value in members},
            node_labels=(node.label,),
        )[node.label]
        if aggregated is None:
            raise ObserveError("complete hierarchy members produced an undefined aggregate")
        assertion = _aggregate_censoring(members)
        return ObservationResolution(
            forecast_key=row.forecast_key,
            target_timestamp=row.target_timestamp,
            actual=aggregated,
            censoring_assertion=assertion,
            availability_bound=None,
        )

    def _ready_keys(
        self,
        rows: tuple[PendingObservation, ...],
        *,
        origin: pd.Timestamp,
    ) -> set:
        resolved_due = {
            row.forecast_key: row
            for row in rows
            if row.target_timestamp < origin and row.resolution is not None
        }
        if self._runtime is None:
            return set(resolved_due)
        scope = self._runtime.manifest.emission_scope
        if scope is EmissionScope.PER_STEP:
            return set(resolved_due)
        if scope is not EmissionScope.WINDOW_SUM:
            raise ObserveError(f"unsupported conformal emission scope: {scope!r}")
        period = _protection_period(self._runtime)
        groups: dict[tuple[str, str, pd.Timestamp], dict[int, PendingObservation]] = {}
        for row in rows:
            key = row.forecast_key
            group_key = (key.series_key, key.model_name, key.origin)
            groups.setdefault(group_key, {})[key.horizon_step] = row
        ready: set = set()
        for members in groups.values():
            leading = tuple(members.get(step) for step in range(1, period + 1))
            if any(value is None for value in leading):
                continue
            complete = tuple(value for value in leading if value is not None)
            if all(value.forecast_key in resolved_due for value in complete):
                ready.update(value.forecast_key for value in complete)
        return ready

    def _deliveries(self, ready: tuple[PendingObservation, ...]) -> tuple[Delivery, ...]:
        if self._runtime is None:
            return ()
        by_partition: dict[str, list[ResolvedObservation]] = {}
        for row in ready:
            issued = row.issued
            if issued is None:
                raise ObserveError(
                    f"deliverable row {row.forecast_key!r} is missing issued conformal facts"
                )
            manifest = self._runtime.manifest
            if (
                issued.method_name != manifest.name
                or issued.emission_form is not manifest.emission_form
                or issued.emission_scope is not manifest.emission_scope
            ):
                raise ObserveError(
                    f"deliverable row {row.forecast_key!r} has conflicting runtime issuance facts"
                )
            resolution = row.resolution
            if resolution is None:
                raise ObserveError("a deliverable row must carry a resolved actual")
            observation = ResolvedObservation(
                forecast_key=row.forecast_key,
                target_timestamp=row.target_timestamp,
                actual=float(resolution.actual),
                point_forecast=row.point_forecast,
                censoring_assertion=resolution.censoring_assertion,
                availability_bound=resolution.availability_bound,
                issued=issued,
            )
            by_partition.setdefault(issued.partition_label, []).append(observation)

        deliveries: list[Delivery] = []
        for label in sorted(by_partition, key=str.encode):
            observations = tuple(sorted(by_partition[label], key=_delivery_order))
            deliveries.append(Delivery(label, observations))
        return tuple(deliveries)

    def _observe_deliveries(
        self,
        deliveries: tuple[Delivery, ...],
    ) -> tuple[tuple, dict[str, bytes]]:
        if self._runtime is None:
            return (), {}
        evolving = dict(self._conformal_states)
        updates: dict[str, bytes] = {}
        annotations = []
        for delivery in deliveries:
            context = self._calibration_context(delivery)
            try:
                effect = self._runtime.observe(
                    delivery,
                    MappingProxyType(dict(evolving)),
                    context=context,
                )
            except ValueError as error:
                raise ObserveError(
                    f"conformal observe failed for partition {delivery.partition_label!r}: {error}"
                ) from error
            if not isinstance(effect, ObserveEffect):
                raise ObserveError("conformal observe must return an ObserveEffect")
            expected = {value.forecast_key for value in delivery.observations}
            actual = {value.forecast_key for value in effect.annotations}
            if actual != expected:
                raise ObserveError(
                    "conformal observe annotations must exactly cover the partition delivery"
                )
            emitted = dict(effect.state_updates)
            if delivery.partition_label not in emitted:
                raise ObserveError("conformal observe omitted the touched partition state update")
            foreign = set(emitted) - {delivery.partition_label, METHOD_SCOPE_LABEL}
            if foreign:
                raise ObserveError(
                    f"conformal observe emitted foreign state updates: {sorted(foreign)}"
                )
            for label, value in emitted.items():
                if label != METHOD_SCOPE_LABEL and label in updates and updates[label] != value:
                    raise ObserveError(f"conformal observe emitted conflicting state for {label!r}")
                evolving[label] = value
                updates[label] = value
            annotations.extend(effect.annotations)
        return tuple(annotations), updates

    def _calibration_context(self, delivery: Delivery) -> CalibrationContext | None:
        if self._runtime is None or not self._runtime.manifest.consumes_calibration_context:
            return None
        nodes = tuple(
            self._node_by_label[value.forecast_key.series_key] for value in delivery.observations
        )
        return CalibrationContext(
            series_keys=tuple(node.label for node in nodes),
            lattice_levels=tuple(node.kind.value for node in nodes),
            aggregate_memberships=tuple(node.members for node in nodes),
        )


def _history_snapshot(
    values: Iterable[ObservedActual] | Mapping[ActualKey, ObservedActual],
) -> tuple[ObservedActual, ...]:
    if isinstance(values, Mapping):
        items = tuple(values.items())
        raw_history: tuple[object, ...] = tuple(value for _key, value in items)
        for key, value in items:
            if not isinstance(value, ObservedActual) or key != value.key:
                raise ObserveError("observed-history mappings must use each record's exact key")
    else:
        raw_history = _snapshot_iterable(values, name="observed history")
    if any(not isinstance(value, ObservedActual) for value in raw_history):
        raise ObserveError("observed history must contain ObservedActual values")
    history = tuple(value for value in raw_history if isinstance(value, ObservedActual))
    keys = tuple(value.key for value in history)
    if len(set(keys)) != len(keys):
        raise ObserveError("observed history contains duplicate keys")
    return history


def _pending_snapshot(
    values: Iterable[PendingObservation],
    *,
    hierarchy: HierarchyIndex,
) -> tuple[PendingObservation, ...]:
    pending = _snapshot_iterable(values, name="pending observations")
    if any(not isinstance(value, PendingObservation) for value in pending):
        raise ObserveError("pending observations must contain PendingObservation values")
    keys = tuple(value.forecast_key for value in pending)
    if len(set(keys)) != len(keys):
        raise ObserveError("pending observations contain duplicate forecast keys")
    known = set(hierarchy.node_labels)
    unknown = sorted(
        {
            value.forecast_key.series_key
            for value in pending
            if value.forecast_key.series_key not in known
        },
        key=str.encode,
    )
    if unknown:
        raise ObserveError(f"pending observations contain unknown hierarchy nodes: {unknown}")
    return pending


def _state_snapshot(values: Mapping[str, bytes | None] | None) -> dict[str, bytes | None]:
    if values is None:
        return {}
    if not isinstance(values, Mapping):
        raise ObserveError("conformal states must be a mapping")
    snapshot = dict(values)
    for label, value in snapshot.items():
        if not isinstance(label, str) or not label:
            raise ObserveError("conformal state labels must be non-empty strings")
        if value is not None and not isinstance(value, bytes):
            raise ObserveError("conformal state values must be immutable bytes or missing")
    return snapshot


def _aggregate_censoring(members: Iterable[ObservedActual]) -> CensoringAssertion | None:
    assertions = tuple(value.censoring_assertion for value in members)
    if CensoringAssertion.CENSORED in assertions:
        return CensoringAssertion.CENSORED
    if all(value is CensoringAssertion.UNCENSORED for value in assertions):
        return CensoringAssertion.UNCENSORED
    return None


def _protection_period(runtime: ConformalRuntime) -> int:
    period = getattr(runtime.config, "protection_period", None)
    if isinstance(period, bool) or not isinstance(period, Integral) or period < 1:
        raise ObserveError("window-sum runtime configuration requires a positive protection_period")
    return int(period)


def _delivery_order(observation: ResolvedObservation) -> tuple:
    key = observation.forecast_key
    return key.origin, key.horizon_step, key.series_key.encode(), key.model_name.encode()


__all__ = ["ObserveLoop"]
