"""Validate and apply points-only reconciliation by isolated cross-section."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import numpy as np
import pandas as pd

from newcalibre.domain import (
    ACTUAL_VALUE,
    HORIZON_STEP,
    MODEL_NAME,
    ORIGIN,
    POINT_FORECAST,
    REQUIRED_FRAME_COLUMNS,
    SERIES_KEY,
    ForecastFrameError,
    HierarchyError,
    HierarchyIndex,
    HierarchyNode,
    HierarchyNodeKind,
    forecast_bound_groups,
)
from newcalibre.reconcile.protocol import ReconcilerDeclaration, ReconciliationContext
from newcalibre.reconcile.summing import SparseSummingMatrix, build_sparse_summing_matrix
from newcalibre.reconcile.tolerance import coherence_tolerance

_CROSS_SECTION_COLUMNS = (MODEL_NAME, ORIGIN, HORIZON_STEP)


class ReconciliationError(ValueError):
    """Report a points-only reconciliation contract violation."""


@dataclass(frozen=True, slots=True)
class _CrossSection:
    identity: tuple[str, pd.Timestamp, int]
    positions: tuple[int, ...]

    @property
    def description(self) -> str:
        model, origin, step = self.identity
        return f"cross-section (model={model!r}, origin={origin}, horizon_step={step})"


def apply_none(
    frame: pd.DataFrame,
    hierarchy: HierarchyIndex | None,
    context: ReconciliationContext,
    *,
    declaration: ReconcilerDeclaration,
) -> pd.DataFrame:
    """Validate an active point frame and return its strict identity."""
    active = _active_inputs(frame, hierarchy, context, declaration=declaration)
    if active is None:
        return frame
    _validated_sections(frame, active)
    return frame


def apply_bottom_up(
    frame: pd.DataFrame,
    hierarchy: HierarchyIndex | None,
    context: ReconciliationContext,
    *,
    declaration: ReconcilerDeclaration,
) -> pd.DataFrame:
    """Append every all-members-present aggregate after unchanged bottom rows."""
    active = _active_inputs(frame, hierarchy, context, declaration=declaration)
    if active is None:
        return frame
    sections = _validated_sections(frame, active, allow_aggregate_rows=True)
    matrix = build_sparse_summing_matrix(active)
    bottom_labels = set(active.bottom_series)
    aggregate_rows: list[pd.DataFrame] = []

    for section in sections:
        section_rows = frame.iloc[list(section.positions)]
        source = section_rows.loc[section_rows[SERIES_KEY].isin(bottom_labels)]
        values = dict(zip(source[SERIES_KEY], source[POINT_FORECAST], strict=True))
        eligible = tuple(
            node
            for node in active.nodes
            if node.kind is not HierarchyNodeKind.BOTTOM
            and all(member in values for member in node.members)
        )
        try:
            aggregated = active.aggregate(
                values,
                node_labels=(node.label for node in eligible),
            )
        except HierarchyError as error:
            raise ReconciliationError(f"{section.description}: {error}") from error
        _verify_cross_section(
            source,
            eligible=eligible,
            aggregated=aggregated,
            matrix=matrix,
            section=section,
        )
        _validate_existing_aggregates(
            section_rows,
            bottom_labels=bottom_labels,
            eligible=eligible,
            aggregated=aggregated,
            matrix=matrix,
            section=section,
        )
        if not eligible:
            continue
        rows = source.iloc[[0] * len(eligible)].copy(deep=True).reset_index(drop=True)
        point_values: list[float] = []
        for node in eligible:
            value = aggregated[node.label]
            point_values.append(np.nan if value is None else float(value))
        rows[SERIES_KEY] = pd.array(
            [node.label for node in eligible],
            dtype=pd.StringDtype(storage="pyarrow"),
        )
        rows[ACTUAL_VALUE] = np.nan
        rows[POINT_FORECAST] = point_values
        aggregate_rows.append(rows)

    bottom_rows = frame.loc[frame[SERIES_KEY].isin(bottom_labels)].copy(deep=True)
    if not aggregate_rows:
        return bottom_rows
    return pd.concat([bottom_rows, *aggregate_rows], ignore_index=True)


def _active_inputs(
    frame: object,
    hierarchy: object,
    context: object,
    *,
    declaration: object,
) -> HierarchyIndex | None:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("reconciliation frame must be a pandas DataFrame")
    if hierarchy is not None and not isinstance(hierarchy, HierarchyIndex):
        raise TypeError("reconciliation hierarchy must be a HierarchyIndex or None")
    if not isinstance(context, ReconciliationContext):
        raise TypeError("reconciliation context must be a ReconciliationContext")
    if not isinstance(declaration, ReconcilerDeclaration):
        raise TypeError("reconciliation declaration must be a ReconcilerDeclaration")
    if hierarchy is None or frame.empty:
        return None
    if declaration.requires_fitted_values and context.fitted_values is None:
        raise ReconciliationError(
            f"strategy {declaration.name!r} requires fitted values before reconciliation"
        )
    return hierarchy


def _validated_sections(
    frame: pd.DataFrame,
    hierarchy: HierarchyIndex,
    *,
    allow_aggregate_rows: bool = False,
) -> tuple[_CrossSection, ...]:
    if frame.columns.has_duplicates:
        raise ReconciliationError("reconciliation frame cannot have duplicate column labels")
    missing = [column for column in REQUIRED_FRAME_COLUMNS if column not in frame.columns]
    if missing:
        raise ReconciliationError(
            "reconciliation frame is missing required columns: " + ", ".join(missing)
        )
    try:
        bound_groups = forecast_bound_groups(frame.columns)
    except ForecastFrameError as error:
        raise ReconciliationError(str(error)) from error
    if bound_groups:
        names = [column for group in bound_groups for column in group]
        raise ReconciliationError(
            f"reconciliation accepts point forecasts only; distributional columns={names}"
        )
    if frame[list(_CROSS_SECTION_COLUMNS)].isna().any(axis=None):
        raise ReconciliationError("reconciliation cross-section identity cannot be missing")

    grouped = frame.groupby(
        list(_CROSS_SECTION_COLUMNS),
        sort=False,
        observed=True,
        dropna=False,
    ).indices
    sections: list[_CrossSection] = []
    for raw_identity, raw_positions in grouped.items():
        if not isinstance(raw_identity, tuple) or len(raw_identity) != 3:
            raise ReconciliationError("reconciliation produced an invalid cross-section key")
        model, origin, step = raw_identity
        if not isinstance(model, str) or not model:
            raise ReconciliationError("reconciliation model names must be non-empty strings")
        if not isinstance(origin, pd.Timestamp):
            raise ReconciliationError("reconciliation origins must be pandas Timestamps")
        if not isinstance(step, Integral) or isinstance(step, bool) or step < 1:
            raise ReconciliationError("reconciliation horizon steps must be positive integers")
        identity = (model, origin, int(step))
        sections.append(
            _CrossSection(
                identity=identity,
                positions=tuple(int(position) for position in raw_positions),
            )
        )
    sections.sort(
        key=lambda section: (
            section.identity[0].encode(),
            section.identity[1].value,
            section.identity[2],
        )
    )

    known = set(hierarchy.node_labels)
    bottoms = set(hierarchy.bottom_series)
    for section in sections:
        source = frame.iloc[list(section.positions)]
        labels = tuple(source[SERIES_KEY])
        if any(not isinstance(label, str) or not label for label in labels):
            raise ReconciliationError(
                f"{section.description} contains a missing or invalid series key"
            )
        duplicate = sorted(
            source.loc[source[SERIES_KEY].duplicated(), SERIES_KEY].unique(),
            key=str.encode,
        )
        if duplicate:
            raise ReconciliationError(
                f"{section.description} contains duplicate node rows: {duplicate}"
            )
        uncovered = sorted(set(labels) - known, key=str.encode)
        if uncovered:
            raise ReconciliationError(
                f"{section.description} contains series not covered by the hierarchy: {uncovered}"
            )
        non_bottom = sorted(set(labels) - bottoms, key=str.encode)
        if non_bottom and not allow_aggregate_rows:
            raise ReconciliationError(
                f"{section.description} requires bottom-node rows only: {non_bottom}"
            )
    return tuple(sections)


def _validate_existing_aggregates(
    source: pd.DataFrame,
    *,
    bottom_labels: set[str],
    eligible: tuple[HierarchyNode, ...],
    aggregated: dict[str, int | float | None],
    matrix: SparseSummingMatrix,
    section: _CrossSection,
) -> None:
    existing = source.loc[~source[SERIES_KEY].isin(bottom_labels)]
    if existing.empty:
        return
    eligible_labels = {node.label for node in eligible}
    ineligible = sorted(set(existing[SERIES_KEY]) - eligible_labels, key=str.encode)
    if ineligible:
        raise ReconciliationError(
            f"{section.description} contains aggregate rows without complete bottom "
            f"membership: {ineligible}"
        )
    try:
        actual = existing[POINT_FORECAST].to_numpy(dtype=np.float64, na_value=np.nan)
    except (TypeError, ValueError, OverflowError) as error:
        raise ReconciliationError(
            f"{section.description} aggregate point forecasts must be real numeric values"
        ) from error
    expected = np.asarray(
        [
            np.nan if aggregated[label] is None else aggregated[label]
            for label in existing[SERIES_KEY]
        ],
        dtype=np.float64,
    )
    finite = np.abs(np.concatenate((actual, expected)))
    finite = finite[np.isfinite(finite)]
    magnitude = float(finite.max()) if finite.size else 0.0
    bottom_values = source.loc[source[SERIES_KEY].isin(bottom_labels), SERIES_KEY]
    bound = coherence_tolerance(
        reduction_width=matrix.subset(bottom_values).reduction_width,
        vector_magnitude=magnitude,
    )
    if not np.allclose(actual, expected, rtol=0.0, atol=bound, equal_nan=True):
        raise ReconciliationError(
            f"{section.description} contains aggregate rows inconsistent with bottom forecasts"
        )


def _verify_cross_section(
    source: pd.DataFrame,
    *,
    eligible: tuple[HierarchyNode, ...],
    aggregated: dict[str, int | float | None],
    matrix: SparseSummingMatrix,
    section: _CrossSection,
) -> None:
    values = dict(zip(source[SERIES_KEY], source[POINT_FORECAST], strict=True))
    subset = matrix.subset(values)
    bottom = np.asarray([values[label] for label in subset.bottom_ids], dtype=np.float64)
    expected = subset.matvec(bottom)
    expected_by_label = dict(zip(subset.node_labels, expected, strict=True))
    actual_labels = (*subset.bottom_ids, *(node.label for node in eligible))
    actual = np.asarray(
        [
            values[label]
            if label in values
            else np.nan
            if aggregated[label] is None
            else aggregated[label]
            for label in actual_labels
        ],
        dtype=np.float64,
    )
    coherent = np.asarray([expected_by_label[label] for label in actual_labels], dtype=np.float64)
    finite = np.abs(np.concatenate((actual, coherent)))
    finite = finite[np.isfinite(finite)]
    magnitude = float(finite.max()) if finite.size else 0.0
    bound = coherence_tolerance(
        reduction_width=subset.reduction_width,
        vector_magnitude=magnitude,
    )
    if not np.allclose(actual, coherent, rtol=0.0, atol=bound, equal_nan=True):
        raise ReconciliationError(
            f"{section.description} failed the derived summing-matrix coherence check"
        )
