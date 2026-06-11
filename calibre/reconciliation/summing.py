"""Generic summing-matrix (S) construction from a hierarchy attribute frame.

The summing matrix maps a bottom-level forecast vector to the full set of nodes
in a cross-sectional hierarchy. It is derived **generically** from the attribute
columns of the hierarchy frame (any cross-sectional level set) — never hard-coded
to a single parent tree. Each attribute column is one grouping dimension; the
distinct values within a column are marginal aggregate nodes, so overlapping
memberships produce a *lattice*, not a single tree, per architecture §9.

Node layout (rows of S), in order:

* bottom identity block — one row per bottom ``unique_id`` (the S columns);
* one aggregate row per distinct value within each attribute column, labelled
  ``"<column>=<value>"``;
* a single grand-total row, labelled :data:`TOTAL_LABEL`.

``S @ b`` for a bottom vector ``b`` therefore reproduces every aggregate as the
sum of its member bottom series and the grand total as ``b.sum()``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from calibre.core.forecast_frame import UNIQUE_ID

TOTAL_LABEL = "__total__"


@dataclass(frozen=True, slots=True)
class HierarchyIndex:
    """Canonical labels and normalized attributes for a hierarchy frame."""

    frame: pd.DataFrame
    attr_cols: tuple[str, ...]
    bottom_ids: tuple[str, ...]
    node_labels: tuple[str, ...]

    def expected_members(self) -> dict[str, pd.Series]:
        """Per-attribute-column member counts, grouped on stringified values.

        The single counting authority every member-completeness site reads
        (``build_node_history``, ``HierarchyActualsSource``, the bottom-up
        reconciler, and the preflight estimate). Each attribute column is
        stringified with ``astype(str)`` before grouping, so the returned
        ``Series`` is keyed by the same stringified values that
        ``build_summing_matrix`` and ``node_labels`` use for aggregate labels.

        ``.size()`` and ``.nunique()`` coincide here: the index frame is
        deduplicated on stringified ``unique_id`` (see :func:`build_hierarchy_index`),
        so each bottom id contributes exactly one row per attribute column, and
        NaN attribute values are rejected at index build (so no group is dropped
        by ``nunique``'s NaN handling). The only behavioral difference from a
        raw ``groupby`` is that values colliding under ``str()`` (e.g. ``1`` and
        ``"1"``) are coherently merged into one group — matching what label
        production and the dense summing matrix already do — and that
        ``category``-dtype phantom groups for unobserved categories are dropped
        by construction.
        """
        return {
            col: self.frame[col].astype(str).groupby(self.frame[col].astype(str)).size()
            for col in self.attr_cols
        }


@dataclass(frozen=True, slots=True)
class SummingMatrix:
    """A dense summing matrix plus the node/bottom labels that index it.

    Attributes:
        S: ``(n_nodes, n_bottom)`` float64 matrix. Row order matches
            ``node_labels``; column order matches ``bottom_ids``.
        bottom_ids: ordered bottom-level ``unique_id`` labels (S columns).
        node_labels: ordered node labels (S rows): bottom identity block first,
            then per-attribute aggregate rows, then the grand total.
    """

    S: np.ndarray
    bottom_ids: tuple[str, ...]
    node_labels: tuple[str, ...]

    @property
    def n_bottom(self) -> int:
        return len(self.bottom_ids)

    @property
    def n_nodes(self) -> int:
        return len(self.node_labels)

    @property
    def total_index(self) -> int:
        return self.node_labels.index(TOTAL_LABEL)

    def subset(self, present_ids: Sequence[str]) -> SummingMatrix:
        """Restrict S to ``present_ids`` (a subset of the bottom ids).

        Columns are sliced to the present bottom ids (preserving canonical
        order) and any aggregate/identity row left with no present member is
        dropped, so a cross-section that forecasts only some series still aligns
        to a coherent summing matrix. The bottom identity block stays the leading
        rows, so the reconciled bottom vector remains ``S[:n_bottom]``.
        """
        wanted = {str(uid) for uid in present_ids}
        unknown = wanted - set(self.bottom_ids)
        if unknown:
            raise ValueError(f"present_ids not in summing matrix bottom ids: {sorted(unknown)}")
        col_idx = [i for i, uid in enumerate(self.bottom_ids) if uid in wanted]
        present = tuple(self.bottom_ids[i] for i in col_idx)
        sub = self.S[:, col_idx]
        keep = sub.sum(axis=1) > 0
        return SummingMatrix(
            S=sub[keep],
            bottom_ids=present,
            node_labels=tuple(label for label, k in zip(self.node_labels, keep, strict=True) if k),
        )


def build_hierarchy_index(hierarchy: pd.DataFrame) -> HierarchyIndex:
    """Validate a hierarchy frame and derive its canonical node labels."""
    if UNIQUE_ID not in hierarchy.columns:
        raise ValueError("hierarchy missing required column: unique_id")
    if hierarchy[UNIQUE_ID].isna().any():
        raise ValueError("hierarchy has null unique_id values")
    attr_cols = tuple(col for col in hierarchy.columns if col != UNIQUE_ID)
    frame = hierarchy.copy()
    frame[UNIQUE_ID] = frame[UNIQUE_ID].astype(str)
    frame = frame.sort_values(UNIQUE_ID, kind="stable").reset_index(drop=True)

    # Detect duplicates on the stringified ids: bottom_ids are derived from the
    # stringified column, so values colliding only as strings (e.g. 1 and "1")
    # must be rejected here too.
    if frame[UNIQUE_ID].duplicated().any():
        duplicates = frame.loc[frame[UNIQUE_ID].duplicated(), UNIQUE_ID]
        raise ValueError(f"hierarchy has duplicate unique_id rows: {sorted(duplicates.unique())}")

    for col in attr_cols:
        if frame[col].isna().any():
            raise ValueError(f"hierarchy attribute column {col!r} has null values")

    bottom_ids = tuple(frame[UNIQUE_ID].tolist())
    if not bottom_ids:
        raise ValueError("hierarchy has no rows")

    labels: list[str] = list(bottom_ids)
    for col in attr_cols:
        values = frame[col].astype(str)
        labels.extend(f"{col}={value}" for value in sorted(values.unique()))
    labels.append(TOTAL_LABEL)

    label_index = pd.Index(labels)
    if label_index.has_duplicates:
        duplicates = sorted(label_index[label_index.duplicated()].unique())
        raise ValueError(
            "hierarchy node labels must be unique; aggregate labels collide with "
            f"bottom unique_id values: {duplicates}"
        )

    return HierarchyIndex(
        frame=frame,
        attr_cols=attr_cols,
        bottom_ids=bottom_ids,
        node_labels=tuple(labels),
    )


def summing_matrix_from_index(hierarchy_index: HierarchyIndex) -> SummingMatrix:
    """Densify a prebuilt :class:`HierarchyIndex` into a :class:`SummingMatrix`.

    The dense S is derived on demand by the strategies that genuinely need it;
    it is never owned by the index or run preparation. Callers that hold the
    threaded index densify through this entry point so no index facts are
    re-derived (same S bytes as the from-frame path).
    """
    frame = hierarchy_index.frame
    attr_cols = hierarchy_index.attr_cols
    bottom_ids = hierarchy_index.bottom_ids
    n_bottom = len(bottom_ids)

    rows: list[np.ndarray] = list(np.eye(n_bottom, dtype=np.float64))

    for col in attr_cols:
        values = frame[col].astype(str)
        for value in sorted(values.unique()):
            rows.append((values == value).to_numpy(dtype=np.float64))

    rows.append(np.ones(n_bottom, dtype=np.float64))

    return SummingMatrix(
        S=np.vstack(rows),
        bottom_ids=bottom_ids,
        node_labels=hierarchy_index.node_labels,
    )


def build_summing_matrix(hierarchy: pd.DataFrame) -> SummingMatrix:
    """Build a :class:`SummingMatrix` from a hierarchy attribute frame.

    ``hierarchy`` must carry a ``unique_id`` column; every other column is
    treated as a cross-sectional grouping dimension (discovered generically). The
    bottom ids are sorted for deterministic node labels. Config-level callers
    that only have the raw frame use this; callers holding the threaded index
    densify through :func:`summing_matrix_from_index` instead.
    """
    return summing_matrix_from_index(build_hierarchy_index(hierarchy))
