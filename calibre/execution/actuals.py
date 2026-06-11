"""Engine-level actual resolution sources.

The engine owns delayed-feedback *timing* (`BackendEngine._resolve_due` decides
which pending ledger rows are due at the current origin); an
:class:`ActualsSource` owns only the *lookup*: given the resolution frame, fill
``y`` for rows that are now observable. Two implementations exist:

* :class:`FrameActualsSource` — the eager path. Wraps a pre-materialized actuals
  frame (flat panels, or a fully expanded node history) and resolves through
  :func:`calibre.evaluation.forecast_metrics.resolve_actuals` unchanged.
* :class:`HierarchyActualsSource` — the lazy hierarchy path. Holds only the
  bottom-level history plus hierarchy facts and computes aggregate node actuals
  on demand for the ledger rows being resolved, instead of prebuilding the full
  node-history frame. Aggregate values follow the same completeness rule as
  ``build_node_history``: a node's actual exists for a date only when every
  bottom member of that node is observed on that date.
"""

from __future__ import annotations

from typing import Protocol

import pandas as pd

from calibre.core.forecast_frame import DS, UNIQUE_ID, Y, validate_actuals_frame
from calibre.evaluation.forecast_metrics import resolve_actuals
from calibre.reconciliation.summing import TOTAL_LABEL, build_hierarchy_index


class ActualsSource(Protocol):
    """Resolve pending ledger rows that are due as of ``current_origin``.

    ``resolve`` mirrors :func:`resolve_actuals`: it fills ``y`` where
    ``ds <= current_origin`` and ``y`` is currently NaN, returning
    ``(updated_ledger, newly_resolved_rows)``. Rows whose actual is not (yet)
    observable stay pending with ``y`` NaN.
    """

    def resolve(
        self,
        ledger_df: pd.DataFrame,
        current_origin: pd.Timestamp,
    ) -> tuple[pd.DataFrame, pd.DataFrame]: ...


class FrameActualsSource:
    """Frame-backed source — the existing eager lookup behavior."""

    def __init__(self, frame: pd.DataFrame) -> None:
        validate_actuals_frame(frame)
        self._frame = frame

    def resolve(
        self,
        ledger_df: pd.DataFrame,
        current_origin: pd.Timestamp,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        return resolve_actuals(ledger_df, self._frame, current_origin)


class HierarchyActualsSource:
    """Resolve hierarchy node actuals on demand from bottom history.

    Holds the bottom-level history and the hierarchy attribute frame. Bottom
    node actuals are direct lookups; ``"<column>=<value>"`` aggregates and the
    grand total are summed over their member bottom series only for the
    requested ``(node, ds)`` pairs, with the same all-members-present
    completeness rule as ``build_node_history`` — so resolution parity with the
    eager node-history frame holds row-for-row.
    """

    def __init__(self, bottom_history: pd.DataFrame, hierarchy: pd.DataFrame) -> None:
        validate_actuals_frame(bottom_history)
        self._index = build_hierarchy_index(hierarchy)

        data = bottom_history[[UNIQUE_ID, DS, Y]].copy()
        data[UNIQUE_ID] = data[UNIQUE_ID].astype(str)
        data[DS] = pd.to_datetime(data[DS]).astype("datetime64[ns]")
        data[Y] = data[Y].astype("float64")

        unknown = set(data[UNIQUE_ID].unique()) - set(self._index.bottom_ids)
        if unknown:
            raise ValueError(
                f"history contains unique_id values not present in hierarchy: {sorted(unknown)}"
            )
        # Duplicate (unique_id, ds) keys make aggregate actuals ill-defined
        # (sums would double-count), so reject them loudly instead of silently
        # picking a row.
        duplicated = data.duplicated(subset=[UNIQUE_ID, DS])
        if duplicated.any():
            keys = (
                data.loc[duplicated, [UNIQUE_ID, DS]]
                .drop_duplicates()
                .itertuples(index=False, name=None)
            )
            raise ValueError(
                "bottom history contains duplicate (unique_id, ds) rows: "
                f"{sorted(str(key) for key in keys)[:10]}"
            )
        self._bottom = data
        self._node_labels = set(self._index.node_labels)
        self._bottom_ids = set(self._index.bottom_ids)
        self._members_by_attr = {
            col: self._index.frame.groupby(col, sort=False)[UNIQUE_ID].nunique()
            for col in self._index.attr_cols
        }
        # Per-run cache of *complete* (node, ds) -> y sums. Bottom history is
        # fixed at construction, so completeness per (node, ds) is fixed for the
        # whole run and a cached entry never invalidates — a future change that
        # let bottom history grow mid-run (streaming history) would break this
        # invariant silently, so it is asserted here in prose. Only complete
        # aggregates are cached; incomplete ones are absent (never cached as
        # absent-forever facts) and recomputed on the fixed-history complete-set
        # check the next time their (node, ds) is requested.
        self._lookup_cache: dict[tuple[str, pd.Timestamp], float] = {}

    def resolve(
        self,
        ledger_df: pd.DataFrame,
        current_origin: pd.Timestamp,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        updated = ledger_df.copy()

        mask_pending = updated[Y].isna() & (updated[DS] <= current_origin)
        if not mask_pending.any():
            return updated, pd.DataFrame(columns=updated.columns)

        pending_idx = updated.index[mask_pending]
        pending_uids = updated.loc[pending_idx, UNIQUE_ID].astype(str)
        unknown = set(pending_uids.unique()) - self._node_labels
        if unknown:
            raise ValueError(
                f"ledger contains unique_id values not present in hierarchy: {sorted(unknown)}"
            )

        pending_ds = pd.to_datetime(updated.loc[pending_idx, DS])
        lookup = self._lookup_for(pending_uids, pending_ds)
        pending_keys = pd.MultiIndex.from_arrays([pending_uids.values, pending_ds.values])
        updated.loc[pending_idx, Y] = lookup.reindex(pending_keys).values

        newly_resolved = updated.loc[pending_idx[updated.loc[pending_idx, Y].notna()]].copy()
        return updated, newly_resolved

    def _lookup_for(self, uids: pd.Series, ds_values: pd.Series) -> pd.Series:
        """Build a ``(unique_id, ds) -> y`` lookup for the requested rows only.

        Wraps :meth:`_compute_lookup` with the per-run complete-sum cache: every
        requested ``(node, ds)`` is computed at most once per run. Already-cached
        pairs are served directly; only the not-yet-computed pairs reach the
        window scan, attribute merge, and group-by — so a forever-pending
        incomplete aggregate that re-enters the due set every origin costs a
        cheap lookup-miss for its already-resolved siblings instead of a full
        rebuild. The returned shape (a Series indexed by ``(unique_id, ds)``
        carrying only the complete entries) is identical to computing without
        the cache.
        """
        # Cache keys are (str uid, pd.Timestamp ds) — the same native types the
        # _compute_lookup result index yields, so store and lookup stay aligned.
        # Deduplicate to the distinct requested pairs (the uncached path returns
        # one entry per (node, ds); resolve() reindexes onto the full pending
        # keys, so a duplicated index here would break that reindex).
        requested_pairs = {
            (str(uid), pd.Timestamp(ds)) for uid, ds in zip(uids, ds_values, strict=True)
        }

        uncached = {pair for pair in requested_pairs if pair not in self._lookup_cache}
        if uncached:
            uncached_uids = pd.Series([uid for uid, _ in uncached], dtype="object")
            uncached_ds = pd.Series([ds for _, ds in uncached], dtype="datetime64[ns]")
            computed = self._compute_lookup(uncached_uids, uncached_ds)
            for key, value in zip(computed.index, computed.to_numpy(), strict=True):
                uid, ds = key
                self._lookup_cache[(str(uid), pd.Timestamp(ds))] = float(value)

        # Serve only the complete cached entries, one row per distinct pair —
        # matching the shape the uncached path produces.
        complete = [
            (uid, ds, self._lookup_cache[(uid, ds)])
            for (uid, ds) in requested_pairs
            if (uid, ds) in self._lookup_cache
        ]
        if not complete:
            return pd.Series(dtype="float64", index=pd.MultiIndex.from_arrays([[], []]))
        index = pd.MultiIndex.from_arrays(
            [[uid for uid, _, _ in complete], [ds for _, ds, _ in complete]]
        )
        return pd.Series([value for _, _, value in complete], index=index, dtype="float64")

    def _compute_lookup(self, uids: pd.Series, ds_values: pd.Series) -> pd.Series:
        """Compute ``(unique_id, ds) -> y`` for the requested rows from bottom
        history (window scan + attribute merge + completeness-gated group-bys).
        Returns only complete entries; no caching."""
        requested_ds = set(ds_values.unique())
        window = self._bottom[self._bottom[DS].isin(requested_ds)]
        requested = set(uids.unique())
        parts: list[pd.Series] = []

        bottom_requested = requested & self._bottom_ids
        if bottom_requested:
            rows = window[window[UNIQUE_ID].isin(bottom_requested)]
            parts.append(rows.set_index([UNIQUE_ID, DS])[Y])

        # A bottom unique_id may itself contain "=", so aggregate labels are
        # matched only among non-bottom requested labels.
        requested_aggregates = requested - self._bottom_ids
        joined: pd.DataFrame | None = None
        for col in self._index.attr_cols:
            wanted_values = {
                label.split("=", 1)[1]
                for label in requested_aggregates
                if label.startswith(f"{col}=")
            }
            if not wanted_values:
                continue
            if joined is None:
                joined = window.merge(
                    self._index.frame[[UNIQUE_ID, *self._index.attr_cols]],
                    on=UNIQUE_ID,
                    how="inner",
                )
            members = joined[joined[col].astype(str).isin(wanted_values)]
            grouped = (
                members.groupby([col, DS], sort=True)
                .agg(**{Y: (Y, "sum"), "_member_count": (UNIQUE_ID, "nunique")})
                .reset_index()
            )
            complete = grouped[
                grouped["_member_count"] == grouped[col].map(self._members_by_attr[col])
            ].copy()
            complete[UNIQUE_ID] = col + "=" + complete[col].astype(str)
            parts.append(complete.set_index([UNIQUE_ID, DS])[Y])

        if TOTAL_LABEL in requested_aggregates:
            total = (
                window.groupby(DS, sort=True)
                .agg(**{Y: (Y, "sum"), "_member_count": (UNIQUE_ID, "nunique")})
                .reset_index()
            )
            total = total[total["_member_count"] == len(self._index.bottom_ids)].copy()
            total[UNIQUE_ID] = TOTAL_LABEL
            parts.append(total.set_index([UNIQUE_ID, DS])[Y])

        if not parts:
            return pd.Series(dtype="float64", index=pd.MultiIndex.from_arrays([[], []]))
        return pd.concat(parts)


def as_actuals_source(actuals: pd.DataFrame | ActualsSource) -> ActualsSource:
    """Coerce the engine's ``actuals`` argument to an :class:`ActualsSource`."""
    if isinstance(actuals, pd.DataFrame):
        return FrameActualsSource(actuals)
    return actuals
