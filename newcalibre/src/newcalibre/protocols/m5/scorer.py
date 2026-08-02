"""Score M5 sales-coverage diagnostics from one generic closed-ledger scan."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import cast

import pandas as pd

from newcalibre.conformal import EmissionForm, resolve_method
from newcalibre.domain import (
    DecisionScopeKind,
    EmissionScope,
    GuaranteeClaim,
    ScoredSeries,
    interval_columns,
)
from newcalibre.engine.reporting import (
    LedgerBatch,
    LedgerBoundScore,
    LedgerColumn,
    LedgerForecastKey,
    LedgerReader,
    LedgerResolution,
    LedgerSelection,
    LedgerSessionMetadata,
)
from newcalibre.protocols.m5.artifacts import M5ArtifactError, _emit_artifacts
from newcalibre.protocols.m5.compiler import _level_from_node_label
from newcalibre.protocols.m5.config import M5ProtocolConfig
from newcalibre.protocols.m5.loader import M5DataError

_LEVELS = ("bottom", "item", "department", "category", "store", "state", "total")
_MASK_IDENTITY = "resolved-and-series-horizon-ready-before-issuance"
_MASK_DEFINITION = (
    "A row is eligible when it is resolved and its node, model, and horizon partition "
    "has the configured minimum number of prior resolved targets strictly before its origin."
)
_BATCH_SIZE = 4096
_MAX_EXAMPLES = 3


class M5ScoringError(ValueError):
    """Report an M5 scoring configuration, reader, or output failure."""


@dataclass(frozen=True, slots=True)
class M5CountSummary:
    """Count the five row states used by every M5 diagnostic reduction."""

    total: int
    resolved: int
    eligible: int
    scored: int
    covered: int


@dataclass(frozen=True, slots=True)
class M5CoverageSummary:
    """Summarize one pooled sales-coverage population and exact-mask result."""

    counts: M5CountSummary
    coverage: float | None
    target: float
    deviation: float | None
    mask_equal: bool
    missing_eligible: int
    early_scored: int


@dataclass(frozen=True, slots=True)
class M5LevelSummary:
    """Summarize pooled and unweighted node sales-coverage for one level."""

    level: str
    node_count: int
    scored_node_count: int
    counts: M5CountSummary
    mean_node_coverage: float | None
    pooled_coverage: float | None
    target: float
    mean_node_deviation: float | None
    pooled_deviation: float | None
    mask_equal: bool
    missing_eligible: int
    early_scored: int


@dataclass(frozen=True, slots=True)
class M5RunContext:
    """Describe the complete configured and observed M5 scoring universe."""

    dataset: str
    phase: str
    session_id: str
    model_name: str
    reconciler: str
    conformal_method: str
    conformal_partition: str
    origin_start: pd.Timestamp | None
    origin_end: pd.Timestamp | None
    origin_count: int
    horizon: int
    target: float
    node_count: int
    expected_row_count: int


@dataclass(frozen=True, slots=True)
class M5Diagnostics:
    """Return compact M5 reductions and the three published artifact paths."""

    status: str
    context: M5RunContext
    population: M5CoverageSummary
    models: Mapping[str, M5CoverageSummary]
    levels: Mapping[str, M5LevelSummary]
    summary_path: Path
    by_node_path: Path
    report_path: Path

    @property
    def paths(self) -> tuple[Path, Path, Path]:
        """Return the exact three diagnostic artifact paths."""
        return self.summary_path, self.by_node_path, self.report_path


@dataclass(slots=True)
class _Aggregate:
    total: int = 0
    resolved: int = 0
    eligible: int = 0
    scored: int = 0
    covered: int = 0
    missing_eligible: int = 0
    early_scored: int = 0
    forced_unequal: bool = False

    def add(
        self,
        *,
        resolved: bool,
        eligible: bool,
        scored: bool,
        covered: bool,
    ) -> None:
        self.total += 1
        self.resolved += resolved
        self.eligible += eligible
        self.scored += scored
        self.covered += covered
        self.missing_eligible += eligible and not scored
        self.early_scored += scored and not eligible

    def summary(self, *, target: float) -> M5CoverageSummary:
        counts = M5CountSummary(
            self.total,
            self.resolved,
            self.eligible,
            self.scored,
            self.covered,
        )
        rate = None if not self.scored else self.covered / self.scored
        return M5CoverageSummary(
            counts=counts,
            coverage=rate,
            target=target,
            deviation=None if rate is None else rate - target,
            mask_equal=(
                not self.forced_unequal and self.missing_eligible == 0 and self.early_scored == 0
            ),
            missing_eligible=self.missing_eligible,
            early_scored=self.early_scored,
        )


@dataclass(slots=True)
class _PartitionProgress:
    delivered: int = 0
    history: int = 0
    seen: int = 0

    def eligibility(
        self,
        *,
        origin_index: int,
        horizon_step: int,
        resolved: bool,
        minimum: int,
    ) -> bool:
        if origin_index != self.seen:
            self.seen = origin_index
            self.history = 0
        if self.seen >= horizon_step and self.history & (1 << (horizon_step - 1)):
            self.delivered += 1
        eligible = resolved and self.delivered >= minimum
        mask = (1 << horizon_step) - 1
        self.history = ((self.history << 1) | int(resolved)) & mask
        self.seen += 1
        return eligible


class _Issues:
    def __init__(self) -> None:
        self.counts: Counter[str] = Counter()
        self.examples: dict[str, list[str]] = {}

    def add(self, reason: str, key: LedgerForecastKey | None = None) -> None:
        self.counts[reason] += 1
        if key is None:
            return
        examples = self.examples.setdefault(reason, [])
        if len(examples) < _MAX_EXAMPLES:
            examples.append(_key_text(key))

    @property
    def total(self) -> int:
        return sum(self.counts.values())


class _Reducer:
    def __init__(
        self,
        *,
        config: M5ProtocolConfig,
        metadata: LedgerSessionMetadata,
        model_name: str,
        target: float,
        emission_form: EmissionForm,
        guarantee_claim: GuaranteeClaim,
        guarantee_currency: object,
    ) -> None:
        self.config = config
        self.metadata = metadata
        self.model_name = model_name
        self.target = target
        self.expected_bound = (interval_columns(target)[1],)
        self.emission_form = emission_form
        self.guarantee_claim = guarantee_claim
        self.guarantee_currency = guarantee_currency
        self.nodes = metadata.series_keys
        self.node_levels: dict[str, str | None] = {}
        self.issues = _Issues()
        for node in self.nodes:
            try:
                self.node_levels[node] = _level_from_node_label(node)
            except M5DataError:
                self.node_levels[node] = None
                self.issues.add("invalid-node-label")
        if set(level for level in self.node_levels.values() if level is not None) != set(_LEVELS):
            self.issues.add("incomplete-level-classes")
        self.population = _Aggregate()
        self.model_aggregates = {model_name: _Aggregate()}
        self.level_aggregates = {level: _Aggregate() for level in _LEVELS}
        self.node_aggregates = {(node, model_name): _Aggregate() for node in self.nodes}
        self.partitions: dict[tuple[str, str, int], _PartitionProgress] = {}
        self.first_origin: pd.Timestamp | None = None
        self.last_origin: pd.Timestamp | None = None
        self.origin_index = -1
        self.rows_in_origin = 0
        self.row_count = 0
        self.previous_key: LedgerForecastKey | None = None

    @property
    def rows_per_origin(self) -> int:
        return len(self.nodes) * self.config.horizon

    def consume_batch(self, batch: LedgerBatch) -> None:
        if batch.session != self.metadata.session:
            raise M5ScoringError("ledger batch session does not match reader metadata")
        expected_columns = {
            LedgerColumn.TARGET_TIMESTAMP.value,
            LedgerColumn.RESOLUTION.value,
            LedgerColumn.SCORES.value,
        }
        if set(batch.columns) != expected_columns:
            raise M5ScoringError("ledger batch does not contain the requested M5 projection")
        for position, key in enumerate(batch.keys):
            target = cast(
                pd.Timestamp,
                batch.columns[LedgerColumn.TARGET_TIMESTAMP.value][position],
            )
            resolution = cast(
                LedgerResolution | None,
                batch.columns[LedgerColumn.RESOLUTION.value][position],
            )
            scores = cast(
                tuple[LedgerBoundScore, ...],
                batch.columns[LedgerColumn.SCORES.value][position],
            )
            self.consume_row(key, target=target, resolution=resolution, scores=scores)

    def consume_row(
        self,
        key: LedgerForecastKey,
        *,
        target: pd.Timestamp,
        resolution: LedgerResolution | None,
        scores: tuple[LedgerBoundScore, ...],
    ) -> None:
        self._start_origin(key)
        self._validate_canonical_key(key)
        self._validate_spine(key)
        self.row_count += 1
        self.rows_in_origin += 1

        node_valid = key.series_key in self.node_levels
        model_valid = key.model_name == self.model_name
        horizon_valid = 1 <= key.horizon_step <= self.config.horizon
        expected_target = key.origin + pd.Timedelta(days=key.horizon_step - 1)
        if target != expected_target:
            self.issues.add("wrong-target-timestamp", key)
        resolved = resolution is not None
        if resolution is not None:
            if resolution.target_timestamp != target:
                self.issues.add("resolution-target-mismatch", key)
            if (
                resolution.censoring_assertion is not None
                or resolution.availability_bound is not None
            ):
                self.issues.add("unexpected-censoring-facts", key)

        eligible = False
        if node_valid and model_valid and horizon_valid:
            partition_key = (key.series_key, key.model_name, key.horizon_step)
            partition = self.partitions.setdefault(partition_key, _PartitionProgress())
            eligible = partition.eligibility(
                origin_index=self.origin_index,
                horizon_step=key.horizon_step,
                resolved=resolved,
                minimum=self.config.minimum_calibration_scores,
            )
        else:
            if not node_valid:
                self.issues.add("wrong-node", key)
            if not model_valid:
                self.issues.add("wrong-model", key)
            if not horizon_valid:
                self.issues.add("wrong-horizon", key)

        score = self._selected_score(scores, key=key)
        scored = False if score is None else score.scored
        covered = False if score is None else score.covered is True
        if score is not None:
            self._validate_score(score, key=key, resolved=resolved)
        if eligible and not scored:
            self.issues.add("missing-eligible-score", key)
        if scored and not eligible:
            self.issues.add("early-ineligible-score", key)

        aggregates = [self.population]
        if model_valid:
            aggregates.append(self.model_aggregates[self.model_name])
        level = self.node_levels.get(key.series_key)
        if level is not None and model_valid:
            aggregates.append(self.level_aggregates[level])
        node_aggregate = self.node_aggregates.get((key.series_key, key.model_name))
        if node_aggregate is not None:
            aggregates.append(node_aggregate)
        for aggregate in aggregates:
            aggregate.add(
                resolved=resolved,
                eligible=eligible,
                scored=scored,
                covered=covered,
            )

    def finish(self) -> None:
        if self.last_origin is not None:
            self._finish_origin()
        if self.origin_index + 1 != self.config.origin_count:
            self.issues.add("wrong-origin-count")
            self.population.forced_unequal = True
        expected_rows = self.rows_per_origin * self.config.origin_count
        if self.row_count != expected_rows:
            self.issues.add("wrong-row-count")
            self.population.forced_unequal = True
        for aggregate in (
            *self.model_aggregates.values(),
            *self.level_aggregates.values(),
            *self.node_aggregates.values(),
        ):
            if self.population.forced_unequal:
                aggregate.forced_unequal = True

    def _start_origin(self, key: LedgerForecastKey) -> None:
        if self.last_origin is None:
            self.first_origin = key.origin
            self.last_origin = key.origin
            self.origin_index = 0
            return
        if key.origin == self.last_origin:
            return
        self._finish_origin()
        expected = self.last_origin + pd.Timedelta(days=1)
        if key.origin != expected:
            self.issues.add("noncontiguous-origin", key)
            self.population.forced_unequal = True
        self.last_origin = key.origin
        self.origin_index += 1
        self.rows_in_origin = 0

    def _finish_origin(self) -> None:
        if self.rows_in_origin != self.rows_per_origin:
            self.issues.add("incomplete-origin-spine")
            self.population.forced_unequal = True

    def _validate_canonical_key(self, key: LedgerForecastKey) -> None:
        previous = self.previous_key
        if previous is not None:
            previous_order = _key_order(previous)
            current_order = _key_order(key)
            if current_order == previous_order:
                self.issues.add("duplicate-row", key)
            elif current_order < previous_order:
                self.issues.add("out-of-order-row", key)
        self.previous_key = key

    def _validate_spine(self, key: LedgerForecastKey) -> None:
        position = self.rows_in_origin
        if position >= self.rows_per_origin:
            self.issues.add("extra-row", key)
            self.population.forced_unequal = True
            return
        horizon_step = position // len(self.nodes) + 1
        node = self.nodes[position % len(self.nodes)]
        if (
            key.series_key != node
            or key.model_name != self.model_name
            or key.horizon_step != horizon_step
        ):
            self.issues.add("row-spine-mismatch", key)
            self.population.forced_unequal = True

    def _selected_score(
        self,
        scores: tuple[LedgerBoundScore, ...],
        *,
        key: LedgerForecastKey,
    ) -> LedgerBoundScore | None:
        if len(scores) != 1:
            self.issues.add("missing-or-duplicate-outcome", key)
            return None
        return scores[0]

    def _validate_score(
        self,
        score: LedgerBoundScore,
        *,
        key: LedgerForecastKey,
        resolved: bool,
    ) -> None:
        descriptor = score.descriptor
        wrong_shape = (
            self.emission_form is not EmissionForm.ONE_SIDED_UPPER
            or score.bound_key != self.expected_bound
            or score.guaranteed_side != "upper"
            or descriptor.type.claim is not self.guarantee_claim
            or descriptor.type.currency is not self.guarantee_currency
        )
        if wrong_shape:
            self.issues.add("wrong-outcome-shape", key)
        if descriptor.level != self.target:
            self.issues.add("wrong-coverage-target", key)
        if score.scored and descriptor.scored_series is not ScoredSeries.RECORDED_SALES:
            self.issues.add("wrong-sales-label", key)
        if descriptor.window is not EmissionScope.PER_STEP:
            self.issues.add("wrong-outcome-window", key)
        if descriptor.scope.kind is not DecisionScopeKind.PER_DECISION_NODE:
            self.issues.add("wrong-outcome-scope", key)
        if score.resolved != resolved:
            self.issues.add("score-resolution-mismatch", key)
        if score.scored and score.covered is None:
            self.issues.add("missing-coverage-result", key)

    def level_summaries(self) -> dict[str, M5LevelSummary]:
        summaries: dict[str, M5LevelSummary] = {}
        for level in _LEVELS:
            pooled = self.level_aggregates[level].summary(target=self.target)
            node_rates = [
                aggregate.covered / aggregate.scored
                for (node, model), aggregate in self.node_aggregates.items()
                if model == self.model_name
                and self.node_levels.get(node) == level
                and aggregate.scored
            ]
            mean = None if not node_rates else math.fsum(node_rates) / len(node_rates)
            node_count = sum(value == level for value in self.node_levels.values())
            summaries[level] = M5LevelSummary(
                level=level,
                node_count=node_count,
                scored_node_count=len(node_rates),
                counts=pooled.counts,
                mean_node_coverage=mean,
                pooled_coverage=pooled.coverage,
                target=self.target,
                mean_node_deviation=None if mean is None else mean - self.target,
                pooled_deviation=pooled.deviation,
                mask_equal=pooled.mask_equal,
                missing_eligible=pooled.missing_eligible,
                early_scored=pooled.early_scored,
            )
        return summaries


def score_m5(
    config: M5ProtocolConfig,
    ledger: LedgerReader,
    *,
    output_dir: Path,
) -> M5Diagnostics:
    """Score one closed M5 ledger scan and atomically emit three diagnostics."""
    if not isinstance(config, M5ProtocolConfig):
        raise M5ScoringError("M5 scoring config must be an M5ProtocolConfig")
    if not isinstance(output_dir, Path):
        raise M5ScoringError("M5 diagnostic output_dir must be a pathlib.Path")
    if output_dir.exists() or output_dir.is_symlink():
        raise M5ScoringError("M5 diagnostic destination must not already exist")
    try:
        metadata = ledger.metadata
    except Exception as error:
        raise M5ScoringError("ledger reader metadata is unavailable") from error
    if not isinstance(metadata, LedgerSessionMetadata):
        raise M5ScoringError("ledger reader metadata must be LedgerSessionMetadata")

    conformal = config.conformal_config
    try:
        runtime = resolve_method(conformal)
        raw_target = conformal["coverage"]
        if isinstance(raw_target, bool) or not isinstance(raw_target, (int, float)):
            raise TypeError("M5 coverage target must be numeric")
        target = float(raw_target)
        model_name = cast(str, config.model_config["model_name"])
        guarantee = runtime.manifest.guarantees[0]
    except Exception as error:
        raise M5ScoringError(
            "M5 configuration does not resolve registered scoring intent"
        ) from error
    if (
        conformal.get("partition_by") != "series-horizon"
        or runtime.manifest.minimum_calibration_scores(runtime.config)
        != config.minimum_calibration_scores
        or runtime.manifest.emission_scope is not EmissionScope.PER_STEP
        or len(runtime.manifest.guarantees) != 1
    ):
        raise M5ScoringError("M5 configuration does not match registered series-horizon intent")

    reducer = _Reducer(
        config=config,
        metadata=metadata,
        model_name=model_name,
        target=target,
        emission_form=runtime.manifest.emission_form,
        guarantee_claim=guarantee.claim,
        guarantee_currency=guarantee.currency,
    )
    selection = LedgerSelection(
        metadata.session,
        (
            LedgerColumn.TARGET_TIMESTAMP,
            LedgerColumn.RESOLUTION,
            LedgerColumn.SCORES,
        ),
        _BATCH_SIZE,
    )
    try:
        batches = ledger.scan(selection)
    except Exception as error:
        raise M5ScoringError("M5 ledger scan could not start") from error
    try:
        for batch in batches:
            if not isinstance(batch, LedgerBatch):
                raise M5ScoringError("M5 ledger scan yielded a non-LedgerBatch value")
            reducer.consume_batch(batch)
    except M5ScoringError:
        raise
    except Exception as error:
        raise M5ScoringError("M5 ledger iteration failed") from error
    reducer.finish()

    population = reducer.population.summary(target=target)
    models = {
        name: aggregate.summary(target=target)
        for name, aggregate in reducer.model_aggregates.items()
    }
    levels = reducer.level_summaries()
    context = M5RunContext(
        dataset=config.dataset,
        phase=config.phase,
        session_id=metadata.session.value,
        model_name=model_name,
        reconciler=config.reconciliation_strategy,
        conformal_method=runtime.manifest.name,
        conformal_partition=config.conformal_partition,
        origin_start=reducer.first_origin,
        origin_end=reducer.last_origin,
        origin_count=reducer.origin_index + 1,
        horizon=config.horizon,
        target=target,
        node_count=len(metadata.series_keys),
        expected_row_count=reducer.rows_per_origin * config.origin_count,
    )
    status = "VALID" if reducer.issues.total == 0 and population.mask_equal else "INVALID"
    summary = _summary_projection(
        status=status,
        context=context,
        population=population,
        models=models,
        levels=levels,
        issues=reducer.issues,
    )
    node_rows = _node_projections(
        status=status,
        context=context,
        reducer=reducer,
    )
    try:
        summary_path, by_node_path, report_path = _emit_artifacts(
            output_dir,
            summary=summary,
            node_rows=node_rows,
        )
    except M5ArtifactError as error:
        raise M5ScoringError(str(error)) from error
    return M5Diagnostics(
        status=status,
        context=context,
        population=population,
        models=MappingProxyType(models),
        levels=MappingProxyType(levels),
        summary_path=summary_path,
        by_node_path=by_node_path,
        report_path=report_path,
    )


def _summary_projection(
    *,
    status: str,
    context: M5RunContext,
    population: M5CoverageSummary,
    models: Mapping[str, M5CoverageSummary],
    levels: Mapping[str, M5LevelSummary],
    issues: _Issues,
) -> dict[str, object]:
    return {
        "schema": 1,
        "status": status,
        "metric": "sales-coverage",
        "context": _context_projection(context),
        "mask": {
            "identity": _MASK_IDENTITY,
            "definition": _MASK_DEFINITION,
            "equal": population.mask_equal,
            "expected_eligible_count": population.counts.eligible,
            "actual_scored_count": population.counts.scored,
            "missing_eligible_count": population.missing_eligible,
            "early_scored_count": population.early_scored,
            "structural_issue_count": issues.total,
            "reasons": dict(sorted(issues.counts.items(), key=lambda item: item[0].encode())),
            "examples": {
                reason: examples
                for reason, examples in sorted(
                    issues.examples.items(),
                    key=lambda item: item[0].encode(),
                )
            },
        },
        "population": _coverage_projection(population),
        "per_model": [
            {"model": name, **_coverage_projection(value)}
            for name, value in sorted(models.items(), key=lambda item: item[0].encode())
        ],
        "levels": [_level_projection(levels[level]) for level in _LEVELS],
    }


def _context_projection(context: M5RunContext) -> dict[str, object]:
    return {
        "dataset": context.dataset,
        "phase": context.phase,
        "session_id": context.session_id,
        "model_name": context.model_name,
        "reconciler": context.reconciler,
        "conformal_method": context.conformal_method,
        "conformal_partition": context.conformal_partition,
        "origin_start": _timestamp_text(context.origin_start),
        "origin_end": _timestamp_text(context.origin_end),
        "origin_count": context.origin_count,
        "horizon": context.horizon,
        "target": context.target,
        "node_count": context.node_count,
        "expected_row_count": context.expected_row_count,
    }


def _coverage_projection(summary: M5CoverageSummary) -> dict[str, object]:
    return {
        "label": "sales-coverage",
        "target": summary.target,
        "coverage": summary.coverage,
        "deviation": summary.deviation,
        "counts": _count_projection(summary.counts),
        "mask_equal": summary.mask_equal,
        "missing_eligible_count": summary.missing_eligible,
        "early_scored_count": summary.early_scored,
    }


def _level_projection(summary: M5LevelSummary) -> dict[str, object]:
    return {
        "level": summary.level,
        "label": "sales-coverage",
        "node_count": summary.node_count,
        "scored_node_count": summary.scored_node_count,
        "target": summary.target,
        "mean_node_coverage": summary.mean_node_coverage,
        "pooled_coverage": summary.pooled_coverage,
        "mean_node_deviation": summary.mean_node_deviation,
        "pooled_deviation": summary.pooled_deviation,
        "counts": _count_projection(summary.counts),
        "mask_equal": summary.mask_equal,
        "missing_eligible_count": summary.missing_eligible,
        "early_scored_count": summary.early_scored,
    }


def _count_projection(counts: M5CountSummary) -> dict[str, int]:
    return {
        "total": counts.total,
        "resolved": counts.resolved,
        "eligible": counts.eligible,
        "scored": counts.scored,
        "covered": counts.covered,
    }


def _node_projections(
    *,
    status: str,
    context: M5RunContext,
    reducer: _Reducer,
) -> list[dict[str, object]]:
    base = {
        "schema": 1,
        "status": status,
        "metric": "sales-coverage",
        "dataset": context.dataset,
        "phase": context.phase,
        "session_id": context.session_id,
        "reconciler": context.reconciler,
        "conformal_method": context.conformal_method,
        "conformal_partition": context.conformal_partition,
        "origin_start": _timestamp_text(context.origin_start),
        "origin_end": _timestamp_text(context.origin_end),
        "origin_count": context.origin_count,
        "horizon": context.horizon,
        "target": context.target,
        "mask_identity": _MASK_IDENTITY,
    }
    rows: list[dict[str, object]] = []
    for (node, model), aggregate in reducer.node_aggregates.items():
        summary = aggregate.summary(target=context.target)
        rows.append(
            {
                **base,
                "level": reducer.node_levels.get(node) or "unknown",
                "node": node,
                "model": model,
                "coverage": summary.coverage,
                "deviation": summary.deviation,
                **_count_projection(summary.counts),
                "mask_equal": summary.mask_equal,
            }
        )
    return rows


def _timestamp_text(value: pd.Timestamp | None) -> str | None:
    return None if value is None else value.date().isoformat()


def _key_order(key: LedgerForecastKey) -> tuple[pd.Timestamp, int, bytes, bytes]:
    return key.origin, key.horizon_step, key.series_key.encode(), key.model_name.encode()


def _key_text(key: LedgerForecastKey) -> str:
    return f"{key.origin.date().isoformat()}|{key.series_key}|{key.model_name}|h{key.horizon_step}"


__all__ = ["M5Diagnostics", "M5ScoringError", "score_m5"]
