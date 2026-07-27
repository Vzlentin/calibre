"""Exercise independent M5 eligibility reduction over the generic ledger reader."""

from __future__ import annotations

import ast
import inspect
from collections.abc import Iterator
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, cast

import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from newcalibre.domain import (
    Calendar,
    DecisionScope,
    DecisionScopeKind,
    EmissionScope,
    GuaranteeClaim,
    GuaranteeCurrency,
    GuaranteeDescriptor,
    GuaranteeType,
    ScoredSeries,
    SessionIdentity,
    interval_columns,
)
from newcalibre.engine import (
    LedgerBatch,
    LedgerBoundScore,
    LedgerForecastKey,
    LedgerResolution,
    LedgerSelection,
    LedgerSessionMetadata,
)
from newcalibre.protocols.m5 import M5Diagnostics, load_m5_config, score_m5
from newcalibre.protocols.m5.scorer import M5ScoringError

_PROJECT_ROOT = Path(__file__).parents[2]
_GATE_C = _PROJECT_ROOT / "benchmarks" / "m5" / "gate-c.yaml"
_MODEL = "seasonal-naive"
_NODES = (
    "__aggregate__:category:s:CATEGORY",
    "__aggregate__:department:s:DEPARTMENT",
    "__aggregate__:item:s:ITEM",
    "__aggregate__:state:s:STATE",
    "__aggregate__:store:s:STORE",
    "__total__",
    "bottom_item_store",
)


def _descriptor(*, scored_series: ScoredSeries = ScoredSeries.RECORDED_SALES):
    return GuaranteeDescriptor(
        type=GuaranteeType(
            claim=GuaranteeClaim.ONE_SIDED_COVERAGE,
            currency=GuaranteeCurrency.FINITE_SAMPLE_MARGINAL,
            declared_slack=None,
        ),
        level=0.9,
        scored_series=scored_series,
        window=EmissionScope.PER_STEP,
        scope=DecisionScope(DecisionScopeKind.PER_DECISION_NODE, None),
    )


def _session() -> SessionIdentity:
    config = load_m5_config(_GATE_C)
    return SessionIdentity.derive(
        tenant="public-m5-fixture",
        series_keys=_NODES,
        calendar=Calendar("D"),
        horizon=config.horizon,
        model_config=config.model_config,
        conformal_config=config.conformal_config,
    )


type _Row = tuple[
    LedgerForecastKey,
    pd.Timestamp,
    LedgerResolution | None,
    tuple[LedgerBoundScore, ...],
]


def _rows(
    *,
    mutation: str | None = None,
    covered: bool = True,
) -> list[_Row]:
    config = load_m5_config(_GATE_C)
    lower, upper = interval_columns(0.9)
    del lower
    rows = []
    first = pd.Timestamp("2026-01-01")
    for origin_index in range(config.origin_count):
        origin = first + pd.Timedelta(days=origin_index)
        for node in sorted(_NODES, key=str.encode):
            for horizon_step in range(1, config.horizon + 1):
                target = origin + pd.Timedelta(days=horizon_step - 1)
                eligible = origin_index >= horizon_step + config.minimum_calibration_scores - 1
                scored = eligible
                if (
                    mutation == "missing-eligible"
                    and node == "bottom_item_store"
                    and (
                        origin_index,
                        horizon_step,
                    )
                    == (20, 1)
                ):
                    scored = False
                if (
                    mutation == "early-ineligible"
                    and node == "bottom_item_store"
                    and (
                        origin_index,
                        horizon_step,
                    )
                    == (0, 1)
                ):
                    scored = True
                score = LedgerBoundScore(
                    bound_key=(upper,),
                    descriptor=_descriptor(),
                    guaranteed_side="upper",
                    resolved=True,
                    scored=scored,
                    value=float(covered) if scored else None,
                    covered=covered if scored else None,
                    unscored_reason=None if scored else "warm-up",
                )
                key = LedgerForecastKey(node, origin, horizon_step, _MODEL)
                resolution = LedgerResolution(target, 1.0, None, None, None)
                rows.append((key, target, resolution, (score,)))
    return rows


@st.composite
def _mask_mismatch_cases(draw: st.DrawFn) -> tuple[str, int, int, tuple[bool, ...]]:
    mismatch = draw(st.sampled_from(("missing", "early")))
    horizon_step = draw(st.integers(min_value=1, max_value=28))
    minimum = load_m5_config(_GATE_C).minimum_calibration_scores
    target_origin = draw(
        st.integers(
            min_value=horizon_step + minimum - 1 if mismatch == "missing" else 0,
            max_value=63,
        )
    )
    resolutions = draw(st.lists(st.booleans(), min_size=64, max_size=64))
    prior_end = max(0, target_origin - horizon_step + 1)
    if mismatch == "missing":
        resolutions[:minimum] = [True] * minimum
    else:
        retained = 0
        for origin_index in range(prior_end):
            if resolutions[origin_index] and retained < minimum - 1:
                retained += 1
            else:
                resolutions[origin_index] = False
    resolutions[target_origin] = True
    return mismatch, horizon_step, target_origin, tuple(resolutions)


def _rows_with_mask_mismatch(
    case: tuple[str, int, int, tuple[bool, ...]],
) -> list[_Row]:
    mismatch, affected_horizon, target_origin, resolutions = case
    config = load_m5_config(_GATE_C)
    first_origin = pd.Timestamp("2026-01-01")
    rows = _rows()
    for position, (key, target, resolution, scores) in enumerate(rows):
        if key.series_key != "bottom_item_store" or key.horizon_step != affected_horizon:
            continue
        origin_index = (key.origin - first_origin).days
        resolved = resolutions[origin_index]
        prior_end = max(0, origin_index - affected_horizon + 1)
        eligible = resolved and sum(resolutions[:prior_end]) >= config.minimum_calibration_scores
        scored = eligible
        if origin_index == target_origin:
            assert eligible == (mismatch == "missing")
            scored = mismatch == "early"
        score = replace(
            scores[0],
            resolved=resolved,
            scored=scored,
            value=1.0 if scored else None,
            covered=True if scored else None,
            unscored_reason=None if scored or not resolved else "warm-up",
        )
        rows[position] = (
            key,
            target,
            resolution if resolved else None,
            (score,),
        )
    return rows


class _Reader:
    def __init__(
        self,
        rows: list[_Row],
        *,
        batch_size: int = 137,
    ) -> None:
        self.metadata = LedgerSessionMetadata(_session(), _session().series_keys)
        self.rows = rows
        self.batch_size = batch_size
        self.scan_calls = 0
        self.iterator_starts = 0

    def scan(self, selection: LedgerSelection) -> Iterator[LedgerBatch]:
        self.scan_calls += 1
        assert selection.session == self.metadata.session
        assert selection.columns == ("target_timestamp", "resolution", "scores")

        def batches() -> Iterator[LedgerBatch]:
            self.iterator_starts += 1
            for offset in range(0, len(self.rows), self.batch_size):
                chunk = self.rows[offset : offset + self.batch_size]
                yield LedgerBatch(
                    session=self.metadata.session,
                    keys=tuple(row[0] for row in chunk),
                    columns={
                        "target_timestamp": tuple(row[1] for row in chunk),
                        "resolution": tuple(row[2] for row in chunk),
                        "scores": tuple(row[3] for row in chunk),
                    },
                    batch_size=selection.batch_size,
                )

        return batches()


def test_exact_cartesian_universe_and_mask_emit_valid_compact_diagnostics(
    tmp_path: Path,
) -> None:
    config = load_m5_config(_GATE_C)
    reader = _Reader(_rows())

    diagnostics = score_m5(config, reader, output_dir=tmp_path / "diagnostics")

    assert isinstance(diagnostics, M5Diagnostics)
    assert diagnostics.status == "VALID"
    assert diagnostics.context.node_count == 7
    assert diagnostics.context.origin_count == 64
    assert diagnostics.context.horizon == 28
    assert diagnostics.population.mask_equal
    assert diagnostics.population.counts.total == 7 * 64 * 28
    assert diagnostics.population.counts.resolved == diagnostics.population.counts.total
    assert diagnostics.population.counts.eligible == 7 * sum(range(27, 55))
    assert diagnostics.population.counts.eligible == diagnostics.population.counts.scored
    assert diagnostics.population.coverage == 1.0
    assert set(diagnostics.levels) == {
        "bottom",
        "item",
        "department",
        "category",
        "store",
        "state",
        "total",
    }
    assert reader.scan_calls == 1
    assert reader.iterator_starts == 1
    assert {path.name for path in diagnostics.paths} == {
        "coverage-summary.json",
        "coverage-by-node.parquet",
        "report.md",
    }
    with pytest.raises(FrozenInstanceError):
        cast(Any, diagnostics).status = "INVALID"


@pytest.mark.parametrize(
    ("mutation", "missing", "early"),
    [("missing-eligible", 1, 0), ("early-ineligible", 0, 1)],
)
def test_mask_mismatches_are_attributed_without_suppressing_artifacts(
    tmp_path: Path,
    mutation: str,
    missing: int,
    early: int,
) -> None:
    config = load_m5_config(_GATE_C)

    diagnostics = score_m5(
        config,
        _Reader(_rows(mutation=mutation)),
        output_dir=tmp_path / mutation,
    )

    assert diagnostics.status == "INVALID"
    assert diagnostics.population.missing_eligible == missing
    assert diagnostics.population.early_scored == early
    assert not diagnostics.population.mask_equal
    assert diagnostics.models[_MODEL].missing_eligible == missing
    assert diagnostics.models[_MODEL].early_scored == early
    assert diagnostics.levels["bottom"].missing_eligible == missing
    assert diagnostics.levels["bottom"].early_scored == early
    assert {path.name for path in diagnostics.paths} == {
        "coverage-summary.json",
        "coverage-by-node.parquet",
        "report.md",
    }


@given(case=_mask_mismatch_cases())
@settings(max_examples=30, deadline=None)
def test_exact_mask_completeness_holds_for_varied_resolution_histories(
    case: tuple[str, int, int, tuple[bool, ...]],
) -> None:
    mismatch, _horizon_step, _target_origin, _resolutions = case
    with TemporaryDirectory() as directory:
        diagnostics = score_m5(
            load_m5_config(_GATE_C),
            _Reader(_rows_with_mask_mismatch(case)),
            output_dir=Path(directory) / "diagnostics",
        )

    missing = int(mismatch == "missing")
    early = int(mismatch == "early")
    assert diagnostics.status == "INVALID"
    assert diagnostics.population.missing_eligible == missing
    assert diagnostics.population.early_scored == early
    assert diagnostics.models[_MODEL].missing_eligible == missing
    assert diagnostics.models[_MODEL].early_scored == early
    assert diagnostics.levels["bottom"].missing_eligible == missing
    assert diagnostics.levels["bottom"].early_scored == early


@pytest.mark.parametrize("covered", [False, True])
def test_coverage_extremes_remain_valid_when_the_mask_matches(
    tmp_path: Path,
    covered: bool,
) -> None:
    diagnostics = score_m5(
        load_m5_config(_GATE_C),
        _Reader(_rows(covered=covered)),
        output_dir=tmp_path / str(covered),
    )

    assert diagnostics.status == "VALID"
    assert diagnostics.population.coverage == float(covered)


def test_zero_scored_node_remains_valid_and_has_null_descriptive_rate(tmp_path: Path) -> None:
    rows = _rows()
    pending_rows: list[_Row] = []
    for key, target, resolution, scores in rows:
        if key.series_key != "bottom_item_store":
            pending_rows.append((key, target, resolution, scores))
            continue
        pending_rows.append(
            (
                key,
                target,
                None,
                (
                    replace(
                        scores[0],
                        resolved=False,
                        scored=False,
                        value=None,
                        covered=None,
                        unscored_reason=None,
                    ),
                ),
            )
        )

    diagnostics = score_m5(
        load_m5_config(_GATE_C),
        _Reader(pending_rows),
        output_dir=tmp_path / "zero-scored",
    )

    assert diagnostics.status == "VALID"
    assert diagnostics.levels["bottom"].counts.scored == 0
    assert diagnostics.levels["bottom"].pooled_coverage is None
    assert diagnostics.levels["bottom"].mean_node_coverage is None


def test_structural_and_descriptor_drift_emit_invalid_diagnostics(tmp_path: Path) -> None:
    config = load_m5_config(_GATE_C)
    missing = _rows()
    missing.pop(100)
    missing_result = score_m5(
        config,
        _Reader(missing),
        output_dir=tmp_path / "missing-row",
    )
    assert missing_result.status == "INVALID"

    wrong_descriptor = _rows()
    key, target, resolution, scores = wrong_descriptor[-1]
    wrong_descriptor[-1] = (
        key,
        target,
        resolution,
        (
            replace(
                scores[0],
                descriptor=_descriptor(scored_series=ScoredSeries.DEMAND_HONEST),
            ),
        ),
    )
    descriptor_result = score_m5(
        config,
        _Reader(wrong_descriptor),
        output_dir=tmp_path / "wrong-descriptor",
    )
    assert descriptor_result.status == "INVALID"

    out_of_order = _rows()
    out_of_order[100], out_of_order[101] = out_of_order[101], out_of_order[100]
    order_result = score_m5(
        config,
        _Reader(out_of_order),
        output_dir=tmp_path / "out-of-order",
    )
    assert order_result.status == "INVALID"


def test_scorer_source_preserves_the_generic_one_pass_boundary() -> None:
    import newcalibre.protocols.m5.scorer as scorer

    source = inspect.getsource(scorer)
    tree = ast.parse(source)
    scan_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "scan"
    ]

    assert len(scan_calls) == 1
    assert "newcalibre.ledger" not in source
    assert "InMemoryLedgerReader" not in source
    assert ".forecasts" not in source
    assert "M5EngineAdapter" not in source
    assert "oracle" not in source.lower()
    assert "confidence" not in source.lower()
    assert "receipt" not in source.lower()
    assert "digest" not in source.lower()


def test_scorer_surface_is_settled_and_rejects_an_existing_destination(tmp_path: Path) -> None:
    signature = inspect.signature(score_m5)
    assert tuple(signature.parameters) == ("config", "ledger", "output_dir")
    assert signature.parameters["output_dir"].kind is inspect.Parameter.KEYWORD_ONLY
    output = tmp_path / "existing"
    output.mkdir()

    with pytest.raises(M5ScoringError, match="destination"):
        score_m5(load_m5_config(_GATE_C), _Reader(_rows()), output_dir=output)
    assert list(output.iterdir()) == []
