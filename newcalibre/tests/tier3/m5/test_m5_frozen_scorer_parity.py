"""Prove disposable count parity with the frozen M5 scorer."""

from __future__ import annotations

import json
import math
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass, replace
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
import pyarrow.parquet as pq
import pytest

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
    LedgerBoundIssuance,
    LedgerForecastKey,
    LedgerResolution,
    LedgerSelection,
    LedgerSessionMetadata,
)
from newcalibre.protocols.m5 import M5Diagnostics, load_m5_config
from tests.import_inspection import imported_modules
from tests.tier3.m5.m5_frozen_export import FrozenM5ExportError, export_frozen_m5_ledger

pytestmark = pytest.mark.tier3

PROJECT_ROOT = Path(__file__).parents[3]
CONFIG = PROJECT_ROOT / "tests" / "fixtures" / "m5" / "reduced-real.yaml"
MODEL = "seasonal-naive"
NODES = (
    "__aggregate__:category:s:CATEGORY",
    "__aggregate__:department:s:DEPARTMENT",
    "__aggregate__:item:s:ITEM",
    "__aggregate__:state:s:STATE",
    "__aggregate__:store:s:STORE",
    "__total__",
    "bottom_item_store",
)
EXPECTED_COLUMNS = ("unique_id", "h", "model_name", "y", "lo_0p9", "hi_0p9")
FROZEN_LEVELS = {
    "bottom": "bottom",
    "item_id": "item",
    "dept_id": "department",
    "cat_id": "category",
    "store_id": "store",
    "state_id": "state",
    "total": "total",
}


@dataclass(frozen=True, slots=True)
class _Counts:
    total: int
    resolved: int
    scored: int
    covered: int


@dataclass(frozen=True, slots=True)
class _Row:
    key: LedgerForecastKey
    issuance: tuple[LedgerBoundIssuance, ...]
    resolution: LedgerResolution | None


class _Reader:
    def __init__(self, rows: tuple[_Row, ...]) -> None:
        session = _session()
        self.metadata = LedgerSessionMetadata(session, session.series_keys)
        self.rows = rows
        self.selections: list[LedgerSelection] = []

    def scan(self, selection: LedgerSelection) -> Iterator[LedgerBatch]:
        self.selections.append(selection)
        for offset in range(0, len(self.rows), selection.batch_size):
            chunk = self.rows[offset : offset + selection.batch_size]
            yield LedgerBatch(
                session=self.metadata.session,
                keys=tuple(row.key for row in chunk),
                columns={
                    "issuances": tuple(row.issuance for row in chunk),
                    "resolution": tuple(row.resolution for row in chunk),
                },
                batch_size=selection.batch_size,
            )


def _descriptor() -> GuaranteeDescriptor:
    return GuaranteeDescriptor(
        type=GuaranteeType(
            claim=GuaranteeClaim.ONE_SIDED_COVERAGE,
            currency=GuaranteeCurrency.FINITE_SAMPLE_MARGINAL,
            declared_slack=None,
        ),
        level=0.9,
        scored_series=ScoredSeries.RECORDED_SALES,
        window=EmissionScope.PER_STEP,
        scope=DecisionScope(DecisionScopeKind.PER_DECISION_NODE, None),
    )


def _session() -> SessionIdentity:
    config = load_m5_config(CONFIG)
    return SessionIdentity.derive(
        tenant="m5-frozen-export-test",
        series_keys=NODES,
        calendar=Calendar("D"),
        horizon=config.horizon,
        model_config=config.model_config,
        conformal_config=config.conformal_config,
    )


def _issuance(value: float | None = 5.0) -> LedgerBoundIssuance:
    upper = interval_columns(0.9)[1]
    finite = value is not None
    return LedgerBoundIssuance(
        bound_key=(upper,),
        bound_values=(value,),
        descriptor=_descriptor(),
        guaranteed_side="upper",
        calibration_ready=finite,
        bounds_finite=finite,
        bounds_null_reason=None if finite else "warm-up",
    )


def _rows() -> tuple[_Row, ...]:
    origin = pd.Timestamp("2026-01-01")
    rows = []
    for index, node in enumerate(NODES):
        target = origin + pd.Timedelta(days=index)
        resolution = (
            None
            if node == "__total__"
            else LedgerResolution(
                target,
                float(index + 1),
                None,
                None,
                None,
            )
        )
        issuance = _issuance(None) if node == "bottom_item_store" else _issuance(10.0)
        rows.append(
            _Row(
                LedgerForecastKey(node, origin, index + 1, MODEL),
                (issuance,),
                resolution,
            )
        )
    return tuple(rows)


def test_disposable_export_has_exact_schema_labels_and_score_mapping(tmp_path: Path) -> None:
    reader = _Reader(_rows())
    output = tmp_path / "frozen-input.parquet"

    export_frozen_m5_ledger(load_m5_config(CONFIG), reader, output, batch_size=2)

    assert tuple(pq.read_schema(output).names) == EXPECTED_COLUMNS
    frame = pd.read_parquet(output)
    assert frame.columns.tolist() == list(EXPECTED_COLUMNS)
    assert frame["unique_id"].tolist() == [
        "cat_id=CATEGORY",
        "dept_id=DEPARTMENT",
        "item_id=ITEM",
        "state_id=STATE",
        "store_id=STORE",
        "__total__",
        "bottom_item_store",
    ]
    assert reader.selections[0].columns == ("issuances", "resolution")
    assert reader.selections[0].batch_size == 2
    total = frame.loc[frame["unique_id"] == "__total__"].iloc[0]
    assert pd.isna(total["y"])
    assert pd.isna(total["lo_0p9"])
    assert total["hi_0p9"] == 10.0
    bottom = frame.loc[frame["unique_id"] == "bottom_item_store"].iloc[0]
    assert bottom["y"] == 7.0
    assert pd.isna(bottom["lo_0p9"])
    assert pd.isna(bottom["hi_0p9"])
    scored = frame.loc[frame["unique_id"] == "dept_id=DEPARTMENT"].iloc[0]
    assert scored["y"] == scored["lo_0p9"] == 2.0
    assert scored["hi_0p9"] == 10.0


@pytest.mark.parametrize("mutation", ["duplicate-row", "missing-issuance", "wrong-model"])
def test_disposable_export_rejects_malformed_rows(
    tmp_path: Path,
    mutation: str,
) -> None:
    rows = list(_rows())
    if mutation == "duplicate-row":
        rows.append(rows[0])
    elif mutation == "missing-issuance":
        rows[0] = replace(rows[0], issuance=())
    else:
        rows[0] = replace(
            rows[0],
            key=replace(rows[0].key, model_name="unexpected-model"),
        )
    output = tmp_path / "rejected.parquet"

    with pytest.raises(FrozenM5ExportError):
        export_frozen_m5_ledger(load_m5_config(CONFIG), _Reader(tuple(rows)), output)

    assert not output.exists()


def test_disposable_export_removes_partial_after_late_batch_failure(tmp_path: Path) -> None:
    rows = list(_rows())
    rows[-1] = replace(
        rows[-1],
        key=replace(rows[-1].key, model_name="unexpected-model"),
    )
    output = tmp_path / "rejected.parquet"

    with pytest.raises(FrozenM5ExportError):
        export_frozen_m5_ledger(
            load_m5_config(CONFIG),
            _Reader(tuple(rows)),
            output,
            batch_size=1,
        )

    assert not output.exists()
    assert not output.with_suffix(f"{output.suffix}.partial").exists()


def test_disposable_export_rejects_non_0p9_scoring_intent(tmp_path: Path) -> None:
    config = load_m5_config(CONFIG)
    changed = config.conformal_config
    changed["coverage"] = 0.8
    # Bypass the stricter loader to witness the exporter's independent boundary.
    object.__setattr__(
        config,
        "_conformal_config_json",
        json.dumps(changed, sort_keys=True, separators=(",", ":")).encode(),
    )

    with pytest.raises(FrozenM5ExportError, match="0.9"):
        export_frozen_m5_ledger(config, _Reader(_rows()), tmp_path / "rejected.parquet")


@pytest.mark.oracle_gate("m5-frozen-scorer-parity")
def test_successor_and_frozen_m5_scorers_have_exact_count_parity(
    frozen_oracle_worktree: Path,
    m5_parity_run,
) -> None:
    diagnostics, reader = m5_parity_run
    assert diagnostics.status == "VALID"
    with TemporaryDirectory() as directory:
        temporary = Path(directory)
        export = temporary / "frozen-input.parquet"
        frozen_output = temporary / "frozen-output"
        export_frozen_m5_ledger(load_m5_config(CONFIG), reader, export)
        artifact = _run_frozen_scorer(frozen_oracle_worktree, export, frozen_output)
        _assert_count_parity(_successor_counts(diagnostics), _frozen_counts(artifact))
        assert export.exists()
        assert artifact.exists()
    assert not export.exists()
    assert not frozen_output.exists()


@pytest.mark.oracle_witness("m5-frozen-scorer-parity")
def test_one_finite_upper_bound_drift_names_the_changed_level_count(
    frozen_oracle_worktree: Path,
    m5_parity_run,
) -> None:
    diagnostics, reader = m5_parity_run
    with TemporaryDirectory() as directory:
        temporary = Path(directory)
        export = temporary / "frozen-input.parquet"
        frozen_output = temporary / "frozen-output"
        export_frozen_m5_ledger(load_m5_config(CONFIG), reader, export)
        frame = pd.read_parquet(export)
        candidates = frame[
            frame["unique_id"].str.contains("=").eq(False)
            & frame["unique_id"].ne("__total__")
            & frame["y"].notna()
            & frame["hi_0p9"].notna()
            & frame["hi_0p9"].ge(frame["y"])
        ]
        assert not candidates.empty
        row_index = candidates.index[0]
        frame.loc[row_index, "hi_0p9"] = frame.loc[row_index, "y"] - 1.0
        frame.to_parquet(export, index=False)

        artifact = _run_frozen_scorer(frozen_oracle_worktree, export, frozen_output)
        with pytest.raises(AssertionError, match="bottom covered count"):
            _assert_count_parity(_successor_counts(diagnostics), _frozen_counts(artifact))
    assert not export.exists()
    assert not frozen_output.exists()


def test_frozen_translation_and_runtime_references_remain_tier3_only() -> None:
    exporter = PROJECT_ROOT / "tests" / "tier3" / "m5" / "m5_frozen_export.py"
    production = PROJECT_ROOT / "src" / "newcalibre"
    exporter_text = exporter.read_text(encoding="utf-8")
    production_sources = tuple(sorted(production.rglob("*.py")))
    production_text = "\n".join(path.read_text(encoding="utf-8") for path in production_sources)
    frozen_imports = [
        (path.relative_to(production), line, module)
        for path in production_sources
        for line, module in imported_modules(
            path.read_text(encoding="utf-8"),
            package=".".join(("newcalibre", *path.relative_to(production).parent.parts)),
        )
        if module == "calibre" or module.startswith("calibre.")
    ]

    assert '"department": "dept_id"' in exporter_text
    assert '"category": "cat_id"' in exporter_text
    assert "score-m5-coverage" not in production_text
    assert frozen_imports == []
    assert "dept_id=" not in production_text
    assert "cat_id=" not in production_text


def _run_frozen_scorer(worktree: Path, export: Path, output: Path) -> Path:
    command = (
        "uv",
        "run",
        "--locked",
        "--no-sync",
        "calibre",
        "score-m5-coverage",
        "--ledger",
        str(export),
        "--coverage",
        "0.9",
        "--output-dir",
        str(output),
        "--report-only",
    )
    completed = subprocess.run(
        command,
        cwd=worktree,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    artifact = output / "coverage-by-node.parquet"
    assert artifact.is_file()
    return artifact


def _successor_counts(diagnostics: M5Diagnostics) -> dict[str, _Counts]:
    def counts(value) -> _Counts:
        return _Counts(
            total=value.total,
            resolved=value.resolved,
            scored=value.scored,
            covered=value.covered,
        )

    return {
        "population": counts(diagnostics.population.counts),
        **{level: counts(summary.counts) for level, summary in diagnostics.levels.items()},
    }


def _frozen_counts(artifact: Path) -> dict[str, _Counts]:
    frame = pd.read_parquet(artifact)
    required = {"level", "total_rows", "resolved_rows", "scored_rows", "coverage"}
    assert required.issubset(frame.columns)
    rows: dict[str, list[_Counts]] = {level: [] for level in FROZEN_LEVELS.values()}
    population: list[_Counts] = []
    for row in frame.itertuples(index=False):
        scored = int(row.scored_rows)
        if scored:
            covered = round(float(row.coverage) * scored)
            assert math.isclose(float(row.coverage), covered / scored, abs_tol=1e-12)
        else:
            assert pd.isna(row.coverage)
            covered = 0
        counts = _Counts(
            total=int(row.total_rows),
            resolved=int(row.resolved_rows),
            scored=scored,
            covered=covered,
        )
        population.append(counts)
        rows[FROZEN_LEVELS[str(row.level)]].append(counts)
    assert all(rows.values())
    return {
        "population": _sum_counts(population),
        **{level: _sum_counts(counts) for level, counts in rows.items()},
    }


def _sum_counts(values: list[_Counts]) -> _Counts:
    return _Counts(
        total=sum(value.total for value in values),
        resolved=sum(value.resolved for value in values),
        scored=sum(value.scored for value in values),
        covered=sum(value.covered for value in values),
    )


def _assert_count_parity(
    successor: dict[str, _Counts],
    frozen: dict[str, _Counts],
) -> None:
    assert successor.keys() == frozen.keys()
    differences: list[str] = []
    for scope in successor:
        for field in ("total", "resolved", "scored", "covered"):
            successor_value = getattr(successor[scope], field)
            frozen_value = getattr(frozen[scope], field)
            if successor_value != frozen_value:
                differences.append(
                    f"{scope} {field} count differs: "
                    f"successor={successor_value}, frozen={frozen_value}"
                )
    assert not differences, "; ".join(differences)
