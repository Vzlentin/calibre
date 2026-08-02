"""Render and atomically publish deterministic M5 diagnostic projections."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast, overload

import pyarrow as pa
import pyarrow.parquet as pq

from newcalibre.domain._canonical_json import CanonicalJsonError, canonical_json_bytes

_SUMMARY_NAME = "coverage-summary.json"
_NODE_NAME = "coverage-by-node.parquet"
_REPORT_NAME = "report.md"
_ARTIFACT_NAMES = (_SUMMARY_NAME, _NODE_NAME, _REPORT_NAME)
_NODE_SCHEMA = pa.schema(
    [
        pa.field("schema", pa.int16(), nullable=False),
        pa.field("status", pa.string(), nullable=False),
        pa.field("metric", pa.string(), nullable=False),
        pa.field("dataset", pa.string(), nullable=False),
        pa.field("phase", pa.string(), nullable=False),
        pa.field("session_id", pa.string(), nullable=False),
        pa.field("reconciler", pa.string(), nullable=False),
        pa.field("conformal_method", pa.string(), nullable=False),
        pa.field("conformal_partition", pa.string(), nullable=False),
        pa.field("origin_start", pa.string(), nullable=True),
        pa.field("origin_end", pa.string(), nullable=True),
        pa.field("origin_count", pa.int64(), nullable=False),
        pa.field("horizon", pa.int64(), nullable=False),
        pa.field("level", pa.string(), nullable=False),
        pa.field("node", pa.string(), nullable=False),
        pa.field("model", pa.string(), nullable=False),
        pa.field("target", pa.float64(), nullable=False),
        pa.field("coverage", pa.float64(), nullable=True),
        pa.field("deviation", pa.float64(), nullable=True),
        pa.field("total", pa.int64(), nullable=False),
        pa.field("resolved", pa.int64(), nullable=False),
        pa.field("eligible", pa.int64(), nullable=False),
        pa.field("scored", pa.int64(), nullable=False),
        pa.field("covered", pa.int64(), nullable=False),
        pa.field("mask_identity", pa.string(), nullable=False),
        pa.field("mask_equal", pa.bool_(), nullable=False),
    ]
)


class M5ArtifactError(ValueError):
    """Report an M5 diagnostic rendering or publication failure."""


@dataclass(frozen=True, slots=True)
class M5ArtifactSet:
    """Carry one validated three-file M5 diagnostic set."""

    root: Path
    summary: Mapping[str, object]
    nodes: pa.Table
    coverage_summary_digest: str
    coverage_by_node_digest: str
    report_digest: str


def load_m5_artifacts(root: Path) -> M5ArtifactSet:
    """Load and cross-validate one exact M5 diagnostic directory."""
    directory = Path(root)
    if directory.is_symlink() or not directory.is_dir():
        raise M5ArtifactError("M5 diagnostic root must be a regular directory")
    paths = {path.name: path for path in directory.iterdir()}
    if set(paths) != set(_ARTIFACT_NAMES) or any(path.is_symlink() for path in paths.values()):
        raise M5ArtifactError("M5 diagnostic root must contain exactly three regular files")
    return validate_m5_artifact_files(
        paths[_SUMMARY_NAME], paths[_NODE_NAME], paths[_REPORT_NAME], root=directory
    )


def validate_m5_artifact_files(
    summary_path: Path,
    by_node_path: Path,
    report_path: Path,
    *,
    root: Path,
) -> M5ArtifactSet:
    """Cross-validate three explicitly named M5 diagnostic files."""
    paths = {
        _SUMMARY_NAME: Path(summary_path),
        _NODE_NAME: Path(by_node_path),
        _REPORT_NAME: Path(report_path),
    }
    if any(path.is_symlink() or not path.is_file() for path in paths.values()):
        raise M5ArtifactError("M5 diagnostic files must be regular files")
    try:
        summary_bytes = paths[_SUMMARY_NAME].read_bytes()
        raw = json.loads(summary_bytes)
        if not isinstance(raw, dict):
            raise TypeError("summary is not an object")
        summary = cast(dict[str, object], raw)
        if summary_bytes != canonical_json_bytes(summary, path="M5 coverage summary") + b"\n":
            raise ValueError("summary is not canonical JSON")
        _validate_summary(summary)
        report_bytes = paths[_REPORT_NAME].read_bytes()
        if report_bytes != _render_report(summary):
            raise ValueError("report does not match the summary projection")
        table = pq.read_table(paths[_NODE_NAME])
        _validate_node_table(table, summary=summary)
    except (
        OSError,
        UnicodeError,
        json.JSONDecodeError,
        CanonicalJsonError,
        TypeError,
        ValueError,
        pa.ArrowException,
    ) as error:
        raise M5ArtifactError("M5 diagnostic artifacts are malformed or inconsistent") from error
    return M5ArtifactSet(
        root=Path(root).resolve(),
        summary=summary,
        nodes=table,
        coverage_summary_digest=hashlib.sha256(summary_bytes).hexdigest(),
        coverage_by_node_digest=hashlib.sha256(paths[_NODE_NAME].read_bytes()).hexdigest(),
        report_digest=hashlib.sha256(report_bytes).hexdigest(),
    )


def _emit_artifacts(
    output_dir: Path,
    *,
    summary: Mapping[str, object],
    node_rows: Sequence[Mapping[str, object]],
) -> tuple[Path, Path, Path]:
    """Build all three projections and publish them through one directory rename."""
    destination = output_dir
    if destination.exists() or destination.is_symlink():
        raise M5ArtifactError("M5 diagnostic destination must not already exist")
    try:
        summary_bytes = canonical_json_bytes(dict(summary), path="M5 coverage summary") + b"\n"
        report_bytes = _render_report(summary)
        table = _node_table(node_rows)
        sink = pa.BufferOutputStream()
        pq.write_table(table, sink, compression="NONE")
        parquet_bytes = sink.getvalue().to_pybytes()
    except (CanonicalJsonError, KeyError, TypeError, ValueError, pa.ArrowException) as error:
        raise M5ArtifactError("M5 diagnostic projections are invalid") from error

    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    except OSError as error:
        raise M5ArtifactError("M5 diagnostic output directory is unavailable") from error
    try:
        (temporary / _SUMMARY_NAME).write_bytes(summary_bytes)
        (temporary / _NODE_NAME).write_bytes(parquet_bytes)
        (temporary / _REPORT_NAME).write_bytes(report_bytes)
        if {path.name for path in temporary.iterdir()} != set(_ARTIFACT_NAMES):
            raise M5ArtifactError("M5 diagnostic staging produced an unexpected file set")
        temporary.replace(destination)
    except Exception as error:
        shutil.rmtree(temporary, ignore_errors=True)
        if isinstance(error, M5ArtifactError):
            raise
        raise M5ArtifactError("M5 diagnostic publication failed") from error

    root = destination.resolve()
    return root / _SUMMARY_NAME, root / _NODE_NAME, root / _REPORT_NAME


def _node_table(rows: Sequence[Mapping[str, object]]) -> pa.Table:
    ordered = sorted(
        (dict(row) for row in rows),
        key=lambda row: (
            cast(str, row["level"]).encode(),
            cast(str, row["node"]).encode(),
            cast(str, row["model"]).encode(),
        ),
    )
    columns = {field.name: [row[field.name] for row in ordered] for field in _NODE_SCHEMA}
    return pa.Table.from_pydict(columns, schema=_NODE_SCHEMA)


def _validate_summary(summary: Mapping[str, object]) -> None:
    _exact_keys(
        summary,
        {"schema", "status", "metric", "context", "mask", "population", "per_model", "levels"},
        name="M5 coverage summary",
    )
    if summary["schema"] != 1 or summary["status"] not in {"VALID", "INVALID"}:
        raise ValueError("M5 summary schema or status is unsupported")
    if summary["metric"] != "sales-coverage":
        raise ValueError("M5 summary metric is not sales-coverage")
    context = _mapping(summary["context"], name="M5 summary context")
    _exact_keys(
        context,
        {
            "dataset",
            "phase",
            "session_id",
            "model_name",
            "reconciler",
            "conformal_method",
            "conformal_partition",
            "origin_start",
            "origin_end",
            "origin_count",
            "horizon",
            "target",
            "node_count",
            "expected_row_count",
        },
        name="M5 summary context",
    )
    for key in (
        "dataset",
        "phase",
        "session_id",
        "model_name",
        "reconciler",
        "conformal_method",
        "conformal_partition",
    ):
        _text(context[key], name=f"M5 context {key}")
    for key in ("origin_start", "origin_end"):
        if context[key] is not None:
            _text(context[key], name=f"M5 context {key}")
    for key in ("origin_count", "horizon", "node_count", "expected_row_count"):
        _nonnegative_integer(context[key], name=f"M5 context {key}")
    _rate(context["target"], name="M5 target", nullable=False)
    mask = _mapping(summary["mask"], name="M5 summary mask")
    _exact_keys(
        mask,
        {
            "identity",
            "definition",
            "equal",
            "expected_eligible_count",
            "actual_scored_count",
            "missing_eligible_count",
            "early_scored_count",
            "structural_issue_count",
            "reasons",
            "examples",
        },
        name="M5 summary mask",
    )
    _text(mask["identity"], name="M5 mask identity")
    _text(mask["definition"], name="M5 mask definition")
    if not isinstance(mask["equal"], bool):
        raise ValueError("M5 mask equality must be boolean")
    for key in (
        "expected_eligible_count",
        "actual_scored_count",
        "missing_eligible_count",
        "early_scored_count",
        "structural_issue_count",
    ):
        _nonnegative_integer(mask[key], name=f"M5 mask {key}")
    reasons = _mapping(mask["reasons"], name="M5 mask reasons")
    examples = _mapping(mask["examples"], name="M5 mask examples")
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value < 1
        for value in reasons.values()
    ):
        raise ValueError("M5 mask reason counts must be positive integers")
    if any(
        not isinstance(value, list) or any(not isinstance(item, str) for item in value)
        for value in examples.values()
    ):
        raise ValueError("M5 mask examples must contain text lists")
    population = _coverage(summary["population"], name="M5 population")
    population_counts = _counts(population["counts"], name="M5 population counts")
    if mask["expected_eligible_count"] != population_counts["eligible"]:
        raise ValueError("M5 mask expected count differs from population")
    if mask["actual_scored_count"] != population_counts["scored"]:
        raise ValueError("M5 mask scored count differs from population")
    if mask["equal"] is not population["mask_equal"]:
        raise ValueError("M5 mask equality differs from population")
    models = summary["per_model"]
    if not isinstance(models, list) or len(models) != 1:
        raise ValueError("M5 summary must contain exactly one model")
    model = _mapping(models[0], name="M5 model summary")
    _text(model.get("model"), name="M5 model name")
    _coverage({key: value for key, value in model.items() if key != "model"}, name="M5 model")
    levels = summary["levels"]
    if not isinstance(levels, list) or len(levels) != 7:
        raise ValueError("M5 summary must contain all seven level classes")
    expected_levels = ("bottom", "item", "department", "category", "store", "state", "total")
    for raw, expected in zip(levels, expected_levels, strict=True):
        level = _mapping(raw, name="M5 level summary")
        _exact_keys(
            level,
            {
                "level",
                "label",
                "node_count",
                "scored_node_count",
                "target",
                "mean_node_coverage",
                "pooled_coverage",
                "mean_node_deviation",
                "pooled_deviation",
                "counts",
                "mask_equal",
                "missing_eligible_count",
                "early_scored_count",
            },
            name="M5 level summary",
        )
        if level["level"] != expected or level["label"] != "sales-coverage":
            raise ValueError("M5 level labels are incomplete or out of order")
        _nonnegative_integer(level["node_count"], name="M5 level node count")
        _nonnegative_integer(level["scored_node_count"], name="M5 scored node count")
        _rate(level["target"], name="M5 level target", nullable=False)
        for key in ("mean_node_coverage", "pooled_coverage"):
            _rate(level[key], name=f"M5 level {key}", nullable=True)
        for key in ("mean_node_deviation", "pooled_deviation"):
            _finite(level[key], name=f"M5 level {key}", nullable=True)
        _counts(level["counts"], name="M5 level counts")
        if not isinstance(level["mask_equal"], bool):
            raise ValueError("M5 level mask equality must be boolean")
        _nonnegative_integer(level["missing_eligible_count"], name="M5 level missing count")
        _nonnegative_integer(level["early_scored_count"], name="M5 level early count")


def _validate_node_table(table: pa.Table, *, summary: Mapping[str, object]) -> None:
    if table.schema != _NODE_SCHEMA:
        raise ValueError("M5 node table schema is not exact")
    context = cast(Mapping[str, object], summary["context"])
    population = cast(Mapping[str, object], summary["population"])
    rows = table.to_pylist()
    if len(rows) != context["node_count"]:
        raise ValueError("M5 node table row count differs from the declared population")
    nodes: set[str] = set()
    totals = {name: 0 for name in ("total", "resolved", "eligible", "scored", "covered")}
    level_counts: dict[str, int] = {}
    level_totals: dict[str, dict[str, int]] = {}
    for row in rows:
        if row["node"] in nodes:
            raise ValueError("M5 node table contains duplicate nodes")
        nodes.add(row["node"])
        if (
            row["schema"] != 1
            or row["status"] != summary["status"]
            or row["metric"] != "sales-coverage"
            or row["dataset"] != context["dataset"]
            or row["phase"] != context["phase"]
            or row["session_id"] != context["session_id"]
            or row["reconciler"] != context["reconciler"]
            or row["conformal_method"] != context["conformal_method"]
            or row["conformal_partition"] != context["conformal_partition"]
            or row["origin_start"] != context["origin_start"]
            or row["origin_end"] != context["origin_end"]
            or row["origin_count"] != context["origin_count"]
            or row["horizon"] != context["horizon"]
            or row["model"] != context["model_name"]
            or row["target"] != context["target"]
        ):
            raise ValueError("M5 node row context differs from the summary")
        if row["level"] not in {
            "bottom",
            "item",
            "department",
            "category",
            "store",
            "state",
            "total",
        }:
            raise ValueError("M5 node row has an unknown level")
        counts = {name: _nonnegative_integer(row[name], name=f"M5 node {name}") for name in totals}
        if not (
            counts["covered"]
            <= counts["scored"]
            <= counts["eligible"]
            <= counts["resolved"]
            <= counts["total"]
        ):
            raise ValueError("M5 node counts are not monotonic")
        coverage = _rate(row["coverage"], name="M5 node coverage", nullable=True)
        deviation = _finite(row["deviation"], name="M5 node deviation", nullable=True)
        expected_coverage = None if not counts["scored"] else counts["covered"] / counts["scored"]
        if coverage != expected_coverage:
            raise ValueError("M5 node coverage does not match counts")
        expected_deviation = None if coverage is None else coverage - cast(float, row["target"])
        if deviation != expected_deviation:
            raise ValueError("M5 node deviation does not match coverage")
        for name, value in counts.items():
            totals[name] += value
        level = cast(str, row["level"])
        level_counts[level] = level_counts.get(level, 0) + 1
        aggregate = level_totals.setdefault(level, {name: 0 for name in totals})
        for name, value in counts.items():
            aggregate[name] += value
    if totals != cast(Mapping[str, object], population["counts"]):
        raise ValueError("M5 node counts do not reduce to the population counts")
    levels = cast(Sequence[Mapping[str, object]], summary["levels"])
    for level in levels:
        name = cast(str, level["level"])
        if (
            level_counts.get(name, 0) != level["node_count"]
            or level_totals.get(name) != level["counts"]
        ):
            raise ValueError("M5 node rows do not reduce to the level summaries")


def _coverage(value: object, *, name: str) -> dict[str, object]:
    payload = _mapping(value, name=name)
    _exact_keys(
        payload,
        {
            "label",
            "target",
            "coverage",
            "deviation",
            "counts",
            "mask_equal",
            "missing_eligible_count",
            "early_scored_count",
        },
        name=name,
    )
    if payload["label"] != "sales-coverage":
        raise ValueError(f"{name} label is not sales-coverage")
    target = _rate(payload["target"], name=f"{name} target", nullable=False)
    counts = _counts(payload["counts"], name=f"{name} counts")
    coverage = _rate(payload["coverage"], name=f"{name} coverage", nullable=True)
    deviation = _finite(payload["deviation"], name=f"{name} deviation", nullable=True)
    expected_coverage = None if not counts["scored"] else counts["covered"] / counts["scored"]
    if coverage != expected_coverage or deviation != (
        None if coverage is None else coverage - target
    ):
        raise ValueError(f"{name} rates do not match counts")
    if not isinstance(payload["mask_equal"], bool):
        raise ValueError(f"{name} mask equality must be boolean")
    for key in ("missing_eligible_count", "early_scored_count"):
        _nonnegative_integer(payload[key], name=f"{name} {key}")
    return payload


def _counts(value: object, *, name: str) -> dict[str, int]:
    payload = _mapping(value, name=name)
    _exact_keys(payload, {"total", "resolved", "eligible", "scored", "covered"}, name=name)
    result = {key: _nonnegative_integer(payload[key], name=f"{name} {key}") for key in payload}
    if not (
        result["covered"]
        <= result["scored"]
        <= result["eligible"]
        <= result["resolved"]
        <= result["total"]
    ):
        raise ValueError(f"{name} are not monotonic")
    return result


def _mapping(value: object, *, name: str) -> dict[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"{name} must be a string-keyed mapping")
    return dict(cast(Mapping[str, object], value))


def _exact_keys(value: Mapping[str, object], keys: set[str], *, name: str) -> None:
    if set(value) != keys:
        raise ValueError(f"{name} must contain exact fields")


def _text(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{name} must be non-empty text")
    return value


def _nonnegative_integer(value: object, *, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise TypeError(f"{name} must be a non-negative integer")
    return value


@overload
def _rate(value: object, *, name: str, nullable: Literal[False]) -> float: ...
@overload
def _rate(value: object, *, name: str, nullable: Literal[True]) -> float | None: ...
def _rate(value: object, *, name: str, nullable: bool) -> float | None:
    result = _finite(value, name=name, nullable=nullable)
    if result is not None and not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be between zero and one")
    return result


def _finite(value: object, *, name: str, nullable: bool) -> float | None:
    if value is None and nullable:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be finite numeric evidence")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _render_report(summary: Mapping[str, object]) -> bytes:
    context = cast(Mapping[str, object], summary["context"])
    population = cast(Mapping[str, object], summary["population"])
    mask = cast(Mapping[str, object], summary["mask"])
    levels = cast(Sequence[Mapping[str, object]], summary["levels"])
    population_counts = cast(Mapping[str, object], population["counts"])
    lines = [
        "# M5 sales-coverage diagnostics",
        "",
        f"**Structural status:** {summary['status']}",
        "",
        (
            "This sales-coverage diagnostic describes recorded sales. "
            "It is not a demand or service guarantee."
        ),
        "",
        "## Run context",
        "",
        f"- Dataset phase: `{context['dataset']}` / `{context['phase']}`",
        f"- Session: `{context['session_id']}`",
        f"- Model: `{context['model_name']}`",
        f"- Reconciler: `{context['reconciler']}`",
        f"- Conformal method: `{context['conformal_method']}`",
        f"- Partition: `{context['conformal_partition']}`",
        (
            f"- Origin window: `{context['origin_start']}` through "
            f"`{context['origin_end']}` ({context['origin_count']} origins)"
        ),
        f"- Horizon: `{context['horizon']}`",
        f"- Target: `{context['target']}`",
        f"- Mask identity: `{mask['identity']}`",
        f"- Exact mask equality: `{mask['equal']}`",
        "",
        "## Population sales-coverage",
        "",
        (
            f"Coverage: `{_display_rate(population['coverage'])}`; signed deviation: "
            f"`{_display_rate(population['deviation'])}`."
        ),
        "",
        "| total | resolved | eligible | scored | covered |",
        "| ---: | ---: | ---: | ---: | ---: |",
        "| "
        + " | ".join(
            str(population_counts[name])
            for name in ("total", "resolved", "eligible", "scored", "covered")
        )
        + " |",
        "",
        "## Sales-coverage by level",
        "",
        (
            "| level | nodes | scored nodes | mean-node rate | pooled rate | target | "
            "mean deviation | pooled deviation | total | resolved | eligible | scored | "
            "covered | mask equal |"
        ),
        (
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | "
            "---: | ---: | ---: | ---: | --- |"
        ),
    ]
    for level in levels:
        counts = cast(Mapping[str, object], level["counts"])
        lines.append(
            "| "
            + " | ".join(
                [
                    str(level["level"]),
                    str(level["node_count"]),
                    str(level["scored_node_count"]),
                    _display_rate(level["mean_node_coverage"]),
                    _display_rate(level["pooled_coverage"]),
                    _display_rate(level["target"]),
                    _display_rate(level["mean_node_deviation"]),
                    _display_rate(level["pooled_deviation"]),
                    *(
                        str(counts[name])
                        for name in ("total", "resolved", "eligible", "scored", "covered")
                    ),
                    str(level["mask_equal"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Machine artifacts",
            "",
            (
                f"- `{_SUMMARY_NAME}` contains the complete sales-coverage summary "
                "and exact-mask result."
            ),
            (
                f"- `{_NODE_NAME}` contains one sales-coverage row per node and model, "
                "including nodes with no scored rows."
            ),
            "",
        ]
    )
    return "\n".join(lines).encode("utf-8")


def _display_rate(value: object) -> str:
    if value is None:
        return "n/a"
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise M5ArtifactError("M5 diagnostic rate must be numeric or null")
    return f"{float(value):.6f}"


__all__ = [
    "M5ArtifactError",
    "M5ArtifactSet",
    "load_m5_artifacts",
    "validate_m5_artifact_files",
]
