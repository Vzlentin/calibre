"""Render and atomically publish deterministic M5 diagnostic projections."""

from __future__ import annotations

import shutil
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

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


class _M5ArtifactError(ValueError):
    """Report an M5 diagnostic rendering or publication failure."""


def _emit_artifacts(
    output_dir: Path,
    *,
    summary: Mapping[str, object],
    node_rows: Sequence[Mapping[str, object]],
) -> tuple[Path, Path, Path]:
    """Build all three projections and publish them through one directory rename."""
    destination = Path(output_dir)
    if destination.exists() or destination.is_symlink():
        raise _M5ArtifactError("M5 diagnostic destination must not already exist")
    try:
        summary_bytes = canonical_json_bytes(dict(summary), path="M5 coverage summary") + b"\n"
        report_bytes = _render_report(summary)
        table = _node_table(node_rows)
        sink = pa.BufferOutputStream()
        pq.write_table(table, sink, compression="NONE")
        parquet_bytes = sink.getvalue().to_pybytes()
    except (CanonicalJsonError, KeyError, TypeError, ValueError, pa.ArrowException) as error:
        raise _M5ArtifactError("M5 diagnostic projections are invalid") from error

    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    except OSError as error:
        raise _M5ArtifactError("M5 diagnostic output directory is unavailable") from error
    try:
        (temporary / _SUMMARY_NAME).write_bytes(summary_bytes)
        (temporary / _NODE_NAME).write_bytes(parquet_bytes)
        (temporary / _REPORT_NAME).write_bytes(report_bytes)
        if {path.name for path in temporary.iterdir()} != set(_ARTIFACT_NAMES):
            raise _M5ArtifactError("M5 diagnostic staging produced an unexpected file set")
        temporary.replace(destination)
    except Exception as error:
        shutil.rmtree(temporary, ignore_errors=True)
        if isinstance(error, _M5ArtifactError):
            raise
        raise _M5ArtifactError("M5 diagnostic publication failed") from error

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
        raise _M5ArtifactError("M5 diagnostic rate must be numeric or null")
    return f"{float(value):.6f}"


__all__: list[str] = []
