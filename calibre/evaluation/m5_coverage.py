from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from calibre.core.forecast_frame import MODEL_NAME, UNIQUE_ID, H, Y, interval_column_names
from calibre.evaluation.forecast_metrics import (
    prepare_interval_coverage_frame,
    summarize_interval_coverage,
)
from calibre.reconciliation.summing import TOTAL_LABEL

COVERAGE_BY_NODE_NAME = "coverage-by-node.parquet"
COVERAGE_REPORT_NAME = "report.md"
COVERAGE_SUMMARY_NAME = "coverage-summary.json"
LEVEL_COLUMN = "level"

PASS = "PASS"
FAIL = "FAIL"
UNKNOWN = "UNKNOWN"


@dataclass(frozen=True, slots=True)
class CoverageThresholds:
    population_tolerance: float = 0.03
    level_tolerance: float = 0.05
    node_outlier_tolerance: float = 0.10
    minimum_scored_ratio: float = 0.50


@dataclass(frozen=True, slots=True)
class M5CoverageArtifacts:
    coverage_by_node_path: Path
    report_path: Path
    summary_path: Path
    coverage_by_node: pd.DataFrame
    population: pd.DataFrame
    per_level: pd.DataFrame
    per_horizon: pd.DataFrame
    summary: dict[str, Any]

    @property
    def acceptance_status(self) -> str:
        return str(self.summary["acceptance_status"])


def infer_m5_level(unique_id: object) -> str:
    label = str(unique_id)
    if label == TOTAL_LABEL:
        return "total"
    if "=" in label:
        return label.split("=", 1)[0]
    return "bottom"


def score_resolved_ledger(
    ledger_path: str | Path,
    *,
    coverage: float = 0.9,
    output_dir: str | Path | None = None,
    thresholds: CoverageThresholds | None = None,
) -> M5CoverageArtifacts:
    thresholds = thresholds or CoverageThresholds()
    path = Path(ledger_path)
    if not path.exists():
        raise FileNotFoundError(f"resolved ledger not found: {path}")
    run_dir = Path(output_dir) if output_dir is not None else path.parent
    ledger = _read_required_ledger_columns(path, coverage=coverage)
    return write_coverage_artifacts(
        ledger,
        output_dir=run_dir,
        coverage=coverage,
        thresholds=thresholds,
    )


def write_coverage_artifacts(
    ledger: pd.DataFrame,
    *,
    output_dir: str | Path,
    coverage: float = 0.9,
    thresholds: CoverageThresholds | None = None,
) -> M5CoverageArtifacts:
    thresholds = thresholds or CoverageThresholds()
    lower_col, upper_col = interval_column_names(coverage)
    required = _required_columns(lower_col=lower_col, upper_col=upper_col)
    _validate_ledger_columns(ledger, required=required)
    if ledger.empty:
        raise ValueError("resolved ledger is empty")
    if ledger[Y].notna().sum() == 0:
        raise ValueError("resolved ledger has no resolved actual rows to score")

    scoring = prepare_interval_coverage_frame(
        ledger,
        coverage=coverage,
        group_by=[UNIQUE_ID, H, MODEL_NAME],
    )
    scoring[LEVEL_COLUMN] = _level_lookup(scoring)

    node = summarize_interval_coverage(
        scoring,
        coverage=coverage,
        group_by=[UNIQUE_ID, LEVEL_COLUMN, H, MODEL_NAME],
    )
    if int(node["scored_rows"].sum()) == 0:
        raise ValueError(
            "resolved ledger has no rows with finite interval bounds "
            f"({lower_col!r}, {upper_col!r})"
        )

    node = _with_outlier_flags(node, coverage=coverage, thresholds=thresholds)
    per_node = summarize_interval_coverage(
        scoring,
        coverage=coverage,
        group_by=[UNIQUE_ID, LEVEL_COLUMN, MODEL_NAME],
    )
    per_node = _with_outlier_flags(per_node, coverage=coverage, thresholds=thresholds)
    population = summarize_interval_coverage(scoring, coverage=coverage, group_by=[])
    per_level = _per_level_summary(scoring, per_node, coverage=coverage)
    per_horizon = summarize_interval_coverage(scoring, coverage=coverage, group_by=[H])
    per_horizon = per_horizon.sort_values(H, kind="stable").reset_index(drop=True)
    summary = _coverage_summary(
        population=population,
        per_level=per_level,
        coverage=coverage,
        thresholds=thresholds,
    )

    run_dir = Path(output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    coverage_by_node_path = run_dir / COVERAGE_BY_NODE_NAME
    report_path = run_dir / COVERAGE_REPORT_NAME
    summary_path = run_dir / COVERAGE_SUMMARY_NAME
    node.to_parquet(coverage_by_node_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(
        _render_report(
            node=node,
            population=population,
            per_level=per_level,
            per_horizon=per_horizon,
            coverage=coverage,
            thresholds=thresholds,
            summary=summary,
        ),
        encoding="utf-8",
    )
    return M5CoverageArtifacts(
        coverage_by_node_path=coverage_by_node_path,
        report_path=report_path,
        summary_path=summary_path,
        coverage_by_node=node,
        population=population,
        per_level=per_level,
        per_horizon=per_horizon,
        summary=summary,
    )


def _required_columns(*, lower_col: str, upper_col: str) -> list[str]:
    return [UNIQUE_ID, H, MODEL_NAME, Y, lower_col, upper_col]


def _read_required_ledger_columns(path: Path, *, coverage: float) -> pd.DataFrame:
    lower_col, upper_col = interval_column_names(coverage)
    required = _required_columns(lower_col=lower_col, upper_col=upper_col)
    schema_columns = _parquet_columns(path)
    missing = [column for column in required if column not in schema_columns]
    if missing:
        raise ValueError(f"resolved ledger missing required column(s): {missing}")
    return pd.read_parquet(path, columns=required)


def _parquet_columns(path: Path) -> list[str]:
    try:
        return list(pq.read_schema(path).names)
    except Exception as exc:
        raise ValueError(f"resolved ledger is not readable parquet: {path}") from exc


def _level_lookup(frame: pd.DataFrame) -> pd.Series:
    level_by_uid = {uid: infer_m5_level(uid) for uid in frame[UNIQUE_ID].drop_duplicates()}
    return frame[UNIQUE_ID].map(level_by_uid)


def _validate_ledger_columns(ledger: pd.DataFrame, *, required: list[str]) -> None:
    missing = [column for column in required if column not in ledger.columns]
    if missing:
        raise ValueError(f"resolved ledger missing required column(s): {missing}")


def _with_outlier_flags(
    frame: pd.DataFrame,
    *,
    coverage: float,
    thresholds: CoverageThresholds,
) -> pd.DataFrame:
    flagged = frame.copy()
    flagged["coverage_error"] = flagged["coverage"] - float(coverage)
    flagged["abs_coverage_error"] = flagged["coverage_error"].abs()
    flagged["is_outlier"] = flagged["scored_rows"].gt(0) & flagged["abs_coverage_error"].gt(
        thresholds.node_outlier_tolerance
    )
    return flagged


def _per_level_summary(
    scoring: pd.DataFrame,
    per_node: pd.DataFrame,
    *,
    coverage: float,
) -> pd.DataFrame:
    pooled = summarize_interval_coverage(
        scoring,
        coverage=coverage,
        group_by=[LEVEL_COLUMN],
    ).rename(
        columns={
            "coverage": "pooled_coverage",
            "mean_interval_width": "pooled_mean_interval_width",
        }
    )
    scored_node = per_node[per_node["scored_rows"].gt(0)]
    average = (
        scored_node.groupby(LEVEL_COLUMN, dropna=False, sort=False)
        .agg(
            average_node_coverage=("coverage", "mean"),
            scored_node_groups=("coverage", "count"),
            outlier_node_groups=("is_outlier", "sum"),
        )
        .reset_index()
    )
    summary = pooled.merge(average, on=LEVEL_COLUMN, how="left")
    summary["scored_node_groups"] = summary["scored_node_groups"].fillna(0).astype("int64")
    summary["outlier_node_groups"] = summary["outlier_node_groups"].fillna(0).astype("int64")
    summary["scored_resolved_ratio"] = _ratio(summary["scored_rows"], summary["resolved_rows"])
    return summary


def _coverage_summary(
    *,
    population: pd.DataFrame,
    per_level: pd.DataFrame,
    coverage: float,
    thresholds: CoverageThresholds,
) -> dict[str, Any]:
    population_row = population.iloc[0]
    population_status = _coverage_status(
        population_row["coverage"],
        target=coverage,
        tolerance=thresholds.population_tolerance,
    )
    level_status = _gate_status(
        [
            _coverage_status(value, target=coverage, tolerance=thresholds.level_tolerance)
            for value in per_level["average_node_coverage"].tolist()
        ]
    )
    population_scored_ratio = _safe_ratio(
        population_row["scored_rows"], population_row["resolved_rows"]
    )
    level_ratios = per_level["scored_resolved_ratio"].tolist()
    completeness_status = _completeness_status(
        [population_scored_ratio, *level_ratios],
        minimum=thresholds.minimum_scored_ratio,
    )
    acceptance_status = _gate_status([population_status, level_status, completeness_status])
    return {
        "acceptance_status": acceptance_status,
        "population_status": population_status,
        "level_status": level_status,
        "completeness_status": completeness_status,
        "target_coverage": float(coverage),
        "population_tolerance": float(thresholds.population_tolerance),
        "level_tolerance": float(thresholds.level_tolerance),
        "minimum_scored_ratio": float(thresholds.minimum_scored_ratio),
        "population_coverage": _json_float(population_row["coverage"]),
        "population_scored_ratio": _json_float(population_scored_ratio),
        "population_total_rows": int(population_row["total_rows"]),
        "population_resolved_rows": int(population_row["resolved_rows"]),
        "population_scored_rows": int(population_row["scored_rows"]),
        "population_unscored_rows": int(population_row["unscored_rows"]),
    }


def _render_report(
    *,
    node: pd.DataFrame,
    population: pd.DataFrame,
    per_level: pd.DataFrame,
    per_horizon: pd.DataFrame,
    coverage: float,
    thresholds: CoverageThresholds,
    summary: dict[str, Any],
) -> str:
    population_row = population.iloc[0]
    outliers = node[node["is_outlier"]].nlargest(
        20,
        ["abs_coverage_error", "scored_rows"],
    )
    width_summary = _width_summary(node)

    lines = [
        "# M5 Coverage Report",
        "",
        f"- Target coverage: {_format_pct(coverage)}",
        (
            "- Population marginal coverage: "
            f"{_format_pct(population_row['coverage'])} "
            f"({int(population_row['scored_rows'])} scored / "
            f"{int(population_row['resolved_rows'])} resolved / "
            f"{int(population_row['total_rows'])} total rows)"
        ),
        (
            "- Acceptance gate: "
            f"{summary['acceptance_status']} "
            f"(population {summary['population_status']}, "
            f"per-level {summary['level_status']}, "
            f"completeness {summary['completeness_status']}; "
            f"population tolerance +/- {_format_pp(thresholds.population_tolerance)}, "
            f"level-average tolerance +/- {_format_pp(thresholds.level_tolerance)}, "
            f"minimum scored/resolved {_format_pct(thresholds.minimum_scored_ratio)})"
        ),
        (
            "- Caveat: rows diagnose realized marginal coverage around reconciled "
            "point forecasts; they are not coherent interval boxes or conditional "
            "per-node guarantees."
        ),
        "",
        "## Per-Level Diagnostics",
        "",
        _markdown_table(
            per_level,
            [
                (LEVEL_COLUMN, "level", str),
                ("average_node_coverage", "avg node coverage", _format_pct),
                ("pooled_coverage", "pooled coverage", _format_pct),
                ("scored_resolved_ratio", "scored/resolved", _format_pct),
                ("scored_rows", "scored rows", _format_int),
                ("unscored_rows", "unscored rows", _format_int),
                ("outlier_node_groups", "outlier groups", _format_int),
            ],
        ),
        "",
        "## Per-Horizon Diagnostics",
        "",
        _markdown_table(
            per_horizon,
            [
                (H, "h", _format_int),
                ("coverage", "coverage", _format_pct),
                ("scored_rows", "scored rows", _format_int),
                ("unscored_rows", "unscored rows", _format_int),
                ("mean_interval_width", "mean width", _format_float),
            ],
        ),
        "",
        "## Interval Width Summary",
        "",
        _markdown_table(
            width_summary,
            [
                ("stat", "stat", str),
                ("value", "value", _format_float),
            ],
        ),
        "",
        "## Per-Node Outliers",
        "",
    ]
    if outliers.empty:
        lines.append("No per-node diagnostic outliers above the configured tolerance.")
    else:
        lines.append(
            _markdown_table(
                outliers,
                [
                    (UNIQUE_ID, "unique_id", str),
                    (LEVEL_COLUMN, "level", str),
                    (H, "h", _format_int),
                    (MODEL_NAME, "model", str),
                    ("coverage", "coverage", _format_pct),
                    ("scored_rows", "scored rows", _format_int),
                    ("abs_coverage_error", "abs error", _format_pp),
                ],
            )
        )
    return "\n".join(lines).rstrip() + "\n"


def _width_summary(node: pd.DataFrame) -> pd.DataFrame:
    widths = node.loc[node["scored_rows"].gt(0), "mean_interval_width"].dropna()
    if widths.empty:
        return pd.DataFrame({"stat": ["count"], "value": [0.0]})
    return pd.DataFrame(
        {
            "stat": ["count", "mean", "median", "p10", "p90", "min", "max"],
            "value": [
                float(len(widths)),
                float(widths.mean()),
                float(widths.median()),
                float(widths.quantile(0.10)),
                float(widths.quantile(0.90)),
                float(widths.min()),
                float(widths.max()),
            ],
        }
    )


def _coverage_status(value: object, *, target: float, tolerance: float) -> str:
    numeric = _as_float(value)
    if numeric is None:
        return UNKNOWN
    return PASS if abs(numeric - float(target)) <= float(tolerance) else FAIL


def _completeness_status(ratios: list[object], *, minimum: float) -> str:
    if not ratios:
        return UNKNOWN
    numeric = [_as_float(value) for value in ratios]
    if any(value is None for value in numeric):
        return UNKNOWN
    return PASS if all(value >= minimum for value in numeric if value is not None) else UNKNOWN


def _gate_status(statuses: list[str]) -> str:
    if not statuses or any(status == UNKNOWN for status in statuses):
        return UNKNOWN
    return PASS if all(status == PASS for status in statuses) else FAIL


def _markdown_table(
    frame: pd.DataFrame,
    columns: list[tuple[str, str, Callable[[object], str]]],
) -> str:
    headers = [label for _, label, _ in columns]
    if frame.empty:
        return "_No rows._"
    rows = []
    for _, row in frame.iterrows():
        rendered = []
        for column, _, formatter in columns:
            value = row[column]
            rendered.append(formatter(value))
        rows.append(rendered)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(values) + " |" for values in rows)
    return "\n".join(lines)


def _ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator.where(denominator.gt(0), np.nan) / denominator.where(denominator.gt(0))


def _safe_ratio(numerator: object, denominator: object) -> float:
    num = _as_float(numerator)
    den = _as_float(denominator)
    if num is None or den is None:
        return float("nan")
    return float(num / den) if den > 0 else np.nan


def _as_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(numeric):
        return None
    return numeric


def _json_float(value: object) -> float | None:
    numeric = _as_float(value)
    return None if numeric is None else float(numeric)


def _format_pct(value: object) -> str:
    numeric = _as_float(value)
    if numeric is None:
        return "n/a"
    return f"{numeric:.2%}"


def _format_pp(value: object) -> str:
    numeric = _as_float(value)
    if numeric is None:
        return "n/a"
    return f"{numeric * 100:.1f} pp"


def _format_float(value: object) -> str:
    numeric = _as_float(value)
    if numeric is None:
        return "n/a"
    return f"{numeric:.4g}"


def _format_int(value: object) -> str:
    numeric = _as_float(value)
    if numeric is None:
        return "0"
    return str(int(numeric))
