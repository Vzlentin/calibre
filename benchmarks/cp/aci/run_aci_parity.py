"""Parity check: Calibre ACI vs the reference conformal-time-series repo."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

if not hasattr(np, "infty"):
    np.infty = np.inf

REPO_ROOT = Path(__file__).resolve().parents[3]
BENCHMARK_DIR = REPO_ROOT / "benchmarks" / "cp" / "aci"
REFERENCE_REPO = BENCHMARK_DIR / "conformal-time-series"
DEFAULT_OUTPUT_DIR = BENCHMARK_DIR / "artifacts" / "amzn_ar"
CONFIG_PATH = REFERENCE_REPO / "tests" / "configs" / "AMZN.yaml"
FORECAST_PATH = REFERENCE_REPO / "tests" / "datasets" / "proc" / "AMZN" / "ar.npz"
TAIL_ALPHA = 0.05
CONFIG = {
    "dataset": "AMZN",
    "model": "ar",
    "ahead": 1,
    "alpha": 0.1,
    "tail_alpha": TAIL_ALPHA,
    "T_burnin": 100,
    "score_function_name": "signed-residual",
    "window_length": 10_000_000,
    "lrs": [0.1, 0.05, 0.01, 0.005, 0.0001],
    "local_quantile_rule": "higher",
    "local_alpha_bounds": None,
}
FLOAT_ATOL = 1e-12

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from calibre.conformal.adaptive import AdaptiveConformalInference  # noqa: E402


@dataclass(slots=True)
class ThresholdPrediction:
    """A one-sided threshold prediction with its alpha and center."""

    threshold: float
    alpha: float
    center: float = 0.0

    def contains(self, value: float) -> bool:
        return bool(float(value) <= float(self.threshold))


def raw_scalar_score(y_true: float, _center: float) -> float:
    """Return the raw scalar score (the value itself)."""
    return float(y_true)


def load_module(module_name: str, path: Path):
    """Import a module by name from an explicit file ``path``."""
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def slugify_lr(lr: float) -> str:
    """Return a filesystem-safe slug for a learning-rate value."""
    return str(lr).replace("-", "neg_").replace(".", "p")


def to_python(value: Any) -> Any:
    """Recursively convert numpy scalars/containers to JSON-safe Python values."""
    if isinstance(value, dict):
        return {key: to_python(val) for key, val in value.items()}
    if isinstance(value, list | tuple):
        return [to_python(val) for val in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating | float):
        number = float(value)
        if math.isnan(number):
            return "nan"
        if math.isinf(number):
            return "inf" if number > 0 else "-inf"
        return number
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def get_reference_commit(reference_repo: Path) -> str:
    """Return the HEAD commit hash of the reference repo."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=reference_repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def warmup_quantile(values: np.ndarray, alpha: float, t_pred: int) -> float:
    """Return the warmup quantile, or +inf before enough samples accrue."""
    if t_pred > math.ceil(1.0 / alpha):
        return float(np.quantile(values[:t_pred], 1.0 - alpha))
    return math.inf


def compare_numeric_arrays(
    reference: np.ndarray, local: np.ndarray, atol: float = FLOAT_ATOL
) -> dict[str, Any]:
    """Compare two numeric arrays and summarize match/diff statistics."""
    reference = np.asarray(reference)
    local = np.asarray(local)
    matches = np.isclose(reference, local, atol=atol, rtol=0.0, equal_nan=True)
    exact = bool(np.all(matches))
    differing = np.flatnonzero(~matches)
    finite_mask = np.isfinite(reference) & np.isfinite(local)
    finite_diff = np.abs(reference[finite_mask] - local[finite_mask])
    summary = {
        "exact_match": exact,
        "first_difference": None,
        "max_abs_diff": float(finite_diff.max()) if finite_diff.size else 0.0,
    }
    if differing.size:
        idx = int(differing[0])
        summary["first_difference"] = {
            "step": idx,
            "reference": to_python(reference[idx]),
            "local": to_python(local[idx]),
        }
    return summary


def compare_boolean_arrays(reference: np.ndarray, local: np.ndarray) -> dict[str, Any]:
    """Compare two boolean arrays and summarize agreement."""
    reference = np.asarray(reference, dtype=bool)
    local = np.asarray(local, dtype=bool)
    matches = reference == local
    exact = bool(np.all(matches))
    differing = np.flatnonzero(~matches)
    summary = {
        "exact_match": exact,
        "first_difference": None,
    }
    if differing.size:
        idx = int(differing[0])
        summary["first_difference"] = {
            "step": idx,
            "reference": bool(reference[idx]),
            "local": bool(local[idx]),
        }
    return summary


def summarize_metrics(interval_frame: pd.DataFrame, burnin: int) -> dict[str, Any]:
    """Summarize coverage and width metrics after a burn-in period."""
    eval_mask = interval_frame["step"] > burnin
    finite_width_mask = eval_mask & np.isfinite(interval_frame["width"])
    coverage_eval = interval_frame.loc[eval_mask, "covered"].mean()
    width_eval = interval_frame.loc[eval_mask, "width"].mean()
    width_eval_finite = (
        interval_frame.loc[finite_width_mask, "width"].mean() if finite_width_mask.any() else np.nan
    )
    return {
        "n_total": int(len(interval_frame)),
        "n_eval": int(eval_mask.sum()),
        "n_eval_finite_width": int(finite_width_mask.sum()),
        "empirical_coverage_eval": float(coverage_eval),
        "average_width_eval": to_python(width_eval),
        "average_width_eval_finite_only": to_python(width_eval_finite),
    }


def choose_diagnosis(
    summary: dict[str, Any], reference_runs: dict[str, dict[str, np.ndarray]]
) -> str:
    """Diagnose where reference and local runs first diverge."""
    q_firsts = []
    alpha_firsts = []
    for tail in ("lower", "upper"):
        q_first = summary[f"q_{tail}"]["first_difference"]
        alpha_first = summary[f"alpha_{tail}"]["first_difference"]
        if q_first is not None:
            q_firsts.append(int(q_first["step"]))
        if alpha_first is not None:
            alpha_firsts.append(int(alpha_first["step"]))
    first_q = min(q_firsts) if q_firsts else None
    first_alpha = min(alpha_firsts) if alpha_firsts else None
    if first_q is not None and (first_alpha is None or first_q < first_alpha):
        adaptive_start = CONFIG["T_burnin"] + 1
        if first_q == adaptive_start:
            return (
                "The first divergence appears in the quantile trajectory at the first adaptive step, "
                "before alpha diverges. This is consistent with the reference repo's `np.quantile(..., "
                "method='higher')` rule differing from the local controller's `finite_sample_radius` rank rule."
            )
        return "The first divergence appears in the quantile trajectory before alpha diverges, which points to a quantile-selection mismatch."
    if first_alpha is not None:
        for tail in ("lower", "upper"):
            ref_alpha = reference_runs[tail]["alpha"]
            if np.any((ref_alpha < 0.0) | (ref_alpha > 1.0)):
                return (
                    "The alpha trajectories diverge after the reference controller leaves the [0, 1] range. "
                    "The local controller clips alpha to its configured bounds, so the two controllers no longer evolve identically."
                )
        return "The alpha trajectory diverges first; inspect the saved alpha traces for the earliest mismatch."
    return "No divergence detected."


def verdict_from_summary(summary: dict[str, Any], metrics_match: bool) -> str:
    """Derive a PASS/FAIL verdict from the comparison summary."""
    exact_fields = (
        summary["q_lower"]["exact_match"],
        summary["q_upper"]["exact_match"],
        summary["alpha_lower"]["exact_match"],
        summary["alpha_upper"]["exact_match"],
        summary["miss"]["exact_match"],
        summary["width"]["exact_match"],
        metrics_match,
    )
    return "exact" if all(exact_fields) else "diverged"


def build_signed_residual_scores(
    y: np.ndarray, forecast: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Build the signed-residual lower/upper score series."""
    lower = forecast - y
    upper = y - forecast
    return lower.astype(float), upper.astype(float)


def run_reference_tail_aci(
    reference_aci,
    scores: np.ndarray,
    alpha: float,
    lr: float,
    window_length: int,
    burnin: int,
) -> dict[str, np.ndarray]:
    """Run the reference tail-ACI implementation and return its outputs."""
    result = reference_aci(
        scores=np.asarray(scores, dtype=float),
        alpha=alpha,
        lr=lr,
        window_length=window_length,
        T_burnin=burnin,
        ahead=1,
    )
    q = np.asarray(result["q"], dtype=float)
    alpha_trace = np.asarray(result["alpha"], dtype=float)
    covered = q >= np.asarray(scores, dtype=float)
    return {
        "q": q,
        "alpha": alpha_trace,
        "covered": covered.astype(bool),
        "miss": (~covered).astype(int),
    }


def run_local_tail_aci(
    scores: np.ndarray,
    alpha: float,
    gamma: float,
    burnin: int,
    quantile_rule: str,
    alpha_bounds: tuple[float, float] | None,
) -> dict[str, np.ndarray]:
    """Run the local Calibre tail-ACI implementation and return its outputs."""
    scores = np.asarray(scores, dtype=float)
    q = np.empty(scores.shape[0], dtype=float)
    alpha_trace = np.full(scores.shape[0], float(alpha), dtype=float)
    covered = np.empty(scores.shape[0], dtype=bool)
    miss = np.empty(scores.shape[0], dtype=int)
    controller: AdaptiveConformalInference | None = None

    for t in range(scores.shape[0]):
        t_pred = t
        if t_pred <= burnin:
            q[t] = warmup_quantile(scores, alpha, t_pred)
            covered[t] = bool(scores[t] <= q[t])
            miss[t] = int(not covered[t])
            continue

        if controller is None:
            controller = AdaptiveConformalInference(
                alpha=alpha,
                gamma=gamma,
                initial_alpha=alpha,
                score=raw_scalar_score,
                initial_scores=scores[:t_pred],
                alpha_bounds=alpha_bounds,
                quantile_rule=quantile_rule,
            )

        alpha_trace[t] = controller.current_alpha
        threshold = controller.get_radius()
        q[t] = threshold
        outcome = controller.observe(
            y_true=float(scores[t]),
            prediction=ThresholdPrediction(threshold=threshold, alpha=controller.current_alpha),
        )
        covered[t] = not bool(outcome["error"])
        miss[t] = int(outcome["error"])

    return {
        "q": q,
        "alpha": alpha_trace,
        "covered": covered,
        "miss": miss,
    }


def build_interval_frame(
    base_frame: pd.DataFrame,
    lower_run: dict[str, np.ndarray],
    upper_run: dict[str, np.ndarray],
) -> pd.DataFrame:
    """Assemble a per-step interval frame from the lower/upper runs."""
    frame = base_frame.copy()
    frame["q_lower"] = lower_run["q"]
    frame["q_upper"] = upper_run["q"]
    frame["alpha_lower"] = lower_run["alpha"]
    frame["alpha_upper"] = upper_run["alpha"]
    frame["lower_tail_covered"] = lower_run["covered"]
    frame["upper_tail_covered"] = upper_run["covered"]
    frame["lower_bound"] = frame["forecast"] - frame["q_lower"]
    frame["upper_bound"] = frame["forecast"] + frame["q_upper"]
    frame["width"] = frame["upper_bound"] - frame["lower_bound"]
    frame["covered"] = (frame["lower_bound"] <= frame["y"]) & (frame["y"] <= frame["upper_bound"])
    frame["miss"] = (~frame["covered"]).astype(int)
    return frame


def metrics_delta(
    reference_metrics: dict[str, Any], local_metrics: dict[str, Any]
) -> dict[str, Any]:
    """Return per-metric reference-minus-local deltas."""

    def delta(lhs: Any, rhs: Any) -> Any:
        if isinstance(lhs, str) or isinstance(rhs, str):
            return "n/a"
        return to_python(float(rhs) - float(lhs))

    return {
        "empirical_coverage_eval_delta": delta(
            reference_metrics["empirical_coverage_eval"],
            local_metrics["empirical_coverage_eval"],
        ),
        "average_width_eval_delta": delta(
            reference_metrics["average_width_eval"],
            local_metrics["average_width_eval"],
        ),
        "average_width_eval_finite_only_delta": delta(
            reference_metrics["average_width_eval_finite_only"],
            local_metrics["average_width_eval_finite_only"],
        ),
    }


def write_report(
    output_dir: Path,
    params: dict[str, Any],
    summaries: list[dict[str, Any]],
) -> None:
    """Write the markdown parity report to ``output_dir``."""
    lines = [
        "# One-Step ACI Parity Report",
        "",
        "## Run Metadata",
        "",
        f"- Reference repo commit: `{params['reference_repo_commit']}`",
        f"- Dataset: `{params['dataset']}`",
        f"- Forecast model artifact: `{params['model']}`",
        f"- Score contract: `{params['score_function_name']}`",
        f"- Burn-in: `{params['T_burnin']}`",
        f"- Nominal alpha: `{params['alpha']}`",
        f"- Tail alpha used for each signed-residual side: `{params['tail_alpha']}`",
        f"- Window length: `{params['window_length']}`",
        "",
        "## Contract Translation",
        "",
        "- Warm start: match the reference warm-up exactly until `t > T_burnin`, then start the local controller with the full pre-adaptive score history.",
        "- Conformity scores: split the reference `signed-residual` score into lower and upper scalar tails and run one local controller per tail.",
        f"- Finite-sample quantile rule: run the local controller with `quantile_rule={params['local_quantile_rule']!r}` to mirror the reference repo's `method='higher'` selection.",
        f"- Alpha bounds: run the local controller with `alpha_bounds={params['local_alpha_bounds']}` so alpha can evolve without clipping, matching the reference update path.",
        "- Quantile representation: combine the two local tail thresholds into the same asymmetric interval form as the reference repo.",
        "- Alpha update timing: both controllers update after observing the current score.",
        "",
        "## Verdicts",
        "",
    ]

    for summary in summaries:
        lines.extend(
            [
                f"### lr = {summary['lr']}",
                "",
                f"- Verdict: `{summary['verdict']}`",
                f"- Diagnosis: {summary['diagnosis']}",
                f"- First lower-tail `q` difference: `{summary['q_lower']['first_difference']}`",
                f"- First upper-tail `q` difference: `{summary['q_upper']['first_difference']}`",
                f"- First miss-sequence difference: `{summary['miss']['first_difference']}`",
                f"- Coverage delta on evaluation region: `{summary['metrics_delta']['empirical_coverage_eval_delta']}`",
                f"- Average width delta on evaluation region: `{summary['metrics_delta']['average_width_eval_delta']}`",
                "",
            ]
        )

    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_reference_dataset(load_dataset, dataset_name: str, reference_repo: Path) -> pd.DataFrame:
    """Load a named dataset from the reference repo into a frame."""
    previous_cwd = Path.cwd()
    try:
        os.chdir(reference_repo / "tests")
        return load_dataset(dataset_name)
    finally:
        os.chdir(previous_cwd)


def main() -> None:
    """Run the ACI parity check end-to-end and write the report."""
    parser = argparse.ArgumentParser(
        description="Run one-step ACI parity against the conformal-time-series reference repo."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where raw outputs and comparison reports will be written.",
    )
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not REFERENCE_REPO.exists():
        raise FileNotFoundError(f"Reference repo not found at {REFERENCE_REPO}")
    if CONFIG["window_length"] < 1_000_000:
        raise ValueError("This harness assumes the AMZN config's effectively unbounded ACI window.")

    methods_module = load_module("cts_methods", REFERENCE_REPO / "core" / "methods.py")
    datasets_module = load_module("cts_datasets", REFERENCE_REPO / "tests" / "datasets.py")
    methods_module.tqdm = lambda iterable: iterable
    reference_aci = methods_module.aci
    load_dataset = datasets_module.load_dataset

    data = load_reference_dataset(load_dataset, CONFIG["dataset"], REFERENCE_REPO)
    forecast_artifact = np.load(FORECAST_PATH)
    forecast = np.asarray(forecast_artifact["forecasts"], dtype=float)
    y = np.asarray(data["y"], dtype=float)
    timestamp = pd.Index(data.index, name="timestamp")
    lower_scores, upper_scores = build_signed_residual_scores(y=y, forecast=forecast)

    base_frame = pd.DataFrame(
        {
            "step": np.arange(y.shape[0], dtype=int),
            "timestamp": timestamp,
            "y": y,
            "forecast": forecast,
            "score_lower": lower_scores,
            "score_upper": upper_scores,
        }
    )
    base_frame.to_csv(output_dir / "base_series.csv", index=False)

    params = {
        **CONFIG,
        "config_path": str(CONFIG_PATH.relative_to(REPO_ROOT)),
        "config_text": CONFIG_PATH.read_text(encoding="utf-8"),
        "forecast_artifact": str(FORECAST_PATH.relative_to(REPO_ROOT)),
        "reference_repo_commit": get_reference_commit(REFERENCE_REPO),
        "output_dir": str(output_dir.relative_to(REPO_ROOT)),
    }
    (output_dir / "params.json").write_text(
        json.dumps(to_python(params), indent=2), encoding="utf-8"
    )

    summaries: list[dict[str, Any]] = []
    for lr in CONFIG["lrs"]:
        reference_runs = {
            "lower": run_reference_tail_aci(
                reference_aci=reference_aci,
                scores=lower_scores,
                alpha=CONFIG["tail_alpha"],
                lr=lr,
                window_length=CONFIG["window_length"],
                burnin=CONFIG["T_burnin"],
            ),
            "upper": run_reference_tail_aci(
                reference_aci=reference_aci,
                scores=upper_scores,
                alpha=CONFIG["tail_alpha"],
                lr=lr,
                window_length=CONFIG["window_length"],
                burnin=CONFIG["T_burnin"],
            ),
        }
        local_runs = {
            "lower": run_local_tail_aci(
                scores=lower_scores,
                alpha=CONFIG["tail_alpha"],
                gamma=lr,
                burnin=CONFIG["T_burnin"],
                quantile_rule=CONFIG["local_quantile_rule"],
                alpha_bounds=CONFIG["local_alpha_bounds"],
            ),
            "upper": run_local_tail_aci(
                scores=upper_scores,
                alpha=CONFIG["tail_alpha"],
                gamma=lr,
                burnin=CONFIG["T_burnin"],
                quantile_rule=CONFIG["local_quantile_rule"],
                alpha_bounds=CONFIG["local_alpha_bounds"],
            ),
        }

        reference_frame = build_interval_frame(
            base_frame, reference_runs["lower"], reference_runs["upper"]
        )
        local_frame = build_interval_frame(base_frame, local_runs["lower"], local_runs["upper"])

        slug = slugify_lr(lr)
        reference_frame.to_csv(output_dir / f"reference_lr_{slug}.csv", index=False)
        local_frame.to_csv(output_dir / f"local_lr_{slug}.csv", index=False)

        reference_metrics = summarize_metrics(reference_frame, CONFIG["T_burnin"])
        local_metrics = summarize_metrics(local_frame, CONFIG["T_burnin"])
        summary = {
            "lr": lr,
            "q_lower": compare_numeric_arrays(
                reference_runs["lower"]["q"], local_runs["lower"]["q"]
            ),
            "q_upper": compare_numeric_arrays(
                reference_runs["upper"]["q"], local_runs["upper"]["q"]
            ),
            "alpha_lower": compare_numeric_arrays(
                reference_runs["lower"]["alpha"], local_runs["lower"]["alpha"]
            ),
            "alpha_upper": compare_numeric_arrays(
                reference_runs["upper"]["alpha"], local_runs["upper"]["alpha"]
            ),
            "miss": compare_boolean_arrays(
                reference_frame["miss"].to_numpy(), local_frame["miss"].to_numpy()
            ),
            "width": compare_numeric_arrays(
                reference_frame["width"].to_numpy(), local_frame["width"].to_numpy()
            ),
            "reference_metrics": reference_metrics,
            "local_metrics": local_metrics,
            "metrics_delta": metrics_delta(reference_metrics, local_metrics),
        }
        summary["diagnosis"] = choose_diagnosis(summary, reference_runs)
        metric_exact = reference_metrics == local_metrics
        summary["verdict"] = verdict_from_summary(summary, metrics_match=metric_exact)
        summaries.append(summary)

    comparison_summary = {
        "params": params,
        "lrs": summaries,
    }
    (output_dir / "comparison_summary.json").write_text(
        json.dumps(to_python(comparison_summary), indent=2),
        encoding="utf-8",
    )
    write_report(output_dir=output_dir, params=params, summaries=summaries)


if __name__ == "__main__":
    main()
