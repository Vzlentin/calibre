"""Implementations behind the CLI subcommands (run, validate, health, ...)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import UUID

import pandas as pd

from calibre.cli.config import (
    BackendConfig,
    load_config,
    load_config_from_mapping,
)
from calibre.conformal.cumulative_risk import CumulativeRiskRuntime
from calibre.core.forecast_frame import DS, UNIQUE_ID, Y
from calibre.core.io import is_local_fs, open_fs
from calibre.core.metrics import set_order_cost
from calibre.evaluation.m5_coverage import (
    CoverageThresholds,
    M5CoverageArtifacts,
    score_resolved_ledger,
)
from calibre.execution.backend import (
    BackendEngine,
    BackendResult,
    ConformalOptions,
    LedgerOutputOptions,
    ReconciliationOptions,
)
from calibre.execution.dataset import DatasetBundle
from calibre.execution.dataset_registry import resolve_dataset_adapter
from calibre.execution.hierarchy_preparation import prepare_run
from calibre.execution.validation import validate_dataset_bundle
from calibre.ordering import OrderPolicy, build_order_policy
from calibre.storage.state import ConformalStateStore
from calibre.tuning import (
    CumulativePinball,
    GlobalTuningTask,
    StudyConfig,
    TuningCandidate,
    optimize_global_task,
    suggest_from_spec,
)

if TYPE_CHECKING:
    import optuna

    from calibre.conformal.runtime import ConformalRuntime

logger = logging.getLogger(__name__)


def _fs_result_uri(fs, path: str) -> str:
    if is_local_fs(fs):
        return path
    return str(fs.unstrip_protocol(path))


_HEALTH_CONFIG: dict[str, Any] = {
    "config_schema": "1.0",
    "dataset": {"adapter": "vn2", "path": "benchmarks/vn2/fixture", "period": 0},
    "tasks": [
        {
            "model": "SeasonalNaive",
            "horizon": 2,
            "config": {"backend": "statsforecast", "season_length": 2},
        }
    ],
    "origins": {"start": "2024-01-29", "end": "2024-01-29", "freq": "W-MON"},
    "output": {"ledger_path": "results/vn2/smoke-ledger.parquet", "streaming": False},
    "execution": {"backend": "local", "seed": 42},
}


def _load_dataset(config: BackendConfig):
    adapter = resolve_dataset_adapter(config.dataset.adapter)
    kwargs = dict(config.dataset.options)
    if config.dataset.period is not None:
        kwargs["period"] = config.dataset.period
    bundle = adapter.load(config.dataset.path, **kwargs)
    validate_dataset_bundle(bundle)
    return bundle


def _enforce_unique_id_limit(bundle: DatasetBundle, max_unique_ids: int | None) -> None:
    if max_unique_ids is None:
        return
    if max_unique_ids < 1:
        raise ValueError("max_unique_ids must be at least 1")
    unique_ids = int(bundle.history[UNIQUE_ID].astype(str).nunique())
    if unique_ids > max_unique_ids:
        raise ValueError(
            f"dataset contains {unique_ids} unique_id values; maximum allowed is {max_unique_ids}"
        )


def _build_order_config(config: BackendConfig) -> OrderPolicy | None:
    if config.ordering is None:
        return None
    return build_order_policy(config.ordering.model_dump())


def _metric_currency(config: BackendConfig) -> str:
    currency = config.dataset.options.get("currency")
    return str(currency) if currency is not None else "EUR"


def _record_order_cost_metric(frame: pd.DataFrame, *, dataset: str, currency: str) -> None:
    if frame.empty:
        return
    if "total_cost" in frame.columns:
        total_cost = float(frame["total_cost"].sum())
    else:
        cost_columns = [
            column
            for column in frame.columns
            if column.endswith("_cost") and pd.api.types.is_numeric_dtype(frame[column])
        ]
        if not cost_columns:
            return
        total_cost = float(frame[cost_columns].sum(numeric_only=True).sum())
    set_order_cost(currency, dataset, total_cost)


def run(
    config_path: str | Path,
    *,
    metrics_port: int | None = None,
    tune: bool = False,
) -> BackendResult | dict[str, Any]:
    """Load the config at ``config_path`` and execute a backtest or HPO search.

    With ``tune=False`` (the default) this runs a backtest; the ``hpo`` block, if
    present, stays inert. With ``tune=True`` it runs the config's ``hpo`` search
    and returns the discovered best model config instead of a backtest result.
    """
    if metrics_port is not None:
        from calibre.core.metrics import serve

        serve(metrics_port)
    config = load_config(config_path)
    if tune:
        return run_tune(config)
    return run_config(config)


def run_config(
    config: BackendConfig,
    *,
    run_id: UUID | None = None,
    conformal_state_store: ConformalStateStore | None = None,
    initial_ledger: pd.DataFrame | None = None,
    max_unique_ids: int | None = None,
) -> BackendResult:
    """Execute a backtest from an already-loaded :class:`BackendConfig`."""
    bundle = _load_dataset(config)
    _enforce_unique_id_limit(bundle, max_unique_ids)
    preparation = prepare_run(config, bundle)
    streaming_output = config.output.ledger_path if config.output.streaming else None
    streaming_order_output = config.output.order_ledger_path if config.output.streaming else None

    # order_conformal claims the single ConformalOptions runtime slot; conformal
    # uses the config slot. ConformalOptions forbids both, and the CLI rejects
    # configuring both — so at most one is non-None here.
    order_runtime: ConformalRuntime | None = (
        CumulativeRiskRuntime(preparation.order_conformal_config)
        if preparation.order_conformal_config is not None
        else None
    )

    engine = BackendEngine(
        execution=config.execution.to_execution_options(freq=config.origins.freq),
        output=LedgerOutputOptions(
            forecast_path=streaming_output,
            order_path=streaming_order_output,
            streaming=config.output.streaming,
        ),
        conformal=ConformalOptions(
            runtime=order_runtime,
            config=preparation.conformal_config if order_runtime is None else None,
            run_id=run_id,
            state_store=conformal_state_store,
            initial_ledger=initial_ledger,
        ),
        reconciliation=ReconciliationOptions(
            reconciler=preparation.reconciler,
            hierarchy_index=preparation.hierarchy_index,
        ),
        order=_build_order_config(config),
    )
    try:
        result = engine.execute(preparation.tasks, preparation.actuals, preparation.origins)
    finally:
        engine.close()

    if not config.output.streaming and config.output.ledger_path is not None:
        result.ledger.to_parquet(config.output.ledger_path)
    if (
        not config.output.streaming
        and result.order_ledger is not None
        and config.output.order_ledger_path is not None
    ):
        result.order_ledger.to_parquet(config.output.order_ledger_path)
    if result.order_ledger is not None:
        _record_order_cost_metric(
            result.order_ledger.to_df(),
            dataset=config.dataset.adapter,
            currency=_metric_currency(config),
        )

    if config.output.streaming:
        logger.info("run complete", extra={"streaming": True})
    else:
        ledger_rows = len(result.ledger.to_df())
        logger.info("run complete", extra={"rows": ledger_rows})
    if config.output.ledger_path is not None:
        logger.info("ledger written", extra={"ledger_path": config.output.ledger_path})
    return result


def _derive_cost_fractile(config: BackendConfig, bundle: DatasetBundle) -> float:
    """Resolve the newsvendor cost fractile (objective ``tau``) for a tune run.

    Sourced from ``hpo.cost_fractile`` when set, else the dataset cost struct's
    ``critical_ratio = Cu / (Cu + Co)`` — never from ``order_conformal.coverage``
    (the orthogonal decision level) nor any ``search_space`` dimension. A global
    study optimises one fractile, so a heterogeneous per-uid cost panel is
    rejected rather than silently reduced to one struct's ratio.
    """
    assert config.hpo is not None
    if config.hpo.cost_fractile is not None:
        return float(config.hpo.cost_fractile)
    if isinstance(bundle.costs, dict):
        raise ValueError(
            "--tune needs a single cost struct to derive the objective fractile, but the "
            "dataset carries a per-uid cost panel; set hpo.cost_fractile to tune a global study"
        )
    tau = float(bundle.costs.critical_ratio)
    if not 0.0 < tau < 1.0:
        raise ValueError(
            f"--tune derived a degenerate objective fractile tau={tau} from the dataset cost "
            "struct: Cu/(Cu+Co) must lie in the open interval (0, 1) (a zero underage or "
            "overage cost collapses the pinball objective); set hpo.cost_fractile instead"
        )
    return tau


@dataclass(frozen=True, slots=True)
class _CliCandidateSpace:
    """Picklable dataset-general candidate factory from a declarative search spec.

    Each trial samples every ``search_space`` key via
    :func:`calibre.tuning.suggest_from_spec`; ``quantile_alpha`` (required) flows
    into ``model_config["quantiles"]`` so the optimizer's returned config carries
    it, with the remaining sampled keys passed through as model-config overrides.
    Defined at module scope (not a closure) so Ray Tune can pickle the searcher.
    """

    search_space: dict[str, dict[str, Any]]

    def __call__(self, trial: optuna.Trial) -> TuningCandidate:
        params = {
            name: suggest_from_spec(trial, name, spec) for name, spec in self.search_space.items()
        }
        quantile_alpha = float(params.pop("quantile_alpha"))
        return TuningCandidate(
            model_config={**params, "quantiles": [quantile_alpha]},
            ordering_config={"quantile": quantile_alpha},
        )


def run_tune(config: BackendConfig) -> dict[str, Any]:
    """Run the config's ``hpo`` search and return the discovered best model config.

    Builds the same :class:`GlobalTuningTask` the benchmark search builds — a
    :class:`CumulativePinball` objective at the cost fractile derived from the
    dataset cost struct, sampled over the config's ``hpo.search_space`` — and
    drives it through :func:`optimize_global_task`. The discovered per-horizon
    quantile lands in the returned config's ``quantiles``.
    """
    if config.hpo is None:
        raise ValueError("--tune requires an hpo config block")
    if len(config.tasks) != 1:
        raise ValueError(
            f"--tune supports a single-task config, got {len(config.tasks)} tasks: the global "
            "study fits one panel, so a multi-task config must be split and tuned per task"
        )
    if "quantile_alpha" not in config.hpo.search_space:
        raise ValueError(
            "hpo.search_space must include a 'quantile_alpha' dimension: it is the "
            "base per-horizon quantile the cost-fractile objective shapes"
        )

    bundle = _load_dataset(config)
    tau = _derive_cost_fractile(config, bundle)

    # The global study fits one panel: bundle.history is the model history, its
    # (uid, ds, y) projection is the realised-demand frame the objective scores,
    # and origins are the same backtest origins a normal run walks.
    base_model_config = {**config.tasks[0].resolved_model_config(), "scope": "global"}
    actuals = bundle.history[[UNIQUE_ID, DS, Y]].copy()
    origins = config.origins.to_list()

    task = GlobalTuningTask(
        history=bundle.history,
        horizon=config.tasks[0].horizon,
        base_model_config=base_model_config,
        search_space=_CliCandidateSpace(config.hpo.search_space),
        actuals=actuals,
        origins=origins,
        objective=CumulativePinball(quantile=0.5, tau=tau),
        study_config=StudyConfig(
            n_trials=config.hpo.budget,
            freq=config.origins.freq,
            seed=config.hpo.seed,
            asha_grace_period=config.hpo.asha_grace_period,
        ),
    )
    best_config = optimize_global_task(task)
    logger.info(
        "tune complete",
        extra={"quantile_alpha": float(best_config["quantiles"][0]), "trials": config.hpo.budget},
    )
    return best_config


def validate(config_path: str | Path) -> BackendConfig:
    """Load and validate a config file, logging a summary."""
    config = load_config(config_path)
    logger.info(
        "config valid",
        extra={"config_schema": config.config_schema, "tasks": len(config.tasks)},
    )
    return config


def score_m5_coverage(
    ledger_path: str | Path,
    *,
    coverage: float = 0.9,
    output_dir: str | Path | None = None,
    thresholds: CoverageThresholds | None = None,
) -> M5CoverageArtifacts:
    """Score M5 interval coverage from a resolved ledger and write artifacts."""
    artifacts = score_resolved_ledger(
        ledger_path,
        coverage=coverage,
        output_dir=output_dir,
        thresholds=thresholds,
    )
    logger.info(
        "m5 coverage artifacts written",
        extra={
            "coverage_by_node_path": str(artifacts.coverage_by_node_path),
            "report_path": str(artifacts.report_path),
            "summary_path": str(artifacts.summary_path),
            "acceptance_status": artifacts.acceptance_status,
        },
    )
    return artifacts


def health() -> dict[str, Any]:
    """Run a fixture backtest end-to-end and return a health payload."""
    import importlib.metadata

    try:
        version = importlib.metadata.version("calibre")
    except importlib.metadata.PackageNotFoundError:
        version = "0.1.0"
    config = load_config_from_mapping(_HEALTH_CONFIG)
    bundle = _load_dataset(config)
    payload = {
        "status": "ok",
        "version": version,
        "config_schema": config.config_schema,
        "fixture_adapter": config.dataset.adapter,
        "fixture_rows": len(bundle.history),
        "fixture_series": int(bundle.history[UNIQUE_ID].astype(str).nunique()),
    }
    return payload


def run_sweep(configs_dir: str | Path) -> list[BackendResult]:
    """Run every YAML config under ``configs_dir`` and return their results."""
    fs, root = open_fs(configs_dir)
    if not fs.exists(root):
        raise FileNotFoundError(f"Config directory not found: {configs_dir}")
    normalized = root.rstrip("/\\")
    configs = sorted(
        {
            *fs.glob(f"{normalized}/*.yaml"),
            *fs.glob(f"{normalized}/*.yml"),
        }
    )
    if not configs:
        raise ValueError(f"No YAML configs found under {configs_dir}")
    return [run_config(load_config(_fs_result_uri(fs, path))) for path in configs]
