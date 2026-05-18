# /goal: Determine and document the perfect execution + tuning stack for Calibre

## Context (read before proposing anything)

Calibre is a demand-planning engine that combines probabilistic forecasting (`statsforecast`, `mlforecast`, `neuralforecast`), conformal prediction intervals, and ordering policies into a single backtestable pipeline. It targets retail clients with many SKUs/series (embarrassingly parallel workloads), is cloud-native (containers, K8s, ECS, Databricks), and values elegance over speculative abstraction.

**Non-negotiable constraints:**
- `BackendEngine.execute(tasks, actuals, origins) -> BackendResult` and `TuningTask` are public surfaces. Their signatures cannot break without a deprecation cycle.
- Model adapters in `calibre/forecasting/{stats,ml,neural}forecast_adapter.py` must remain backend-blind. No Ray imports in adapters.
- VN2 cost gate: `benchmarks/vn2/config/winning.yaml` must reproduce `total_cost = 4992.20` on every milestone.
- The cloud MVP (PRs #26–31, May 17–18) stays: slim/full Docker split, FastAPI surface, SQLAlchemy + Alembic state store, GHCR publishing, ECS Terraform, fsspec IO consolidation, `RunStore`/options dataclasses.
- Image weight budget: slim image ~980 MB today, target ~150 MB. Full image stays wherever it lands after the migration.

**Current pain points in the code:**
- `calibre/execution/backend.py` uses `fugue.api.transform` for per-series fan-out, forcing: parquet round-trips via `ForecastTask.to_uri`/`materialize`, base64-pickled `model_config` columns, dynamic schema string construction (`_collect_quantile_columns`), and a `_run_direct` single-node bypass because Fugue overhead is wasteful at small scale.
- `calibre/tuning/optimizer.py` runs `study.optimize(_objective, n_trials=...)` sequentially. Each trial spins up its own Fugue context. No parallel HPO, no early stopping, no resource sharing.
- The two distributed frameworks (Optuna + Fugue/Dask) cannot nest, so the system serializes at the trial level. VN2 (~600 series × ~50 trials × N origins ≈ 150k fit-predicts) runs sequentially today.

**Existing canonical plan:** Read `~/.claude/plans/i-want-to-switch-polished-ocean.md` (written by Claude Code). It proposes a full rip: replace Fugue/Dask/Spark with Ray Core + Ray Tune (OptunaSearch + ASHA), make `BackendEngine.execute()` a generator yielding `OriginResult` per origin, delete all Fugue accommodation code, and add `tests/integration/test_ray.py` / `test_ray_tune.py`.

**Vault context:** The Obsidian vault at `~/obsidian-vault/vault/Val/Wiki/` contains:
- `Calibre Conformal Mission.md`: Calibre's thesis is making conformal prediction competitive with direct quantile for inventory decisions. VN2 Silver achieved (< EUR 5,000 with capped cumulative CRC). Gold target: < EUR 4,831. Diamond: < EUR 4,677.
- `Autoreason for Calibre.md`: tournament-based recipe refinement over conformal methods, score functions, and stock policies.
- The vault catalogs ~20 conformal methods (ACI, PID, BCI, AcMCP, MSCP, SPCI, HopCPT, EnbPI, CopulaCPTS, etc.) that Calibre implements or may implement.

**Real use cases driving the stack:**
1. **Per-series forecasting**: 100–10,000 SKUs, each fits its own model (SeasonalNaive, local LightGBM). Embarrassingly parallel.
2. **Global forecasting**: One model across all series (global LightGBM quantile regressor). Not fan-out parallel.
3. **Hyperparameter tuning**: Panel-level Optuna sweep over lag sets, quantile alphas, tree hyperparams. Needs trial parallelism + early stopping.
4. **Conformal calibration**: Sequential per-origin `observe → apply → observe`. Mutable state (`_issued_count`). Must stay on the driver.
5. **Inventory simulation**: Walk-forward decision loop with R,S or newsvendor policies. Cost is the only metric that matters.
6. **Cloud deployment**: Stateless containers, object-store URIs, K8s Jobs, ECS Fargate, Databricks notebooks. No persistent local disk.

## Task

Determine and document the perfect execution + tuning stack for Calibre. Do not write implementation code. Produce a written decision document that answers:

### 1. Execution layer: what replaces Fugue?

Evaluate these options against the use cases and constraints:
- **Ray Core** (`@ray.remote` tasks + ObjectRefs)
- **Dask-native** (drop Fugue, use `dask.distributed` directly)
- ** concurrent.futures.ProcessPoolExecutor** (single-node only)
- **Keep Fugue** (status quo)

For the winner, specify:
- How `_run_parallel` (local scope, per-uid fan-out) is implemented
- How `_run_direct` (global scope, single-node) is implemented
- Whether `_run_global_distributed` survives or dies
- How `ForecastTaskRef` / `ForecastTask` hand-off works (URI vs ObjectRef)
- How `ExecutionOptions` and `_resolve_execution_engine` change
- Cluster lifecycle: who calls `ray.init()` / `ray.shutdown()`, and when
- The single-node fast path for `uv run calibre run ...` (< 10 tasks): is it Ray or ProcessPoolExecutor?

### 2. Tuning layer: what replaces sequential Optuna?

Evaluate:
- **Ray Tune + OptunaSearch** (distributed scheduling, Optuna keeps sampling)
- **Optuna + `joblib.Parallel`** (multi-core, no distributed)
- **Optuna + Dask integration** (`optuna.integration.DaskStorage`)
- **Ray Tune native search** (drop Optuna, use BayesOpt / HyperOpt / BOHB)

For the winner, specify:
- How `TuningTask.search_space: Callable[[optuna.Trial], dict]` is preserved (conditional sampling must survive)
- How ASHA early stopping hooks into the per-origin loop
- How `MLflowLoggerCallback` replaces `safe_log_metric` / `optuna_mlflow_callback`
- How `RunStore` (PR #31) integrates with trial persistence
- Resource budget per trial (`cpu`, `gpu`) and how nested parallelism (trials × per-uid tasks) is prevented from over-subscribing

### 3. Ray depth: how deep does Ray go?

Evaluate each layer and give a **do / defer / skip** verdict:

| Layer | Technology | Verdict |
|-------|-----------|---------|
| 0 | Ray Core + Tune (execution + HPO) | ? |
| 1 | Ray Data (replace `pd.read_parquet` in loading) | ? |
| 2 | Ray Actors (`ConformalRuntimeActor`, `ForecastCacheActor`) | ? |
| 3 | Ray Serve (replace FastAPI) | ? |
| 4 | Ray Train (distributed LightGBM/XGBoost fitting) | ? |
| 5 | Ray-native state store (replace SQLAlchemy/Alembic) | ? |
| 6 | Ray Jobs (replace CLI / ECS orchestration) | ? |

Justify each verdict with Calibre-specific reasoning (panel size, latency requirements, operational complexity, pre-1.0 stage).

### 4. Data layer: what stays, what changes?

- `fsspec` for object-store IO: keep or replace?
- `pandas` as the in-memory frame format: keep or move to Arrow / Ray Data?
- Parquet as the serialization format: keep or add IPC / Plasma?
- `ForecastTaskRef` (URI-based materialization): keep, extend, or replace with `ray.put()`?

### 5. Observability: unified or separate?

- Ray Dashboard vs Dask Dashboard vs Spark UI: which survives?
- MLflow for experiment tracking: keep, extend, or replace with Ray's native experiment tracking?
- Prometheus metrics from the FastAPI service: keep or move to Ray's metrics?

### 6. Packaging: image size and extras

- Current `pyproject.toml` has `[dask]`, `[spark]`, `[ml]`, `[neural]`, `[benchmarks]` extras. What should the post-migration extras look like?
- Slim image excludes MLForecast/NeuralForecast; full image includes everything. Does Ray go in both or only full?
- Measured size estimate: dropping Fugue/Java saves ~X MB; adding Ray costs ~Y MB. Net?

### 7. Staged migration sequence

If the decision is to migrate (likely), define phases:
- Phase A: add new backend alongside old (`execution.engine: ray`)
- Phase B: port tuning to new scheduler
- Phase C: observability migration
- Phase D: drop old backend (gated on what?)

Each phase needs:
- Entry criteria
- Exit criteria (tests + cost gate)
- Rollback plan
- Effort estimate in working days

### 8. Risk register

List the top 5 risks with likelihood, impact, and mitigation. Must include:
- Ray cluster setup cost per CLI invocation
- ASHA pruning good trials because origin ordering is non-stationary
- Conditional Optuna search spaces that don't translate to the chosen scheduler
- Plasma store memory pressure on large panels
- Windows dev experience (if applicable)

## Output format

Write the decision document as `docs/stack-decision.md` in the repo. It must be self-contained: a reader with no knowledge of this conversation can understand why every choice was made.

Also append a one-page executive summary to `docs/stack-decision-summary.md` with:
- Before / after stack comparison table
- Effort estimate and timeline
- Go / no-go recommendation

## Rules

- **Do not write implementation code.** No edits to `backend.py`, `optimizer.py`, `pyproject.toml`, or tests. Research and document only.
- **Do not run tests or benchmarks.** Analysis only.
- **Read the existing canonical plan** at `~/.claude/plans/i-want-to-switch-polished-ocean.md` before forming opinions. Cite it where you agree or disagree.
- **Read the vault** at `~/obsidian-vault/vault/Val/Wiki/_wiki/Calibre Conformal Mission.md` and `Autoreason for Calibre.md` for mission context.
- **Read the code** in `calibre/execution/backend.py`, `calibre/tuning/optimizer.py`, `calibre/tuning/task.py`, `calibre/cli/commands.py`, `calibre/core/forecast_task.py`, and `pyproject.toml` to ground decisions in actual lines.
- **Show your reasoning.** Every verdict needs a "because" clause tied to Calibre's use cases.
- **Be direct.** No filler, no hedging, no "Great question!" If something is wrong with the canonical plan, say so plainly.
