# Calibre Component Orchestration

## Overview

Calibre is a demand planning library. Users call `run_backtest()` or `run_forecast()` from
`pipeline/runner.py`. Forecasts can optionally be wrapped in conformal prediction intervals and
then fed into an order policy (R,S / R,s,S / Newsvendor) to produce order quantities.

## Component Diagram

```mermaid
flowchart TD
    User(["User"])

    subgraph pipeline ["pipeline/"]
        Runner["runner.py\nrun_backtest() / run_forecast()"]
        Loading["loading.py\nload_period() / melt_wide_sales()"]
        Tasks["tasks.py\nbuild_tasks()"]
    end

    subgraph tasks_pkg ["tasks/"]
        ForecastTask["forecast_task.py\nForecastTask (frozen dataclass)"]
        TuningTask["tuning_task.py\nTuningTask\n(Optuna HPO)"]
    end

    subgraph engine ["engine/"]
        Backend["backend.py\nBackendEngine\n.execute() → BackendResult"]
        FLedger["ledger.py\nForecastLedger\n.append() / .update_resolved()"]
        OLedger["ledger.py\nOrderLedger\n.append()"]
        BaseLedger["ledger.py\n_BaseLedger\n.to_df() / .to_parquet()"]
        Scoring["scoring.py\nresolve_actuals()\ncompute_row_errors()\ncompute_metrics()"]
    end

    subgraph models ["models/"]
        Registry["registry.py\nresolve_adapter()"]
        SF["statsforecast.py\nStatsForecastAdapter"]
        ML["mlforecast.py\nMLForecastAdapter"]
        NF["neuralforecast.py\nNeuralForecastAdapter"]
        Protocol["base.py\nModelAdapter protocol\nfit() / predict()"]
    end

    subgraph conformal ["conformal/"]
        CRuntime["runtime.py\nConformalRuntime\n.apply() / .observe()"]
        CPConfig["runtime.py\nConformalPolicyConfig\n(method, coverage, gamma)"]
        ACI["aci.py\nAdaptiveConformalInference\n(per-horizon online)"]
        MSCP["mscp.py\nMultiStepSplitConformal\n(multi-step batch)"]
        CPolicies["policies.py\nOnlineConformalController"]
        CScores["scores.py\nabsolute_error (score fn)"]
        CIntervals["intervals.py\nsymmetric_intervals()"]
    end

    subgraph order ["order/"]
        OConfig["config.py\nOrderPolicyConfig\napply_order_policy()"]
        RS["rs.py\napply_rs_policy()\n(R,S order-up-to)"]
        RSS["rss.py\napply_rss_policy()\n(R,s,S)"]
        NV["newsvendor.py\napply_newsvendor_policy()\n(critical ratio)"]
    end

    subgraph contracts ["contracts/"]
        Contract["forecast_frame.py\nvalidate_forecast_frame()\nSchema: unique_id, ds, y, y_hat,\nh, forecast_origin, model_name\n+ conformal interval columns"]
    end

    Metrics["metrics.py\nmae, rmse, smape, wape, ... (40+)"]
    Result(["PipelineResult\n(ledger, scores, sales, order_ledger)"])

    %% Top-level flow
    User -->|"model_configs, conformal_config,\norder_config, data_dir, horizon"| Runner
    Runner --> Loading
    Loading -->|"sales DataFrame [unique_id, ds, y]"| Tasks
    Runner --> Tasks
    Tasks -->|"List[ForecastTask]"| ForecastTask
    ForecastTask --> Backend
    Runner -->|"conformal_config\norder_config"| Backend

    %% Tuning path (optional)
    TuningTask -.->|"Optuna HPO wraps\nBackendEngine"| Backend

    %% Engine: fit/predict loop
    Backend -->|"model_config"| Registry
    Registry --> SF & ML & NF
    SF & ML & NF -.->|"implements"| Protocol
    SF & ML & NF -->|"predictions [ds, y_hat, h]"| Backend

    %% Conformal layer
    Backend -->|"point forecasts"| CRuntime
    CPConfig --> CRuntime
    CRuntime --> ACI & MSCP
    ACI & MSCP --> CPolicies
    CPolicies --> CScores & CIntervals
    CRuntime -->|"enriched with\n[lo, hi, alpha, cal_state]"| Backend

    %% Order layer
    Backend -->|"conformal frame"| OConfig
    OConfig --> RS & RSS & NV
    RS & RSS & NV -->|"order_qty rows"| OLedger

    %% Ledger / validation
    Backend -->|"append predictions"| FLedger
    FLedger -->|"validate schema"| Contract
    FLedger & OLedger -.->|"extends"| BaseLedger

    %% Actuals resolution loop
    Backend -->|"resolve / score"| Scoring
    Scoring -->|"fill y, error cols"| FLedger
    CRuntime -->|".observe() nonconformity scores"| FLedger

    %% Metrics
    FLedger -->|"full forecast df"| Scoring
    Scoring -->|"apply"| Metrics
    Metrics -->|"scores df"| Runner

    Runner --> Result
```

## Backtest Walk-Forward Loop

For each origin timestamp `BackendEngine.execute()` does:

```
1. _resolve_ledger()         ← fill actuals & observe nonconformity scores from prior round
2. _run_origin()             ← fit adapters on truncated history, predict h steps
3. ConformalRuntime.apply()  ← attach [lo, hi, alpha, calibration_state] per series/model
4. apply_order_policy()      ← derive order_qty from conformal intervals → OrderLedger
5. ForecastLedger.append()   ← validate & store predictions
6. _resolve_ledger()         ← fill actuals that just became known at this origin
```

## Data Flow

| Stage | Input | Output |
|-------|-------|--------|
| **Load** | `data_dir`, `period` | `sales: DataFrame[unique_id, ds, y]` |
| **Build Tasks** | `sales`, `model_configs`, `horizon` | `List[ForecastTask]` |
| **Fit & Predict** | `ForecastTask` (history truncated to origin) | `DataFrame[ds, y_hat, h]` |
| **Conformal** | point forecasts per series/model | adds `y_lo_{cov}`, `y_hi_{cov}`, `conformal_alpha`, `calibration_state` |
| **Order Policy** | conformal frame + inventory params | `order_qty` per series/origin |
| **Validate** | enriched predictions | validated forecast frame |
| **Resolve Actuals** | ledger + sales | fills `y` where `ds ≤ origin` |
| **Observe Scores** | resolved rows + conformal state | updates `nonconformity_score`, adapts `alpha` |
| **Score Errors** | resolved rows | adds `error`, `abs_error`, `pct_error` |
| **Metrics** | scored ledger grouped by `[unique_id, h]` | `mae`, `rmse`, `smape`, `wape` |

## Key Architectural Patterns

- **Adapter + Registry**: `resolve_adapter(model_config)` dynamically instantiates the right backend. New forecasting libraries only need a new adapter + registry entry.
- **Conformal layer**: `ConformalRuntime` is stateful per `(unique_id, model_name)`. ACI updates its alpha each round (online); MSCP uses a rolling calibration window. Both are transparent to the rest of the pipeline.
- **Order layer**: `apply_order_policy()` dispatches to R,S / R,s,S / Newsvendor. All three consume the conformal upper/lower bounds from the forecast frame — orders are decoupled from the forecast model.
- **Dual ledger**: `ForecastLedger` accumulates validated predictions; `OrderLedger` accumulates order decisions. Both extend `_BaseLedger` for unified `to_df()` / `to_parquet()` export.
- **Deferred actuals + observe loop**: Predictions are stored with `y=NaN`. Each origin, newly-matured actuals are filled in and fed back through `ConformalRuntime.observe()` to update calibration state before the next round's `apply()`.
- **Distributed execution**: Fugue partitions `_run_origin` by `unique_id` for parallel processing.
- **HPO via TuningTask**: `TuningTask` wraps `BackendEngine` in an Optuna study, enabling per-series hyperparameter search without touching the core pipeline.
