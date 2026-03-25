# Calibre Component Orchestration

## Overview

Calibre is a demand planning / forecasting library. Users call `run_backtest()` or `run_forecast()` from `pipeline/runner.py`. Everything else is internal orchestration.

## Component Diagram

```mermaid
flowchart TD
    User(["User"])

    subgraph pipeline ["pipeline/"]
        Runner["runner.py\nrun_backtest() / run_forecast()"]
        Loading["loading.py\nload_period()\nmelt_wide_sales()"]
        Tasks["tasks.py\nbuild_tasks()"]
    end

    subgraph tasks_pkg ["tasks/"]
        ForecastTask["forecast_task.py\nForecastTask\n(frozen dataclass)"]
    end

    subgraph engine ["engine/"]
        Backend["backend.py\nBackendEngine\n.execute()"]
        Ledger["ledger.py\nLedger\n.append() / .to_df()"]
        Scoring["scoring.py\nresolve_actuals()\ncompute_row_errors()\ncompute_metrics()"]
    end

    subgraph models ["models/"]
        Registry["registry.py\nresolve_adapter()"]
        SF["statsforecast.py\nStatsForecastAdapter"]
        ML["mlforecast.py\nMLForecastAdapter"]
        NF["neuralforecast.py\nNeuralForecastAdapter"]
        Protocol["base.py\nModelAdapter protocol\nfit() / predict()"]
    end

    subgraph contracts ["contracts/"]
        Contract["forecast_frame.py\nvalidate_forecast_frame()\nSchema: unique_id, ds, y,\ny_hat, h, forecast_origin,\nmodel_name"]
    end

    Metrics["metrics.py\nmae, rmse, smape,\nwape, ... (40+)"]

    Result(["PipelineResult\n(ledger, scores, sales)"])

    %% Main flow
    User -->|"model_configs, data_dir,\nhorizon, origins"| Runner
    Runner --> Loading
    Runner --> Tasks
    Loading -->|"sales DataFrame\n[unique_id, ds, y]"| Tasks
    Tasks -->|"List[ForecastTask]"| ForecastTask
    ForecastTask -->|"tasks"| Backend
    Runner --> Backend

    %% Engine internals
    Backend -->|"model_config"| Registry
    Registry --> SF & ML & NF
    SF & ML & NF -.->|"implements"| Protocol
    SF & ML & NF -->|"predictions\n[ds, y_hat, h]"| Backend
    Backend -->|"append rows"| Ledger
    Ledger -->|"validate schema"| Contract
    Backend -->|"fill actuals"| Scoring
    Scoring -->|"resolved rows"| Ledger

    %% Metrics
    Ledger -->|"full forecast df"| Scoring
    Scoring -->|"apply"| Metrics
    Metrics -->|"scores df"| Runner

    %% Result
    Runner --> Result
```

## Data Flow

| Stage | Input | Output |
|-------|-------|--------|
| **Load** | `data_dir`, `period` | `sales: DataFrame[unique_id, ds, y]` |
| **Build Tasks** | `sales`, `model_configs`, `horizon` | `List[ForecastTask]` |
| **Fit & Predict** | `ForecastTask` (history truncated to origin) | `predictions: DataFrame[ds, y_hat, h]` |
| **Validate** | predictions + metadata columns | validated forecast frame |
| **Resolve Actuals** | ledger + sales | fills `y` where `ds ≤ origin` |
| **Score** | complete forecast rows | `error`, `abs_error`, `pct_error` |
| **Metrics** | scored ledger grouped by `[unique_id, h]` | `mae`, `rmse`, `smape`, `wape` |

## Key Orchestration Patterns

- **Adapter + Registry**: `resolve_adapter(model_config)` dynamically instantiates the right backend. Adding a new forecasting library requires only a new adapter and registry entry.
- **Backtest loop**: `BackendEngine.execute()` iterates over origins, truncating history at each one to simulate "data available at time T".
- **Contract validation**: Every batch appended to the `Ledger` is validated against the `forecast_frame` schema — errors surface early.
- **Deferred actuals**: Predictions are stored with `y=NaN`; actuals are back-filled once known (origin passes), enabling ongoing evaluation.
- **Distributed execution**: Fugue partitions task execution by `unique_id` for parallel processing.
