# Electricity forecasting with Calibre

[`01_electricity_application_oventi.ipynb`](01_electricity_application_oventi.ipynb)
walks through hourly electricity forecasting and 90% prediction intervals with
Calibre. It compares moving average, MSTL, CatBoost and online gradient descent
with EnbPI residual calibration, ACI and Conformal PID. SARIMA provides a native
Gaussian interval baseline.

## Run

From the repository root, create a dedicated environment and install the local
Calibre package alongside the notebook dependencies:

```sh
uv venv notebooks/.venv --python 3.12
uv pip install --python notebooks/.venv/bin/python -e . -r notebooks/requirements.txt
uv run --no-project notebooks/.venv/bin/python -m ipykernel install \
  --user --name calibre-electricity --display-name 'Python (Calibre electricity)'
uv run --no-project notebooks/.venv/bin/python -m jupyterlab \
  notebooks/01_electricity_application_oventi.ipynb
```

Select **Python (Calibre electricity)** and run all cells. In VS Code, select the
same kernel or `notebooks/.venv/bin/python`. The notebook can run from either the
repository root or `notebooks/`. It uses the current `main` APIs and does not
require another pull request.

The first run downloads the public LSTNet electricity matrix to
`data/electricity_notebook/`. The source is pinned to a commit and checked against
a SHA-256 digest on every read. Data and generated results stay out of Git.

## What it covers

- Data quality checks and descriptive statistics for 321 clients, with displays
  limited to the first five. The matrix has no source timestamps or verifiable
  physical units, so plots use observation indexes.
- Five chronological train/calibration/test folds, one-step forecasts and SARIMA
  order selection on the first training block only.
- Thirteen forecast/interval pairs evaluated on the same targets using RMSE, MAE,
  bias, coverage and width. The final figures show one forecaster at a time.
- CSV tables, PNG figures and a configuration/version manifest in
  `results/electricity_notebook/<UTC>/`.

`electricity_application.py` contains the data loading, fold orchestration and
reporting for this example. Point forecasts use Calibre's `StatsForecastModel`
and `MLForecast` adapters through `rolling_forecasts`. Calibration uses `replay`
and the library's `EnbPI`, `ACI` and `ConformalPID` implementations, including PID's
built-in AR(1) scorecasting. No forecasting or calibration origin loop is replaced.

EnbPI is used only for sequential residual calibration, not the complete
bootstrap/OOB algorithm. The uncertainty estimates for mean fold metrics use a
paired circular block bootstrap and remain exploratory with few dependent folds.
Empty and unbounded prediction sets remain in the evaluation. There is no claim
of conditional or distribution-free coverage under arbitrary dependence.

The notebook uses the Oventi palette. To display the logo, supply the original JPEG
at `notebooks/assets/OVENTI-logo.jpeg`. It is optional and is not recreated.
