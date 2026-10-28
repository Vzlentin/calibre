"""Notebook-only data, orchestration and reporting; all origin loops live in Calibre."""

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from time import perf_counter
from urllib.request import urlopen

import matplotlib.pyplot as plt  # ty: ignore[unresolved-import]
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor  # ty: ignore[unresolved-import]
from sklearn.linear_model import SGDRegressor
from statsforecast.models import ARIMA, MSTL, AutoETS, WindowAverage

from calibre import (
    ACI,
    Absolute,
    Asymmetric,
    BottomUp,
    ConformalPID,
    EnbPI,
    Hierarchy,
    Panel,
    SplitQuantile,
    Step,
    initial_state,
    replay,
    rolling_forecasts,
)
from calibre.backtest.forecasts import Forecasts
from calibre.forecast.models.mlforecast import MLForecast
from calibre.forecast.models.statsforecast import StatsForecastModel
from calibre.metrics import coverage

OVENTI = {
    "light": "#3737AC",
    "medium": "#252673",
    "dark": "#131439",
    "teal": "#009C9E",
    "orange": "#F79021",
}
SOURCE_URL = (
    "https://raw.githubusercontent.com/laiguokun/multivariate-time-series-data/"
    "7f402f185cc2435b5e66aed13a3b560ed142e023/electricity/electricity.txt.gz"
)
SOURCE_SHA256 = "3c4c069588198c1fcc95cace7bb69c99922129edfd673b7286661dad20badefa"
FAMILIES = ("Moving average", "Trend-season MSTL", "SARIMA", "CatBoost", "Online GD")
METHODS = ("EnbPI (residual-only)", "ACI", "Conformal PID")
METRICS = ("rmse", "mae", "bias", "coverage", "width")


@dataclass(frozen=True)
class Config:
    n_split: int = 5
    train_size: int = 720
    calibration_size: int = 168
    test_size: int = 168
    level: float = 0.90
    n_lags: int = 24
    period: int = 24
    moving_window: int = 24
    sarima_d: int = 1
    sarima_method: str = "CSS-ML"
    seasonal_order: tuple[int, int, int] = (1, 0, 0)
    catboost_iterations: int = 200
    catboost_depth: int = 5
    ogd_eta: float = 0.001
    ogd_l2: float = 0.001
    aci_gamma: float = 0.005
    pid_rate: float = 0.05
    pid_integral: float = 0.10
    pid_score_window: int = 64
    seed: int = 42

    def __post_init__(self):
        integers = (
            self.n_split,
            self.train_size,
            self.calibration_size,
            self.test_size,
            self.n_lags,
            self.period,
            self.moving_window,
            self.catboost_iterations,
            self.catboost_depth,
            self.pid_score_window,
        )
        if any(not isinstance(v, int) or isinstance(v, bool) or v < 1 for v in integers):
            raise ValueError("Window lengths, fold counts and model sizes must be positive ints")
        if self.train_size <= max(self.n_lags + 2, 2 * self.period, self.moving_window):
            raise ValueError("Training is too short for the lags/decomposition")
        if not 0 < self.level < 1:
            raise ValueError("level must be in (0, 1)")
        if self.calibration_size < int(np.ceil(self.level / (1 - self.level))) + 1:
            raise ValueError("Calibration is too short for the nominal split-conformal rank")


@dataclass(frozen=True)
class Split:
    fold: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int


def chronological_splits(n, config):
    first = n - config.n_split * config.test_size
    start = first - config.calibration_size - config.train_size
    if start < 0:
        raise ValueError("Not enough observations for the requested rolling windows")
    return [
        Split(
            fold=k,
            train_start=first + k * config.test_size - config.calibration_size - config.train_size,
            train_end=first + k * config.test_size - config.calibration_size,
            test_start=first + k * config.test_size,
            test_end=first + (k + 1) * config.test_size,
        )
        for k in range(config.n_split)
    ]


def load_electricity(directory):
    """Download atomically, verify both cached and new files, retain original row order."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "electricity.txt.gz"
    if not path.exists():
        temporary = path.with_suffix(".gz.part")
        try:
            with urlopen(SOURCE_URL, timeout=120) as response, temporary.open("wb") as output:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
            if sha256(temporary.read_bytes()).hexdigest() != SOURCE_SHA256:
                raise ValueError("Downloaded source checksum mismatch")
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    if sha256(path.read_bytes()).hexdigest() != SOURCE_SHA256:
        raise ValueError(
            f"Invalid cached source: {path}; inspect/remove it before downloading again"
        )
    frame = pd.read_csv(path, header=None, compression="gzip")
    if frame.shape != (26304, 321):
        raise ValueError(f"Unexpected pinned-source shape: {frame.shape}")
    frame.columns = [f"client_{k + 1:04d}" for k in range(frame.shape[1])]
    frame.index.name = "observation"
    return frame


def describe_data(frame):
    """Complete per-client descriptive table, with missingness and invalid values visible."""
    finite = frame.where(np.isfinite(frame))
    description = finite.describe(percentiles=[0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]).T
    description = description.rename(columns={"std": "std_sample"})
    description["std_population"] = finite.std(ddof=0)
    description["variance_sample"] = finite.var(ddof=1)
    description["range"] = finite.max() - finite.min()
    description["iqr"] = finite.quantile(0.75) - finite.quantile(0.25)
    description["skewness"] = finite.skew()
    description["excess_kurtosis"] = finite.kurt()
    description["zeros"] = frame.eq(0).sum()
    description["zero_fraction"] = description["zeros"] / len(frame)
    description["negative"] = frame.lt(0).sum()
    description["missing"] = frame.isna().sum()
    description["infinite"] = np.isinf(frame).sum()
    description["n_unique"] = frame.nunique()
    quality = pd.Series(
        {
            "observations": len(frame),
            "client_columns": frame.shape[1],
            "numeric_columns": len(frame.select_dtypes(include="number").columns),
            "missing_cells": int(frame.isna().sum().sum()),
            "infinite_cells": int(np.isinf(frame).sum().sum()),
            "constant_clients": int((finite.nunique() <= 1).sum()),
            "repeated_rows_not_removed": int(frame.duplicated().sum()),
            "index_unique": frame.index.is_unique,
            "index_increasing": frame.index.is_monotonic_increasing,
            "source_dates_available": False,
        },
        name="Data quality",
    )
    return quality, description


def identify_sarima(training, config, max_p=2, max_q=2):
    """Small fixed AIC grid on the first training block only; never read a test value."""
    rows = []
    for p in range(max_p + 1):
        for q in range(max_q + 1):
            order = (p, config.sarima_d, q)
            fitted = ARIMA(
                order=order,
                seasonal_order=config.seasonal_order,
                season_length=config.period,
                method=config.sarima_method,
            ).fit(np.asarray(training, dtype=float))
            rows.append(
                {
                    "p": p,
                    "d": config.sarima_d,
                    "q": q,
                    "aic": fitted.model_["aic"],
                    "aicc": fitted.model_["aicc"],
                    "bic": fitted.model_["bic"],
                }
            )
    table = pd.DataFrame(rows).sort_values("aic", ignore_index=True)
    if not np.isfinite(table.aic).any():
        raise ValueError("No finite training AIC in the predefined SARIMA grid")
    best = table.loc[table.aic.idxmin()]
    return tuple(int(best[k]) for k in ("p", "d", "q")), table


class VendorWindow:
    """Thin vendor-call proxy: bound past history and retain native ARIMA intervals.

    StatsForecastModel still calls forecast once per series/origin. No extra origin loop,
    point model, fitted state, or conformal implementation is introduced here.
    """

    def __init__(self, model, history, level=None):
        self.model, self.history, self.level = model, history, level
        self.bounds = []

    def forecast(self, y, h):
        kwargs = {} if self.level is None else {"level": [100 * self.level]}
        # Panel stores float32; likelihood optimization needs a float64 working array.
        history = np.asarray(y[-self.history :], dtype=np.float64)
        result = self.model.forecast(y=history, h=h, **kwargs)
        if self.level is not None:
            key = f"{100 * self.level:g}"
            # StatsForecast keys render an integral float level as, e.g., '90.0'.
            lower_key = next(
                k for k in result if k.startswith("lo-") and float(k[3:]) == float(key)
            )
            upper_key = next(
                k for k in result if k.startswith("hi-") and float(k[3:]) == float(key)
            )
            self.bounds.append((np.asarray(result[lower_key]), np.asarray(result[upper_key])))
        return result


def make_forecasts(family, values, config, sarima_order=(1, 1, 1)):
    """Forecast calibration + test once; return original-scale Calibre arrays.

    OGD re-fits a deterministic single-pass SGD on each expanding prefix. With a fixed
    training scaler and constant step, this exactly replays online gradient updates.
    CatBoost trees stay fixed within a fold. Local StatsForecast models re-fit at each
    prediction, as specified by the repository adapter (even with refit_every=None).
    """
    values = np.asarray(values, dtype=float)
    series = np.array(["target"])
    hierarchy = Hierarchy.flat(series)
    center, scale = 0.0, 1.0
    recorder = None
    refit_every = None
    if family == "Moving average":
        vendor = WindowAverage(window_size=config.moving_window)
    elif family == "Trend-season MSTL":
        vendor = MSTL(season_length=config.period, trend_forecaster=AutoETS(model="AAN"))
    elif family == "SARIMA":
        vendor = ARIMA(
            order=sarima_order,
            seasonal_order=config.seasonal_order,
            season_length=config.period,
            method=config.sarima_method,
        )
    elif family == "CatBoost":
        model = MLForecast(
            CatBoostRegressor(
                iterations=config.catboost_iterations,
                depth=config.catboost_depth,
                learning_rate=0.05,
                loss_function="RMSE",
                random_seed=config.seed,
                verbose=False,
                allow_writing_files=False,
                thread_count=1,
            ),
            lags=list(range(1, config.n_lags + 1)),
            lookback=config.n_lags,
            num_threads=1,
        )
    elif family == "Online GD":
        center = float(values[: config.train_size].mean())
        scale = float(values[: config.train_size].std())
        if scale == 0:
            scale = 1.0  # Declared unit scale for a constant initial training history.
        model = MLForecast(
            SGDRegressor(
                loss="squared_error",
                penalty="l2",
                alpha=config.ogd_l2,
                learning_rate="constant",
                eta0=config.ogd_eta,
                max_iter=1,
                tol=None,
                shuffle=False,
                random_state=config.seed,
            ),
            lags=list(range(1, config.n_lags + 1)),
            lookback=config.n_lags,
            num_threads=1,
        )
        refit_every = 1
    else:
        raise ValueError(f"Unknown forecaster: {family}")
    if family in FAMILIES[:3]:
        recorder = VendorWindow(
            vendor, config.train_size, config.level if family == "SARIMA" else None
        )
        model = StatsForecastModel(recorder)
    # These dates satisfy Panel's regular-calendar contract, but are NOT source dates.
    # No date features are used; every public plot/output retains observation indexes.
    periods = pd.date_range("2000-01-01", periods=len(values), freq="h")
    panel = Panel(series, periods, ((values - center) / scale)[None, :], "h")
    origins = np.arange(config.train_size - 1, len(values) - 1)
    forecasts = rolling_forecasts(
        panel,
        hierarchy,
        model,
        BottomUp(hierarchy),
        origins,
        horizon=1,
        refit_every=refit_every,
    )
    forecasts = Forecasts(forecasts.origins, (forecasts.points * scale + center).astype(np.float32))
    actuals = values[None, :].astype(np.float32)
    bounds = None
    if family == "SARIMA":
        assert recorder is not None
        bounds = np.array(recorder.bounds)[:, :, 0]
    return forecasts, actuals, bounds


def run_family(family, y, splits, config, sarima_order=(1, 1, 1)):
    """Outer folds only: use Calibre's two origin loops for forecasts and calibration."""
    frames = []
    for split in splits:
        started = perf_counter()
        values = y[split.train_start : split.test_end]
        forecasts, actuals, native_bounds = make_forecasts(family, values, config, sarima_order)
        start = config.calibration_size
        test = Forecasts(forecasts.origins[start:], forecasts.points[start:])
        expected = np.arange(split.test_start, split.test_end)
        np.testing.assert_array_equal(test.origins + split.train_start + 1, expected)
        pairs = []
        if family == "SARIMA":
            pairs.append(
                (
                    "Gaussian (native)",
                    test.points[:, 0, 0],
                    native_bounds[start:, 0],
                    native_bounds[start:, 1],
                )
            )
        else:
            # These two methods warm up causally through the entire calibration block.
            for method, score, calibrator in (
                (METHODS[0], Asymmetric(), EnbPI(config.level, window=config.calibration_size)),
                (
                    METHODS[1],
                    Absolute(),
                    ACI(
                        SplitQuantile(config.level, window=config.calibration_size),
                        gamma=config.aci_gamma,
                    ),
                ),
            ):
                result = replay(
                    forecasts, actuals, target=Step(), score=score, calibrator=calibrator
                )
                pairs.append(
                    (
                        method,
                        result.point[start:, 0, 0],
                        result.lower[start:, 0, 0],
                        result.upper[start:, 0, 0],
                    )
                )
            pid = ConformalPID(
                config.level,
                learning_rate=config.pid_rate,
                integral_gain=config.pid_integral,
                window=config.calibration_size,
                scorecast_window=config.pid_score_window,
                scorecast=True,
                ahead=1,
            )
            state = initial_state(Step(), pid, n_nodes=1, horizon=1)
            # The final seed target is known AT the first test origin. Nothing later is read.
            known_actuals = actuals[:, : int(test.origins[0]) + 1]
            scores = np.abs(forecasts.residuals(known_actuals)[:start])
            if not np.isfinite(scores).all():
                raise ValueError("PID seed includes a target not known at the first test origin")
            state["calibrator"] = pid.seed(state["calibrator"], scores)
            result = replay(
                test, actuals, target=Step(), score=Absolute(), calibrator=pid, state=state
            )
            pairs.append(
                (METHODS[2], result.point[:, 0, 0], result.lower[:, 0, 0], result.upper[:, 0, 0])
            )
        for method, point, lower, upper in pairs:
            frames.append(
                pd.DataFrame(
                    {
                        "family": family,
                        "method": method,
                        "fold": split.fold,
                        "step": expected,
                        "truth": actuals[0, test.origins + 1],
                        "point": point,
                        "lower": lower,
                        "upper": upper,
                    }
                )
            )
        print(f"{family} | fold {split.fold + 1}/{len(splits)} | {perf_counter() - started:.1f}s")
    result = pd.concat(frames, ignore_index=True)
    if result[["truth", "point", "lower", "upper"]].isna().any().any():
        raise ValueError("Undefined predictions or intervals in an evaluation block")
    return result


def interval_lengths(lower, upper):
    """Lebesgue length: empty sets 0, unbounded sets inf; never discard saturation dates."""
    lower, upper = np.asarray(lower), np.asarray(upper)
    empty = lower > upper
    with np.errstate(invalid="ignore"):
        length = upper - lower
    return np.where(empty, 0.0, length)


def fold_metrics(predictions):
    rows = []
    for (family, method, fold), group in predictions.groupby(
        ["family", "method", "fold"], sort=False
    ):
        error = group.point.to_numpy(dtype=float) - group.truth.to_numpy(dtype=float)
        lengths = interval_lengths(group.lower, group.upper)
        rows.append(
            {
                "family": family,
                "method": method,
                "fold": fold,
                "H": len(group),
                "rmse": np.sqrt(np.mean(error**2)),
                "mae": np.mean(np.abs(error)),
                "bias": np.mean(error),
                "coverage": coverage(
                    group.truth.to_numpy(), group.lower.to_numpy(), group.upper.to_numpy()
                ),
                "width": np.mean(lengths),
                "n_empty": int((group.lower > group.upper).sum()),
                "n_unbounded": int(np.isinf(lengths).sum()),
            }
        )
    return pd.DataFrame(rows)


def summarize(folds, confidence=0.95, n_bootstrap=2000, block_length=2, seed=42):
    """Paired circular block bootstrap of chronological FOLD metrics, not IID target CIs."""
    n = folds.fold.nunique()
    if n < 2 or not 1 <= block_length <= n or n_bootstrap < 1 or not 0 < confidence < 1:
        raise ValueError("Need >=2 folds, 1<=block_length<=n_folds, B>=1, confidence in (0,1)")
    rng = np.random.default_rng(seed)
    starts = rng.integers(n, size=(n_bootstrap, int(np.ceil(n / block_length))))
    indices = ((starts[..., None] + np.arange(block_length)) % n).reshape(n_bootstrap, -1)[:, :n]
    rows = []
    expected = np.sort(folds.fold.unique())
    for (family, method), group in folds.groupby(["family", "method"], sort=False):
        group = group.sort_values("fold")
        np.testing.assert_array_equal(group.fold, expected)
        row = {
            "family": family,
            "method": method,
            "n_folds": n,
            "n_targets": int(group.H.sum()),
            "n_empty": int(group.n_empty.sum()),
            "n_unbounded": int(group.n_unbounded.sum()),
        }
        for metric in METRICS:
            values = group[metric].to_numpy(dtype=float)
            means = values[indices].mean(axis=1)
            row[metric] = values.mean()
            # Order-statistic percentiles preserve inf rather than interpolating inf-inf.
            lo, hi = np.quantile(
                means, [(1 - confidence) / 2, (1 + confidence) / 2], method="inverted_cdf"
            )
            row[f"{metric}_lo"], row[f"{metric}_hi"] = lo, hi
        rows.append(row)
    return pd.DataFrame(rows)


class ExperimentReporter:
    """Oventi notebook figures; intentionally not a Calibre package API."""

    def __init__(self, level=0.9, plot_fold=0, rolling_window=24, output=None):
        self.level, self.plot_fold, self.rolling_window = level, plot_fold, rolling_window
        self.output = None if output is None else Path(output)
        if self.output is not None:
            self.output.mkdir(parents=True, exist_ok=True)

    def save(self, fig, name):
        if self.output is not None:
            fig.savefig(self.output / f"{name}.png", bbox_inches="tight", facecolor="white")

    def plot_pairs(self, predictions):
        selected = predictions.loc[predictions.fold == self.plot_fold]
        if selected.empty:
            raise ValueError("The preselected plot fold is absent")
        for i, ((family, method), group) in enumerate(
            selected.groupby(["family", "method"], sort=False)
        ):
            group = group.sort_values("step")
            fig = plt.figure(figsize=(15, 7.4), layout="constrained")
            grid = fig.add_gridspec(2, 2, height_ratios=[2, 1])
            ax, hist, cov = (
                fig.add_subplot(grid[0, :]),
                fig.add_subplot(grid[1, 0]),
                fig.add_subplot(grid[1, 1]),
            )
            x = group.step.to_numpy()
            ax.plot(x, group.truth, color=OVENTI["dark"], lw=1.2, label="Observed")
            ax.plot(x, group.point, color=OVENTI["teal"], lw=1, ls="--", label="Forecast, h=1")
            ordinary = (
                np.isfinite(group.lower) & np.isfinite(group.upper) & (group.lower <= group.upper)
            )
            ax.fill_between(
                x,
                group.lower,
                group.upper,
                where=ordinary,
                alpha=0.20,
                color=OVENTI["light"],
                label=f"{self.level:.0%} prediction interval",
            )
            lengths = interval_lengths(group.lower, group.upper)
            unbounded, empty = np.isinf(lengths), group.lower.to_numpy() > group.upper.to_numpy()
            if unbounded.any():
                ax.scatter(
                    x[unbounded],
                    group.point.to_numpy()[unbounded],
                    color=OVENTI["orange"],
                    marker="^",
                    label=f"Unbounded set ({unbounded.sum()})",
                    zorder=4,
                )
            if empty.any():
                ax.scatter(
                    x[empty],
                    group.point.to_numpy()[empty],
                    color=OVENTI["orange"],
                    marker="x",
                    label=f"Empty set ({empty.sum()})",
                    zorder=4,
                )
            ax.set(
                title=f"OVENTI | {family} × {method} | fold {self.plot_fold}",
                xlabel="Observation index (hourly cadence; no source dates)",
                ylabel="Published load scale",
            )
            ax.legend(loc="upper left", ncols=2, fontsize=8)
            error = group.point - group.truth
            hist.hist(error, bins=25, color=OVENTI["light"], alpha=0.8)
            hist.axvline(0, color=OVENTI["dark"], ls="--")
            hist.axvline(error.mean(), color=OVENTI["orange"], label=f"Bias = {error.mean():.2f}")
            hist.set(title="Forecast errors", xlabel="Forecast − observed", ylabel="Count")
            hist.legend(fontsize=8)
            covered = ((group.lower <= group.truth) & (group.truth <= group.upper)).astype(float)
            rolling = covered.rolling(self.rolling_window, min_periods=self.rolling_window).mean()
            cov.plot(
                x, rolling, color=OVENTI["teal"], label=f"Trailing {self.rolling_window} targets"
            )
            cov.axhline(self.level, color=OVENTI["orange"], ls="--", label="Nominal coverage")
            cov.set(
                title="Rolling coverage (test targets only)",
                xlabel="Observation index",
                ylim=(-0.02, 1.02),
            )
            cov.legend(fontsize=8)
            self.save(fig, f"pair_{family.replace(' ', '_')}_{i:02d}")
            plt.show()

    def plot_comparison(self, all_summary, rmse_limit=None, auto_symlog=True):
        """Return one five-metric figure per forecaster, comparing only its interval methods."""
        figures = {}
        for family, group in all_summary.groupby("family", sort=False):
            # Filtering is strictly local to figures, never to tables or exports.
            table = group.copy()
            if rmse_limit is not None:
                table = table.loc[table.rmse <= rmse_limit].copy()
            if table.empty:
                print(f"{family}: all methods excluded by the figure-only RMSE threshold.")
                continue
            figures[family] = self._plot_forecaster_comparison(
                family, table, excluded=len(group) - len(table), auto_symlog=auto_symlog
            )
        if not figures:
            raise ValueError("No pair remains below the figure-only RMSE threshold")
        return figures

    def _plot_forecaster_comparison(self, family, table, excluded, auto_symlog):
        fig, axes = plt.subplots(3, 2, figsize=(13, 11.5), layout="constrained")
        labels = table.method.tolist()
        method_colors = {
            METHODS[0]: OVENTI["light"],
            METHODS[1]: OVENTI["teal"],
            METHODS[2]: OVENTI["orange"],
            "Gaussian (native)": OVENTI["medium"],
        }
        x = np.arange(len(table))
        for ax, metric in zip(axes.flat, METRICS, strict=False):
            means = table[metric].to_numpy(dtype=float)
            low, high = table[f"{metric}_lo"].to_numpy(), table[f"{metric}_hi"].to_numpy()
            finite = np.isfinite(means)
            ax.bar(x[finite], means[finite], color=[method_colors[m] for m in table.method[finite]])
            # Draw endpoints directly: a percentile CI need not contain the original mean.
            interval_finite = finite & np.isfinite(low) & np.isfinite(high)
            ax.vlines(
                x[interval_finite],
                low[interval_finite],
                high[interval_finite],
                color=OVENTI["dark"],
                lw=1.3,
            )
            ax.hlines(
                low[interval_finite],
                x[interval_finite] - 0.14,
                x[interval_finite] + 0.14,
                color=OVENTI["dark"],
            )
            ax.hlines(
                high[interval_finite],
                x[interval_finite] - 0.14,
                x[interval_finite] + 0.14,
                color=OVENTI["dark"],
            )
            for index in np.flatnonzero(~finite | ~np.isfinite(low) | ~np.isfinite(high)):
                ax.text(
                    index,
                    0.96,
                    "∞ / CI unbounded",
                    transform=ax.get_xaxis_transform(),
                    rotation=90,
                    va="top",
                    ha="center",
                    fontsize=7,
                )
            if metric == "coverage":
                ax.axhline(self.level, color=OVENTI["dark"], ls="--", label="Nominal coverage")
                ax.set_ylim(0, 1.04)
                ax.legend(fontsize=8)
            elif auto_symlog:
                magnitudes = np.abs(
                    np.concatenate([means[finite], low[np.isfinite(low)], high[np.isfinite(high)]])
                )
                positive = magnitudes[magnitudes > 0]
                if len(positive) and positive.max() / np.median(positive) > 100:
                    ax.set_yscale("symlog", linthresh=float(np.median(positive)))
            ax.set_title(f"{metric.upper()} — mean and 95% block-bootstrap CI", fontsize=10)
            ax.set_xticks(x, labels, rotation=12, ha="right", fontsize=9)
            ax.set_xlim((-0.6, len(table) - 0.4) if len(table) > 1 else (-1.6, 1.6))
        axes.flat[-1].axis("off")
        scope = (
            "EnbPI: residual-only calibration, not the full OOB algorithm."
            if METHODS[0] in labels
            else "SARIMA: native Gaussian PI, without conformal calibration."
        )
        axes.flat[-1].text(
            0,
            0.9,
            f"{len(table)} methods shown; {excluded} excluded in this figure only.\n"
            "Bars: mean fold metrics. CIs: paired circular blocks of folds.\n"
            "Few folds ⇒ exploratory uncertainty, not exact confidence.\n"
            "Empty-set width = 0; unbounded width = ∞. No dates discarded.\n"
            f"{scope}",
            va="top",
            fontsize=9,
            color=OVENTI["dark"],
            linespacing=1.8,
        )
        fig.suptitle(f"OVENTI | Calibre — {family}", fontsize=15)
        self.save(fig, f"final_comparison_{family.lower().replace(' ', '_')}")
        plt.show()
        return fig
