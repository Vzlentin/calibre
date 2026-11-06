"""Electricity folds and calibrator comparisons using Calibre's two origin loops."""

from dataclasses import dataclass
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import SGDRegressor
from statsforecast.models import ARIMA, MSTL, AutoETS, WindowAverage
from threadpoolctl import threadpool_limits

from calibre import (
    ACI,
    AR1,
    Absolute,
    Asymmetric,
    BottomUp,
    Calibrator,
    ConformalPID,
    EnbPI,
    ForecastScorecaster,
    Hierarchy,
    Panel,
    Score,
    SplitQuantile,
    State,
    Step,
    initial_state,
    replay,
    rolling_forecasts,
)
from calibre.backtest import Forecasts
from calibre.conformal.calibrators.spci import SPCI
from calibre.forecast.models.base import Forecaster
from calibre.forecast.models.mlforecast import MLForecast
from calibre.forecast.models.statsforecast import StatsForecastModel
from calibre.metrics import coverage, width
from compare import compare

POINT_METRICS = ("rmse", "mae", "bias")
INTERVAL_METRICS = ("coverage", "width")


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
    booster_iterations: int = 200
    booster_depth: int = 5
    ogd_eta: float = 0.001
    ogd_l2: float = 0.001
    aci_gamma: float = 0.005
    pid_rate: float = 0.05
    pid_integral: float = 0.10
    pid_score_window: int = 64
    spci_n_lags: int = 24
    spci_min_train_samples: int = 20
    spci_n_estimators: int = 100
    spci_max_depth: int = 5
    spci_min_samples_leaf: int = 5
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
            self.booster_iterations,
            self.booster_depth,
            self.pid_score_window,
            self.spci_n_lags,
            self.spci_min_train_samples,
            self.spci_n_estimators,
            self.spci_max_depth,
            self.spci_min_samples_leaf,
        )
        if any(not isinstance(v, int) or isinstance(v, bool) or v < 1 for v in integers):
            raise ValueError("Window lengths, fold counts and model sizes must be positive ints")
        if self.train_size <= max(self.n_lags + 2, 2 * self.period, self.moving_window):
            raise ValueError("Training is too short for the lags/decomposition")
        if not 0 < self.level < 1:
            raise ValueError("level must be in (0, 1)")
        if self.calibration_size < int(np.ceil(self.level / (1 - self.level))) + 1:
            raise ValueError("Calibration is too short for the nominal split-conformal rank")
        if self.spci_n_lags + self.spci_min_train_samples > self.calibration_size:
            raise ValueError("Calibration must be at least spci_n_lags + spci_min_train_samples")


@dataclass(frozen=True)
class Split:
    fold: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int


def chronological_splits(n: int, config: Config) -> list[Split]:
    """Contiguous test blocks, each preceded by training and calibration windows."""
    first = n - config.n_split * config.test_size
    if first - config.calibration_size - config.train_size < 0:
        raise ValueError("Not enough observations for the requested rolling windows")
    return [
        Split(
            fold=k,
            train_start=start - config.calibration_size - config.train_size,
            train_end=start - config.calibration_size,
            test_start=start,
            test_end=start + config.test_size,
        )
        for k, start in enumerate(range(first, n, config.test_size))
    ]


def identify_sarima(training, config, max_p=2, max_q=2):
    """Choose from a fixed AIC grid using only the first training block."""
    rows = []
    for p in range(max_p + 1):
        for q in range(max_q + 1):
            fitted = ARIMA(
                order=(p, config.sarima_d, q),
                seasonal_order=config.seasonal_order,
                season_length=config.period,
                method=config.sarima_method,
            ).fit(np.asarray(training, dtype=float))
            rows.append(
                {
                    "p": p,
                    "d": config.sarima_d,
                    "q": q,
                    **{key: fitted.model_[key] for key in ("aic", "aicc", "bic")},
                }
            )
    table = pd.DataFrame(rows).sort_values("aic", ignore_index=True)
    if not np.isfinite(table.aic).any():
        raise ValueError("No finite training AIC in the predefined SARIMA grid")
    best = table.loc[table.aic.idxmin()]
    return tuple(int(best[k]) for k in ("p", "d", "q")), table


@dataclass(frozen=True)
class ForecastSpec:
    model: Forecaster
    refit_every: int | None = None
    center: float = 0.0
    scale: float = 1.0


def make_forecasters(training, config: Config, sarima_order) -> dict[str, ForecastSpec]:
    """Build once per fold; only Online GD uses training-only scaling and refits."""
    lags = list(range(1, config.n_lags + 1))
    return {
        "Moving average": ForecastSpec(
            StatsForecastModel(WindowAverage(window_size=config.moving_window))
        ),
        "Trend-season MSTL": ForecastSpec(
            StatsForecastModel(
                MSTL(season_length=config.period, trend_forecaster=AutoETS(model="AAN"))
            )
        ),
        "SARIMA": ForecastSpec(
            StatsForecastModel(
                ARIMA(
                    order=sarima_order,
                    seasonal_order=config.seasonal_order,
                    season_length=config.period,
                    method=config.sarima_method,
                )
            )
        ),
        "HistGradientBoosting": ForecastSpec(
            MLForecast(
                HistGradientBoostingRegressor(
                    max_iter=config.booster_iterations,
                    max_depth=config.booster_depth,
                    learning_rate=0.05,
                    random_state=config.seed,
                    early_stopping=False,
                ),
                lags=lags,
                lookback=config.n_lags,
                num_threads=1,
            )
        ),
        "Online GD": ForecastSpec(
            MLForecast(
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
                lags=lags,
                lookback=config.n_lags,
                num_threads=1,
            ),
            refit_every=1,
            center=float(np.mean(training)),
            scale=float(np.std(training)) or 1.0,
        ),
    }


def make_forecasts(
    spec: ForecastSpec,
    values: np.ndarray,
    config: Config,
) -> tuple[Forecasts, np.ndarray]:
    """Forecast calibration/test once, restoring original units.

    Histogram boosting stays frozen; local vendor models use expanding history.
    Online GD replays deterministic single-pass updates on each expanding prefix.
    """
    series = np.array(["target"])
    hierarchy = Hierarchy.flat(series)
    # Positional dates satisfy Panel's calendar contract, not recovered source dates.
    periods = pd.date_range("2000-01-01", periods=len(values), freq="h")
    panel = Panel(series, periods, ((values - spec.center) / spec.scale)[None, :], "h")
    # HistGradientBoosting otherwise uses every OpenMP thread even on this small series.
    with threadpool_limits(limits=1):
        forecasts = rolling_forecasts(
            panel,
            hierarchy,
            spec.model,
            BottomUp(hierarchy),
            np.arange(config.train_size - 1, len(values) - 1),
            horizon=1,
            refit_every=spec.refit_every,
        )
    return Forecasts(
        forecasts.origins, (forecasts.points * spec.scale + spec.center).astype(np.float32)
    ), values[None, :].astype(np.float32)


@dataclass(frozen=True)
class CalibratorSpec:
    score: Score
    calibrator: Calibrator


def make_calibrators(config: Config) -> dict[str, CalibratorSpec]:
    """Configuration-only instances, shared across fresh fold states."""
    return {
        "EnbPI (residual-only)": CalibratorSpec(
            Asymmetric(), EnbPI(config.level, window=config.calibration_size)
        ),
        "ACI": CalibratorSpec(
            Absolute(),
            ACI(
                SplitQuantile(config.level, window=config.calibration_size), gamma=config.aci_gamma
            ),
        ),
        "Conformal PID": CalibratorSpec(
            Absolute(),
            ConformalPID(
                config.level,
                learning_rate=config.pid_rate,
                integral_gain=config.pid_integral,
                window=config.calibration_size,
                scorecast_window=config.pid_score_window,
                scorecaster=ForecastScorecaster(AR1()),
                ahead=1,
            ),
        ),
        "SPCI": CalibratorSpec(
            Asymmetric(),
            SPCI(
                config.level,
                window=config.calibration_size,
                n_lags=config.spci_n_lags,
                ahead=1,
                min_train_samples=config.spci_min_train_samples,
                n_estimators=config.spci_n_estimators,
                max_depth=config.spci_max_depth,
                min_samples_leaf=config.spci_min_samples_leaf,
                random_state=config.seed,
                n_jobs=1,
            ),
        ),
    }


def prepare_states(
    calibration: Forecasts,
    actuals: np.ndarray,
    first_test_origin: int,
    calibrators: dict[str, CalibratorSpec],
) -> dict[str, State]:
    """Replay EnbPI/ACI with pending ledgers; seed SPCI/PID from known forecast errors."""
    known_actuals = actuals[:, : first_test_origin + 1]
    residuals = calibration.residuals(known_actuals)
    if not np.isfinite(residuals).all():
        raise ValueError("Calibration seed includes a target not known at first test origin")
    states = {}
    for name, spec in calibrators.items():
        calibrator = spec.calibrator
        if isinstance(calibrator, (SPCI, ConformalPID)):
            state = initial_state(
                Step(),
                calibrator,
                actuals.shape[0],
                calibration.points.shape[2],
                score=spec.score,
            )
            if isinstance(calibrator, SPCI):
                state["calibrator"] = calibrator.seed(
                    state["calibrator"], residuals, origins=calibration.origins.astype(np.int64)
                )
            else:
                state["calibrator"] = calibrator.seed(state["calibrator"], np.abs(residuals))
        else:
            state = replay(
                calibration, known_actuals, target=Step(), score=spec.score, calibrator=calibrator
            ).state
        states[name] = state
    return states


@dataclass(frozen=True)
class TestLosses:
    """Chronological test losses, with unique forecaster errors and paired interval losses.

    Point contributions [T, F, 3] are squared, absolute and signed errors.
    Interval contributions [T, F, M, 2] are endpoint coverage and library width;
    ready [T, F, M] records finite issued parameters. All data are caller-owned arrays.
    """

    steps: np.ndarray
    fold_sizes: np.ndarray
    families: np.ndarray
    methods: np.ndarray
    point: np.ndarray
    interval: np.ndarray
    ready: np.ndarray


def run(
    y,
    splits: list[Split],
    config: Config,
    sarima_order=(1, 1, 1),
) -> tuple[pd.DataFrame, pd.DataFrame, TestLosses]:
    """Return separate fold tables and per-target losses for paired bootstrap analysis.

    Standard compare readiness masks infinite thresholds, including PID saturation;
    the loss record keeps every target and its readiness, without changing compare.
    """
    y = np.asarray(y, dtype=float)
    if y.ndim != 1 or not np.isfinite(y).all():
        raise ValueError("Expected one finite electricity series")
    if not splits:
        raise ValueError("At least one chronological test fold is required")
    calibrators = make_calibrators(config)
    methods = np.array(list(calibrators))
    method_indexes = {name: i for i, name in enumerate(methods)}
    fold_sizes = np.array([s.test_end - s.test_start for s in splits], dtype=np.int64)
    steps = np.concatenate([np.arange(s.test_start, s.test_end) for s in splits])
    if (fold_sizes <= 0).any() or not np.all(np.diff(steps) == 1):
        raise ValueError("Test folds must be positive, adjacent chronological windows")
    losses = None
    offset = 0
    point_rows, interval_tables = [], []
    for split, size in zip(splits, fold_sizes, strict=True):
        values = y[split.train_start : split.test_end]
        models = make_forecasters(values[: config.train_size], config, sarima_order)
        if losses is None:
            families = np.array(list(models))
            losses = TestLosses(
                steps,
                fold_sizes,
                families,
                methods,
                np.empty((len(steps), len(families), len(POINT_METRICS)), dtype=np.float64),
                np.empty(
                    (len(steps), len(families), len(methods), len(INTERVAL_METRICS)),
                    dtype=np.float64,
                ),
                np.empty((len(steps), len(families), len(methods)), dtype=bool),
            )
        window = slice(offset, offset + size)
        for family_index, (family, spec) in enumerate(models.items()):
            started = perf_counter()
            forecasts, actuals = make_forecasts(spec, values, config)
            start = config.calibration_size
            calibration = Forecasts(forecasts.origins[:start], forecasts.points[:start])
            test = Forecasts(forecasts.origins[start:], forecasts.points[start:])
            expected = np.arange(split.test_start, split.test_end)
            np.testing.assert_array_equal(test.origins + split.train_start + 1, expected)
            error = test.points[:, 0, 0].astype(float) - actuals[0, test.origins + 1]
            losses.point[window, family_index] = np.column_stack((error**2, np.abs(error), error))
            point_rows.append(
                {
                    "family": family,
                    "fold": split.fold,
                    "H": len(expected),
                    "rmse": np.sqrt(np.mean(error**2)),
                    "mae": np.mean(np.abs(error)),
                    "bias": np.mean(error),
                }
            )
            states = prepare_states(calibration, actuals, int(test.origins[0]), calibrators)
            for score_type in (Asymmetric, Absolute):
                group = {
                    name: entry.calibrator
                    for name, entry in calibrators.items()
                    if isinstance(entry.score, score_type)
                }
                table, runs = compare(
                    test,
                    actuals,
                    group,
                    Step(),
                    score_type(),
                    config.level,
                    states={name: states[name] for name in group},
                )
                for name, result in runs.items():
                    method_index = method_indexes[name]
                    truth = result.target[:, 0, 0, None].astype(np.float64)
                    lower = result.lower[:, 0, 0, None].astype(np.float64)
                    upper = result.upper[:, 0, 0, None].astype(np.float64)
                    losses.interval[window, family_index, method_index, 0] = coverage(
                        truth, lower, upper, axis=-1
                    )
                    losses.interval[window, family_index, method_index, 1] = width(
                        truth, lower, upper, axis=-1
                    )
                    losses.ready[window, family_index, method_index] = (
                        np.isfinite(result.threshold).reshape(len(expected), -1).all(axis=1)
                    )
                interval_tables.append(
                    table.reset_index().assign(family=family, fold=split.fold, H=len(expected))
                )
            print(
                f"{family} | fold {split.fold + 1}/{len(splits)} | {perf_counter() - started:.1f}s"
            )
        offset += size
    point_folds = pd.DataFrame(point_rows, columns=["family", "fold", "H", *POINT_METRICS])
    interval_folds = pd.concat(interval_tables, ignore_index=True)[
        ["family", "method", "fold", "H", "ready", *INTERVAL_METRICS, "interval_score"]
    ]
    assert losses is not None
    return point_folds, interval_folds, losses
