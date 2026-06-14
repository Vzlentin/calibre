"""Point-forecast error metrics over aligned actual/predicted arrays.

A registry (:data:`METRICS`) maps metric names to callables; :func:`evaluate`
and :func:`evaluate_all` compute a subset or the full set. Percentage-based
metrics are returned as fractions (not multiplied by 100).
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import numpy as np

logger = logging.getLogger(__name__)

EPSILON = 1e-10


def _error(actual: np.ndarray, predicted: np.ndarray):
    """Return the raw forecast error (actual minus predicted)."""
    return actual - predicted


def _percentage_error(actual: np.ndarray, predicted: np.ndarray):
    """Return the percentage error as a fraction (not multiplied by 100)."""
    return _error(actual, predicted) / (actual + EPSILON)


def _naive_forecasting(actual: np.ndarray, seasonality: int = 1):
    """Return the seasonal-naive forecast that repeats prior samples."""
    return actual[:-seasonality]


def _relative_error(actual: np.ndarray, predicted: np.ndarray, benchmark: np.ndarray | None = None):
    """Return the error relative to a benchmark (seasonal-naive by default)."""
    if benchmark is None or isinstance(benchmark, int):
        # If no benchmark prediction provided - use naive forecasting
        seasonality = 1 if not isinstance(benchmark, int) else benchmark
        return _error(actual[seasonality:], predicted[seasonality:]) / (
            _error(actual[seasonality:], _naive_forecasting(actual, seasonality)) + EPSILON
        )

    return _error(actual, predicted) / (_error(actual, benchmark) + EPSILON)


def _bounded_relative_error(
    actual: np.ndarray, predicted: np.ndarray, benchmark: np.ndarray | None = None
):
    """Return the bounded relative error in ``[0, 1]``."""
    if benchmark is None or isinstance(benchmark, int):
        # If no benchmark prediction provided - use naive forecasting
        seasonality = 1 if not isinstance(benchmark, int) else benchmark

        abs_err = np.abs(_error(actual[seasonality:], predicted[seasonality:]))
        abs_err_bench = np.abs(
            _error(actual[seasonality:], _naive_forecasting(actual, seasonality))
        )
    else:
        abs_err = np.abs(_error(actual, predicted))
        abs_err_bench = np.abs(_error(actual, benchmark))

    return abs_err / (abs_err + abs_err_bench + EPSILON)


def _geometric_mean(a, axis=0, dtype=None):
    """Return the geometric mean along ``axis``."""
    if not isinstance(a, np.ndarray):  # if not an ndarray object attempt to convert it
        log_a = np.log(np.array(a, dtype=dtype))
    elif dtype:  # Must change the default dtype allowing array type
        if isinstance(a, np.ma.MaskedArray):
            log_a = np.log(np.ma.asarray(a, dtype=dtype))
        else:
            log_a = np.log(np.asarray(a, dtype=dtype))
    else:
        log_a = np.log(a)
    return np.exp(log_a.mean(axis=axis))


def mse(actual: np.ndarray, predicted: np.ndarray):
    """Return the mean squared error."""
    return np.mean(np.square(_error(actual, predicted)))


def rmse(actual: np.ndarray, predicted: np.ndarray):
    """Return the root mean squared error."""
    return np.sqrt(mse(actual, predicted))


def nrmse(actual: np.ndarray, predicted: np.ndarray):
    """Return the range-normalized root mean squared error."""
    return rmse(actual, predicted) / (actual.max() - actual.min())


def me(actual: np.ndarray, predicted: np.ndarray):
    """Return the mean error (bias)."""
    return np.mean(_error(actual, predicted))


def mae(actual: np.ndarray, predicted: np.ndarray):
    """Return the mean absolute error."""
    return np.mean(np.abs(_error(actual, predicted)))


mad = mae  # Mean Absolute Deviation (it is the same as MAE)


def gmae(actual: np.ndarray, predicted: np.ndarray):
    """Return the geometric mean absolute error."""
    return _geometric_mean(np.abs(_error(actual, predicted)))


def mdae(actual: np.ndarray, predicted: np.ndarray):
    """Return the median absolute error."""
    return np.median(np.abs(_error(actual, predicted)))


def mpe(actual: np.ndarray, predicted: np.ndarray):
    """Return the mean percentage error (as a fraction)."""
    return np.mean(_percentage_error(actual, predicted))


def mape(actual: np.ndarray, predicted: np.ndarray):
    """Return the mean absolute percentage error (as a fraction).

    Properties:
        + Easy to interpret
        + Scale independent
        - Biased, not symmetric
        - Undefined when actual[t] == 0

    The result is NOT multiplied by 100.
    """
    return np.mean(np.abs(_percentage_error(actual, predicted)))


def mdape(actual: np.ndarray, predicted: np.ndarray):
    """Return the median absolute percentage error (as a fraction)."""
    return np.median(np.abs(_percentage_error(actual, predicted)))


def smape(actual: np.ndarray, predicted: np.ndarray):
    """Return the symmetric mean absolute percentage error (as a fraction)."""
    return np.mean(
        2.0 * np.abs(actual - predicted) / ((np.abs(actual) + np.abs(predicted)) + EPSILON)
    )


def smdape(actual: np.ndarray, predicted: np.ndarray):
    """Return the symmetric median absolute percentage error (as a fraction)."""
    return np.median(
        2.0 * np.abs(actual - predicted) / ((np.abs(actual) + np.abs(predicted)) + EPSILON)
    )


def maape(actual: np.ndarray, predicted: np.ndarray):
    """Return the mean arctangent absolute percentage error (as a fraction)."""
    return np.mean(np.arctan(np.abs((actual - predicted) / (actual + EPSILON))))


def mase(actual: np.ndarray, predicted: np.ndarray, seasonality: int = 1):
    """Return the mean absolute scaled error.

    The baseline is the seasonal-naive forecast shifted by ``seasonality``.
    """
    return mae(actual, predicted) / mae(
        actual[seasonality:], _naive_forecasting(actual, seasonality)
    )


def std_ae(actual: np.ndarray, predicted: np.ndarray):
    """Return the standard deviation of the absolute error."""
    __mae = mae(actual, predicted)
    return np.sqrt(np.sum(np.square(_error(actual, predicted) - __mae)) / (len(actual) - 1))


def std_ape(actual: np.ndarray, predicted: np.ndarray):
    """Return the standard deviation of the percentage error."""
    __mape = mape(actual, predicted)
    return np.sqrt(
        np.sum(np.square(_percentage_error(actual, predicted) - __mape)) / (len(actual) - 1)
    )


def rmspe(actual: np.ndarray, predicted: np.ndarray):
    """Return the root mean squared percentage error (as a fraction)."""
    return np.sqrt(np.mean(np.square(_percentage_error(actual, predicted))))


def rmdspe(actual: np.ndarray, predicted: np.ndarray):
    """Return the root median squared percentage error (as a fraction)."""
    return np.sqrt(np.median(np.square(_percentage_error(actual, predicted))))


def rmsse(actual: np.ndarray, predicted: np.ndarray, seasonality: int = 1):
    """Return the root mean squared scaled error."""
    q = np.abs(_error(actual, predicted)) / mae(
        actual[seasonality:], _naive_forecasting(actual, seasonality)
    )
    return np.sqrt(np.mean(np.square(q)))


def inrse(actual: np.ndarray, predicted: np.ndarray):
    """Return the integral normalized root squared error."""
    return np.sqrt(
        np.sum(np.square(_error(actual, predicted))) / np.sum(np.square(actual - np.mean(actual)))
    )


def rrse(actual: np.ndarray, predicted: np.ndarray):
    """Return the root relative squared error."""
    return np.sqrt(
        np.sum(np.square(actual - predicted)) / np.sum(np.square(actual - np.mean(actual)))
    )


def mre(actual: np.ndarray, predicted: np.ndarray, benchmark: np.ndarray | None = None):
    """Return the mean relative error."""
    return np.mean(_relative_error(actual, predicted, benchmark))


def rae(actual: np.ndarray, predicted: np.ndarray):
    """Return the relative absolute error (a.k.a. approximation error)."""
    return np.sum(np.abs(actual - predicted)) / (np.sum(np.abs(actual - np.mean(actual))) + EPSILON)


def mrae(actual: np.ndarray, predicted: np.ndarray, benchmark: np.ndarray | None = None):
    """Return the mean relative absolute error."""
    return np.mean(np.abs(_relative_error(actual, predicted, benchmark)))


def mdrae(actual: np.ndarray, predicted: np.ndarray, benchmark: np.ndarray | None = None):
    """Return the median relative absolute error."""
    return np.median(np.abs(_relative_error(actual, predicted, benchmark)))


def gmrae(actual: np.ndarray, predicted: np.ndarray, benchmark: np.ndarray | None = None):
    """Return the geometric mean relative absolute error."""
    return _geometric_mean(np.abs(_relative_error(actual, predicted, benchmark)))


def mbrae(actual: np.ndarray, predicted: np.ndarray, benchmark: np.ndarray | None = None):
    """Return the mean bounded relative absolute error."""
    return np.mean(_bounded_relative_error(actual, predicted, benchmark))


def umbrae(actual: np.ndarray, predicted: np.ndarray, benchmark: np.ndarray | None = None):
    """Return the unscaled mean bounded relative absolute error."""
    __mbrae = mbrae(actual, predicted, benchmark)
    return __mbrae / (1 - __mbrae)


def mda(actual: np.ndarray, predicted: np.ndarray):
    """Return the mean directional accuracy."""
    return np.mean(
        (np.sign(actual[1:] - actual[:-1]) == np.sign(predicted[1:] - actual[:-1])).astype(int)
    )


def pinball_linear(actual: np.ndarray, predicted: np.ndarray, tau: float = 0.5):
    """Return the pinball (quantile) loss at quantile ``tau``."""
    return np.mean(np.maximum(tau * (actual - predicted), (tau - 1) * (actual - predicted)))


def wape(actual: np.ndarray, predicted: np.ndarray):
    """Return the weighted absolute percentage error."""
    return mae(actual, predicted) / np.mean(actual)


METRICS: dict[str, Callable[..., float]] = {
    "mse": mse,
    "rmse": rmse,
    "nrmse": nrmse,
    "me": me,
    "mae": mae,
    "mad": mad,
    "gmae": gmae,
    "mdae": mdae,
    "mpe": mpe,
    "mape": mape,
    "mdape": mdape,
    "smape": smape,
    "smdape": smdape,
    "maape": maape,
    "mase": mase,
    "std_ae": std_ae,
    "std_ape": std_ape,
    "rmspe": rmspe,
    "rmdspe": rmdspe,
    "rmsse": rmsse,
    "inrse": inrse,
    "rrse": rrse,
    "mre": mre,
    "rae": rae,
    "mrae": mrae,
    "mdrae": mdrae,
    "gmrae": gmrae,
    "mbrae": mbrae,
    "umbrae": umbrae,
    "mda": mda,
    "wape": wape,
}


def evaluate(actual: np.ndarray, predicted: np.ndarray, metrics=("mae", "mse", "smape", "umbrae")):
    """Compute the named metrics, returning NaN for any that raise.

    Args:
        actual: Observed values.
        predicted: Forecast values aligned with ``actual``.
        metrics: Metric names to compute (keys of :data:`METRICS`).

    Returns:
        A name-to-value mapping; failed metrics map to ``np.nan`` and are logged.
    """
    results = {}
    for name in metrics:
        try:
            results[name] = METRICS[name](actual, predicted)
        except Exception as err:
            results[name] = np.nan
            logger.warning("Unable to compute metric %s: %s", name, err)
    return results


def evaluate_all(actual: np.ndarray, predicted: np.ndarray):
    """Compute every metric in :data:`METRICS` for the given series."""
    return evaluate(actual, predicted, metrics=set(METRICS.keys()))
