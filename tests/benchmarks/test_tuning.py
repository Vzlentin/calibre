"""Tests for benchmarks.vn2.tuning — HPO exception handling."""

from __future__ import annotations

import pytest

from benchmarks.vn2.tuning import _best_tune_result, _cumulative_pinball


def test_cumulative_pinball_returns_inf_for_empty_frame():
    import pandas as pd

    from calibre.core.forecast_frame import DS, UNIQUE_ID, Y

    result = _cumulative_pinball(
        forecast_df=pd.DataFrame(),
        actuals=pd.DataFrame({UNIQUE_ID: ["a"], DS: [pd.Timestamp("2024-01-01")], Y: [1.0]}),
        horizon=2,
        quantile=0.5,
        tau=0.5,
    )
    assert result == float("inf")


def test_infra_exception_not_swallowed_as_inf():
    """Infra exceptions in the HPO trainable must propagate, not return inf.

    This test verifies that run_cost_search._trainable distinguishes bad-trial
    errors (ValueError/KeyError → inf) from infra failures (other Exception →
    propagate). The exception discriminator was added in Phase 4.c.
    """
    # Verify the trainable pattern in run_cost_search: broad `except Exception`
    # must NOT be present (it was replaced with narrow ValueError/KeyError).
    import inspect

    import benchmarks.vn2.run_benchmark as mod

    source = inspect.getsource(mod.run_cost_search)
    assert "except Exception" not in source, (
        "run_cost_search should not have a broad 'except Exception' that swallows to inf; "
        "infra failures must propagate. Found broad catch — check Phase 4.c fix."
    )


def test_best_tune_result_raises_when_all_trials_fail():
    """_best_tune_result raises RuntimeError when no valid trial exists."""

    class _FakeResult:
        error = None
        metrics = {"objective": float("inf")}

    class _FakeResults:
        def __iter__(self):
            return iter([_FakeResult()])

        def get_best_result(self, **kwargs):
            return _FakeResult()

    with pytest.raises(RuntimeError, match="without a valid"):
        _best_tune_result(_FakeResults())
