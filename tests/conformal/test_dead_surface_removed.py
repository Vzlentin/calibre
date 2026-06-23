"""Guard that the deleted conformal/ordering surface stays un-importable.

The dead split/ACI/policies/intervals cluster plus the ``CumulativeBoundRule``,
``ScaledAbsoluteErrorScore``/``scaled_absolute_error``, and
``MultiStepIntervalPrediction`` symbols were removed (issue #241). These checks
fail loud if any of them silently reappears.
"""

from __future__ import annotations

import importlib

import pytest

REMOVED_CONFORMAL_NAMES = (
    "CumulativeSplitConformalInference",
    "MultiStepSplitConformalInference",
    "AdaptiveConformalInference",
    "MultiStepAdaptiveConformalInference",
    "OnlineConformalController",
    "ScaledAbsoluteErrorScore",
    "scaled_absolute_error",
    "MultiStepIntervalPrediction",
    "symmetric_interval",
    "symmetric_intervals",
)

REMOVED_ORDERING_NAMES = ("CumulativeBoundRule",)

REMOVED_MODULES = (
    "calibre.conformal.split",
    "calibre.conformal.adaptive",
    "calibre.conformal.policies",
    "calibre.conformal.intervals",
)


@pytest.mark.parametrize("name", REMOVED_CONFORMAL_NAMES)
def test_removed_conformal_names_are_gone(name: str) -> None:
    conformal = importlib.import_module("calibre.conformal")
    assert not hasattr(conformal, name)
    assert name not in conformal.__all__


@pytest.mark.parametrize("name", REMOVED_ORDERING_NAMES)
def test_removed_ordering_names_are_gone(name: str) -> None:
    ordering = importlib.import_module("calibre.ordering")
    assert not hasattr(ordering, name)
    assert name not in ordering.__all__


@pytest.mark.parametrize("module", REMOVED_MODULES)
def test_removed_modules_are_gone(module: str) -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module)


def test_live_surface_still_importable() -> None:
    from calibre.conformal import (
        AbsoluteErrorScore,
        IntervalPrediction,
        absolute_error,
        absolute_error_score,
    )

    assert AbsoluteErrorScore is not None
    assert IntervalPrediction is not None
    assert absolute_error is not None
    assert absolute_error_score is not None
