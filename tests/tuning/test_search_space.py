"""Unit tests for the declarative search-space spec parser."""

from __future__ import annotations

import optuna
import pytest

from calibre.tuning import suggest_from_spec


def _recording_trial(params: dict[str, object]) -> optuna.trial.FixedTrial:
    return optuna.trial.FixedTrial(params)


def test_categorical_dispatches_to_suggest_categorical() -> None:
    trial = _recording_trial({"q": 0.51})
    spec = {"type": "categorical", "choices": [0.45, 0.51, 0.59]}

    assert suggest_from_spec(trial, "q", spec) == 0.51


def test_int_honours_step() -> None:
    trial = _recording_trial({"n": 250})
    spec = {"type": "int", "low": 200, "high": 800, "step": 50}

    value = suggest_from_spec(trial, "n", spec)

    assert value == 250
    assert trial.distributions["n"].step == 50


def test_int_defaults_step_to_one() -> None:
    trial = _recording_trial({"k": 17})
    spec = {"type": "int", "low": 10, "high": 60}

    value = suggest_from_spec(trial, "k", spec)

    assert value == 17
    assert trial.distributions["k"].step == 1


def test_float_honours_log() -> None:
    trial = _recording_trial({"lr": 0.05})
    spec = {"type": "float", "low": 0.02, "high": 0.10, "log": True}

    value = suggest_from_spec(trial, "lr", spec)

    assert value == 0.05
    assert trial.distributions["lr"].log is True


def test_float_honours_step() -> None:
    trial = _recording_trial({"s": 0.8})
    spec = {"type": "float", "low": 0.6, "high": 1.0, "step": 0.1}

    value = suggest_from_spec(trial, "s", spec)

    assert value == 0.8
    assert trial.distributions["s"].step == pytest.approx(0.1)


def test_unknown_type_raises() -> None:
    trial = _recording_trial({})

    with pytest.raises(ValueError, match="Unknown HPO spec type"):
        suggest_from_spec(trial, "x", {"type": "boolean"})
