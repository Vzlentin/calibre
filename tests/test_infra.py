"""Direct locks on the shared helpers in ``tests/infra.py``."""

from __future__ import annotations

import os

import pytest

from tests.infra import restore_cwd


def test_restore_cwd_restores_after_chdir(tmp_path) -> None:
    original = os.getcwd()
    with restore_cwd():
        os.chdir(tmp_path)
        assert os.getcwd() != original
    assert os.getcwd() == original


def test_restore_cwd_restores_on_exception(tmp_path) -> None:
    original = os.getcwd()
    with pytest.raises(RuntimeError, match="boom"), restore_cwd():
        os.chdir(tmp_path)
        raise RuntimeError("boom")
    assert os.getcwd() == original


def test_tuning_evaluation_seam_is_exported() -> None:
    from calibre.tuning import OBJECTIVE_METRIC, ORIGIN_INDEX, evaluate_candidate
    from calibre.tuning import __all__ as tuning_all

    assert {"OBJECTIVE_METRIC", "ORIGIN_INDEX", "evaluate_candidate"} <= set(tuning_all)
    assert OBJECTIVE_METRIC == "objective"
    assert ORIGIN_INDEX == "origin_index"
    assert callable(evaluate_candidate)
