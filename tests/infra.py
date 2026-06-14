"""Shared test infrastructure helpers (not fixtures; import directly)."""

from __future__ import annotations

import os
from contextlib import contextmanager

import numpy as np


def closed_form_min_trace(S: np.ndarray, w_diag: np.ndarray, base: np.ndarray) -> np.ndarray:
    """Dense MinT projection with diagonal W: S (S' W^-1 S)^-1 S' W^-1 y.

    Reference implementation for reconciliation agreement pins; deliberately
    independent of the production solver path.
    """
    weighted = S.T / w_diag
    bottom = np.linalg.solve(weighted @ S, weighted @ base)
    return S @ bottom


@contextmanager
def restore_cwd():
    """Save the working directory and restore it on exit, even on error.

    A ``chdir`` inside the block cannot leak to the caller.

    Backs the autouse ``_restore_cwd`` fixture in ``tests/conftest.py``: some
    test dependencies (e.g. Ray Tune trials) chdir and don't always restore,
    which would otherwise leak a stale cwd into subsequent tests.
    """
    original = os.getcwd()
    try:
        yield
    finally:
        os.chdir(original)
