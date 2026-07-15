"""Shared test infrastructure helpers (not fixtures; import directly)."""

from __future__ import annotations

import importlib.util
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType

import numpy as np


def load_script_module(path: Path) -> ModuleType:
    """Import a standalone script (outside any package) as a module.

    Used for repo automation scripts that tests exercise directly, e.g.
    ``.github/scripts/``. The module is registered in ``sys.modules`` under
    its stem so dataclasses and pickling resolve it.
    """
    name = path.stem
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


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
