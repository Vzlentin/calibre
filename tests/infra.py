"""Shared test infrastructure helpers (not fixtures; import directly)."""

from __future__ import annotations

import os
from contextlib import contextmanager


@contextmanager
def restore_cwd():
    """Restore the original working directory on exit, even on error.

    Backs the autouse ``_restore_cwd`` fixture in ``tests/conftest.py``: some
    test dependencies (e.g. Ray Tune trials) chdir and don't always restore,
    which would otherwise leak a stale cwd into subsequent tests.
    """
    original = os.getcwd()
    try:
        yield
    finally:
        os.chdir(original)
