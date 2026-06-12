"""Shared test infrastructure helpers (not fixtures; import directly)."""

from __future__ import annotations

import os
from contextlib import contextmanager


@contextmanager
def restore_cwd():
    """Save the working directory and restore it on exit, even on error —
    a ``chdir`` inside the block cannot leak to the caller.

    Backs the autouse ``_restore_cwd`` fixture in ``tests/conftest.py``: some
    test dependencies (e.g. Ray Tune trials) chdir and don't always restore,
    which would otherwise leak a stale cwd into subsequent tests.
    """
    original = os.getcwd()
    try:
        yield
    finally:
        os.chdir(original)
