"""Shared API error types and exception-to-HTTP mapping."""

from __future__ import annotations

import traceback


def format_error(exc: Exception) -> str:
    """Render an exception as a single-line ``Type: message`` string for API error fields."""
    return "".join(traceback.format_exception_only(type(exc), exc)).strip()
