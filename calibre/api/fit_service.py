"""Backward-compatible imports for fit lifecycle helpers.

The implementation lives in :mod:`calibre.execution.model_lifecycle` so
adapter execution stays behind a non-HTTP boundary.
"""

from __future__ import annotations

from calibre.execution.model_lifecycle import (
    AdapterResolver,
    fit_model_artifacts,
    fit_tasks_for_record,
    model_config_for_fit,
    predict_from_artifacts,
    validate_fit_record,
)

__all__ = [
    "AdapterResolver",
    "fit_model_artifacts",
    "fit_tasks_for_record",
    "model_config_for_fit",
    "predict_from_artifacts",
    "validate_fit_record",
]
