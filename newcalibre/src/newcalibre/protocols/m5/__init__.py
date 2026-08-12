"""Expose M5 loading, verification, generic execution, and scoring."""

from newcalibre.protocols.m5.artifacts import (
    M5ArtifactError,
    M5ArtifactSet,
    load_m5_artifacts,
    validate_m5_artifact_files,
)
from newcalibre.protocols.m5.config import load_m5_config
from newcalibre.protocols.m5.inventory import verify_m5_inputs
from newcalibre.protocols.m5.runner import M5RunResult, run_m5
from newcalibre.protocols.m5.scorer import M5Diagnostics, score_m5

__all__ = [
    "M5ArtifactError",
    "M5ArtifactSet",
    "M5Diagnostics",
    "M5RunResult",
    "load_m5_artifacts",
    "load_m5_config",
    "run_m5",
    "score_m5",
    "validate_m5_artifact_files",
    "verify_m5_inputs",
]
