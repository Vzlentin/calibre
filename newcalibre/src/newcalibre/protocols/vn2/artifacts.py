"""Expose the stable VN2 result-bundle interface over private implementation modules."""

from newcalibre.protocols.vn2._artifact_contracts import (
    CONFIG_PATH as CONFIG_PATH,
)
from newcalibre.protocols.vn2._artifact_contracts import (
    GITHUB_REPOSITORY as GITHUB_REPOSITORY,
)
from newcalibre.protocols.vn2._artifact_contracts import (
    INPUT_INVENTORY_PATH as INPUT_INVENTORY_PATH,
)
from newcalibre.protocols.vn2._artifact_contracts import (
    LOCK_PATH as LOCK_PATH,
)
from newcalibre.protocols.vn2._artifact_contracts import (
    RESULT_KIND as RESULT_KIND,
)
from newcalibre.protocols.vn2._artifact_contracts import (
    THREAD_VARIABLES,
    VN2EvidenceEnvironment,
    VN2ResultBundle,
    VN2ResultError,
    VN2ResultFile,
    VN2ResultManifest,
)
from newcalibre.protocols.vn2._artifact_projection import (
    capture_vn2_evidence_environment,
    emit_vn2_result_bundle,
)
from newcalibre.protocols.vn2._artifact_validation import validate_vn2_result_bundle

__all__ = [
    "THREAD_VARIABLES",
    "VN2EvidenceEnvironment",
    "VN2ResultBundle",
    "VN2ResultError",
    "VN2ResultFile",
    "VN2ResultManifest",
    "capture_vn2_evidence_environment",
    "emit_vn2_result_bundle",
    "validate_vn2_result_bundle",
]
