"""Public facade for strict VN2 Gate-A tracking proposals.

All records are derived from validated successor evidence and are published only
beneath the real successor ``newcalibre/artifacts`` tree.
"""

from newcalibre.protocols.vn2._tracking_contracts import (
    TRACKING_KIND,
    TRACKING_SCHEMA,
    AppendDecision,
    TrackingComparison,
    TrackingError,
    VN2TrackingRecord,
)
from newcalibre.protocols.vn2._tracking_contracts import (
    _ga1_digest as tracking_ga1_digest,
)
from newcalibre.protocols.vn2._tracking_contracts import (
    _regular_file_sha256_if_exists as regular_file_sha256_if_exists,
)
from newcalibre.protocols.vn2._tracking_persistence import write_proposal_record
from newcalibre.protocols.vn2._tracking_projection import build_tracking_record
from newcalibre.protocols.vn2._tracking_promotion import (
    TRACKING_SERIES_PATH,
    PromotionReceipt,
    build_promotion_receipt,
    load_promotion_metadata,
    parse_promotion_receipt,
    promotion_receipt_path,
    validate_promotion_paths,
    validate_tracking_promotion,
    write_promotion_receipt,
)
from newcalibre.protocols.vn2._tracking_validation import (
    compare_tracking_records,
    decide_append,
    parse_tracking_history,
    parse_tracking_record,
    require_exact_recomputation,
    resolve_tracking_history_mode,
)

# These are direct aliases so callers keep the original signatures and callable
# identities; supply the public-facing descriptions on those underlying callables.
tracking_ga1_digest.__doc__ = (
    "Return the canonical GA1 comparability digest for a tracking payload."
)
regular_file_sha256_if_exists.__doc__ = (
    "Hash a regular non-symlink file, returning None only when it is absent."
)

__all__ = [
    "AppendDecision",
    "PromotionReceipt",
    "TRACKING_KIND",
    "TRACKING_SCHEMA",
    "TRACKING_SERIES_PATH",
    "TrackingComparison",
    "TrackingError",
    "VN2TrackingRecord",
    "build_promotion_receipt",
    "build_tracking_record",
    "compare_tracking_records",
    "decide_append",
    "load_promotion_metadata",
    "parse_promotion_receipt",
    "parse_tracking_history",
    "parse_tracking_record",
    "promotion_receipt_path",
    "regular_file_sha256_if_exists",
    "require_exact_recomputation",
    "resolve_tracking_history_mode",
    "tracking_ga1_digest",
    "validate_promotion_paths",
    "validate_tracking_promotion",
    "write_promotion_receipt",
    "write_proposal_record",
]
