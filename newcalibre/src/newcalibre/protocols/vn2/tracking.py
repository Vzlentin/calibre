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
from newcalibre.protocols.vn2._tracking_persistence import write_proposal_record
from newcalibre.protocols.vn2._tracking_projection import build_tracking_record
from newcalibre.protocols.vn2._tracking_validation import (
    compare_tracking_records,
    decide_append,
    parse_tracking_history,
    parse_tracking_record,
)

__all__ = [
    "AppendDecision",
    "TRACKING_KIND",
    "TRACKING_SCHEMA",
    "TrackingComparison",
    "TrackingError",
    "VN2TrackingRecord",
    "build_tracking_record",
    "compare_tracking_records",
    "decide_append",
    "parse_tracking_history",
    "parse_tracking_record",
    "write_proposal_record",
]
