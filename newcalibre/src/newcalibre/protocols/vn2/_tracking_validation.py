"""Private parsing and comparison operations for VN2 tracking."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from newcalibre.protocols.vn2._tracking_contracts import (
    _GA1_FIELDS,
    AppendDecision,
    TrackingComparison,
    TrackingError,
    VN2TrackingRecord,
    _canonical_record_bytes,
    _comparability_key,
    _parse_json,
    _read_bytes,
)


def parse_tracking_record(value: bytes | bytearray | str | Path) -> VN2TrackingRecord:
    """Parse exactly one canonical JSONL record."""
    payload = _read_bytes(value, name="tracking record")
    if not payload.endswith(b"\n") or payload.endswith(b"\r\n"):
        raise TrackingError("tracking record must end with exactly one LF")
    body = payload[:-1]
    if not body or b"\n" in body or b"\r" in body:
        raise TrackingError("tracking record must contain exactly one JSON object line")
    parsed = _parse_json(body, name="tracking record")
    if _canonical_record_bytes(parsed) != payload:
        raise TrackingError("tracking record bytes are not canonical")
    return VN2TrackingRecord._from_parsed(parsed)


def parse_tracking_history(value: bytes | bytearray | str | Path) -> tuple[VN2TrackingRecord, ...]:
    """Parse canonical history rows and reject repeated identities."""
    payload = _read_bytes(value, name="tracking history")
    if not payload:
        raise TrackingError("tracking history must contain at least one record")
    if payload.endswith(b"\r\n") or b"\r" in payload:
        raise TrackingError("tracking history must use LF-only line endings")
    rows = payload.split(b"\n")
    if rows[-1] != b"":
        raise TrackingError("tracking history must end with LF")
    rows = rows[:-1]
    if not rows or any(not row for row in rows):
        raise TrackingError("tracking history contains a blank line")
    records = tuple(parse_tracking_record(row + b"\n") for row in rows)
    identities = [record.identity for record in records]
    if len(set(identities)) != len(identities):
        raise TrackingError("tracking history contains a duplicate identity")
    return records


def decide_append(
    record: VN2TrackingRecord, history: Sequence[VN2TrackingRecord]
) -> AppendDecision:
    """Return append/no-op/conflict semantics for one valid history."""
    for existing in history:
        if existing.identity != record.identity:
            continue
        if existing.to_bytes() == record.to_bytes():
            return AppendDecision("noop", record)
        return AppendDecision("conflict", record)
    return AppendDecision("append", record)


def compare_tracking_records(
    current: VN2TrackingRecord,
    prior: VN2TrackingRecord,
) -> TrackingComparison:
    """Compare costs only when every exact GA1 field matches."""
    current_key = _comparability_key(current.payload)
    prior_key = _comparability_key(prior.payload)
    mismatches = tuple(field for field in _GA1_FIELDS if current_key[field] != prior_key[field])
    if mismatches:
        return TrackingComparison(False, mismatches, None, None)
    delta = current.total_cost - prior.total_cost
    return TrackingComparison(True, (), delta, delta > 0.0)


def resolve_tracking_history_mode(
    *,
    mode: str,
    require_history: str,
    history: Sequence[VN2TrackingRecord],
) -> Literal["mint", "compare", "skip"]:
    """Resolve whether one Tier 4 run must compare canonical history."""
    if mode not in {"mint", "verify"}:
        raise TrackingError("tracking mode must be 'mint' or 'verify'")
    if require_history not in {"true", "false"}:
        raise TrackingError("tracking history requirement must be 'true' or 'false'")
    if mode == "mint":
        return "mint"
    if history:
        return "compare"
    if require_history == "true":
        raise TrackingError("verify mode requires non-empty canonical tracking history")
    return "skip"


def require_exact_recomputation(
    current: VN2TrackingRecord,
    history: Sequence[VN2TrackingRecord],
) -> TrackingComparison:
    """Require the latest historical record to compare at exact zero delta."""
    if not history:
        raise TrackingError("tracking verification requires non-empty canonical history")
    comparison = compare_tracking_records(current, history[-1])
    if not comparison.comparable:
        mismatches = ", ".join(comparison.mismatched_fields)
        raise TrackingError(f"latest tracking record is not GA1-comparable: {mismatches}")
    if comparison.total_cost_delta != 0.0:
        raise TrackingError(
            "recomputed objective does not exactly match the latest tracking record"
        )
    return comparison
