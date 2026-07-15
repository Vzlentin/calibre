"""Exercise the strict successor VN2 tracking record contract."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from newcalibre.domain._canonical_json import canonical_json_bytes
from newcalibre.protocols.vn2.tracking import (
    TRACKING_KIND,
    TRACKING_SCHEMA,
    TrackingError,
    VN2TrackingRecord,
    compare_tracking_records,
    decide_append,
    parse_tracking_history,
    parse_tracking_record,
    write_tracking_record,
)

pytestmark = pytest.mark.tier1

CANDIDATE = "a" * 40
WORKFLOW = "b" * 40
RUN_ID = "123456"
DIGEST = "1" * 64


def _environment() -> dict[str, object]:
    return {
        "arch": "x86_64",
        "cpu_model": "fixture cpu",
        "numpy": "2.3.1",
        "numpy_config": "OpenBLAS fixture",
        "os": {"id": "ubuntu", "pretty_name": "Ubuntu 24.04", "version_id": "24.04"},
        "python": "3.12.13",
        "runner_image": "ubuntu24/20260701.1",
        "thread_policy": {"OMP_NUM_THREADS": "1"},
    }


def _record(*, total_cost: float = 3.0, candidate: str = CANDIDATE) -> VN2TrackingRecord:
    environment = _environment()
    environment_digest = hashlib.sha256(
        canonical_json_bytes(environment, path="environment.facts")
    ).hexdigest()
    toolchain_digest = hashlib.sha256(
        canonical_json_bytes(
            {
                "numpy": environment["numpy"],
                "numpy_config": environment["numpy_config"],
                "python": environment["python"],
                "schema": 1,
            },
            path="tracking toolchain",
        )
    ).hexdigest()
    config_digest = "2" * 64
    input_digest = "3" * 64
    identity = hashlib.sha256(
        canonical_json_bytes(
            {
                "artifact_kind": "vn2-gate-a-results",
                "candidate_sha": candidate,
                "config_digest": config_digest,
                "environment_digest": environment_digest,
                "input_inventory_digest": input_digest,
            },
            path="tracking identity",
        )
    ).hexdigest()
    payload = {
        "schema": TRACKING_SCHEMA,
        "record_kind": TRACKING_KIND,
        "identity": identity,
        "subject": {"repository": "Vzlentin/calibre", "candidate_sha": candidate},
        "workflow": {
            "definition_ref": "Vzlentin/calibre/.github/workflows/newcalibre.yml@main",
            "definition_sha": WORKFLOW,
            "run_id": RUN_ID,
            "run_url": f"https://github.com/Vzlentin/calibre/actions/runs/{RUN_ID}",
        },
        "result_artifact": {
            "id": "789012",
            "name": f"vn2-acceptance-{candidate}",
            "digest": DIGEST,
        },
        "result_bundle": {
            "artifact_kind": "vn2-gate-a-results",
            "manifest_sha256": DIGEST,
            "inner_bundle_digest": DIGEST,
            "provenance_digest": DIGEST,
            "files": {
                "environment.json": DIGEST,
                "r1-orders.jsonl": DIGEST,
                "r2-cost-ledger.jsonl": DIGEST,
                "r3-final-triple.json": DIGEST,
                "r4-cost-trajectory.json": DIGEST,
            },
        },
        "evidence": {
            "config": {"path": "benchmarks/vn2/protocol.yaml", "digest": config_digest},
            "input_inventory": {
                "path": "benchmarks/vn2/vn2-input-digests.json",
                "digest": input_digest,
            },
            "lockfile": {"path": "uv.lock", "digest": "4" * 64},
            "promoted_capture": {
                "artifact_id": "789013",
                "artifact_digest": DIGEST,
                "artifact_name": f"oracle-capture-{candidate}",
                "capture_digest": "5" * 64,
                "manifest_sha256": DIGEST,
                "inner_bundle_digest": DIGEST,
                "environment_digest": DIGEST,
                "producer_sha": candidate,
                "workflow_sha": WORKFLOW,
                "workflow_run_id": RUN_ID,
                "run_url": f"https://github.com/Vzlentin/calibre/actions/runs/{RUN_ID}",
            },
            "actuals_semantics": "censored_sales_surrogate",
            "session": {"id": DIGEST, "series_count": 1, "series_identity_digest": DIGEST},
        },
        "environment": {
            "facts": environment,
            "digest": environment_digest,
            "toolchain_digest": toolchain_digest,
        },
        "objective": {
            "holding_cost": 1.0,
            "shortage_cost": total_cost - 1.0,
            "total_cost": total_cost,
        },
    }
    return VN2TrackingRecord(payload)


def test_canonical_record_round_trip_and_history_duplicate_refusal() -> None:
    record = _record()
    assert parse_tracking_record(record.to_bytes()).to_bytes() == record.to_bytes()
    with pytest.raises(TrackingError, match="duplicate identity"):
        parse_tracking_history(record.to_bytes() + record.to_bytes())


def test_comparison_is_informational_and_exact_key_mismatches_have_no_delta() -> None:
    current = _record(total_cost=4.0)
    prior = _record(total_cost=3.0)
    comparison = compare_tracking_records(current, prior)
    assert comparison.comparable
    changed = _record(total_cost=4.0, candidate="c" * 40)
    changed_comparison = compare_tracking_records(changed, prior)
    assert changed_comparison.comparable
    assert changed_comparison.total_cost_delta == 1.0


def test_append_idempotency_and_atomic_writer(tmp_path: Path) -> None:
    record = _record()
    assert decide_append(record, ()).action == "append"
    assert decide_append(record, (record,)).action == "noop"
    path = tmp_path / "proposal.jsonl"
    assert write_tracking_record(record, path)
    assert not write_tracking_record(record, path)
    with pytest.raises(TrackingError, match="conflicts"):
        write_tracking_record(_record(total_cost=4.0), path)


def test_strict_codec_refuses_pretty_json_and_crlf() -> None:
    record = _record()
    body = record.to_bytes()[:-1]
    assert parse_tracking_record(body + b"\n").to_bytes() == record.to_bytes()
    with pytest.raises(TrackingError, match="canonical"):
        parse_tracking_record((record.to_json().rstrip("\n").replace(":", ": ") + "\n").encode())
    with pytest.raises(TrackingError, match="LF"):
        parse_tracking_record(body + b"\r\n")
