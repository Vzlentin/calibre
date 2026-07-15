"""Exercise the strict successor VN2 tracking record contract."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pytest

from newcalibre.domain._canonical_json import canonical_json_bytes
from newcalibre.protocols.vn2 import VN2ResultError, build_tracking_record
from newcalibre.protocols.vn2.tracking import (
    TRACKING_KIND,
    TRACKING_SCHEMA,
    TrackingError,
    VN2TrackingRecord,
    compare_tracking_records,
    decide_append,
    parse_tracking_history,
    parse_tracking_record,
    write_proposal_record,
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
        "thread_policy": {
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
        },
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


def _json_payload(record: VN2TrackingRecord) -> dict[str, object]:
    return json.loads(record.to_json())


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
    root = tmp_path / "newcalibre"
    (root / "artifacts").mkdir(parents=True)
    (root / "pyproject.toml").write_text("[project]\nname = 'newcalibre'\n")
    path = root / "artifacts" / "proposal.jsonl"
    assert write_proposal_record(record, path)
    assert not write_proposal_record(record, path)
    with pytest.raises(TrackingError, match="conflicts"):
        write_proposal_record(_record(total_cost=4.0), path)


def test_strict_codec_refuses_pretty_json_and_crlf() -> None:
    record = _record()
    body = record.to_bytes()[:-1]
    assert parse_tracking_record(body + b"\n").to_bytes() == record.to_bytes()
    with pytest.raises(TrackingError, match="canonical"):
        parse_tracking_record((record.to_json().rstrip("\n").replace(":", ": ") + "\n").encode())
    with pytest.raises(TrackingError, match="LF"):
        parse_tracking_record(body + b"\r\n")


def test_tracking_record_owns_nested_values_and_exposes_no_aliases() -> None:
    payload = _json_payload(_record())
    record = VN2TrackingRecord(payload)
    payload["subject"]["repository"] = "evil"  # type: ignore[index]
    assert record.payload["subject"]["repository"] == "Vzlentin/calibre"  # type: ignore[index]
    with pytest.raises(TypeError):
        record.payload["subject"]["repository"] = "evil"  # type: ignore[index]
    import newcalibre.protocols.vn2.tracking as tracking

    assert not hasattr(tracking, "TrackingRecord")
    assert not hasattr(tracking, "append_decision")
    assert not hasattr(tracking, "build_proposed_record")
    assert not hasattr(tracking, "compare_records")
    assert not hasattr(tracking, "write_tracking_record")


@pytest.mark.parametrize(
    "payload",
    [
        b"",
        b"{}\n\n",
        b"\xef\xbb\xbf{}\n",
        b"\xff{}\n",
        b"{}",
    ],
)
def test_tracking_codec_rejects_malformed_line_shapes(payload: bytes) -> None:
    with pytest.raises(TrackingError):
        parse_tracking_record(payload)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.pop("subject"),
        lambda value: value.__setitem__("unknown", None),
        lambda value: value["environment"].__setitem__("facts", []),
        lambda value: value["objective"].__setitem__("total_cost", math.nan),
        lambda value: value["evidence"]["promoted_capture"].__setitem__(
            "artifact_name", "oracle-capture-" + "c" * 40
        ),
        lambda value: value["workflow"].__setitem__(
            "definition_ref", "evil.example/workflow.yml@main"
        ),
    ],
)
def test_tracking_constructor_refuses_schema_and_fact_corruption(mutation) -> None:
    payload = _json_payload(_record())
    mutation(payload)
    with pytest.raises(TrackingError):
        VN2TrackingRecord(payload)


def test_tracking_history_and_publication_paths_are_successor_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "newcalibre"
    (root / "artifacts").mkdir(parents=True)
    (root / "pyproject.toml").write_text("[project]\nname = 'newcalibre'\n")
    monkeypatch.chdir(root)
    record = _record()
    relative = Path("artifacts") / "nested" / "proposal.jsonl"
    assert write_proposal_record(record, relative)
    with pytest.raises(TrackingError):
        write_proposal_record(record, tmp_path / "outside" / "proposal.jsonl")
    with pytest.raises(TrackingError, match="tracked history"):
        write_proposal_record(record, root / "stage3" / "tracking" / "series.jsonl")


def test_builder_translates_domain_validator_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    import newcalibre.protocols.vn2._tracking_projection as projection

    def fail(*args: object, **kwargs: object) -> object:
        raise VN2ResultError("fixture validator failure")

    monkeypatch.setattr(projection, "validate_vn2_result_bundle", fail)
    with pytest.raises(TrackingError) as caught:
        build_tracking_record(
            Path("result"),
            Path("captures"),
            candidate_sha=CANDIDATE,
            definition_ref="Vzlentin/calibre/.github/workflows/newcalibre.yml@main",
            definition_sha=WORKFLOW,
            run_id=RUN_ID,
            run_url=f"https://github.com/Vzlentin/calibre/actions/runs/{RUN_ID}",
            result_artifact_id="789012",
            result_artifact_name=f"vn2-acceptance-{CANDIDATE}",
            result_artifact_digest=DIGEST,
        )
    assert isinstance(caught.value.__cause__, VN2ResultError)
