"""Run the full VN2 protocol and emit the self-validating R1-R4 bundle."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import pytest

from newcalibre.domain import GuaranteeDescriptor
from newcalibre.protocols.vn2 import (
    VN2ProtocolConfig,
    VN2RunResult,
    capture_vn2_evidence_environment,
    emit_vn2_result_bundle,
    load_vn2_config,
    load_vn2_dataset,
    run_vn2,
    validate_vn2_result_bundle,
    verify_vn2_inputs,
)

pytestmark = [
    pytest.mark.tier4,
    pytest.mark.skipif(
        sys.platform == "win32",
        reason="Gate-A VN2 evidence is ratified only on Ubuntu 24.04 x86_64",
    ),
]

PROJECT_ROOT = Path(__file__).parents[2]
REPOSITORY_ROOT = PROJECT_ROOT.parent
CONFIG_PATH = PROJECT_ROOT / "benchmarks" / "vn2" / "protocol.yaml"
INVENTORY_PATH = PROJECT_ROOT / "benchmarks" / "vn2" / "vn2-input-digests.json"
LOCK_PATH = PROJECT_ROOT / "uv.lock"
DATA_PATH = PROJECT_ROOT / "data" / "vn2"
BUNDLE_PATH = PROJECT_ROOT / "artifacts" / "vn2"
TRACKING_PATH = REPOSITORY_ROOT / "stage3" / "evidence" / "tracking" / "series.jsonl"


def test_full_vn2_run_emits_and_revalidates_exact_r1_r4_bundle() -> None:
    candidate_sha = _required_environment("VN2_CANDIDATE_SHA")
    workflow_sha = _required_environment("VN2_WORKFLOW_SHA")
    run_id = _required_environment("VN2_RUN_ID")
    run_url = _required_environment("VN2_RUN_URL")
    mode = _required_environment("VN2_MODE")
    assert mode in {"mint", "verify"}
    assert not BUNDLE_PATH.exists(), "Tier 4 requires a clean ignored artifact directory"

    tracking_before = _optional_digest(TRACKING_PATH)
    verify_vn2_inputs(DATA_PATH, INVENTORY_PATH)
    config = load_vn2_config(CONFIG_PATH)
    dataset = load_vn2_dataset(DATA_PATH, INVENTORY_PATH, config)
    result = run_vn2(dataset)

    assert len(result.series_identities) == 599
    assert len(result.orders) == 599 * 6
    assert len(result.settlements) == 599 * 8

    emitted = emit_vn2_result_bundle(
        BUNDLE_PATH,
        result=result,
        config=config,
        candidate_sha=candidate_sha,
        workflow_sha=workflow_sha,
        run_id=run_id,
        run_url=run_url,
        config_path=CONFIG_PATH,
        input_inventory_path=INVENTORY_PATH,
        lock_path=LOCK_PATH,
        environment=capture_vn2_evidence_environment(),
    )
    provenance_digest = emitted.manifest.provenance_digest
    assert (BUNDLE_PATH / "r1-orders.jsonl").read_bytes() == _expected_r1_jsonl(
        result,
        config,
        provenance_digest=provenance_digest,
    )
    assert (BUNDLE_PATH / "r2-cost-ledger.jsonl").read_bytes() == _expected_r2_jsonl(
        result,
        config,
        provenance_digest=provenance_digest,
    )
    validated = validate_vn2_result_bundle(
        BUNDLE_PATH,
        expected_candidate_sha=candidate_sha,
        expected_workflow_sha=workflow_sha,
        expected_run_id=run_id,
        expected_config_path=CONFIG_PATH,
        expected_input_inventory_path=INVENTORY_PATH,
        expected_lock_path=LOCK_PATH,
    )

    assert validated == emitted
    trajectory = json.loads((BUNDLE_PATH / "r4-cost-trajectory.json").read_text(encoding="utf-8"))
    assert [row["round"] for row in trajectory["decision_rounds"]] == list(range(1, 7))
    assert len(trajectory["drain_remainder"]["periods"]) == 2
    assert all("r5" not in path.name.casefold() for path in BUNDLE_PATH.rglob("*"))
    assert _optional_digest(TRACKING_PATH) == tracking_before


def _required_environment(name: str) -> str:
    value = os.environ.get(name)
    assert value, f"{name} must be set by the evidence workflow"
    return value


def _optional_digest(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def _expected_r1_jsonl(
    result: VN2RunResult,
    config: VN2ProtocolConfig,
    *,
    provenance_digest: str,
) -> bytes:
    identities = result.series_identities
    series_keys = tuple(sorted(identities, key=str.encode))
    orders = {(order.series_key, order.origin): order for order in result.orders}
    rows: list[dict[str, object]] = []
    for round_number, origin in enumerate(config.decision_origins, start=1):
        for series_key in series_keys:
            order = orders[(series_key, origin)]
            evidence = order.evidence
            assert evidence is not None
            store, product = identities[series_key]
            rows.append(
                {
                    "actuals_semantics": config.actuals_semantics.value,
                    "arrival_period": order.arrival_period.isoformat(),
                    "bindings": [
                        {"bound": binding.bound, "name": binding.name, "value": binding.value}
                        for binding in evidence.bindings
                    ],
                    "consumed_claim": evidence.effective_descriptor.type.claim.value,
                    "effective_descriptor": _expected_descriptor(evidence.effective_descriptor),
                    "model_name": order.model_name,
                    "origin": order.origin.isoformat(),
                    "product": product,
                    "provenance_digest": provenance_digest,
                    "quantity": order.quantity,
                    "raw_target": evidence.raw_target,
                    "reorder_point": evidence.reorder_point,
                    "round": round_number,
                    "schema": 1,
                    "series_key": series_key,
                    "session_id": result.session.value,
                    "source_columns": list(evidence.source_columns),
                    "source_descriptor": _expected_descriptor(evidence.source_descriptor),
                    "store": store,
                    "target": evidence.target,
                }
            )
    return _canonical_jsonl(rows)


def _expected_r2_jsonl(
    result: VN2RunResult,
    config: VN2ProtocolConfig,
    *,
    provenance_digest: str,
) -> bytes:
    identities = result.series_identities
    series_keys = tuple(sorted(identities, key=str.encode))
    settlements = {(record.series_key, record.period): record for record in result.settlements}
    rows: list[dict[str, object]] = []
    for period_index, period in enumerate(config.realized_periods, start=1):
        for series_key in series_keys:
            record = settlements[(series_key, period)]
            store, product = identities[series_key]
            rows.append(
                {
                    "actuals_semantics": record.actuals_semantics.value,
                    "arrivals": record.arrivals,
                    "closing_backorders": record.transition.closing_backorders,
                    "currency": config.currency,
                    "demand": record.transition.demand,
                    "end_inventory": record.transition.closing_on_hand,
                    "holding_basis": record.holding.basis,
                    "holding_cost": record.holding.amount,
                    "holding_rate": record.holding.rate,
                    "missed_sales": record.transition.unmet_demand,
                    "on_order": record.inventory_position.on_order,
                    "period": period.isoformat(),
                    "period_index": period_index,
                    "product": product,
                    "provenance_digest": provenance_digest,
                    "sales": record.transition.fulfilled_demand,
                    "schema": 1,
                    "series_key": series_key,
                    "session_id": result.session.value,
                    "shortage_basis": record.shortage.basis,
                    "shortage_cost": record.shortage.amount,
                    "shortage_rate": record.shortage.rate,
                    "start_inventory": record.transition.available_inventory,
                    "stockout_rule": record.transition.rule.value,
                    "store": store,
                }
            )
    return _canonical_jsonl(rows)


def _expected_descriptor(descriptor: GuaranteeDescriptor) -> dict[str, object]:
    return {
        "level": descriptor.level,
        "scope": {
            "class_system_name": descriptor.scope.class_system_name,
            "kind": descriptor.scope.kind.value,
        },
        "scored_series": descriptor.scored_series.value,
        "type": {
            "claim": descriptor.type.claim.value,
            "currency": (
                None if descriptor.type.currency is None else descriptor.type.currency.value
            ),
            "declared_slack": descriptor.type.declared_slack,
        },
        "window": descriptor.window.value,
    }


def _canonical_jsonl(rows: list[dict[str, object]]) -> bytes:
    return b"".join(
        (
            json.dumps(
                row,
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
        for row in rows
    )
