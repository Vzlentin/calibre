"""Test the compact VN2 workflows and trusted-base append check."""

from __future__ import annotations

import hashlib
from pathlib import Path

import yaml

ROOT = Path(__file__).parents[1]
REGRESSION = ROOT / ".github" / "workflows" / "newcalibre.yml"
APPEND_CHECK = ROOT / ".github" / "workflows" / "vn2-evidence.yml"
ATTRIBUTES = ROOT / ".gitattributes"
CAPTURE = ROOT / "stage3" / "evidence" / "captures" / "vn2"


def _workflow(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _runs(workflow: dict) -> str:
    return "\n".join(
        str(step.get("run", ""))
        for job in workflow["jobs"].values()
        for step in job.get("steps", [])
    )


def test_regression_workflow_has_only_pr_main_and_regression_lanes() -> None:
    """Keep Tier 0/1 on PR, Tier 2 on main, and one VN2 regression job."""
    workflow = _workflow(REGRESSION)

    assert set(workflow["jobs"]) == {
        "newcalibre-lint",
        "newcalibre-unit",
        "newcalibre-consistency",
        "vn2-regression",
    }
    assert workflow["permissions"] == {"contents": "read"}
    assert workflow["jobs"]["newcalibre-lint"]["if"] == "github.event_name == 'pull_request'"
    assert workflow["jobs"]["newcalibre-unit"]["if"] == "github.event_name == 'pull_request'"
    assert workflow["jobs"]["newcalibre-consistency"]["if"] == "github.event_name == 'push'"


def test_regression_uses_one_inventory_and_successor_verifier() -> None:
    """Key acquisition and verification directly on the canonical inventory."""
    text = REGRESSION.read_text(encoding="utf-8")
    runs = _runs(_workflow(REGRESSION))

    assert "hashFiles('newcalibre/benchmarks/vn2/vn2-input-digests.json')" in text
    assert "newcalibre/scripts/vn2_data.py download" in runs
    assert "newcalibre/scripts/vn2_data.py verify" in runs
    assert "stage3/evidence/" + "vn2-input-digests.json" not in text
    assert ".github/scripts/stage3_" + "vn2_data.py" not in text


def test_regression_runs_replay_witness_and_generic_acceptance() -> None:
    """Retain independent Tier 3 and generic-engine Tier 4 behavior."""
    runs = _runs(_workflow(REGRESSION))

    assert "pytest newcalibre/tests/tier3" in runs
    assert "pytest newcalibre/tests/tier4" in runs
    assert "vn2_tracking.py build" in runs
    assert "actions/upload-artifact@v4" in REGRESSION.read_text(encoding="utf-8")
    assert "gh pr" not in runs
    assert "git push" not in runs


def test_append_check_executes_only_trusted_base_code() -> None:
    """Fetch head only as Git data and invoke the base validator read-only."""
    workflow = _workflow(APPEND_CHECK)
    text = APPEND_CHECK.read_text(encoding="utf-8")
    job = workflow["jobs"]["validate-append"]
    steps = job["steps"]
    checkout = steps[0]
    bootstrap = next(step for step in steps if step.get("id") == "validator")
    validation = next(
        step for step in steps if step.get("name") == "Validate exact append with base code"
    )
    runs = _runs(workflow)

    assert workflow["permissions"] == {"contents": "read"}
    assert checkout["with"]["ref"] == "${{ github.event.pull_request.base.sha }}"
    assert checkout["with"]["persist-credentials"] is False
    assert 'git fetch --no-tags origin "$HEAD_SHA"' in runs
    assert 'git show "$HEAD_SHA:$tracking"' in runs
    assert "vn2_tracking.py validate-append --help" in bootstrap["run"]
    assert "available=false" in bootstrap["run"]
    assert validation["if"] == "steps.validator.outputs.available == 'true'"
    assert "vn2_tracking.py validate-append" in validation["run"]
    assert "pull_request" + "_target" not in text
    assert "actions/download-artifact" not in text
    assert "secrets." not in text
    assert "write" not in str(workflow["permissions"]).lower()
    assert "git checkout" not in runs


def test_capture_is_exactly_manifest_plus_six_unchanged_orders() -> None:
    """Keep the six canonical order payload bytes under one compact manifest."""
    files = {path.relative_to(CAPTURE).as_posix() for path in CAPTURE.rglob("*") if path.is_file()}
    assert files == {
        "manifest.json",
        *(f"orders/round-{round_number}.json" for round_number in range(1, 7)),
    }
    expected = (
        "95484ebab662b7409a88b4c4a0beb676ef29f5710db68c7e70d92de47344544e",
        "c15c986c3bd248bcf29d5c36276949b8e3528ece936ec5ed43b106bdaabbb3bb",
        "2a6b15bbed2a1320050b02d1f950800cae3ccb85f637e018facbee6351de682d",
        "2583264e3f69fc7650de548970105ffaf34d566ec555b39819effb84b8f5cb46",
        "7c46d3519cafb5a3c59f0a619d5cfb83c4307a3545b0ed8aa075904bce825c91",
        "2b904b1f34d377808adffbff079880b32de06edd3d30178f628eaa993870020a",
    )
    actual = tuple(
        hashlib.sha256((CAPTURE / "orders" / f"round-{number}.json").read_bytes()).hexdigest()
        for number in range(1, 7)
    )
    assert actual == expected


def test_attributes_cover_only_live_durable_evidence() -> None:
    """Preserve compact evidence bytes without the deleted duplicate inventory."""
    text = ATTRIBUTES.read_text(encoding="utf-8")

    assert "stage3/evidence/captures/** -text" in text
    assert "stage3/evidence/tracking/** -text" in text
    assert "newcalibre/benchmarks/vn2/vn2-input-digests.json -text" in text
    assert "stage3/evidence/" + "vn2-input-digests.json" not in text


def test_deleted_privileged_automation_is_absent() -> None:
    """Do not retain promotion, aggregation, custody, or bootstrap entry points."""
    deleted = (
        ".github/workflows/gate-" + "a.yml",
        ".github/workflows/oracle-" + "capture.yml",
        ".github/workflows/s3-activation-" + "gate.yml",
        ".github/workflows/stage3-" + "bootstrap.yml",
        ".github/scripts/stage3_" + "gate_report.py",
        ".github/scripts/stage3_tracking_" + "admission.py",
        "tests/test_stage3_tracking_" + "admission.py",
        "stage3/evidence/" + "vn2-input-digests.json",
    )

    assert all(not (ROOT / path).exists() for path in deleted)
