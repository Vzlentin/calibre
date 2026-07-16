"""Exercise the Stage 3 VN2 and oracle workflow contracts."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import textwrap
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path

import pytest
import yaml

from tests.infra import load_script_module

SCRIPTS_DIR = Path(__file__).parents[1] / ".github" / "scripts"
GATE_WORKFLOW = Path(__file__).parents[1] / ".github" / "workflows" / "gate-a.yml"
SUCCESSOR_WORKFLOW = Path(__file__).parents[1] / ".github" / "workflows" / "newcalibre.yml"
ACTIVATION_WORKFLOW = Path(__file__).parents[1] / ".github" / "workflows" / "s3-activation-gate.yml"
BOOTSTRAP_VN2_INVENTORY = (
    Path(__file__).parents[1] / "stage3" / "evidence" / "vn2-input-digests.json"
)
SUCCESSOR_VN2_INVENTORY = (
    Path(__file__).parents[1] / "newcalibre" / "benchmarks" / "vn2" / "vn2-input-digests.json"
)
REPO_ROOT = Path(__file__).parents[1]
TIER3_ORACLE_INVENTORY = REPO_ROOT / "newcalibre" / "tests" / "tier3" / "oracle_inventory.json"
REQUIRED_TIER3_OUTCOMES = json.loads(TIER3_ORACLE_INVENTORY.read_text(encoding="utf-8"))["named"]
GIT_ATTRIBUTES = REPO_ROOT / ".gitattributes"
VN2_THREAD_ENV = {
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}
VN2_CACHE_KEY = "stage3-vn2-${{ hashFiles('stage3/evidence/vn2-input-digests.json') }}"
VN2_ACQUIRE = (
    "uv run --no-project python .github/scripts/stage3_vn2_data.py download "
    "--target newcalibre/data/vn2 --if-missing"
)
VN2_VERIFY = (
    "uv sync --project newcalibre --locked --group dev\n"
    "uv run --project newcalibre --locked --no-sync python "
    "newcalibre/scripts/vn2_data.py verify "
    "--target newcalibre/data/vn2 \\\n"
    "  --inventory newcalibre/benchmarks/vn2/vn2-input-digests.json"
)
DIGEST_BOUND_ATTRIBUTE_EXAMPLES = {
    "stage3/evidence/captures/** -text": (
        "stage3/evidence/captures/ba45e9463e6b9d2921ca0d9e9692d2645a228058/files.sha256"
    ),
    "stage3/evidence/vn2-input-digests.json -text": ("stage3/evidence/vn2-input-digests.json"),
    "stage3/evidence/tracking/** -text": ("stage3/evidence/tracking/series.jsonl"),
    "newcalibre/benchmarks/vn2/vn2-input-digests.json -text": (
        "newcalibre/benchmarks/vn2/vn2-input-digests.json"
    ),
    "benchmarks/vn2/config/vn2-winning-loop.yaml -text": (
        "benchmarks/vn2/config/vn2-winning-loop.yaml"
    ),
    "newcalibre/benchmarks/vn2/protocol.yaml -text": ("newcalibre/benchmarks/vn2/protocol.yaml"),
}
stage3_clock = load_script_module(SCRIPTS_DIR / "stage3_clock.py")
stage3_gate_report = load_script_module(SCRIPTS_DIR / "stage3_gate_report.py")
stage3_oracle_evidence = load_script_module(SCRIPTS_DIR / "stage3_oracle_evidence.py")


def _write_report_tracking_evidence(root: Path, candidate: str) -> tuple[Path, Path]:
    root.mkdir(parents=True, exist_ok=True)

    def canonical(value: object) -> bytes:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()

    workflow = {
        "definition_ref": ("Vzlentin/calibre/.github/workflows/newcalibre.yml@refs/heads/main"),
        "definition_sha": candidate,
        "run_id": "123456",
        "run_url": "https://github.com/Vzlentin/calibre/actions/runs/123456",
    }
    result_artifact = {
        "digest": "d" * 64,
        "id": "789012",
        "name": f"vn2-acceptance-{candidate}",
    }
    environment_facts = {
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
    config_digest = "2" * 64
    input_digest = "3" * 64
    lock_digest = "4" * 64
    capture_digest = "5" * 64
    environment_digest = hashlib.sha256(
        canonical(
            {
                "actuals_semantics": "censored_sales_surrogate",
                "architecture": environment_facts["arch"],
                "config_digest": config_digest,
                "input_inventory_digest": input_digest,
                "lockfile_digest": lock_digest,
                "os_id": "ubuntu",
                "os_version": "24.04",
                "promoted_capture_digest": capture_digest,
            }
        )
    ).hexdigest()
    toolchain_digest = hashlib.sha256(
        canonical(
            {
                "numpy": environment_facts["numpy"],
                "numpy_config": environment_facts["numpy_config"],
                "python": environment_facts["python"],
                "schema": 1,
            }
        )
    ).hexdigest()
    identity = hashlib.sha256(
        canonical(
            {
                "artifact_kind": "vn2-gate-a-results",
                "candidate_sha": candidate,
                "config_digest": config_digest,
                "environment_digest": environment_digest,
                "input_inventory_digest": input_digest,
            }
        )
    ).hexdigest()
    record = {
        "environment": {
            "digest": environment_digest,
            "facts": environment_facts,
            "toolchain_digest": toolchain_digest,
        },
        "evidence": {
            "actuals_semantics": "censored_sales_surrogate",
            "config": {"digest": config_digest, "path": "benchmarks/vn2/protocol.yaml"},
            "input_inventory": {
                "digest": input_digest,
                "path": "benchmarks/vn2/vn2-input-digests.json",
            },
            "lockfile": {"digest": lock_digest, "path": "uv.lock"},
            "promoted_capture": {
                "artifact_digest": "1" * 64,
                "artifact_id": "789013",
                "artifact_name": f"oracle-capture-{candidate}",
                "capture_digest": capture_digest,
                "environment_digest": "1" * 64,
                "inner_bundle_digest": "1" * 64,
                "manifest_sha256": "1" * 64,
                "producer_sha": candidate,
                "run_url": "https://github.com/Vzlentin/calibre/actions/runs/123456",
                "workflow_run_id": "123456",
                "workflow_sha": candidate,
            },
            "session": {
                "id": "1" * 64,
                "series_count": 1,
                "series_identity_digest": "1" * 64,
            },
        },
        "identity": identity,
        "objective": {
            "holding_cost": 1.0,
            "shortage_cost": 2.0,
            "total_cost": 3.0,
        },
        "record_kind": "vn2-gate-a-tracking-record",
        "result_artifact": result_artifact,
        "result_bundle": {
            "artifact_kind": "vn2-gate-a-results",
            "files": {
                "r1-orders.jsonl": "1" * 64,
                "r2-cost-ledger.jsonl": "1" * 64,
                "r3-final-triple.json": "1" * 64,
                "r4-cost-trajectory.json": "1" * 64,
            },
            "inner_bundle_digest": "1" * 64,
            "manifest_sha256": "1" * 64,
            "provenance_digest": "1" * 64,
        },
        "schema": 1,
        "subject": {
            "candidate_sha": candidate,
            "repository": "Vzlentin/calibre",
        },
        "workflow": workflow,
    }
    record_bytes = (
        json.dumps(
            record,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
        + b"\n"
    )
    series = root / "series.jsonl"
    series.write_bytes(record_bytes)
    receipt = {
        "candidate_sha": candidate,
        "proposal_artifact": {
            "digest": "f" * 64,
            "id": "789013",
            "name": f"vn2-tracking-proposal-{candidate}",
        },
        "receipt_kind": "vn2-tracking-promotion-receipt",
        "record_sha256": hashlib.sha256(record_bytes).hexdigest(),
        "repository": "Vzlentin/calibre",
        "result_artifact": result_artifact,
        "schema": 1,
        "workflow": workflow,
    }
    receipt_path = root / f"{candidate}-receipt.json"
    receipt_path.write_text(
        json.dumps(
            receipt,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return series, receipt_path


def test_tracking_activation_gate_is_base_controlled_before_mint() -> None:
    assert ACTIVATION_WORKFLOW.is_file()
    workflow_text = ACTIVATION_WORKFLOW.read_text(encoding="utf-8")
    workflow = yaml.safe_load(workflow_text)
    triggers = workflow[True]
    assert set(triggers) == {"pull_request_target"}
    trigger = triggers["pull_request_target"]
    assert trigger["branches"] == ["main"]
    assert trigger["types"] == ["opened", "synchronize", "reopened"]
    assert "paths" not in trigger
    assert workflow["permissions"] == {
        "actions": "read",
        "contents": "read",
        "issues": "read",
        "pull-requests": "read",
    }
    assert set(workflow["jobs"]) == {"s3-activation-gate"}
    job = workflow["jobs"]["s3-activation-gate"]
    assert "if" not in job
    checkout = job["steps"][0]
    assert re.fullmatch(r"actions/checkout@[0-9a-f]{40}", checkout["uses"])
    assert checkout["with"] == {
        "ref": "${{ github.event.pull_request.base.sha }}",
        "fetch-depth": 1,
        "persist-credentials": False,
    }
    assert "github.event.pull_request.head.sha" not in json.dumps(checkout)
    fetch = next(
        step for step in job["steps"] if step.get("name") == "Fetch PR head as untrusted Git data"
    )
    assert '"refs/pull/$PR_NUMBER/head"' in fetch["run"]
    assert 'test "$(git rev-parse FETCH_HEAD)" = "$HEAD_SHA"' in fetch["run"]
    setup = next(step for step in job["steps"] if "setup-uv@" in step.get("uses", ""))
    assert re.fullmatch(r"astral-sh/setup-uv@[0-9a-f]{40}", setup["uses"])
    tracking_change = next(
        step for step in job["steps"] if step.get("name") == "Detect tracking tree changes"
    )
    assert 'git diff --quiet --no-renames "$BASE_SHA" "$HEAD_SHA"' in tracking_change["run"]
    promotion = next(
        step
        for step in job["steps"]
        if step.get("name") == "Inspect and confine tracking promotion"
    )
    assert ".github/scripts/stage3_tracking_admission.py inspect" in promotion["run"]
    assert promotion["if"] == "steps.tracking-change.outputs.present == 'true'"
    assert "python3 - <<" not in workflow_text
    download = next(
        step
        for step in job["steps"]
        if step.get("name") == "Download live same-run artifact metadata and archives"
    )
    assert ".github/scripts/stage3_tracking_admission.py admit" in download["run"]
    assert '--result-destination "$RUNNER_TEMP/result"' in download["run"]
    assert '--proposal-destination "$RUNNER_TEMP/proposal"' in download["run"]
    assert "unzip " not in download["run"]
    validator = next(
        step
        for step in job["steps"]
        if step.get("name") == "Validate live artifacts and exact append bytes"
    )
    assert "newcalibre/scripts/vn2_tracking.py promote" in validator["run"]
    assert '--default-branch-sha "$DEFAULT_SHA"' in validator["run"]
    assert (
        "s3-activation-gate"
        not in yaml.safe_load(SUCCESSOR_WORKFLOW.read_text(encoding="utf-8"))["jobs"]
    )


def _copy_evidence_scripts(root: Path) -> tuple[Path, Path]:
    script_dir = root / ".github" / "scripts"
    script_dir.mkdir(parents=True)
    oracle_script = script_dir / "stage3_oracle_evidence.py"
    gate_script = script_dir / "stage3_gate_report.py"
    shutil.copy2(SCRIPTS_DIR / oracle_script.name, oracle_script)
    shutil.copy2(SCRIPTS_DIR / gate_script.name, gate_script)
    shutil.copy2(SCRIPTS_DIR / "stage3_clock.py", script_dir / "stage3_clock.py")
    return oracle_script, gate_script


def _copied_gate_subprocess_env() -> dict[str, str]:
    env = {**os.environ}
    successor_src = str(REPO_ROOT / "newcalibre" / "src")
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        os.pathsep.join((successor_src, existing_pythonpath))
        if existing_pythonpath
        else successor_src
    )
    return env


def _write_named_junit(path: Path, *, skipped: bool = False) -> None:
    suite = ET.Element(
        "testsuite",
        {
            "name": "tier3",
            "tests": "2",
            "failures": "0",
            "errors": "0",
            "skipped": "2" if skipped else "0",
        },
    )
    for required in REQUIRED_TIER3_OUTCOMES.values():
        classname, name = required["node"].rsplit("::", 1)
        case = ET.SubElement(suite, "testcase", {"classname": classname, "name": name})
        if skipped:
            ET.SubElement(case, "skipped", {"message": "promoted captures absent"})
    ET.ElementTree(suite).write(path, encoding="utf-8", xml_declaration=True)


def _prepare_emitter_workspace(root: Path, *, include_captures: bool) -> Path:
    workspace = root / "emitter-workspace"
    required_files = (
        "benchmarks/vn2/config/vn2-winning-loop.yaml",
        "newcalibre/benchmarks/vn2/protocol.yaml",
        "newcalibre/benchmarks/vn2/vn2-input-digests.json",
        "newcalibre/uv.lock",
        "stage3/evidence/vn2-input-digests.json",
    )
    for relative in required_files:
        target = workspace / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO_ROOT / relative, target)
    if include_captures:
        captures = workspace / "stage3" / "evidence" / "captures"
        shutil.copytree(REPO_ROOT / "stage3" / "evidence" / "captures", captures)
    return workspace


def _write_oracle_environment(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "arch": "x86_64",
                "cpu_model": "Evidence CPU",
                "numpy": "2.4.6",
                "numpy_config": "OpenBLAS 0.3.31",
                "os": {
                    "id": "ubuntu",
                    "pretty_name": "Ubuntu 24.04.4 LTS",
                    "version_id": "24.04",
                },
                "python": "3.12.12",
                "runner_image": "ubuntu24/20260705.1",
                "thread_policy": {name: "1" for name in stage3_oracle_evidence.THREAD_VARIABLES},
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_tracking_verify_and_gate_candidate_modes_are_explicit() -> None:
    gate = yaml.safe_load(GATE_WORKFLOW.read_text(encoding="utf-8"))
    triggers = gate[True]
    assert set(triggers) == {"pull_request_target", "workflow_dispatch"}
    assert triggers["pull_request_target"] == {
        "types": ["closed"],
        "branches": ["main"],
        "paths": [
            "stage3/evidence/tracking/series.jsonl",
            "stage3/evidence/tracking/*-receipt.json",
        ],
    }
    event_candidate = (
        "${{ github.event_name == 'pull_request_target' && "
        "github.event.pull_request.merge_commit_sha || inputs.candidate_sha }}"
    )
    assert gate["env"]["GATE_CANDIDATE_SHA"] == event_candidate
    merged_only = (
        "github.event_name != 'pull_request_target' || github.event.pull_request.merged == true"
    )
    for name in ("lint", "unit-isolated", "consistency", "oracle", "vn2-verify"):
        assert gate["jobs"][name]["if"] == merged_only
    assert gate["jobs"]["aggregate"]["if"] == f"always() && ({merged_only})"
    assert gate["jobs"]["vn2-verify"]["env"]["VN2_REQUIRE_HISTORY"] == "true"

    successor = yaml.safe_load(SUCCESSOR_WORKFLOW.read_text(encoding="utf-8"))
    vn2 = successor["jobs"]["vn2-acceptance"]
    assert (
        vn2["env"]["VN2_REQUIRE_HISTORY"]
        == "${{ github.event_name == 'schedule' && 'false' || 'true' }}"
    )
    steps = vn2["steps"]
    snapshot = next(step for step in steps if step.get("name") == "Snapshot tracked history state")
    tier4 = next(step for step in steps if step.get("name") == "Tier 4 run")
    guard = next(step for step in steps if step.get("name") == "Final tracked history guard")
    assert steps.index(snapshot) < steps.index(tier4) < steps.index(guard)
    assert "sha256sum" in snapshot["run"]
    assert 'test "$INITIAL_STATE" = present' in guard["run"]
    assert 'test "$INITIAL_STATE" = absent' in guard["run"]


@pytest.mark.parametrize(
    ("workflow_path", "job_candidate", "checkout_candidate"),
    [
        (
            SUCCESSOR_WORKFLOW,
            "${{ github.event_name == 'workflow_dispatch' && inputs.candidate_sha || github.sha }}",
            "${{ github.event_name == 'workflow_dispatch' && inputs.candidate_sha || github.sha }}",
        ),
        (
            GATE_WORKFLOW,
            "${{ github.event_name == 'pull_request_target' && "
            "github.event.pull_request.merge_commit_sha || inputs.candidate_sha }}",
            "${{ env.GATE_CANDIDATE_SHA }}",
        ),
    ],
)
def test_oracle_workflows_bind_exact_candidate_environment_and_vn2_inputs(
    workflow_path: Path,
    job_candidate: str,
    checkout_candidate: str,
) -> None:
    workflow = yaml.safe_load(workflow_path.read_text(encoding="utf-8"))
    job = workflow["jobs"]["oracle"]
    steps = job["steps"]
    checkout = next(step for step in steps if step.get("uses") == "actions/checkout@v4")
    checked_out_head = next(
        step for step in steps if step.get("name") == "Verify checked-out HEAD equals candidate"
    )
    harness = next(
        step for step in steps if step.get("name") == "Require successor + Tier 3 harness"
    )
    setup = next(step for step in steps if step.get("uses") == "astral-sh/setup-uv@v4")
    preflight = next(
        step for step in steps if step.get("name") == "Preflight - evidence environment (ADR 0001)"
    )
    cache = next(step for step in steps if step.get("uses") == "actions/cache@v4")
    acquire = next(
        step for step in steps if step.get("name") == "Acquire VN2 inputs with bootstrap tooling"
    )
    verify = next(
        step for step in steps if step.get("name") == "Verify VN2 inputs with successor tooling"
    )
    provenance = next(step for step in steps if step.get("name") == "Record numerical provenance")
    tier3 = next(step for step in steps if step.get("id") == "tier3")

    assert job["runs-on"] == "ubuntu-24.04"
    assert {key: job["env"][key] for key in VN2_THREAD_ENV} == VN2_THREAD_ENV
    assert job["env"]["ORACLE_CANDIDATE_SHA"] == job_candidate
    assert checkout["with"]["ref"] == checkout_candidate
    assert "if" not in checked_out_head
    assert checked_out_head["run"] == ('test "$(git rev-parse HEAD)" = "$ORACLE_CANDIDATE_SHA"')
    assert "if" not in harness
    assert "newcalibre/tests/tier3/oracle_inventory.json" in harness["run"]
    assert "newcalibre/tests/tier3/test_conditional_replay.py" in harness["run"]

    preflight_script = preflight["run"]
    for required in (
        'test "$ID" = "ubuntu"',
        'test "$VERSION_ID" = "24.04"',
        'test "$(uname -m)" = "x86_64"',
        'test "${ImageOS:-}" = "ubuntu24"',
        'test -n "${ImageVersion:-}"',
        "uv run --no-project --python 3.12 python - <<'PY'",
        "sys.version_info[:2] == (3, 12)",
        "sha256sum newcalibre/uv.lock",
    ):
        assert required in preflight_script
    assert "python3 - <<'PY'" not in preflight_script

    assert cache["with"]["path"] == "newcalibre/data/vn2"
    assert cache["with"]["key"] == VN2_CACHE_KEY
    assert "restore-keys" not in cache["with"]
    assert acquire["run"] == VN2_ACQUIRE
    assert verify["run"].strip() == VN2_VERIFY
    assert "numpy.__version__" in provenance["run"]
    assert "numpy.show_config()" in provenance["run"]
    assert 'Path("uv.lock").read_bytes()' in provenance["run"]
    assert "sha256" in provenance["run"]
    assert tier3["working-directory"] == "newcalibre"
    assert "uv run --locked --no-sync pytest tests/tier3" in tier3["run"]
    assert "oracle-pytest-${ORACLE_CANDIDATE_SHA}.xml" in tier3["run"]
    assert steps.index(setup) < steps.index(preflight)
    assert steps.index(cache) < steps.index(acquire) < steps.index(verify)
    assert steps.index(verify) < steps.index(provenance) < steps.index(tier3)


def test_scheduled_oracle_skips_only_absent_captures_and_manual_runs_fail_closed() -> None:
    workflow = yaml.safe_load(SUCCESSOR_WORKFLOW.read_text(encoding="utf-8"))
    job = workflow["jobs"]["oracle"]
    steps = job["steps"]
    harness = next(
        step for step in steps if step.get("name") == "Require successor + Tier 3 harness"
    )
    presence = next(step for step in steps if step.get("name") == "Promoted captures presence")
    skipped = next(step for step in steps if step.get("name") == "Tier 3 visibly skipped")
    manual_refusal = next(
        step for step in steps if step.get("name") == "Require promoted captures (manual runs)"
    )

    assert "if" not in harness
    assert "newcalibre/pyproject.toml" in harness["run"]
    assert "stage3/evidence/vn2-input-digests.json" in harness["run"]
    assert "newcalibre/benchmarks/vn2/vn2-input-digests.json" in harness["run"]
    assert "newcalibre/tests/tier3/oracle_inventory.json" in harness["run"]
    assert "newcalibre/tests/tier3/test_conditional_replay.py" in harness["run"]
    assert "test ! -L stage3/evidence/captures" in presence["run"]
    assert "symlinked promoted captures root refused" in presence["run"]
    assert "exit 1" in presence["run"]
    assert presence["run"].index("! -L") < presence["run"].index("-d")
    assert presence["run"].index("symlinked promoted captures root refused") < presence[
        "run"
    ].index("present=false")
    assert 'echo "present=true" >> "$GITHUB_OUTPUT"' in presence["run"]
    assert 'echo "present=false" >> "$GITHUB_OUTPUT"' in presence["run"]
    assert skipped["if"] == (
        "github.event_name == 'schedule' && steps.captures.outputs.present != 'true'"
    )
    assert skipped["run"] == 'echo "::notice::tier 3 skipped - promoted captures absent"'
    assert manual_refusal["if"] == (
        "github.event_name == 'workflow_dispatch' && steps.captures.outputs.present != 'true'"
    )
    assert "exit 1" in manual_refusal["run"]
    guarded = [
        step
        for step in steps
        if step.get("uses") in {"astral-sh/setup-uv@v4", "actions/cache@v4"}
        or step.get("name")
        in {
            "Preflight - evidence environment (ADR 0001)",
            "Acquire VN2 inputs with bootstrap tooling",
            "Verify VN2 inputs with successor tooling",
            "Record numerical provenance",
            "Tier 3 replay + divergence + witness enforcement",
        }
    ]
    assert guarded
    assert all(step["if"] == "steps.captures.outputs.present == 'true'" for step in guarded)


def test_gate_a_oracle_fails_closed_and_remains_an_aggregate_dependency() -> None:
    workflow = yaml.safe_load(GATE_WORKFLOW.read_text(encoding="utf-8"))
    oracle = workflow["jobs"]["oracle"]
    precondition = next(
        step
        for step in oracle["steps"]
        if step.get("name") == "Gate precondition - promoted captures present"
    )

    assert oracle["if"] == (
        "github.event_name != 'pull_request_target' || github.event.pull_request.merged == true"
    )
    assert "if" not in precondition
    assert "test ! -L stage3/evidence/captures" in precondition["run"]
    assert precondition["run"].index("test ! -L") < precondition["run"].index("test -d")
    assert "exit 1" in precondition["run"]
    assert "oracle" in workflow["jobs"]["aggregate"]["needs"]


def test_digest_bound_evidence_and_configs_are_never_text_normalized() -> None:
    rules = {
        line.strip()
        for line in GIT_ATTRIBUTES.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }

    assert rules == set(DIGEST_BOUND_ATTRIBUTE_EXAMPLES)

    for rule, evidence_path in DIGEST_BOUND_ATTRIBUTE_EXAMPLES.items():
        evidence = REPO_ROOT / evidence_path
        if os.path.lexists(evidence):
            assert evidence.is_file()
        else:
            assert rule == "stage3/evidence/tracking/** -text"
        checked = subprocess.run(
            ["git", "check-attr", "text", "--", evidence_path],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        assert checked.stdout.strip() == f"{evidence_path}: text: unset"


def test_vn2_acceptance_has_an_exact_successor_owned_consumption_boundary() -> None:
    workflow = yaml.safe_load(SUCCESSOR_WORKFLOW.read_text(encoding="utf-8"))
    triggers = workflow[True]  # PyYAML 1.1 parses the YAML key `on` as boolean true.
    job = workflow["jobs"]["vn2-acceptance"]
    steps = job["steps"]
    checkout = next(step for step in steps if step.get("uses") == "actions/checkout@v4")
    setup = next(step for step in steps if step.get("uses") == "astral-sh/setup-uv@v4")
    checked_out_head = next(
        step for step in steps if step.get("name") == "Verify checked-out HEAD equals candidate"
    )
    preflight = next(
        step for step in steps if step.get("name") == "Preflight — evidence environment (ADR 0001)"
    )
    harness = next(
        step for step in steps if step.get("name") == "Require successor + Tier 4 harness"
    )
    acquire = next(
        step for step in steps if step.get("name") == "Acquire VN2 inputs with bootstrap tooling"
    )
    verify = next(
        step for step in steps if step.get("name") == "Verify VN2 inputs with successor tooling"
    )
    provenance = next(step for step in steps if step.get("name") == "Record numerical provenance")
    tier4 = next(step for step in steps if step.get("name") == "Tier 4 run")
    mutation = next(
        step
        for step in steps
        if step.get("name") == "Refuse tracked checkout mutation before upload"
    )
    result_upload = next(step for step in steps if step.get("name") == "Upload digest-bound bundle")
    result_download = next(
        step
        for step in steps
        if step.get("name") == "Download immutable result bundle for proposal"
    )
    proposal = next(
        step
        for step in steps
        if step.get("name") == "Build proposal record from immutable result (manual vn2-mint only)"
    )
    revalidate = next(
        step
        for step in steps
        if step.get("name")
        == "Revalidate proposal record from immutable result (manual vn2-mint only)"
    )
    proposal_upload = next(
        step
        for step in steps
        if step.get("name") == "Upload proposal record (manual vn2-mint only)"
    )
    proposal_download = next(
        step
        for step in steps
        if step.get("name") == "Download immutable proposal record (manual vn2-mint only)"
    )
    uploaded_validation = next(
        step
        for step in steps
        if step.get("name") == "Validate uploaded proposal record (manual vn2-mint only)"
    )
    final_guard = next(step for step in steps if step.get("name") == "Final tracked history guard")

    assert workflow["permissions"] == {"contents": "read"}
    assert job["if"] == (
        "github.event_name == 'schedule' || "
        "(github.event_name == 'workflow_dispatch' && startsWith(inputs.lane, 'vn2'))"
    )
    assert triggers["schedule"] == [{"cron": "0 5 * * *"}]
    dispatch_inputs = triggers["workflow_dispatch"]["inputs"]
    assert dispatch_inputs["candidate_sha"]["required"] is True
    assert dispatch_inputs["lane"]["options"] == ["oracle", "vn2-verify", "vn2-mint"]
    assert checkout["with"]["ref"] == (
        "${{ github.event_name == 'workflow_dispatch' && inputs.candidate_sha || github.sha }}"
    )
    assert "if" not in checked_out_head
    assert checked_out_head["run"] == ('test "$(git rev-parse HEAD)" = "$VN2_CANDIDATE_SHA"')
    assert steps.index(setup) < steps.index(preflight)
    assert {key: job["env"][key] for key in VN2_THREAD_ENV} == VN2_THREAD_ENV
    assert job["env"]["VN2_CANDIDATE_SHA"] == (
        "${{ github.event_name == 'workflow_dispatch' && inputs.candidate_sha || github.sha }}"
    )
    assert job["env"]["VN2_WORKFLOW_SHA"] == "${{ github.workflow_sha }}"
    assert job["env"]["VN2_RUN_ID"] == "${{ github.run_id }}"
    assert job["env"]["VN2_RUN_URL"] == (
        "https://github.com/${{ github.repository }}/actions/runs/${{ github.run_id }}"
    )
    assert job["env"]["VN2_WORKFLOW_REF"] == "${{ github.workflow_ref }}"
    assert job["env"]["VN2_MODE"] == (
        "${{ github.event_name == 'workflow_dispatch' && inputs.lane == 'vn2-mint' "
        "&& 'mint' || 'verify' }}"
    )
    assert "github.workflow_sha" not in job["env"]["VN2_CANDIDATE_SHA"]
    assert "inputs.lane == 'vn2-mint'" in job["env"]["VN2_MODE"]
    preflight_script = preflight["run"]
    for required in (
        'test "$ID" = "ubuntu"',
        'test "$VERSION_ID" = "24.04"',
        'test "$(uname -m)" = "x86_64"',
        'test "${ImageOS:-}" = "ubuntu24"',
        'test -n "${ImageVersion:-}"',
        "uv run --no-project --python 3.12 python - <<'PY'",
        "sys.version_info[:2] == (3, 12)",
        "sha256sum newcalibre/uv.lock",
    ):
        assert required in preflight_script
    assert "python3 - <<'PY'" not in preflight_script
    assert harness["run"].strip() == (
        "set -euo pipefail\n"
        "test -f newcalibre/pyproject.toml \\\n"
        '  || { echo "missing successor project: newcalibre/pyproject.toml"; exit 1; }\n'
        "test -f stage3/evidence/vn2-input-digests.json \\\n"
        '  || { echo "missing VN2 digest inventory: '
        'stage3/evidence/vn2-input-digests.json"; exit 1; }\n'
        "test -f newcalibre/tests/tier4/test_vn2_acceptance.py \\\n"
        '  || { echo "missing Tier 4 acceptance test: '
        'newcalibre/tests/tier4/test_vn2_acceptance.py"; exit 1; }'
    )
    assert all(step.get("name") != "Tier 4 visibly skipped" for step in steps)
    assert acquire["run"] == VN2_ACQUIRE
    assert verify["run"].strip() == VN2_VERIFY
    assert "numpy.__version__" in provenance["run"]
    assert "numpy.show_config()" in provenance["run"]
    assert 'Path("uv.lock").read_bytes()' in provenance["run"]
    assert "sha256" in provenance["run"]

    assert result_upload["if"] == "github.event_name == 'workflow_dispatch'"
    assert result_upload["id"] == "result_upload"
    assert result_upload["uses"] == "actions/upload-artifact@v4"
    assert result_upload["with"] == {
        "name": "vn2-acceptance-${{ env.VN2_CANDIDATE_SHA }}",
        "path": "newcalibre/artifacts/vn2/",
        "if-no-files-found": "error",
    }
    assert result_download["if"] == "env.VN2_MODE == 'mint'"
    assert result_download["uses"] == "actions/download-artifact@v4"
    assert result_download["with"] == {
        "artifact-ids": "${{ steps.result_upload.outputs.artifact-id }}",
        "path": "${{ runner.temp }}/vn2-result-${{ env.VN2_CANDIDATE_SHA }}",
        "merge-multiple": True,
    }

    proposal_steps = [proposal, revalidate, proposal_upload, proposal_download, uploaded_validation]
    assert all(step["if"] == "env.VN2_MODE == 'mint'" for step in proposal_steps)
    assert proposal_upload["id"] == "proposal_upload"
    assert proposal_upload["uses"] == "actions/upload-artifact@v4"
    assert proposal_upload["with"] == {
        "name": "vn2-tracking-proposal-${{ env.VN2_CANDIDATE_SHA }}",
        "path": "newcalibre/artifacts/vn2-tracking/proposed-record.jsonl",
        "if-no-files-found": "error",
    }
    assert proposal_download["uses"] == "actions/download-artifact@v4"
    assert proposal_download["with"] == {
        "artifact-ids": "${{ steps.proposal_upload.outputs.artifact-id }}",
        "path": "${{ runner.temp }}/vn2-proposal-${{ env.VN2_CANDIDATE_SHA }}",
        "merge-multiple": True,
    }
    result_root = '--result-root "${RUNNER_TEMP}/vn2-result-${VN2_CANDIDATE_SHA}"'
    for step in (proposal, revalidate, uploaded_validation):
        assert result_root in step["run"]
        assert (
            '--result-artifact-id "${{ steps.result_upload.outputs.artifact-id }}"' in step["run"]
        )
        assert (
            '--result-artifact-digest "${{ steps.result_upload.outputs.artifact-digest }}"'
            in step["run"]
        )
        assert '--result-artifact-name "vn2-acceptance-$VN2_CANDIDATE_SHA"' in step["run"]
    assert "--proposal artifacts/vn2-tracking/proposed-record.jsonl" in revalidate["run"]
    assert (
        '--proposal "${RUNNER_TEMP}/vn2-proposal-${VN2_CANDIDATE_SHA}/proposed-record.jsonl"'
        in uploaded_validation["run"]
    )

    assert final_guard["if"] == "always()"
    final_script = final_guard["run"]
    assert "set -euo pipefail" in final_script
    assert 'test ! -L "$path"' in final_script
    assert 'test "$INITIAL_STATE" = present' in final_script
    assert 'test "$INITIAL_STATE" = absent' in final_script
    assert 'test "$(git rev-parse HEAD)" = "$VN2_CANDIDATE_SHA"' in final_script
    assert "git status --porcelain --untracked-files=no" in final_script
    status_command = "git status --porcelain --untracked-files=no"
    assert sum(status_command in step.get("run", "") for step in (mutation, final_guard)) == 2
    assert steps.index(tier4) < steps.index(mutation) < steps.index(result_upload)
    assert (
        steps.index(result_upload)
        < steps.index(result_download)
        < steps.index(proposal)
        < steps.index(revalidate)
        < steps.index(proposal_upload)
        < steps.index(proposal_download)
        < steps.index(uploaded_validation)
        < steps.index(final_guard)
    )
    assert all("git push" not in step.get("run", "") for step in steps)


def test_gate_a_vn2_verify_binds_one_candidate_and_the_evidence_environment() -> None:
    workflow = yaml.safe_load(GATE_WORKFLOW.read_text(encoding="utf-8"))
    job = workflow["jobs"]["vn2-verify"]
    steps = job["steps"]
    checkout = next(step for step in steps if step.get("uses") == "actions/checkout@v4")
    setup = next(step for step in steps if step.get("uses") == "astral-sh/setup-uv@v4")
    preflight = next(
        step for step in steps if step.get("name") == "Preflight — evidence environment (ADR 0001)"
    )
    provenance = next(step for step in steps if step.get("name") == "Record numerical provenance")
    tier4 = next(
        step
        for step in steps
        if step.get("name") == "Tier 4 verify-only at C1 (recompute and self-validate)"
    )
    mutation = next(
        step for step in steps if step.get("name") == "Refuse tracked checkout mutation"
    )

    assert checkout["with"]["ref"] == "${{ env.GATE_CANDIDATE_SHA }}"
    assert steps.index(setup) < steps.index(preflight)
    assert {key: job["env"][key] for key in VN2_THREAD_ENV} == VN2_THREAD_ENV
    assert job["env"]["VN2_CANDIDATE_SHA"] == (
        "${{ github.event_name == 'pull_request_target' && "
        "github.event.pull_request.merge_commit_sha || inputs.candidate_sha }}"
    )
    assert job["env"]["VN2_WORKFLOW_SHA"] == "${{ github.workflow_sha }}"
    assert job["env"]["VN2_RUN_ID"] == "${{ github.run_id }}"
    assert job["env"]["VN2_RUN_URL"] == (
        "https://github.com/${{ github.repository }}/actions/runs/${{ github.run_id }}"
    )
    assert job["env"]["VN2_MODE"] == "verify"
    assert job["env"]["VN2_REQUIRE_HISTORY"] == "true"
    assert "github.workflow_sha" not in job["env"]["VN2_CANDIDATE_SHA"]
    for required in (
        'test "$ID" = "ubuntu"',
        'test "$VERSION_ID" = "24.04"',
        'test "$(uname -m)" = "x86_64"',
        'test "${ImageOS:-}" = "ubuntu24"',
        'test -n "${ImageVersion:-}"',
        "uv run --no-project --python 3.12 python - <<'PY'",
        "sys.version_info[:2] == (3, 12)",
        "sha256sum newcalibre/uv.lock",
    ):
        assert required in preflight["run"]
    assert "python3 - <<'PY'" not in preflight["run"]
    assert "numpy.__version__" in provenance["run"]
    assert "numpy.show_config()" in provenance["run"]
    assert 'Path("uv.lock").read_bytes()' in provenance["run"]
    assert "sha256" in provenance["run"]
    assert steps.index(tier4) < steps.index(mutation)
    assert "git status --porcelain --untracked-files=no" in mutation["run"]


def test_successor_vn2_inventory_is_the_exact_bootstrap_approved_blob() -> None:
    assert SUCCESSOR_VN2_INVENTORY.read_bytes() == BOOTSTRAP_VN2_INVENTORY.read_bytes()
    inventory = json.loads(SUCCESSOR_VN2_INVENTORY.read_text(encoding="utf-8"))
    assert len(inventory["files"]) == 12


@pytest.mark.parametrize("workflow_path", [SUCCESSOR_WORKFLOW, GATE_WORKFLOW])
def test_oracle_jobs_always_publish_candidate_qualified_evidence(workflow_path: Path) -> None:
    workflow = yaml.safe_load(workflow_path.read_text(encoding="utf-8"))
    steps = workflow["jobs"]["oracle"]["steps"]
    emit = next(
        step for step in steps if step.get("name") == "Emit candidate-qualified oracle evidence"
    )
    validate = next(
        step for step in steps if step.get("name") == "Validate candidate-qualified oracle evidence"
    )
    upload = next(
        step for step in steps if step.get("name") == "Upload oracle evidence and pytest outcomes"
    )

    assert emit["if"] == "always()"
    assert emit["env"]["ORACLE_TEST_OUTCOME"] == "${{ steps.tier3.outcome }}"
    assert ".github/scripts/stage3_oracle_evidence.py emit" in emit["run"]
    assert "oracle-evidence-${ORACLE_CANDIDATE_SHA}.json" in emit["run"]
    assert "oracle-pytest-${ORACLE_CANDIDATE_SHA}.xml" in emit["run"]
    assert validate["if"] == "always()"
    assert ".github/scripts/stage3_oracle_evidence.py validate" in validate["run"]
    assert upload["if"] == "always()"
    assert upload["uses"] == "actions/upload-artifact@v4"
    assert upload["with"]["name"] == "oracle-evidence-${{ env.ORACLE_CANDIDATE_SHA }}"
    assert upload["with"]["if-no-files-found"] == "error"
    assert "oracle-evidence-${{ env.ORACLE_CANDIDATE_SHA }}.json" in upload["with"]["path"]
    assert "oracle-pytest-${{ env.ORACLE_CANDIDATE_SHA }}.xml" in upload["with"]["path"]


def test_gate_aggregate_ingests_and_binds_the_exact_oracle_artifact() -> None:
    workflow = yaml.safe_load(GATE_WORKFLOW.read_text(encoding="utf-8"))
    aggregate = workflow["jobs"]["aggregate"]
    steps = aggregate["steps"]
    download = next(step for step in steps if step.get("name") == "Download exact oracle evidence")
    report = next(
        step for step in steps if step.get("name") == "Emit the single-SHA technical Gate report"
    )
    upload = next(step for step in steps if step.get("name") == "Upload Gate report")
    enforce = next(
        step for step in steps if step.get("name") == "Enforce Gate report outcome after upload"
    )

    assert "oracle" in aggregate["needs"]
    assert aggregate["if"] == (
        "always() && (github.event_name != 'pull_request_target' || "
        "github.event.pull_request.merged == true)"
    )
    assert download["if"] == "always()"
    assert download["continue-on-error"] is True
    assert download["uses"] == "actions/download-artifact@v4"
    assert download["with"]["name"] == "oracle-evidence-${{ env.GATE_CANDIDATE_SHA }}"
    assert download["with"]["path"] == "${{ runner.temp }}/oracle-evidence"
    assert "--oracle-evidence" in report["run"]
    assert "oracle-evidence-${{ env.GATE_CANDIDATE_SHA }}.json" in report["run"]
    assert "--oracle-junit" in report["run"]
    assert "oracle-pytest-${{ env.GATE_CANDIDATE_SHA }}.xml" in report["run"]
    assert report["if"] == "always()"
    assert '--lane "lint=${{ needs.lint.result }}"' in report["run"]
    assert "--lane \"unit-isolated=${{ needs['unit-isolated'].result }}\"" in report["run"]
    assert '--lane "consistency=${{ needs.consistency.result }}"' in report["run"]
    assert '--lane "oracle=${{ needs.oracle.result }}"' in report["run"]
    assert "--lane \"vn2-verify=${{ needs['vn2-verify'].result }}\"" in report["run"]
    assert "--no-fail" in report["run"]
    assert upload["if"] == "always()"
    assert enforce["if"] == "always()"
    assert "--check-report" in enforce["run"]
    assert steps.index(report) < steps.index(upload) < steps.index(enforce)


@pytest.mark.parametrize("oracle_result", ["failure", "skipped"])
def test_gate_report_persists_failed_or_skipped_oracle_lane_before_enforcement(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    oracle_result: str,
) -> None:
    candidate = "c" * 40
    output = tmp_path / "gate-report.json"
    tracking, _ = _write_report_tracking_evidence(
        tmp_path / "tracking",
        candidate,
    )
    inventory = tmp_path / "vn2-input-digests.json"
    inventory.write_text("{}\n", encoding="utf-8")
    captures = tmp_path / "captures"
    captures.mkdir()
    monkeypatch.setattr(stage3_gate_report, "TRACKING_SERIES", tracking)
    monkeypatch.setattr(stage3_gate_report, "INPUT_INVENTORY", inventory)
    monkeypatch.setattr(stage3_gate_report, "CAPTURES_DIR", captures)
    monkeypatch.setattr(
        stage3_gate_report,
        "find_activation_record",
        lambda: {
            "merge_sha": "a" * 40,
            "merged_at": "2026-07-09T23:54:34Z",
            "deadline": "2026-08-20T23:54:34Z",
            "pr": 331,
        },
    )

    class FixedDateTime(datetime):
        @classmethod
        def now(cls, tz: object = None) -> FixedDateTime:
            return cls(2026, 7, 15, 12, 0, tzinfo=UTC)

    monkeypatch.setattr(stage3_gate_report, "datetime", FixedDateTime)

    def fake_git(*args: str) -> str:
        if args == ("rev-parse", "HEAD"):
            return candidate
        if args[:3] == ("diff", "--no-renames", "--name-only"):
            return ""
        raise AssertionError(f"unexpected git call: {args!r}")

    monkeypatch.setattr(stage3_gate_report, "git", fake_git)
    monkeypatch.setattr(
        stage3_gate_report.platform,
        "freedesktop_os_release",
        lambda: {"PRETTY_NAME": "Ubuntu 24.04.4 LTS"},
    )
    for name, value in {
        "CANDIDATE_SHA": candidate,
        "GITHUB_EVENT_NAME": "workflow_dispatch",
        "GITHUB_REF": "refs/heads/main",
        "GITHUB_REPOSITORY": "Vzlentin/calibre",
        "GITHUB_RUN_ATTEMPT": "1",
        "GITHUB_RUN_ID": "123456789",
        "GITHUB_SERVER_URL": "https://github.com",
        "GITHUB_WORKFLOW_REF": "Vzlentin/calibre/.github/workflows/gate-a.yml@refs/heads/main",
        "GITHUB_WORKFLOW_SHA": "b" * 40,
    }.items():
        monkeypatch.setenv(name, value)
    lane_results = {
        "lint": "success",
        "unit-isolated": "success",
        "consistency": "success",
        "oracle": oracle_result,
        "vn2-verify": "success",
    }
    arguments = ["stage3_gate_report.py", "--out", str(output), "--no-fail"]
    for name, result in lane_results.items():
        arguments.extend(("--lane", f"{name}={result}"))
    monkeypatch.setattr(sys, "argv", arguments)

    assert stage3_gate_report.main() == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["problems"] == [f"lane oracle result={oracle_result}"]
    assert {name: lane["result"] for name, lane in report["lanes"].items()} == lane_results
    for name, lane in report["lanes"].items():
        assert lane["job"] == name
        assert lane["candidate_sha"] == candidate
        assert lane["workflow_sha"] == "b" * 40
        assert lane["run_id"] == "123456789"

    monkeypatch.setattr(
        sys,
        "argv",
        ["stage3_gate_report.py", "--check-report", str(output)],
    )
    assert stage3_gate_report.main() == 1


def test_gate_report_requires_canonical_record_and_matching_receipt(tmp_path: Path) -> None:
    candidate = "c" * 40
    series, receipt = _write_report_tracking_evidence(tmp_path / "tracking", candidate)
    problems: list[str] = []
    c0, record_digest, receipt_digest = stage3_gate_report._tracking_evidence(
        series,
        problems=problems,
    )
    assert c0 == candidate
    assert record_digest == hashlib.sha256(series.read_bytes()).hexdigest()
    assert receipt_digest == hashlib.sha256(receipt.read_bytes()).hexdigest()
    assert problems == []

    receipt_value = json.loads(receipt.read_text(encoding="utf-8"))
    receipt_value["record_sha256"] = "0" * 64
    receipt.write_text(
        json.dumps(receipt_value, separators=(",", ":"), sort_keys=True) + "\n",
        encoding="utf-8",
    )
    problems = []
    c0, _, receipt_digest = stage3_gate_report._tracking_evidence(
        series,
        problems=problems,
    )
    assert c0 is None
    assert receipt_digest is None
    assert problems == ["tracking promotion receipt does not bind the latest canonical record"]


def test_gate_report_records_missing_oracle_artifact_as_a_negative() -> None:
    binding, problems = stage3_gate_report._oracle_binding(
        Path("missing-oracle-evidence.json"),
        Path("missing-oracle-pytest.xml"),
        candidate="c" * 40,
        required_outcomes=REQUIRED_TIER3_OUTCOMES,
    )

    assert binding is None
    assert problems == ["aggregate is missing the oracle evidence JSON"]


def test_required_tier3_inventory_is_exact_and_evidence_aligned() -> None:
    inventory = json.loads(TIER3_ORACLE_INVENTORY.read_text(encoding="utf-8"))

    assert set(inventory) == {"named", "schema", "tier"}
    assert inventory["schema"] == 1
    assert inventory["tier"] == "tier3"
    assert inventory["named"] == {
        "gate": {
            "id": "vn2-conditional-replay",
            "node": (
                "tests.tier3.test_conditional_replay::"
                "test_promoted_orders_match_independent_conditional_replay"
            ),
        },
        "witness": {
            "id": "vn2-conditional-replay",
            "node": (
                "tests.tier3.test_conditional_replay::"
                "test_conditional_replay_rejects_one_successor_order_unit"
            ),
        },
    }
    evidence_problems: list[str] = []
    gate_problems: list[str] = []
    assert (
        stage3_oracle_evidence._load_required_tier3_outcomes(
            TIER3_ORACLE_INVENTORY,
            problems=evidence_problems,
        )
        == inventory["named"]
    )
    assert (
        stage3_gate_report._load_required_oracle_outcomes(
            TIER3_ORACLE_INVENTORY,
            problems=gate_problems,
        )
        == inventory["named"]
    )
    assert evidence_problems == gate_problems == []


@pytest.mark.parametrize(
    ("inventory_text", "expected_problem"),
    [
        pytest.param(None, "missing required Tier 3 oracle inventory", id="missing"),
        pytest.param("{", "invalid JSON in required Tier 3 oracle inventory", id="malformed"),
        pytest.param(
            "{}\n",
            "required Tier 3 oracle inventory has an invalid root schema",
            id="wrong-schema",
        ),
    ],
)
def test_inventory_failures_write_negative_oracle_and_gate_artifacts_in_subprocesses(
    tmp_path: Path,
    inventory_text: str | None,
    expected_problem: str,
) -> None:
    fixture_root = tmp_path / "script-repo"
    oracle_script, gate_script = _copy_evidence_scripts(fixture_root)
    inventory_path = fixture_root / "newcalibre" / "tests" / "tier3" / "oracle_inventory.json"
    if inventory_text is not None:
        inventory_path.parent.mkdir(parents=True)
        inventory_path.write_text(inventory_text, encoding="utf-8")

    candidate = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    environment_path = tmp_path / "environment.json"
    environment_path.write_text("{}\n", encoding="utf-8")
    evidence_path = tmp_path / f"oracle-evidence-{candidate}.json"
    junit_path = tmp_path / f"oracle-pytest-{candidate}.xml"
    emitted = subprocess.run(
        [
            sys.executable,
            str(oracle_script),
            "emit",
            "--candidate-sha",
            candidate,
            "--environment",
            str(environment_path),
            "--junit",
            str(junit_path),
            "--out",
            str(evidence_path),
            "--test-outcome",
            "success",
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert emitted.returncode == 0, emitted.stderr
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    assert evidence["status"] == "failed"
    assert any(expected_problem in problem for problem in evidence["problems"])

    gate_path = tmp_path / "gate-report.json"
    wrapper = textwrap.dedent(
        """
        import importlib.util
        import sys
        from pathlib import Path

        script_path = Path(sys.argv[1])
        output_path = Path(sys.argv[2])
        candidate = sys.argv[3]
        sys.path.insert(0, str(script_path.parent))
        spec = importlib.util.spec_from_file_location("stage3_gate_report_subprocess", script_path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        module.TRACKING_SERIES = output_path.parent / "missing-series.jsonl"
        module.INPUT_INVENTORY = output_path.parent / "missing-inputs.json"
        module.CAPTURES_DIR = output_path.parent / "missing-captures"
        module.find_activation_record = lambda: None
        module.git = lambda *args: candidate if args == ("rev-parse", "HEAD") else ""
        module.platform.freedesktop_os_release = lambda: {"PRETTY_NAME": "test OS"}
        sys.argv = [str(script_path), "--out", str(output_path), "--no-fail"]
        raise SystemExit(module.main())
        """
    )
    gate_env = {**_copied_gate_subprocess_env(), "CANDIDATE_SHA": candidate}
    gate_emitted = subprocess.run(
        [sys.executable, "-c", wrapper, str(gate_script), str(gate_path), candidate],
        cwd=REPO_ROOT,
        env=gate_env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert gate_emitted.returncode == 0, gate_emitted.stderr
    gate_report = json.loads(gate_path.read_text(encoding="utf-8"))
    assert any(expected_problem in problem for problem in gate_report["problems"])


@pytest.mark.parametrize("inventory_text", [None, "{"])
def test_gate_report_check_mode_is_inventory_independent_in_a_subprocess(
    tmp_path: Path,
    inventory_text: str | None,
) -> None:
    fixture_root = tmp_path / "script-repo"
    _, gate_script = _copy_evidence_scripts(fixture_root)
    if inventory_text is not None:
        inventory_path = fixture_root / "newcalibre" / "tests" / "tier3" / "oracle_inventory.json"
        inventory_path.parent.mkdir(parents=True)
        inventory_path.write_text(inventory_text, encoding="utf-8")
    report_path = tmp_path / "clean-report.json"
    report_path.write_text(
        json.dumps(
            {
                "lanes": {name: {"result": "success"} for name in stage3_gate_report.LANE_NAMES},
                "problems": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    checked = subprocess.run(
        [sys.executable, str(gate_script), "--check-report", str(report_path)],
        cwd=REPO_ROOT,
        env=_copied_gate_subprocess_env(),
        check=False,
        capture_output=True,
        text=True,
    )

    assert checked.returncode == 0, checked.stderr
    assert json.loads(checked.stdout) == {"problems": []}


@pytest.mark.parametrize("present_roles", [("gate",), ("witness",), ()])
def test_oracle_outcomes_refuse_one_half_or_empty_required_inventory(
    tmp_path: Path,
    present_roles: tuple[str, ...],
) -> None:
    junit_path = tmp_path / "oracle-pytest.xml"
    suite = ET.Element("testsuite", {"name": "tier3"})
    for role in present_roles:
        required = REQUIRED_TIER3_OUTCOMES[role]
        classname, name = required["node"].rsplit("::", 1)
        ET.SubElement(suite, "testcase", {"classname": classname, "name": name})
    ET.ElementTree(suite).write(junit_path, encoding="utf-8", xml_declaration=True)
    problems: list[str] = []

    outcomes = stage3_oracle_evidence._pytest_outcomes(
        junit_path,
        required_outcomes=REQUIRED_TIER3_OUTCOMES,
        problems=problems,
    )

    assert set(outcomes["named"]) == set(present_roles)
    assert (
        stage3_oracle_evidence._evidence_status(
            "success",
            outcomes["named"],
            capture_root_absent=False,
            problems=problems,
        )
        == "failed"
    )
    for missing_role in {"gate", "witness"} - set(present_roles):
        assert f"pytest JUnit must contain exactly one named {missing_role} outcome" in problems


def test_oracle_outcomes_refuse_required_tests_from_a_renamed_module(tmp_path: Path) -> None:
    junit_path = tmp_path / "oracle-pytest.xml"
    suite = ET.Element("testsuite", {"name": "tier3"})
    for required in REQUIRED_TIER3_OUTCOMES.values():
        _, name = required["node"].rsplit("::", 1)
        ET.SubElement(
            suite,
            "testcase",
            {"classname": "tests.tier3.renamed_replay", "name": name},
        )
    ET.ElementTree(suite).write(junit_path, encoding="utf-8", xml_declaration=True)
    problems: list[str] = []

    outcomes = stage3_oracle_evidence._pytest_outcomes(
        junit_path,
        required_outcomes=REQUIRED_TIER3_OUTCOMES,
        problems=problems,
    )

    assert outcomes["named"] == {}
    assert (
        stage3_oracle_evidence._evidence_status(
            "success",
            outcomes["named"],
            capture_root_absent=False,
            problems=problems,
        )
        == "failed"
    )
    assert problems == [
        "pytest JUnit must contain exactly one named gate outcome",
        "pytest JUnit must contain exactly one named witness outcome",
    ]


def test_skipped_placeholder_preserves_required_named_nodes(tmp_path: Path) -> None:
    junit_path = tmp_path / "oracle-pytest.xml"
    stage3_oracle_evidence._placeholder_junit(
        junit_path,
        skipped=True,
        required_outcomes=REQUIRED_TIER3_OUTCOMES,
    )
    problems: list[str] = []

    outcomes = stage3_oracle_evidence._pytest_outcomes(
        junit_path,
        required_outcomes=REQUIRED_TIER3_OUTCOMES,
        problems=problems,
    )

    assert problems == []
    assert outcomes["named"] == {
        role: {**required, "status": "skipped"}
        for role, required in REQUIRED_TIER3_OUTCOMES.items()
    }
    assert (
        stage3_oracle_evidence._evidence_status(
            "skipped",
            outcomes["named"],
            capture_root_absent=True,
            problems=problems,
        )
        == "skipped"
    )


def test_oracle_evidence_emitter_binds_full_provenance_and_named_outcomes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    candidate = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    workflow_sha = "b" * 40
    for name, value in {
        "GITHUB_EVENT_NAME": "workflow_dispatch",
        "GITHUB_JOB": "oracle",
        "GITHUB_REF": "refs/heads/feat/s3-u7c-conditional-replay",
        "GITHUB_REPOSITORY": "Vzlentin/calibre",
        "GITHUB_RUN_ATTEMPT": "1",
        "GITHUB_RUN_ID": "123456789",
        "GITHUB_SERVER_URL": "https://github.com",
        "GITHUB_SHA": candidate,
        "GITHUB_WORKFLOW_REF": (
            "Vzlentin/calibre/.github/workflows/gate-a.yml@"
            "refs/heads/feat/s3-u7c-conditional-replay"
        ),
        "GITHUB_WORKFLOW_SHA": workflow_sha,
    }.items():
        monkeypatch.setenv(name, value)
    monkeypatch.chdir(REPO_ROOT)

    environment_path = tmp_path / "environment.json"
    environment_path.write_text(
        json.dumps(
            {
                "arch": "x86_64",
                "cpu_model": "Evidence CPU",
                "numpy": "2.4.6",
                "numpy_config": "OpenBLAS 0.3.31",
                "os": {
                    "id": "ubuntu",
                    "pretty_name": "Ubuntu 24.04.4 LTS",
                    "version_id": "24.04",
                },
                "python": "3.12.12",
                "runner_image": "ubuntu24/20260705.1",
                "thread_policy": {name: "1" for name in stage3_oracle_evidence.THREAD_VARIABLES},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    junit_path = tmp_path / f"oracle-pytest-{candidate}.xml"
    suite = ET.Element(
        "testsuite",
        {"name": "tier3", "tests": "2", "failures": "0", "errors": "0"},
    )
    for required in REQUIRED_TIER3_OUTCOMES.values():
        classname, name = required["node"].rsplit("::", 1)
        ET.SubElement(
            suite,
            "testcase",
            {"classname": classname, "name": name},
        )
    ET.ElementTree(suite).write(junit_path, encoding="utf-8", xml_declaration=True)
    evidence_path = tmp_path / f"oracle-evidence-{candidate}.json"

    report = stage3_oracle_evidence.emit_evidence(
        candidate_sha=candidate,
        environment_path=environment_path,
        junit_path=junit_path,
        output_path=evidence_path,
        test_outcome="success",
    )

    assert report["problems"] == []
    assert report["status"] == "passed"
    assert report["identity"]["candidate_sha"] == candidate
    assert report["identity"]["head_sha"] == candidate
    assert report["identity"]["workflow_sha"] == workflow_sha
    assert report["environment"]["arch"] == "x86_64"
    assert report["environment"]["python"] == "3.12.12"
    assert report["environment"]["uv"].startswith("uv ")
    assert report["environment"]["lock_sha256"]
    assert report["environment"]["numerical_stack"]["numpy_config"] == "OpenBLAS 0.3.31"
    assert set(report["digests"]) == {
        "capture_inner_bundle",
        "capture_manifest",
        "capture_payload",
        "input_inventory",
        "oracle_config",
        "receipt",
        "receipt_artifact",
        "successor_config",
        "successor_input_inventory",
    }
    assert report["actuals_semantics"] == "censored_sales_surrogate"
    assert report["tests"]["named"] == {
        role: {**required, "status": "passed"} for role, required in REQUIRED_TIER3_OUTCOMES.items()
    }
    assert (
        stage3_oracle_evidence.validate_evidence(
            evidence_path,
            junit_path,
            allow_skipped=False,
        )
        == []
    )
    binding, binding_problems = stage3_gate_report._oracle_binding(
        evidence_path,
        junit_path,
        candidate=candidate,
        required_outcomes=REQUIRED_TIER3_OUTCOMES,
    )
    assert binding_problems == []
    assert binding["evidence_sha256"] == stage3_oracle_evidence.sha256_file(evidence_path)
    assert binding["junit_sha256"] == stage3_oracle_evidence.sha256_file(junit_path)
    assert binding["outcomes"] == report["tests"]["named"]

    report["tests"]["named"]["gate"]["node"] = "tests.tier3.renamed_replay::test_gate"
    evidence_path.write_text(json.dumps(report) + "\n", encoding="utf-8")
    assert (
        "tests.named.gate.node must equal "
        "tests.tier3.test_conditional_replay::"
        "test_promoted_orders_match_independent_conditional_replay"
        in stage3_oracle_evidence.validate_evidence(
            evidence_path,
            junit_path,
            allow_skipped=False,
        )
    )
    _, gate_problems = stage3_gate_report._oracle_binding(
        evidence_path,
        junit_path,
        candidate=candidate,
        required_outcomes=REQUIRED_TIER3_OUTCOMES,
    )
    assert "oracle evidence lacks a passing named gate outcome" in gate_problems


@pytest.mark.skipif(sys.platform != "linux", reason="symlink evidence checks bite on Linux")
@pytest.mark.parametrize(
    ("link_kind", "expected_problem"),
    [
        ("captures-root", "promoted capture root must not be a symbolic link"),
        ("bundle-root", "promoted capture bundle root must not be a symbolic link"),
        ("receipt", "promoted capture receipt must not be a symbolic link"),
        ("manifest", "capture manifest must not be a symbolic link"),
    ],
)
def test_oracle_emitter_writes_a_failed_artifact_for_symlinked_capture_evidence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    link_kind: str,
    expected_problem: str,
) -> None:
    workspace = _prepare_emitter_workspace(tmp_path, include_captures=True)
    captures = workspace / "stage3" / "evidence" / "captures"
    bundle = next(path for path in captures.iterdir() if len(path.name) == 40)
    receipt = captures / f"{bundle.name}-receipt.json"
    if link_kind == "captures-root":
        target = workspace / "real-captures"
        captures.rename(target)
        captures.symlink_to(target, target_is_directory=True)
    elif link_kind == "bundle-root":
        target = workspace / "real-bundle"
        bundle.rename(target)
        bundle.symlink_to(target, target_is_directory=True)
    elif link_kind == "receipt":
        target = workspace / "real-receipt.json"
        receipt.rename(target)
        receipt.symlink_to(target)
    else:
        manifest = bundle / "manifest.json"
        target = bundle / "real-manifest.json"
        manifest.rename(target)
        manifest.symlink_to(target)

    candidate = "c" * 40
    monkeypatch.setattr(stage3_oracle_evidence, "_git_head", lambda: candidate)
    monkeypatch.chdir(workspace)
    environment_path = tmp_path / "environment.json"
    _write_oracle_environment(environment_path)
    junit_path = tmp_path / f"oracle-pytest-{candidate}.xml"
    _write_named_junit(junit_path)
    evidence_path = tmp_path / f"oracle-evidence-{candidate}.json"

    report = stage3_oracle_evidence.emit_evidence(
        candidate_sha=candidate,
        environment_path=environment_path,
        junit_path=junit_path,
        output_path=evidence_path,
        test_outcome="success",
    )

    assert report["status"] == "failed"
    assert any(expected_problem in problem for problem in report["problems"])
    assert json.loads(evidence_path.read_text(encoding="utf-8")) == report


def test_oracle_emitter_allows_skip_only_for_a_genuinely_absent_capture_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    workspace = _prepare_emitter_workspace(tmp_path, include_captures=False)
    candidate = "c" * 40
    workflow_sha = "b" * 40
    for name, value in {
        "GITHUB_EVENT_NAME": "schedule",
        "GITHUB_JOB": "oracle",
        "GITHUB_REF": "refs/heads/main",
        "GITHUB_REPOSITORY": "Vzlentin/calibre",
        "GITHUB_RUN_ATTEMPT": "1",
        "GITHUB_RUN_ID": "123456789",
        "GITHUB_SERVER_URL": "https://github.com",
        "GITHUB_SHA": candidate,
        "GITHUB_WORKFLOW_REF": "Vzlentin/calibre/.github/workflows/newcalibre.yml@refs/heads/main",
        "GITHUB_WORKFLOW_SHA": workflow_sha,
    }.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(stage3_oracle_evidence, "_git_head", lambda: candidate)
    monkeypatch.chdir(workspace)
    environment_path = tmp_path / "environment.json"
    junit_path = tmp_path / f"oracle-pytest-{candidate}.xml"
    evidence_path = tmp_path / f"oracle-evidence-{candidate}.json"

    report = stage3_oracle_evidence.emit_evidence(
        candidate_sha=candidate,
        environment_path=environment_path,
        junit_path=junit_path,
        output_path=evidence_path,
        test_outcome="skipped",
    )

    assert report["status"] == "skipped"
    assert report["problems"] == []
    assert (
        stage3_oracle_evidence.validate_evidence(
            evidence_path,
            junit_path,
            allow_skipped=True,
        )
        == []
    )
    report["problems"] = ["invalid capture evidence"]
    evidence_path.write_text(json.dumps(report) + "\n", encoding="utf-8")
    assert (
        "passing or skipped oracle evidence must contain no recorded problems"
        in stage3_oracle_evidence.validate_evidence(
            evidence_path,
            junit_path,
            allow_skipped=True,
        )
    )
    monkeypatch.setenv("GITHUB_EVENT_NAME", "workflow_dispatch")
    manual_evidence_path = tmp_path / "manual" / f"oracle-evidence-{candidate}.json"
    manual_report = stage3_oracle_evidence.emit_evidence(
        candidate_sha=candidate,
        environment_path=environment_path,
        junit_path=junit_path,
        output_path=manual_evidence_path,
        test_outcome="skipped",
    )
    assert manual_report["status"] == "failed"
    assert (
        "only scheduled runs may skip an absent promoted capture root" in manual_report["problems"]
    )


@pytest.mark.skipif(sys.platform != "linux", reason="symlink evidence checks bite on Linux")
@pytest.mark.parametrize(
    ("link_kind", "expected_problem"),
    [
        ("captures-root", "promoted capture root must not be a symbolic link"),
        ("bundle-root", "promoted capture entry must not be a symbolic link"),
        ("manifest", "capture manifest must not be a symbolic link"),
    ],
)
def test_gate_manifest_digest_collection_refuses_symlinked_evidence(
    tmp_path: Path,
    link_kind: str,
    expected_problem: str,
) -> None:
    captures = tmp_path / "captures"
    bundle = captures / ("a" * 40)
    bundle.mkdir(parents=True)
    manifest = bundle / "manifest.json"
    manifest.write_text("{}\n", encoding="utf-8")
    if link_kind == "captures-root":
        target = tmp_path / "real-captures"
        captures.rename(target)
        captures.symlink_to(target, target_is_directory=True)
    elif link_kind == "bundle-root":
        target = tmp_path / "real-bundle"
        bundle.rename(target)
        bundle.symlink_to(target, target_is_directory=True)
    else:
        target = bundle / "real-manifest.json"
        manifest.rename(target)
        manifest.symlink_to(target)
    problems: list[str] = []

    digests = stage3_gate_report._capture_manifest_digests(captures, problems=problems)

    assert digests == {}
    assert any(expected_problem in problem for problem in problems)
