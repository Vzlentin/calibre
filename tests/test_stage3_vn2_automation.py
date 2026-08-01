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
M5_RUNBOOK = ROOT / "benchmarks" / "m5" / "README.md"


def _workflow(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _runs(workflow: dict) -> str:
    return "\n".join(_job_runs(job) for job in workflow["jobs"].values())


def _job_runs(job: dict) -> str:
    return "\n".join(str(step.get("run", "")) for step in job.get("steps", []))


def _protocol_jobs(workflow: dict) -> dict:
    return {
        name: workflow["jobs"][name]
        for name in ("vn2-acceptance", "m5-acceptance", "reference-gates")
    }


def test_workflow_has_pr_main_and_protocol_scoped_lanes() -> None:
    """Keep existing lanes and split scheduled acceptance by evidence surface."""
    workflow = _workflow(REGRESSION)

    assert set(workflow[True]) == {
        "push",
        "pull_request",
        "schedule",
        "workflow_dispatch",
    }
    assert workflow[True]["push"] == {"branches": ["main"]}
    assert workflow[True]["pull_request"] == {"branches": ["main"]}
    assert workflow[True]["schedule"] == [{"cron": "0 5 * * *"}]
    assert set(workflow["jobs"]) == {
        "newcalibre-lint",
        "newcalibre-unit",
        "newcalibre-consistency",
        "vn2-acceptance",
        "m5-acceptance",
        "reference-gates",
    }
    assert workflow["permissions"] == {"contents": "read"}
    assert workflow["jobs"]["newcalibre-lint"]["if"] == "github.event_name == 'pull_request'"
    assert workflow["jobs"]["newcalibre-unit"]["if"] == "github.event_name == 'pull_request'"
    assert workflow["jobs"]["newcalibre-consistency"]["if"] == "github.event_name == 'push'"
    protocol_jobs = _protocol_jobs(workflow)
    protocol_condition = (
        "github.event_name == 'schedule' || github.event_name == 'workflow_dispatch'"
    )
    assert all(job["if"] == protocol_condition for job in protocol_jobs.values())
    assert all("environment" not in job for job in protocol_jobs.values())
    type_step = next(
        step
        for step in workflow["jobs"]["newcalibre-lint"]["steps"]
        if "ty check" in step.get("run", "")
    )
    assert type_step["working-directory"] == "newcalibre"
    assert type_step["run"] == "uv run --locked --no-sync ty check src/newcalibre/"


def test_protocol_jobs_pin_manual_candidates_end_to_end() -> None:
    """Bind each manual protocol run to one validated full candidate SHA."""
    workflow = _workflow(REGRESSION)
    candidate = (
        "${{ github.event_name == 'workflow_dispatch' && inputs.candidate_sha || github.sha }}"
    )

    for name, env_name in (
        ("vn2-acceptance", "VN2_CANDIDATE_SHA"),
        ("m5-acceptance", "M5_CANDIDATE_SHA"),
        ("reference-gates", "REFERENCE_CANDIDATE_SHA"),
    ):
        job = workflow["jobs"][name]
        validation = next(
            step for step in job["steps"] if step.get("name") == "Validate manual candidate"
        )
        checkout = next(step for step in job["steps"] if step.get("uses") == "actions/checkout@v4")
        runs = _job_runs(job)

        assert job["env"][env_name] == candidate
        assert validation["if"] == "github.event_name == 'workflow_dispatch'"
        assert f'[[ "${env_name}" =~ ^[0-9a-f]{{40}}$ ]]' in validation["run"]
        assert checkout["with"]["ref"] == candidate
        assert f'test "$(git rev-parse HEAD)" = "${env_name}"' in runs


def test_protocol_jobs_use_exact_inventories_and_successor_verifiers() -> None:
    """Acquire only inventory-owned basenames and verify every restored dataset."""
    text = REGRESSION.read_text(encoding="utf-8")
    workflow = _workflow(REGRESSION)
    protocol_jobs = _protocol_jobs(workflow)
    protocol_text = str(protocol_jobs)
    runs = "\n".join(_job_runs(job) for job in protocol_jobs.values())
    vn2 = workflow["jobs"]["vn2-acceptance"]
    m5 = workflow["jobs"]["m5-acceptance"]
    vn2_cache = next(step for step in vn2["steps"] if step.get("id") == "vn2-cache")
    m5_cache = next(step for step in m5["steps"] if step.get("id") == "m5-cache")
    vn2_acquisition = next(
        step
        for step in vn2["steps"]
        if step.get("name") == "Acquire and verify VN2 inputs with successor tooling"
    )
    m5_acquisition = next(
        step
        for step in m5["steps"]
        if step.get("name") == "Acquire and verify M5 inputs with successor tooling"
    )

    assert vn2_cache["with"] == {
        "path": "newcalibre/data/vn2",
        "key": "vn2-${{ hashFiles('newcalibre/benchmarks/vn2/vn2-input-digests.json') }}",
    }
    assert m5_cache["with"] == {
        "path": "newcalibre/data/m5",
        "key": "m5-${{ hashFiles('newcalibre/benchmarks/m5/m5-inputs.json') }}",
    }
    for protocol, cache_id, acquisition in (
        ("vn2", "vn2-cache", vn2_acquisition),
        ("m5", "m5-cache", m5_acquisition),
    ):
        assert acquisition["env"] == {
            "OVENTI_DATASET_BASE_URL": "${{ vars.OVENTI_DATASET_BASE_URL }}"
        }
        assert "if" not in acquisition
        assert f"steps.{cache_id}.outputs.cache-hit" in acquisition["run"]
        assert "jq -r '.files[].name'" in acquisition["run"]
        assert '[[ "$name" != */* && "$name" != "." && "$name" != ".." ]]' in acquisition["run"]
        assert f'"${{OVENTI_DATASET_BASE_URL%/}}/{protocol}/$name"' in acquisition["run"]
        assert "curl --fail --location --retry 3" in acquisition["run"]
    assert "newcalibre/scripts/vn2_data.py verify" in runs
    assert "newcalibre/scripts/m5_data.py verify" in runs
    assert "newcalibre/scripts/vn2_data.py download" not in runs
    assert "newcalibre/scripts/m5_data.py download" not in runs
    assert "benchmarks/vn2/vn2_file_links.json" not in text
    assert "restore-keys" not in protocol_text
    assert "stage3/evidence/" + "vn2-input-digests.json" not in protocol_text
    assert ".github/scripts/stage3_" + "vn2_data.py" not in protocol_text
    assert "secrets." not in protocol_text
    assert "id-token" not in protocol_text


def test_protocol_jobs_select_directories_and_report_m5_sizing() -> None:
    """Run exact protocol directories and expose standard-runner headroom."""
    workflow = _workflow(REGRESSION)
    text = REGRESSION.read_text(encoding="utf-8")
    runs = _runs(workflow)
    vn2_runs = _job_runs(workflow["jobs"]["vn2-acceptance"])
    m5_runs = _job_runs(workflow["jobs"]["m5-acceptance"])
    reference = workflow["jobs"]["reference-gates"]
    reference_runs = _job_runs(reference)
    m5_sizing = next(
        step
        for step in workflow["jobs"]["m5-acceptance"]["steps"]
        if step.get("name") == "Run reduced real-M5 acceptance and measure runner headroom"
    )["run"]
    pytest_commands = "\n".join(line for line in runs.splitlines() if "pytest " in line)

    assert "pytest newcalibre/tests/tier3" in vn2_runs
    assert "pytest newcalibre/tests/tier4/vn2" in vn2_runs
    assert "pytest newcalibre/tests/tier4/m5" in m5_runs
    assert "pytest newcalibre/tests/tier4/reference" in reference_runs
    assert "pytest newcalibre/tests/tier4\n" not in runs
    assert "-m tier4" not in runs
    assert "test_vn2_acceptance.py" not in pytest_commands
    assert "test_vn2_gate_b_advisory.py" not in pytest_commands
    assert "test_m5_reduced_acceptance.py" not in pytest_commands
    assert "actions/cache" not in str(reference)
    assert "OVENTI_DATASET_BASE_URL" not in str(reference)
    assert "curl " not in reference_runs
    assert "M5 acceptance elapsed seconds:" in m5_runs
    assert "M5 aggregate peak job memory (process RSS) KiB:" in m5_runs
    assert "M5 minimum memory headroom KiB:" in m5_runs
    assert "M5 minimum free disk KiB:" in m5_runs
    assert "minimum_headroom_kib=$(( 4 * 1024 * 1024 ))" in m5_runs
    assert "20 * 60" in m5_runs
    assert "pytest newcalibre/tests/tier4/m5 &" in m5_sizing
    assert "test_pid=$!" in m5_sizing
    assert 'wait "$test_pid" || test_status=$?' in m5_sizing
    assert "(( test_status == 0 && sizing_status == 0 ))" in m5_sizing
    assert m5_sizing.index("test_pid=$!") < m5_sizing.index('wait "$test_pid"')
    assert m5_sizing.index('wait "$test_pid"') < m5_sizing.index(
        "(( test_status == 0 && sizing_status == 0 ))"
    )
    assert "vn2_tracking.py build" in vn2_runs
    assert "actions/upload-artifact@v4" in text
    assert "gh pr" not in runs
    assert "git push" not in runs
    assert "vn2-regression:" not in text


def test_m5_runbook_documents_only_the_generic_verified_origin_contract() -> None:
    """Describe generic public acquisition without making staging a fallback."""
    text = M5_RUNBOOK.read_text(encoding="utf-8")

    assert "OVENTI_DATASET_BASE_URL" in text
    assert "newcalibre/scripts/m5_data.py verify" in text
    assert "newcalibre/benchmarks/m5/m5-inputs.json" in text
    assert "github.io" not in text
    assert "calibre-protocol-data-temp" not in text


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
