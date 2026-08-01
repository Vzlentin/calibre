"""Test the compact VN2 workflows and trusted-base append check."""

from __future__ import annotations

import hashlib
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).parents[1]
REGRESSION = ROOT / ".github" / "workflows" / "newcalibre.yml"
FROZEN_CI = ROOT / ".github" / "workflows" / "ci.yml"
APPEND_CHECK = ROOT / ".github" / "workflows" / "vn2-evidence.yml"
ATTRIBUTES = ROOT / ".gitattributes"
CAPTURE = ROOT / "stage3" / "evidence" / "captures" / "vn2"
M5_RUNBOOK = ROOT / "benchmarks" / "m5" / "README.md"
M5_MONITOR = ROOT / ".github" / "scripts" / "run-m5-acceptance.sh"
README = ROOT / "README.md"
PERFORMANCE_SPEC = ROOT / "docs" / "spec" / "30-performance.md"


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


def test_performance_spec_uses_consistent_resident_memory_contract() -> None:
    """Pin public resident-memory, cgroup-v2, and bounded scaling terminology."""
    source = PERFORMANCE_SPEC.read_text(encoding="utf-8")

    assert "RSS" not in source
    assert source.count("peak_process_resident_bytes") >= 2
    assert source.count("peak_job_memory_bytes") >= 3
    assert "cgroup-v2 charged job peak" in source
    assert "full-M5 serial path" in source
    assert "exactly `profile.json` and `environment.json`" in source


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


def test_pr_unit_lane_provisions_only_loopback_in_its_network_namespace() -> None:
    """Bring loopback up without restoring external connectivity."""
    workflow = _workflow(REGRESSION)
    step = next(
        step
        for step in workflow["jobs"]["newcalibre-unit"]["steps"]
        if step.get("name") == "Run Tier 1 without repository, captures, or network"
    )
    run = step["run"]

    assert "sudo unshare -n bash -c" in run
    assert run.index("ip link set lo up") < run.index('exec sudo -u "$1"')
    assert 'socket.create_connection((\\"1.1.1.1\\", 443), timeout=3)' in run
    assert "uv run --no-sync pytest tests/tier1" in run


@pytest.mark.skipif(sys.platform != "linux", reason="requires a Linux network namespace")
def test_ray_starts_on_loopback_inside_a_hermetic_network_namespace(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Execute the production Ray startup inside an isolated Linux namespace."""
    sudo = shutil.which("sudo")
    unshare = shutil.which("unshare")
    uv = shutil.which("uv")
    if sudo is None or unshare is None or uv is None:
        pytest.skip("requires sudo, unshare, and uv")
    capability = subprocess.run(
        [sudo, "-n", unshare, "-n", "true"],
        check=False,
        capture_output=True,
        text=True,
    )
    if capability.returncode != 0:
        pytest.skip("requires permission to create a network namespace")

    witness = """
import socket

import ray

from newcalibre.engine.ray import RayDispatch

try:
    socket.create_connection(("1.1.1.1", 443), timeout=1)
except OSError:
    pass
else:
    raise AssertionError("isolated namespace reached an external address")

dispatch = RayDispatch()
try:
    dispatch._ensure_runtime()
    nodes = [node for node in ray.nodes() if node.get("Alive")]
    assert len(nodes) == 1
    assert nodes[0]["NodeManagerAddress"] == "127.0.0.01"
finally:
    dispatch.shutdown()
"""
    ambient_python = tmp_path / "ambient-python"
    ambient_python.write_text("#!/bin/sh\nexit 86\n", encoding="utf-8")
    ambient_python.chmod(0o755)
    monkeypatch.setattr(sys, "executable", str(ambient_python))
    ambient = subprocess.run(
        [sys.executable, "-c", witness],
        check=False,
        capture_output=True,
        text=True,
    )
    assert ambient.returncode == 86

    namespace = f"""
set -euo pipefail
ip link set lo up
exec {shlex.quote(sudo)} -n -u {shlex.quote(os.environ.get("USER", ""))} \
  env PATH={shlex.quote(os.environ["PATH"])} HOME={shlex.quote(str(Path.home()))} \
  {shlex.quote(uv)} run --project {shlex.quote(str(ROOT / "newcalibre"))} \
  --locked --no-sync python -c {shlex.quote(witness)}
"""
    completed = subprocess.run(
        [sudo, "-n", unshare, "-n", "bash", "-c", namespace],
        cwd=ROOT / "newcalibre",
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr


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
    assert "benchmarks/vn2/vn2_file_" + "links.json" not in text
    assert "restore-" + "keys" not in protocol_text
    assert "stage3/evidence/" + "vn2-input-digests.json" not in protocol_text
    assert ".github/scripts/stage3_" + "vn2_data.py" not in protocol_text
    assert "secrets." not in protocol_text
    assert "id-" + "token" not in protocol_text


def test_frozen_jobs_use_one_exact_inventory_cache_and_verifier() -> None:
    """Acquire frozen VN2 inputs by safe inventory basename and always verify."""
    workflow = _workflow(FROZEN_CI)
    exact_cache = {
        "path": "data/vn2",
        "key": "vn2-${{ hashFiles('newcalibre/benchmarks/vn2/vn2-input-digests.json') }}",
    }

    for job_name in ("test", "docker-build"):
        job = workflow["jobs"][job_name]
        scope = next(step for step in job["steps"] if step.get("id") == "scope")
        gate = next(step for step in job["steps"] if step.get("id") == "gate")
        cache = next(step for step in job["steps"] if step.get("id") == "vn2-cache")
        acquisition = next(
            step
            for step in job["steps"]
            if step.get("name") == "Acquire and verify VN2 inputs with successor tooling"
        )

        assert cache["with"] == exact_cache
        assert scope["uses"] == "tj-actions/changed-files@v45"
        assert "steps.scope.outputs.only_changed != 'true'" in gate["run"]
        assert cache["if"] == "steps.gate.outputs.run_frozen == 'true'"
        assert acquisition["if"] == "steps.gate.outputs.run_frozen == 'true'"
        assert acquisition["env"] == {
            "OVENTI_DATASET_BASE_URL": "${{ vars.OVENTI_DATASET_BASE_URL }}"
        }
        assert "set -euo pipefail" in acquisition["run"]
        assert "inventory=newcalibre/benchmarks/vn2/vn2-input-digests.json" in acquisition["run"]
        assert "target=data/vn2" in acquisition["run"]
        assert '[[ "${{ steps.vn2-cache.outputs.cache-hit }}" != "true" ]]' in acquisition["run"]
        assert "jq -r '.files[] | [.name, .bytes] | @tsv'" in acquisition["run"]
        assert "while IFS=$'\\t' read -r name expected_bytes; do" in acquisition["run"]
        basename_guard = '[[ "$name" != */* && "$name" != "." && "$name" != ".." ]]'
        assert basename_guard in acquisition["run"]
        assert '"${OVENTI_DATASET_BASE_URL%/}/vn2/$name"' in acquisition["run"]
        assert "curl --fail --location --retry 3" in acquisition["run"]
        assert "--max-time 60" in acquisition["run"]
        assert '--max-filesize "$expected_bytes"' in acquisition["run"]
        assert "--remove-on-error" in acquisition["run"]
        assert "newcalibre/scripts/vn2_data.py verify" in acquisition["run"]
        verify = acquisition["run"].index("newcalibre/scripts/vn2_data.py verify")
        cache_miss = acquisition["run"].index("cache-hit")
        cache_miss_end = acquisition["run"].index("\nfi\n", cache_miss)
        guard = acquisition["run"].index(basename_guard)
        curl = acquisition["run"].index("curl --fail --location --retry 3")
        assert cache_miss < guard < curl < cache_miss_end < verify


def test_frozen_ci_retains_lanes_commands_and_verified_docker_mount() -> None:
    """Preserve frozen CI surfaces while moving acquisition onto the runner."""
    workflow = _workflow(FROZEN_CI)
    text = FROZEN_CI.read_text(encoding="utf-8")
    test_runs = _job_runs(workflow["jobs"]["test"])
    docker_runs = _job_runs(workflow["jobs"]["docker-build"])

    assert workflow[True] == {
        "push": {"branches": ["main"]},
        "pull_request": {"branches": ["main"]},
        "schedule": [{"cron": "0 2 * * *"}],
    }
    assert set(workflow["jobs"]) == {
        "lint-and-type-check",
        "test",
        "s3-ingestion",
        "m5-coverage-neutrality",
        "docker-build",
    }
    assert "uv run pytest\n" in test_runs
    assert "docker build -t calibre:full ." in docker_runs
    assert 'docker run --rm -v "$PWD/data/vn2:/app/data/vn2:ro"' in docker_runs
    assert "-m benchmarks.vn2 --config /app/benchmarks/vn2/config/winning.yaml" in docker_runs
    assert "docker build -f Dockerfile.slim -t calibre:slim ." in docker_runs
    push_commands = {
        line.strip() for line in docker_runs.splitlines() if line.strip().startswith("docker push ")
    }
    assert push_commands == {
        'docker push "$image_repo:${tag_prefix}-full"',
        'docker push "$image_repo:${tag_prefix}-slim"',
        'docker push "$image_repo:${sha_tag}-full"',
        'docker push "$image_repo:${sha_tag}-slim"',
    }
    assert "docker run --rm --user root" not in docker_runs
    forbidden = (
        "restore-" + "keys",
        "azure" + "/login",
        "id-" + "token",
        "download_" + "vn2_data",
        "vn2_file_" + "links",
    )
    assert all(term not in text for term in forbidden)


def test_predecessor_vn2_acquisition_is_deleted_and_undocumented() -> None:
    """Leave the generic verified origin as the only VN2 acquisition contract."""
    obsolete = (
        ROOT / "benchmarks" / "vn2" / ("download_" + "vn2_data.py"),
        ROOT / "benchmarks" / "vn2" / ("vn2_file_" + "links.json"),
    )
    text = README.read_text(encoding="utf-8")

    assert all(not path.exists() for path in obsolete)
    assert "OVENTI_DATASET_BASE_URL" in text
    assert "newcalibre/benchmarks/vn2/vn2-input-digests.json" in text
    assert "newcalibre/scripts/vn2_data.py verify" in text
    assert "download_" + "vn2_data" not in text
    assert "vn2_file_" + "links" not in text
    assert "git" + "hub.io" not in text
    assert "calibre-protocol-data-" + "temp" not in text


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
    m5_monitor = M5_MONITOR.read_text(encoding="utf-8")
    pytest_commands = "\n".join(line for line in runs.splitlines() if "pytest " in line)

    assert "pytest newcalibre/tests/tier3/vn2" in vn2_runs
    assert "pytest newcalibre/tests/tier4/vn2" in vn2_runs
    assert "pytest newcalibre/tests/tier3/m5" in m5_runs
    assert "pytest newcalibre/tests/tier4/m5" in m5_runs
    assert "pytest newcalibre/tests/tier4/reference" in reference_runs
    assert "pytest newcalibre/tests/tier3\n" not in runs
    assert "pytest newcalibre/tests/tier4\n" not in runs
    assert "-m tier4" not in runs
    assert "test_vn2_acceptance.py" not in pytest_commands
    assert "test_vn2_gate_b_advisory.py" not in pytest_commands
    assert "test_m5_reduced_acceptance.py" not in pytest_commands
    assert "test_m5_frozen_scorer_parity.py" not in pytest_commands
    assert "actions/cache" not in str(reference)
    assert "OVENTI_DATASET_BASE_URL" not in str(reference)
    assert "curl " not in reference_runs
    assert "bash .github/scripts/run-m5-acceptance.sh" in m5_sizing
    assert "pytest newcalibre/tests/tier4/m5" in m5_sizing
    assert "M5 acceptance elapsed seconds:" in m5_monitor
    assert "M5 aggregate peak job memory (process RSS) KiB:" in m5_monitor
    assert "M5 minimum memory headroom KiB:" in m5_monitor
    assert "M5 minimum free disk KiB:" in m5_monitor
    assert "minimum_headroom_kib=$(( 4 * 1024 * 1024 ))" in m5_monitor
    assert "maximum_elapsed_seconds=$(( 20 * 60 ))" in m5_monitor
    assert 'wait "$test_pid" || test_status=$?' in m5_monitor
    assert 'exit "$test_status"' in m5_monitor
    assert 'exit "$sizing_status"' in m5_monitor
    assert "vn2_tracking.py build" in vn2_runs
    assert "actions/upload-artifact@v4" in text
    assert "gh pr" not in runs
    assert "git push" not in runs
    assert "vn2-regression:" not in text


def test_protocol_jobs_keep_tier3_prerequisites_and_frozen_oracle_scoped() -> None:
    """Bind each Tier-3 directory to its protocol prerequisites and exact oracle."""
    workflow = _workflow(REGRESSION)
    vn2 = workflow["jobs"]["vn2-acceptance"]
    m5 = workflow["jobs"]["m5-acceptance"]
    vn2_runs = _job_runs(vn2)
    m5_runs = _job_runs(m5)
    vn2_precondition = next(
        step
        for step in vn2["steps"]
        if step.get("name") == "Require canonical VN2 evidence and harnesses"
    )["run"]
    m5_precondition = next(
        step
        for step in m5["steps"]
        if step.get("name") == "Require canonical M5 inputs, config, and harnesses"
    )["run"]
    preparation = next(
        step for step in m5["steps"] if step.get("name") == "Prepare exact frozen M5 scorer"
    )["run"]
    cleanup = next(step for step in m5["steps"] if step.get("name") == "Remove frozen M5 scorer")

    assert "newcalibre/tests/tier3/vn2/oracle_inventory.json" in vn2_precondition
    assert "newcalibre/tests/tier3/vn2/test_conditional_replay.py" in vn2_precondition
    assert "newcalibre/tests/tier3/m5/oracle_inventory.json" in m5_precondition
    assert "newcalibre/tests/tier3/m5/m5_frozen_export.py" in m5_precondition
    assert "newcalibre/tests/tier3/m5/test_m5_frozen_scorer_parity.py" in m5_precondition
    assert "newcalibre/tests/tier4/m5/test_m5_reduced_acceptance.py" in m5_precondition
    assert "newcalibre/data/m5" not in vn2_runs
    assert "newcalibre/scripts/m5_data.py" not in vn2_runs
    assert "newcalibre/data/vn2" not in m5_runs
    assert "newcalibre/scripts/vn2_data.py" not in m5_runs
    assert "CALIBRE_FROZEN_ORACLE_WORKTREE" not in vn2_runs
    assert "oracle-freeze-2026-07-06" in preparation
    assert "686a1b284a4f4879123b4095d306f07b88d2ddc3" in preparation
    assert "git worktree add --detach" in preparation
    assert 'cd "$oracle_worktree"' in preparation
    assert "uv sync --locked" in preparation
    assert "CALIBRE_FROZEN_ORACLE_WORKTREE" in preparation
    assert cleanup["if"] == "always()"
    assert "git worktree remove --force" in cleanup["run"]
    assert not any(step.get("uses") == "actions/upload-artifact@v4" for step in m5["steps"])


def test_m5_monitor_executes_child_and_sizing_exit_contracts(tmp_path: Path) -> None:
    """Propagate child and sizing failures while allowing a complete success."""
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_df = fake_bin / "df"
    fake_df.write_text(
        "#!/bin/sh\n"
        "printf 'Filesystem 1024-blocks Used Available Capacity Mounted on\\n'\n"
        "printf 'fake 6291456 1 6291455 1%% /\\n'\n",
        encoding="utf-8",
    )
    fake_df.chmod(0o755)

    def run(*, child_status: int, memory_headroom_kib: int) -> subprocess.CompletedProcess[str]:
        meminfo = tmp_path / f"meminfo-{child_status}-{memory_headroom_kib}"
        meminfo.write_text(f"MemAvailable: {memory_headroom_kib} kB\n", encoding="utf-8")
        env = {
            **os.environ,
            "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
            "M5_MEMINFO_PATH": str(meminfo),
            "M5_DISK_PATH": str(tmp_path),
            "M5_MONITOR_INTERVAL_SECONDS": "0",
        }
        return subprocess.run(
            ["bash", str(M5_MONITOR), "bash", "-c", f"exit {child_status}"],
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )

    child_failure = run(child_status=7, memory_headroom_kib=5 * 1024 * 1024)
    sizing_failure = run(child_status=0, memory_headroom_kib=1)
    success = run(child_status=0, memory_headroom_kib=5 * 1024 * 1024)

    assert child_failure.returncode == 7
    assert "M5 acceptance child status: 7" in child_failure.stdout
    assert sizing_failure.returncode == 1
    assert "M5 acceptance sizing status: 1" in sizing_failure.stdout
    assert success.returncode == 0
    assert "M5 acceptance child status: 0" in success.stdout
    assert "M5 acceptance sizing status: 0" in success.stdout


def test_m5_runbook_documents_only_the_generic_verified_origin_contract() -> None:
    """Describe generic public acquisition without making staging a fallback."""
    text = M5_RUNBOOK.read_text(encoding="utf-8")

    assert "OVENTI_DATASET_BASE_URL" in text
    assert "newcalibre/scripts/m5_data.py verify" in text
    assert "newcalibre/benchmarks/m5/m5-inputs.json" in text
    assert "git" + "hub.io" not in text
    assert "calibre-protocol-data-" + "temp" not in text


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
