"""Exercise the Stage 3 tracker and Gate automation contracts."""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType

import pytest
import yaml

SCRIPTS_DIR = Path(__file__).parents[1] / ".github" / "scripts"
GATE_WORKFLOW = Path(__file__).parents[1] / ".github" / "workflows" / "gate-a.yml"
CLOCK_WORKFLOW = Path(__file__).parents[1] / ".github" / "workflows" / "stage3-clock.yml"
ROOT_WORKFLOW = Path(__file__).parents[1] / ".github" / "workflows" / "ci.yml"
SUCCESSOR_WORKFLOW = Path(__file__).parents[1] / ".github" / "workflows" / "newcalibre.yml"
BOOTSTRAP_VN2_INVENTORY = (
    Path(__file__).parents[1] / "stage3" / "evidence" / "vn2-input-digests.json"
)
SUCCESSOR_VN2_INVENTORY = (
    Path(__file__).parents[1] / "newcalibre" / "benchmarks" / "vn2" / "vn2-input-digests.json"
)
TIER2_ARTIFACT_CHECK = (
    "set -euo pipefail\n"
    "for run in tier2-run1 tier2-run2; do\n"
    "  for artifact in resumed-ledger.bin same-seed-ledger.bin; do\n"
    '    test -s "${RUNNER_TEMP}/${run}/${artifact}"\n'
    "  done\n"
    "done\n"
    'diff -r "${RUNNER_TEMP}/tier2-run1" "${RUNNER_TEMP}/tier2-run2"'
)

ACTIVATION_BODY = (
    "<!-- s3-clock-activation -->\n"
    "## Clock activation record (immutable)\n\n"
    "Written by `stage3-clock` from GitHub's merge event for PR #331 "
    "(S3-U1 first landing). This record — not the milestone — is authoritative.\n\n"
    "```json\n"
    "{\n"
    '  "merge_sha": "3fcd8b9f88366132f4e45fe52f064cd83e0a9023",\n'
    '  "merged_at": "2026-07-09T23:54:34Z",\n'
    '  "deadline": "2026-08-20T23:54:34Z",\n'
    '  "pr": 331\n'
    "}\n"
    "```\n"
)
ACTIVATION_RECORD = {
    "merge_sha": "3fcd8b9f88366132f4e45fe52f064cd83e0a9023",
    "merged_at": "2026-07-09T23:54:34Z",
    "deadline": "2026-08-20T23:54:34Z",
    "pr": 331,
}
BLOCKER_MARKER = "<!-- s3-blocker -->"
NOW = datetime(2026, 7, 30, 6, 30, tzinfo=UTC)
STARTED_AT = "s3-started-at: 2026-07-10T00:00:00Z"


def _comment(
    body: str,
    *,
    actor: str = "owner",
    created_at: str = "2026-07-30T06:30:00Z",
    updated_at: str | None = None,
) -> dict[str, object]:
    timestamps = {
        "created_at": created_at,
        "updated_at": updated_at or created_at,
    }
    if actor == "actions":
        return {
            "body": body,
            "author_association": "NONE",
            "user": {"login": "github-actions[bot]", "type": "Bot"},
            "performed_via_github_app": {"slug": "github-actions"},
            **timestamps,
        }
    associations = {
        "owner": "OWNER",
        "member": "MEMBER",
        "collaborator": "COLLABORATOR",
        "outsider": "NONE",
    }
    association = associations[actor]
    return {
        "body": body,
        "author_association": association,
        "user": {"login": actor, "type": "User"},
        "performed_via_github_app": None,
        **timestamps,
    }


def _control_body(marker: str, record: dict[str, object]) -> str:
    return f"{marker}\n```json\n{json.dumps(record, indent=2)}\n```"


def _gate_decision_body(
    *,
    decision: str = "go",
    disposition: str = "mint-b1-b5",
    report_emitted_at: str = "2026-07-30T06:00:00Z",
    candidate_sha: str = "c" * 40,
    report_sha256: str = "d" * 64,
) -> str:
    return _control_body(
        stage3_clock.DECISION_MARKER,
        {
            "kind": "gate",
            "decision": decision,
            "disposition": disposition,
            "candidate_sha": candidate_sha,
            "report_sha256": report_sha256,
            "report_emitted_at": report_emitted_at,
        },
    )


def _early_halt_body(disposition: str = "abandon") -> str:
    return _control_body(
        stage3_clock.DECISION_MARKER,
        {
            "kind": "early-halt",
            "decision": "no-go",
            "disposition": disposition,
        },
    )


def _expiry_body() -> str:
    return _control_body(
        stage3_clock.EXPIRY_MARKER,
        {
            "activation_merge_sha": ACTIVATION_RECORD["merge_sha"],
            "deadline": ACTIVATION_RECORD["deadline"],
        },
    )


def _load_script(name: str) -> ModuleType:
    path = SCRIPTS_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


stage3_clock = _load_script("stage3_clock")
stage3_gate_report = _load_script("stage3_gate_report")


def test_gate_report_reads_the_real_multiline_activation_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    forged_outsider = ACTIVATION_BODY.replace(
        "2026-08-20T23:54:34Z",
        "9999-12-31T23:59:59Z",
    )
    forged_actions = ACTIVATION_BODY.replace(
        ACTIVATION_RECORD["merge_sha"],
        "0" * 40,
    )
    monkeypatch.setattr(
        stage3_clock,
        "activation_record_matches_pull_request",
        lambda record: record == ACTIVATION_RECORD,
    )
    monkeypatch.setattr(
        stage3_clock,
        "issue_comments",
        lambda issue: [
            _comment(forged_outsider, actor="outsider"),
            _comment(forged_actions, actor="actions"),
            _comment(ACTIVATION_BODY, actor="actions"),
        ],
    )

    record = stage3_gate_report.find_activation_record()

    assert record == ACTIVATION_RECORD
    assert stage3_gate_report.record_is_schema_complete(record)
    assert stage3_gate_report.find_activation_record is stage3_clock.find_activation_record


def _activation_provenance_fixtures() -> tuple[dict, list[dict], list[dict]]:
    pr = {
        "merged": True,
        "merged_at": ACTIVATION_RECORD["merged_at"],
        "merge_commit_sha": ACTIVATION_RECORD["merge_sha"],
        "base": {"ref": "main"},
    }
    pr_events = [
        {
            "event": "labeled",
            "created_at": "2026-07-09T23:05:58Z",
            "label": {"name": stage3_clock.CLOCK_START_LABEL},
        },
        {
            "event": "merged",
            "created_at": ACTIVATION_RECORD["merged_at"],
            "commit_id": ACTIVATION_RECORD["merge_sha"],
        },
    ]
    u1_timeline = [
        {
            "event": "cross-referenced",
            "created_at": "2026-07-09T23:05:58Z",
            "source": {
                "issue": {
                    "number": 331,
                    "repository_url": "https://api.github.com/repos/Vzlentin/calibre",
                    "pull_request": {},
                }
            },
        }
    ]
    return pr, pr_events, u1_timeline


def _install_activation_api_fixtures(
    monkeypatch: pytest.MonkeyPatch,
    pr: dict,
    pr_events: list[dict],
    u1_timeline: list[dict],
) -> tuple[list[str], list[str]]:
    api_paths: list[str] = []
    paginated_paths: list[str] = []

    def gh(path: str) -> dict:
        api_paths.append(path)
        assert path == "repos/Vzlentin/calibre/pulls/331"
        return pr

    def paginated(path: str) -> list[dict]:
        paginated_paths.append(path)
        fixtures = {
            "repos/Vzlentin/calibre/issues/331/events?per_page=100": pr_events,
            "repos/Vzlentin/calibre/issues/302/timeline?per_page=100": u1_timeline,
        }
        assert path in fixtures
        return fixtures[path]

    monkeypatch.setattr(stage3_clock, "repo", lambda: "Vzlentin/calibre")
    monkeypatch.setattr(stage3_clock, "gh_api", gh)
    monkeypatch.setattr(stage3_clock, "gh_api_paginated", paginated)
    return api_paths, paginated_paths


def test_activation_record_is_bound_to_exact_immutable_api_facts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pr, pr_events, u1_timeline = _activation_provenance_fixtures()
    api_paths, paginated_paths = _install_activation_api_fixtures(
        monkeypatch,
        pr,
        pr_events,
        u1_timeline,
    )

    assert stage3_clock.activation_record_matches_pull_request(ACTIVATION_RECORD)
    assert api_paths == ["repos/Vzlentin/calibre/pulls/331"]
    assert paginated_paths == [
        "repos/Vzlentin/calibre/issues/331/events?per_page=100",
        "repos/Vzlentin/calibre/issues/302/timeline?per_page=100",
    ]


@pytest.mark.parametrize(
    "tamper",
    [
        "not-merged",
        "pr-sha",
        "pr-time",
        "base",
        "missing-merge-event",
        "merge-event-sha",
        "merge-event-time",
        "missing-label",
        "label-after-merge",
        "unlabeled-at-merge",
        "link-after-merge",
        "link-not-pr",
        "link-wrong-pr",
        "link-wrong-repo",
    ],
)
def test_activation_record_rejects_each_tampered_provenance_fact(
    monkeypatch: pytest.MonkeyPatch,
    tamper: str,
) -> None:
    pr, pr_events, u1_timeline = copy.deepcopy(_activation_provenance_fixtures())
    if tamper == "not-merged":
        pr["merged"] = False
    elif tamper == "pr-sha":
        pr["merge_commit_sha"] = "0" * 40
    elif tamper == "pr-time":
        pr["merged_at"] = "2026-07-09T23:54:33Z"
    elif tamper == "base":
        pr["base"] = {"ref": "feature"}
    elif tamper == "missing-merge-event":
        pr_events[:] = [event for event in pr_events if event["event"] != "merged"]
    elif tamper == "merge-event-sha":
        pr_events[1]["commit_id"] = "0" * 40
    elif tamper == "merge-event-time":
        pr_events[1]["created_at"] = "2026-07-09T23:54:33Z"
    elif tamper == "missing-label":
        pr_events[:] = [event for event in pr_events if event["event"] != "labeled"]
    elif tamper == "label-after-merge":
        pr_events[0]["created_at"] = "2026-07-09T23:54:35Z"
    elif tamper == "unlabeled-at-merge":
        pr_events.insert(
            1,
            {
                "event": "unlabeled",
                "created_at": "2026-07-09T23:54:00Z",
                "label": {"name": stage3_clock.CLOCK_START_LABEL},
            },
        )
    elif tamper == "link-after-merge":
        u1_timeline[0]["created_at"] = "2026-07-09T23:54:35Z"
    elif tamper == "link-not-pr":
        del u1_timeline[0]["source"]["issue"]["pull_request"]
    elif tamper == "link-wrong-pr":
        u1_timeline[0]["source"]["issue"]["number"] = 999
    elif tamper == "link-wrong-repo":
        u1_timeline[0]["source"]["issue"]["repository_url"] = (
            "https://api.github.com/repos/attacker/fork"
        )

    _install_activation_api_fixtures(monkeypatch, pr, pr_events, u1_timeline)
    assert not stage3_clock.activation_record_matches_pull_request(ACTIVATION_RECORD)


def test_latest_same_second_label_event_owns_the_merge_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pr, pr_events, u1_timeline = _activation_provenance_fixtures()
    pr_events[0:1] = [
        {
            "event": "labeled",
            "created_at": "2026-07-09T23:54:00Z",
            "label": {"name": stage3_clock.CLOCK_START_LABEL},
        },
        {
            "event": "unlabeled",
            "created_at": "2026-07-09T23:54:00Z",
            "label": {"name": stage3_clock.CLOCK_START_LABEL},
        },
        {
            "event": "labeled",
            "created_at": "2026-07-09T23:54:00Z",
            "label": {"name": stage3_clock.CLOCK_START_LABEL},
        },
    ]
    _install_activation_api_fixtures(monkeypatch, pr, pr_events, u1_timeline)

    assert stage3_clock.activation_record_matches_pull_request(ACTIVATION_RECORD)


@pytest.mark.parametrize("late_fact", ["label", "u1-link"])
def test_activate_refuses_mutable_predicates_added_only_after_merge(
    monkeypatch: pytest.MonkeyPatch,
    late_fact: str,
) -> None:
    pr, pr_events, u1_timeline = _activation_provenance_fixtures()
    pr.update(
        {
            "labels": [{"name": stage3_clock.CLOCK_START_LABEL}],
            "title": "S3-U1a scaffold",
            "body": "Part of #302",
        }
    )
    if late_fact == "label":
        pr_events[0]["created_at"] = "2026-07-09T23:54:35Z"
    else:
        u1_timeline[0]["created_at"] = "2026-07-09T23:54:35Z"
    _install_activation_api_fixtures(monkeypatch, pr, pr_events, u1_timeline)
    monkeypatch.setattr(stage3_clock, "find_activation_record", lambda: None)
    monkeypatch.setattr(
        stage3_clock,
        "post_issue_comment",
        lambda *args: pytest.fail(f"unexpected activation write: {args}"),
    )

    assert stage3_clock.cmd_activate(331) == 1


@pytest.mark.parametrize(
    "record",
    [
        {**ACTIVATION_RECORD, "deadline": "9999-12-31T23:59:59Z"},
        {**ACTIVATION_RECORD, "deadline": "2026-08-20"},
        {**ACTIVATION_RECORD, "merged_at": "2026-07-09T23:54:34+00:00"},
        {
            **ACTIVATION_RECORD,
            "merged_at": "9999-12-01T00:00:00Z",
            "deadline": "9999-12-31T00:00:00Z",
        },
        {**ACTIVATION_RECORD, "pr": True},
        {**ACTIVATION_RECORD, "extra": "not allowed"},
    ],
)
def test_activation_record_requires_the_exact_bot_written_clock_invariant(
    record: dict[str, object],
) -> None:
    assert not stage3_clock.record_is_schema_complete(record)


def test_tracker_marker_authorities_are_explicit() -> None:
    owner = _comment("marker")
    member = {**owner, "author_association": "MEMBER"}
    collaborator = {**owner, "author_association": "COLLABORATOR"}
    outsider = _comment("marker", actor="outsider")
    actions = _comment("marker", actor="actions")
    fake_actions = {
        **actions,
        "performed_via_github_app": {"slug": "untrusted-app"},
    }

    assert all(
        stage3_clock.comment_from_tracker_operator(comment)
        for comment in (owner, member, collaborator)
    )
    assert not stage3_clock.comment_from_tracker_operator(outsider)
    assert stage3_clock.comment_from_github_actions(actions)
    assert not stage3_clock.comment_from_github_actions(fake_actions)


def test_control_markers_must_be_the_exact_first_line() -> None:
    quoted_activation = _comment(
        f"<!-- s3-heartbeat 2026-W31 -->\n{ACTIVATION_BODY}",
        actor="actions",
    )
    embedded_blocker = _comment(f"progress note\n{_blocker_body('waiting', '2026-08-01')}")

    assert stage3_clock.find_activation_record([quoted_activation]) is None
    assert not stage3_clock.blocker_suspends_stall([embedded_blocker], today=NOW.date())


def test_gate_workflow_grants_report_read_access_and_uploads_negative_reports() -> None:
    workflow = yaml.safe_load(GATE_WORKFLOW.read_text(encoding="utf-8"))
    aggregate = workflow["jobs"]["aggregate"]

    assert aggregate["permissions"] == {
        "contents": "read",
        "issues": "read",
        "pull-requests": "read",
    }
    emit = next(
        step
        for step in aggregate["steps"]
        if step.get("name") == "Emit the single-SHA technical Gate report"
    )
    upload = next(
        step for step in aggregate["steps"] if step.get("uses") == "actions/upload-artifact@v4"
    )
    assert '"${RUNNER_TEMP}/gate-report.json"' in emit["run"]
    assert upload["if"] == "always()"
    assert upload["with"]["path"] == "${{ runner.temp }}/gate-report.json"


@pytest.mark.parametrize(
    ("workflow_path", "job_name"),
    [
        (SUCCESSOR_WORKFLOW, "newcalibre-consistency"),
        (GATE_WORKFLOW, "consistency"),
    ],
)
def test_consistency_workflows_require_nonempty_tier2_artifacts_before_diff(
    workflow_path: Path,
    job_name: str,
) -> None:
    workflow = yaml.safe_load(workflow_path.read_text(encoding="utf-8"))
    comparison = next(
        step
        for step in workflow["jobs"][job_name]["steps"]
        if step.get("name") == "Require Tier-2 artifacts and compare executions"
    )

    assert comparison["run"].strip() == TIER2_ARTIFACT_CHECK


def test_vn2_acceptance_has_an_exact_successor_owned_consumption_boundary() -> None:
    workflow = yaml.safe_load(SUCCESSOR_WORKFLOW.read_text(encoding="utf-8"))
    job = workflow["jobs"]["vn2-acceptance"]
    steps = job["steps"]
    presence = next(step for step in steps if step.get("name") == "Successor + harness presence")
    skipped = next(step for step in steps if step.get("name") == "Tier 4 visibly skipped")
    tier4_index = next(
        index for index, step in enumerate(steps) if step.get("name") == "Tier 4 run"
    )
    acquire = steps[tier4_index - 2]
    verify = steps[tier4_index - 1]
    tier4 = steps[tier4_index]

    assert job["if"] == (
        "github.event_name == 'schedule' || "
        "(github.event_name == 'workflow_dispatch' && startsWith(inputs.lane, 'vn2'))"
    )
    assert presence["run"].strip() == (
        "ready=false\n"
        "[ -f newcalibre/pyproject.toml ] && "
        "[ -f stage3/evidence/vn2-input-digests.json ] \\\n"
        "  && [ -f newcalibre/tests/tier4/test_vn2_acceptance.py ] && ready=true\n"
        'echo "ready=$ready" >> "$GITHUB_OUTPUT"'
    )
    assert skipped["run"] == (
        'echo "::notice::tier 4 skipped — acceptance contract or digest inventory absent"'
    )
    assert skipped["if"] == "steps.present.outputs.ready != 'true'"
    guarded_tail = {
        "astral-sh/setup-uv@v4": "steps.present.outputs.ready == 'true'",
        "Restore VN2 data (exact digest-inventory key, no fallback)": (
            "steps.present.outputs.ready == 'true'"
        ),
        "Acquire VN2 inputs with bootstrap tooling": "steps.present.outputs.ready == 'true'",
        "Verify VN2 inputs with successor tooling": "steps.present.outputs.ready == 'true'",
        "Tier 4 run": "steps.present.outputs.ready == 'true'",
        "Upload digest-bound bundle": (
            "steps.present.outputs.ready == 'true' && github.event_name == 'workflow_dispatch'"
        ),
    }
    for step in steps[steps.index(skipped) + 1 :]:
        identity = step.get("name", step.get("uses"))
        assert step["if"] == guarded_tail[identity]

    assert acquire["name"] == "Acquire VN2 inputs with bootstrap tooling"
    assert acquire["run"] == (
        "uv run --no-project python .github/scripts/stage3_vn2_data.py download "
        "--target newcalibre/data/vn2 --if-missing"
    )
    assert verify["name"] == "Verify VN2 inputs with successor tooling"
    assert verify["run"].strip() == (
        "uv sync --project newcalibre --locked --group dev\n"
        "uv run --project newcalibre --locked --no-sync python "
        "newcalibre/scripts/vn2_data.py verify "
        "--target newcalibre/data/vn2 \\\n"
        "  --inventory newcalibre/benchmarks/vn2/vn2-input-digests.json"
    )
    assert tier4["working-directory"] == "newcalibre"
    assert tier4["run"] == "uv run --locked --no-sync pytest tests/tier4"


def test_required_successor_unit_covers_every_stage3_contract_pull_request() -> None:
    successor = yaml.safe_load(SUCCESSOR_WORKFLOW.read_text(encoding="utf-8"))
    triggers = successor[True]  # PyYAML 1.1 parses the YAML key `on` as boolean true.
    pull_request = triggers["pull_request"]
    unit = successor["jobs"]["newcalibre-unit"]

    assert pull_request == {"branches": ["main"]}
    assert "paths" not in pull_request
    assert "paths-ignore" not in pull_request
    assert unit["if"] == "github.event_name == 'push' || github.event_name == 'pull_request'"
    assert "needs" not in unit
    contract = next(
        step for step in unit["steps"] if step.get("name") == "Stage 3 automation contracts"
    )
    assert contract["if"] == "steps.present.outputs.exists == 'true'"
    assert contract["run"].strip() == (
        "uv sync --project newcalibre --locked --group dev\n"
        "uv run --project newcalibre --locked --no-sync pytest tests/test_stage3_automation.py"
    )
    build_export = next(
        step for step in unit["steps"] if step.get("name") == "Build successor-only export"
    )
    provision_export = next(
        step
        for step in unit["steps"]
        if step.get("name") == "Provision export venv (network still on)"
    )
    assert unit["steps"].index(build_export) < unit["steps"].index(contract)
    assert unit["steps"].index(contract) < unit["steps"].index(provision_export)

    workflow = yaml.safe_load(ROOT_WORKFLOW.read_text(encoding="utf-8"))
    assert "stage3-automation-contract" not in workflow["jobs"]

    expected_successor_only_patterns = {
        "newcalibre/**",
        "stage3/**",
        ".github/workflows/newcalibre.yml",
        ".github/scripts/stage3_*.py",
        "tests/test_stage3_automation.py",
    }
    scoped_job_names = {
        "lint-and-type-check",
        "test",
        "s3-ingestion",
        "docker-build",
    }
    assert {
        name
        for name, job in workflow["jobs"].items()
        if any(step.get("name") == "Detect successor-only change" for step in job["steps"])
    } == scoped_job_names
    for name in scoped_job_names:
        scoped_job = workflow["jobs"][name]
        scope = next(
            step
            for step in scoped_job["steps"]
            if step.get("name") == "Detect successor-only change"
        )
        configured_patterns = set(scope["with"]["files"].splitlines())
        assert expected_successor_only_patterns <= configured_patterns


def test_successor_vn2_inventory_is_the_exact_bootstrap_approved_blob() -> None:
    assert SUCCESSOR_VN2_INVENTORY.read_bytes() == BOOTSTRAP_VN2_INVENTORY.read_bytes()


def test_clock_manual_triggers_execute_only_the_default_branch_workflow() -> None:
    workflow = yaml.safe_load(CLOCK_WORKFLOW.read_text(encoding="utf-8"))
    triggers = workflow[True]  # PyYAML 1.1 parses the YAML key `on` as boolean true.

    assert "workflow_dispatch" not in triggers
    assert triggers["repository_dispatch"]["types"] == [
        "stage3-check-deadline",
        "stage3-heartbeat",
    ]
    for action, job_name, command in [
        ("stage3-check-deadline", "check-deadline", "check-deadline"),
        ("stage3-heartbeat", "heartbeat", "heartbeat"),
    ]:
        job = workflow["jobs"][job_name]
        assert f"github.event.action == '{action}'" in job["if"]
        required_permissions = {"contents": "read", "issues": "write"}
        if job_name == "check-deadline":
            required_permissions["pull-requests"] = "read"
        assert job["permissions"] == required_permissions
        assert any(f"stage3_clock.py {command}" in step.get("run", "") for step in job["steps"])
        checkout = next(step for step in job["steps"] if step.get("uses") == "actions/checkout@v4")
        assert "ref" not in checkout.get("with", {})
        assert "client_payload" not in json.dumps(job)

    assert workflow["jobs"]["activate"]["permissions"] == {
        "contents": "read",
        "issues": "write",
        "pull-requests": "read",
    }
    successor = yaml.safe_load(SUCCESSOR_WORKFLOW.read_text(encoding="utf-8"))
    assert successor["jobs"]["s3-activation-gate"]["permissions"] == {
        "contents": "read",
        "issues": "read",
        "pull-requests": "read",
    }


def test_clock_help_preserves_copy_paste_control_templates(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(sys, "argv", ["stage3_clock.py", "--help"])

    with pytest.raises(SystemExit) as exit_info:
        stage3_clock.main()

    assert exit_info.value.code == 0
    help_text = capsys.readouterr().out
    assert (
        "<!-- s3-blocker -->\n"
        "    ```json\n"
        '    {"description": "<why work is blocked>", "next_review": "<YYYY-MM-DD>"}' in help_text
    )
    assert "rescope-once-max-3-weeks``, ``respec``, or\n``abandon``" in help_text
    assert "event_type=stage3-check-deadline" in help_text


def test_negative_gate_report_is_written_before_main_returns_nonzero(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output = tmp_path / "gate-report.json"
    monkeypatch.setattr(stage3_gate_report, "TRACKING_SERIES", tmp_path / "missing-series.jsonl")
    monkeypatch.setattr(stage3_gate_report, "INPUT_INVENTORY", tmp_path / "missing-inputs.json")
    monkeypatch.setattr(stage3_gate_report, "CAPTURES_DIR", tmp_path / "missing-captures")
    monkeypatch.setattr(stage3_gate_report, "find_activation_record", lambda: None)
    monkeypatch.setattr(stage3_gate_report, "git", lambda *args: "a" * 40)
    monkeypatch.setattr(
        stage3_gate_report.platform,
        "freedesktop_os_release",
        lambda: {"PRETTY_NAME": "test runner"},
    )
    monkeypatch.setenv("CANDIDATE_SHA", "b" * 40)
    monkeypatch.setattr(sys, "argv", ["stage3_gate_report.py", "--out", str(output)])

    assert stage3_gate_report.main() == 1
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["candidate_sha"] == "b" * 40
    assert report["problems"] == [
        "gate precondition unmet: no promoted tracking record (U9b)",
        "no schema-complete activation record on the Gate issue",
    ]


def test_deadline_check_uses_one_gate_comment_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    class FixedDateTime(datetime):
        @classmethod
        def now(cls, tz: object = None) -> FixedDateTime:
            return cls(2026, 7, 30, 6, 30, tzinfo=UTC)

    reads: list[int] = []

    def read_comments(issue: int) -> list[dict[str, object]]:
        reads.append(issue)
        return [_comment(ACTIVATION_BODY, actor="actions")]

    monkeypatch.setattr(stage3_clock, "datetime", FixedDateTime)
    monkeypatch.setattr(stage3_clock, "issue_comments", read_comments)
    monkeypatch.setattr(
        stage3_clock,
        "activation_record_matches_pull_request",
        lambda record: True,
    )
    monkeypatch.setattr(stage3_clock, "milestone_issues", lambda: [])
    monkeypatch.setattr(
        stage3_clock,
        "post_issue_comment",
        lambda *args: pytest.fail(f"unexpected comment write: {args}"),
    )

    assert stage3_clock.cmd_check_deadline() == 0
    assert reads == [stage3_clock.GATE_ISSUE]


@pytest.mark.parametrize(
    "body",
    [
        _gate_decision_body(decision="go", disposition="mint-b1-b5"),
        _gate_decision_body(
            decision="no-go",
            disposition="rescope-once-max-3-weeks",
        ),
        _gate_decision_body(decision="no-go", disposition="respec"),
        _gate_decision_body(decision="no-go", disposition="abandon"),
        _early_halt_body("rescope-once-max-3-weeks"),
        _early_halt_body("respec"),
        _early_halt_body("abandon"),
    ],
)
def test_every_closed_gate_decision_disposition_is_accepted(body: str) -> None:
    comment = _comment(body)
    deadline = datetime(2026, 8, 20, 23, 54, 34, tzinfo=UTC)

    assert stage3_clock.comment_is_timely_owner_decision(comment, deadline=deadline)


def test_gate_decision_accepts_exact_report_and_deadline_boundaries() -> None:
    boundary = "2026-08-20T23:54:34Z"
    comment = _comment(
        _gate_decision_body(report_emitted_at=boundary),
        created_at=boundary,
    )

    assert stage3_clock.comment_is_timely_owner_decision(
        comment,
        deadline=datetime(2026, 8, 20, 23, 54, 34, tzinfo=UTC),
    )


@pytest.mark.parametrize(
    "record",
    [
        {
            "kind": "early-halt",
            "decision": "go",
            "disposition": "abandon",
        },
        {
            "kind": "early-halt",
            "decision": "no-go",
            "disposition": "abandon",
            "extra": "not closed",
        },
    ],
)
def test_early_halt_record_rejects_go_or_extra_fields(record: dict[str, object]) -> None:
    assert not stage3_clock.decision_record_is_complete(
        record,
        comment_created_at=NOW,
    )


def test_rendered_expiry_round_trips_as_the_same_activation_record() -> None:
    deadline = datetime(2026, 8, 20, 23, 54, 34, tzinfo=UTC)
    comment = _comment(
        stage3_clock.expiry_comment(ACTIVATION_RECORD),
        actor="actions",
        created_at="2026-08-21T00:00:00Z",
    )

    assert stage3_clock.comment_is_valid_expiry(
        comment,
        activation_record=ACTIVATION_RECORD,
        deadline=deadline,
    )


@pytest.mark.parametrize(
    "tamper",
    ["sha", "deadline", "edited", "extra", "missing"],
)
def test_expiry_rejects_tampered_or_edited_records(tamper: str) -> None:
    record = {
        "activation_merge_sha": ACTIVATION_RECORD["merge_sha"],
        "deadline": ACTIVATION_RECORD["deadline"],
    }
    updated_at = None
    if tamper == "sha":
        record["activation_merge_sha"] = "0" * 40
    elif tamper == "deadline":
        record["deadline"] = "2026-08-20T23:54:35Z"
    elif tamper == "edited":
        updated_at = "2026-08-21T00:00:01Z"
    elif tamper == "extra":
        record["extra"] = "not closed"
    else:
        del record["deadline"]
    comment = _comment(
        _control_body(stage3_clock.EXPIRY_MARKER, record),
        actor="actions",
        created_at="2026-08-21T00:00:00Z",
        updated_at=updated_at,
    )

    assert not stage3_clock.comment_is_valid_expiry(
        comment,
        activation_record=ACTIVATION_RECORD,
        deadline=datetime(2026, 8, 20, 23, 54, 34, tzinfo=UTC),
    )


def test_deadline_command_posts_a_round_trippable_expiry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FixedDateTime(datetime):
        @classmethod
        def now(cls, tz: object = None) -> FixedDateTime:
            return cls(2026, 8, 21, 0, 0, tzinfo=UTC)

    posts: list[tuple[object, ...]] = []
    monkeypatch.setattr(stage3_clock, "datetime", FixedDateTime)
    monkeypatch.setattr(stage3_clock, "issue_comments", lambda issue: [])
    monkeypatch.setattr(stage3_clock, "find_activation_record", lambda comments: ACTIVATION_RECORD)
    monkeypatch.setattr(stage3_clock, "milestone_issues", lambda: [])
    monkeypatch.setattr(stage3_clock, "post_issue_comment", lambda *args: posts.append(args))

    assert stage3_clock.cmd_check_deadline() == 0
    assert len(posts) == 1
    issue, body = posts[0]
    assert issue == stage3_clock.GATE_ISSUE
    emitted = _comment(
        body,
        actor="actions",
        created_at="2026-08-21T00:00:00Z",
    )
    assert stage3_clock.comment_is_valid_expiry(
        emitted,
        activation_record=ACTIVATION_RECORD,
        deadline=datetime(2026, 8, 20, 23, 54, 34, tzinfo=UTC),
    )


@pytest.mark.parametrize(
    ("gate_comments", "expected_posts"),
    [
        (
            [
                _comment(_gate_decision_body(), actor="outsider"),
                _comment(_expiry_body(), actor="outsider"),
            ],
            1,
        ),
        ([_comment(_gate_decision_body(), actor="actions")], 1),
        ([_comment(_gate_decision_body(), actor="member")], 1),
        ([_comment(_gate_decision_body(), actor="collaborator")], 1),
        ([_comment(_expiry_body())], 1),
        (
            [
                _comment(
                    _gate_decision_body(),
                    created_at="2026-08-20T23:54:35Z",
                )
            ],
            1,
        ),
        (
            [
                _comment(
                    _gate_decision_body(),
                    updated_at="2026-07-31T06:30:00Z",
                )
            ],
            1,
        ),
        ([_comment(f"decision notes\n{_gate_decision_body()}")], 1),
        (
            [
                _comment(
                    f"<!-- s3-heartbeat 2026-W31 -->\n{_expiry_body()}",
                    actor="actions",
                    created_at="2026-08-21T00:00:00Z",
                )
            ],
            1,
        ),
        ([_comment(stage3_clock.DECISION_MARKER)], 1),
        (
            [
                _comment(
                    _gate_decision_body(report_emitted_at="2026-07-31T00:00:00Z"),
                )
            ],
            1,
        ),
        ([_comment(_gate_decision_body(disposition="abandon"))], 1),
        (
            [
                _comment(
                    _control_body(
                        stage3_clock.DECISION_MARKER,
                        {
                            "kind": "gate",
                            "decision": [],
                            "disposition": "mint-b1-b5",
                            "candidate_sha": "c" * 40,
                            "report_sha256": "d" * 64,
                            "report_emitted_at": "2026-07-30T06:00:00Z",
                        },
                    )
                )
            ],
            1,
        ),
        (
            [
                _comment(
                    _control_body(
                        stage3_clock.DECISION_MARKER,
                        {
                            "kind": "early-halt",
                            "decision": "no-go",
                            "disposition": [],
                        },
                    )
                )
            ],
            1,
        ),
        (
            [
                _comment(
                    _expiry_body(),
                    actor="actions",
                    created_at="2026-08-20T23:00:00Z",
                )
            ],
            1,
        ),
        ([_comment(_gate_decision_body())], 0),
        ([_comment(_gate_decision_body(decision="no-go", disposition="respec"))], 0),
        ([_comment(_early_halt_body())], 0),
        (
            [
                _comment(
                    _expiry_body(),
                    actor="actions",
                    created_at="2026-08-21T00:00:00Z",
                )
            ],
            0,
        ),
    ],
)
def test_deadline_markers_only_suppress_escalation_from_their_authority(
    monkeypatch: pytest.MonkeyPatch,
    gate_comments: list[dict[str, object]],
    expected_posts: int,
) -> None:
    class FixedDateTime(datetime):
        @classmethod
        def now(cls, tz: object = None) -> FixedDateTime:
            return cls(2026, 8, 21, 0, 0, tzinfo=UTC)

    posts: list[tuple[object, ...]] = []
    monkeypatch.setattr(stage3_clock, "datetime", FixedDateTime)
    monkeypatch.setattr(stage3_clock, "issue_comments", lambda issue: gate_comments)
    monkeypatch.setattr(stage3_clock, "find_activation_record", lambda comments: ACTIVATION_RECORD)
    monkeypatch.setattr(stage3_clock, "milestone_issues", lambda: [])
    monkeypatch.setattr(
        stage3_clock,
        "post_issue_comment",
        lambda *args: posts.append(args),
    )

    assert stage3_clock.cmd_check_deadline() == 0
    assert len(posts) == expected_posts


@pytest.mark.parametrize(
    ("marker", "actor", "expected_posts"),
    [
        ("<!-- s3-heartbeat 2026-W31 -->", "outsider", 1),
        ("<!-- s3-heartbeat 2026-W31 -->", "owner", 1),
        ("<!-- s3-heartbeat 2026-W30 -->", "actions", 1),
        ("weekly notes\n<!-- s3-heartbeat 2026-W31 -->", "actions", 1),
        ("<!-- s3-heartbeat 2026-W31 -->", "actions", 0),
    ],
)
def test_heartbeat_idempotency_only_trusts_actions_comments(
    monkeypatch: pytest.MonkeyPatch,
    marker: str,
    actor: str,
    expected_posts: int,
) -> None:
    class FixedDateTime(datetime):
        @classmethod
        def now(cls, tz: object = None) -> FixedDateTime:
            return cls(2026, 7, 30, 6, 30, tzinfo=UTC)

    posts: list[tuple[object, ...]] = []
    monkeypatch.setattr(stage3_clock, "datetime", FixedDateTime)
    monkeypatch.setattr(
        stage3_clock, "issue_comments", lambda issue: [_comment(marker, actor=actor)]
    )
    monkeypatch.setattr(stage3_clock, "milestone_issues", lambda: [])
    monkeypatch.setattr(
        stage3_clock,
        "post_issue_comment",
        lambda *args: posts.append(args),
    )

    assert stage3_clock.cmd_heartbeat() == 0
    assert len(posts) == expected_posts


def _blocker_body(description: object, next_review: object, **extra: object) -> str:
    record = {"description": description, "next_review": next_review, **extra}
    return f"{BLOCKER_MARKER}\n```json\n{json.dumps(record)}\n```"


@pytest.mark.parametrize(
    ("comments", "expected"),
    [
        ([_blocker_body("waiting for owner input", "2026-07-31")], True),
        ([_blocker_body("waiting for {owner} input", "2026-07-31")], True),
        ([_blocker_body("waiting for owner input", "2026-07-30")], False),
        ([_blocker_body("waiting for owner input", "2026-07-29")], False),
        ([_blocker_body("", "2026-07-31")], False),
        ([_blocker_body(7, "2026-07-31")], False),
        ([_blocker_body("waiting", "2026-7-31")], False),
        ([_blocker_body("waiting", "2026-07-31T12:00:00Z")], False),
        ([_blocker_body("waiting", "2026-02-30")], False),
        ([_blocker_body("waiting", 20260731)], False),
        ([_blocker_body("waiting", "2026-07-31", owner="agent")], False),
        ([f'{BLOCKER_MARKER}\n```json\n{{"description": "waiting"}}\n```'], False),
        ([f"{BLOCKER_MARKER}\n```json\nnot-json\n```"], False),
        (
            [
                _blocker_body("older valid blocker", "2026-08-01"),
                f"{BLOCKER_MARKER}\n```json\nnot-json\n```",
            ],
            False,
        ),
        (
            [
                _blocker_body("current blocker", "2026-08-01"),
                "unrelated newer progress note",
            ],
            True,
        ),
        (
            [
                _blocker_body("expired blocker", "2026-07-29"),
                _blocker_body("replacement blocker", "2026-08-01"),
            ],
            True,
        ),
    ],
)
def test_blocker_uses_only_the_newest_marker_comment_and_requires_current_schema(
    comments: list[str],
    expected: bool,
) -> None:
    records = [_comment(body) for body in comments]

    assert stage3_clock.blocker_suspends_stall(records, today=NOW.date()) is expected


def test_untrusted_comments_cannot_extend_blockers_or_reset_start_time() -> None:
    old_start = _comment(STARTED_AT)
    forged_start = _comment("s3-started-at: 2099-01-01T00:00:00Z", actor="outsider")
    forged_blocker = _comment(
        _blocker_body("outsider delay", "9999-12-31"),
        actor="outsider",
    )

    assert not stage3_clock.blocker_suspends_stall([forged_blocker], today=NOW.date())
    assert stage3_clock.started_at([old_start, forged_start], now=NOW) == datetime(
        2026,
        7,
        10,
        tzinfo=UTC,
    )


def test_future_or_embedded_start_markers_cannot_replace_the_latest_valid_start() -> None:
    valid = _comment(STARTED_AT, created_at="2026-07-10T00:00:01Z")
    embedded = _comment("progress note\ns3-started-at: 2026-07-29T00:00:00Z")
    future = _comment(
        "s3-started-at: 9999-01-01T00:00:00Z",
        created_at="2026-07-30T06:30:00Z",
    )

    assert stage3_clock.started_at([valid, embedded, future], now=NOW) == datetime(
        2026,
        7,
        10,
        tzinfo=UTC,
    )


def _issue(
    *labels: str,
    state: str = "open",
    title: str = "S3-U4: Ledger + scoring predicates",
    pull_request: bool = False,
) -> dict[str, object]:
    issue: dict[str, object] = {
        "number": 305,
        "title": title,
        "state": state,
        "labels": [{"name": label} for label in labels],
    }
    if pull_request:
        issue["pull_request"] = {"url": "https://api.github.test/pulls/305"}
    return issue


def _run_deadline_check(
    monkeypatch: pytest.MonkeyPatch,
    *,
    issue: dict[str, object],
    comments: list[str],
    now: datetime = NOW,
) -> list[tuple[str, tuple[object, ...]]]:
    class FixedDateTime(datetime):
        @classmethod
        def now(cls, tz: object = None) -> FixedDateTime:
            return cls(
                now.year,
                now.month,
                now.day,
                now.hour,
                now.minute,
                now.second,
                tzinfo=UTC,
            )

    monkeypatch.setattr(stage3_clock, "datetime", FixedDateTime)
    monkeypatch.setattr(stage3_clock, "repo", lambda: "Vzlentin/calibre")
    monkeypatch.setattr(stage3_clock, "find_activation_record", lambda comments: ACTIVATION_RECORD)
    monkeypatch.setattr(stage3_clock, "milestone_issues", lambda: [issue])
    monkeypatch.setattr(
        stage3_clock,
        "issue_comments",
        lambda number: (
            [] if number == stage3_clock.GATE_ISSUE else [_comment(body) for body in comments]
        ),
    )
    actions: list[tuple[str, tuple[object, ...]]] = []
    monkeypatch.setattr(
        stage3_clock,
        "gh_api",
        lambda *args: actions.append(("label", args)),
    )
    monkeypatch.setattr(
        stage3_clock,
        "post_issue_comment",
        lambda *args: actions.append(("comment", args)),
    )

    assert stage3_clock.cmd_check_deadline() == 0
    return actions


@pytest.mark.parametrize(
    "comments",
    [
        [STARTED_AT],
        [STARTED_AT, "blocked, but without a machine-readable record"],
        [STARTED_AT, _blocker_body("waiting for input", "2026-07-30")],
    ],
)
def test_stale_active_issue_stalls_without_a_current_structured_blocker(
    monkeypatch: pytest.MonkeyPatch,
    comments: list[str],
) -> None:
    actions = _run_deadline_check(
        monkeypatch,
        issue=_issue("s3:in-progress", "s3:blocked"),
        comments=comments,
    )

    assert [kind for kind, _ in actions] == ["label", "comment"]


def test_current_structured_blocker_suspends_until_its_review_date(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    actions = _run_deadline_check(
        monkeypatch,
        issue=_issue("s3:in-progress", "s3:blocked"),
        comments=[STARTED_AT, _blocker_body("waiting for input", "2026-07-31")],
    )

    assert actions == []


def test_blocker_record_without_the_blocked_label_does_not_suspend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    actions = _run_deadline_check(
        monkeypatch,
        issue=_issue("s3:in-progress"),
        comments=[STARTED_AT, _blocker_body("waiting for input", "2026-07-31")],
    )

    assert [kind for kind, _ in actions] == ["label", "comment"]


@pytest.mark.parametrize(
    "issue",
    [
        _issue("s3:in-progress", state="closed"),
        _issue("s3:queued", "s3:in-progress"),
        _issue("s3:in-progress", title="S3-U0: Bootstrap"),
        _issue("s3:in-progress", title="S3-U9: Cost-regression tracking"),
        _issue("s3:in-progress", pull_request=True),
        _issue("s3:in-progress", "s3:stalled?"),
    ],
)
def test_non_active_or_already_stalled_items_never_write_another_stall(
    monkeypatch: pytest.MonkeyPatch,
    issue: dict[str, object],
) -> None:
    assert _run_deadline_check(monkeypatch, issue=issue, comments=[STARTED_AT]) == []


def test_exactly_fourteen_days_is_not_more_than_the_stall_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = datetime(2026, 7, 24, 0, 0, tzinfo=UTC)

    assert (
        _run_deadline_check(
            monkeypatch,
            issue=_issue("s3:in-progress"),
            comments=[STARTED_AT],
            now=now,
        )
        == []
    )


def test_issue_without_a_start_marker_cannot_stall(monkeypatch: pytest.MonkeyPatch) -> None:
    assert (
        _run_deadline_check(
            monkeypatch,
            issue=_issue("s3:in-progress"),
            comments=["progress without activation metadata"],
        )
        == []
    )


def test_latest_start_marker_owns_a_restarted_active_episode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    actions = _run_deadline_check(
        monkeypatch,
        issue=_issue("s3:in-progress"),
        comments=[STARTED_AT, "s3-started-at: 2026-07-25T00:00:00Z"],
    )

    assert actions == []


def test_invalid_newest_start_marker_does_not_abort_or_replace_the_valid_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    actions = _run_deadline_check(
        monkeypatch,
        issue=_issue("s3:in-progress"),
        comments=[STARTED_AT, "s3-started-at: 2026-02-30T00:00:00Z"],
    )

    assert [kind for kind, _ in actions] == ["label", "comment"]
