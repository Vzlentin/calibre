"""Test the retained Stage 3 clock and heartbeat automation."""

from __future__ import annotations

import importlib.util
import json
from datetime import UTC, date, datetime
from pathlib import Path
from types import ModuleType

import pytest
import yaml

ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / ".github" / "scripts" / "stage3_clock.py"
WORKFLOW = ROOT / ".github" / "workflows" / "stage3-clock.yml"


def _load_script(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


stage3_clock = _load_script(SCRIPT)


def _comment(marker: str, record: dict[str, object], *, actor: str = "OWNER") -> dict:
    body = f"{marker}\n```json\n{json.dumps(record)}\n```"
    return {
        "body": body,
        "author_association": actor,
        "created_at": "2026-07-16T12:00:00Z",
        "updated_at": "2026-07-16T12:00:00Z",
    }


def test_clock_workflow_retains_only_deadline_and_heartbeat_authority() -> None:
    """Keep the pending owner-decision clock without deleted evidence workflows."""
    workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))

    assert set(workflow["jobs"]) == {"activate", "check-deadline", "heartbeat"}
    assert workflow["permissions"] == {"contents": "read"}
    assert set(workflow[True]["repository_dispatch"]["types"]) == {
        "stage3-check-deadline",
        "stage3-heartbeat",
    }


def test_activation_record_requires_exact_immutable_shape() -> None:
    """Accept only a six-week activation record bound to a full merge SHA."""
    record = {
        "merge_sha": "a" * 40,
        "merged_at": "2026-07-01T12:00:00Z",
        "deadline": "2026-08-12T12:00:00Z",
        "pr": 331,
    }

    assert stage3_clock.record_is_schema_complete(record)
    assert not stage3_clock.record_is_schema_complete({**record, "extra": True})
    assert not stage3_clock.record_is_schema_complete(
        {**record, "deadline": "2026-08-11T12:00:00Z"}
    )


def test_gate_decision_requires_owner_timeliness_and_closed_disposition() -> None:
    """Retain the pending owner go/no-go decision contract."""
    record = {
        "kind": "gate",
        "decision": "go",
        "disposition": "mint-b1-b5",
        "candidate_sha": "a" * 40,
        "report_sha256": "b" * 64,
        "report_emitted_at": "2026-07-16T11:00:00Z",
    }
    comment = _comment(stage3_clock.DECISION_MARKER, record)
    deadline = datetime(2026, 7, 17, tzinfo=UTC)

    assert stage3_clock.comment_is_timely_owner_decision(comment, deadline=deadline)
    assert not stage3_clock.comment_is_timely_owner_decision(
        {**comment, "author_association": "MEMBER"}, deadline=deadline
    )
    assert not stage3_clock.decision_record_is_complete(
        {**record, "disposition": "abandon"},
        comment_created_at=datetime(2026, 7, 16, 12, tzinfo=UTC),
    )


def test_expiry_comment_round_trips_the_activation_identity() -> None:
    """Bind a default no-go escalation to the exact activation and deadline."""
    activation = {
        "merge_sha": "a" * 40,
        "merged_at": "2026-06-01T12:00:00Z",
        "deadline": "2026-07-13T12:00:00Z",
        "pr": 331,
    }
    comment = {
        **_comment(
            stage3_clock.EXPIRY_MARKER,
            {
                "activation_merge_sha": activation["merge_sha"],
                "deadline": activation["deadline"],
            },
            actor="NONE",
        ),
        "created_at": "2026-07-14T12:00:00Z",
        "updated_at": "2026-07-14T12:00:00Z",
        "user": {"login": "github-actions[bot]", "type": "Bot"},
        "performed_via_github_app": {"slug": "github-actions"},
    }

    assert stage3_clock.comment_is_valid_expiry(
        comment,
        activation_record=activation,
        deadline=datetime(2026, 7, 13, 12, tzinfo=UTC),
    )


def test_blocker_suspension_requires_operator_and_future_review() -> None:
    """Suspend the stall tripwire only for a current operator blocker."""
    record = {"description": "awaiting owner decision", "next_review": "2026-07-20"}
    valid = _comment(stage3_clock.BLOCKER_MARKER, record, actor="MEMBER")
    outsider = _comment(stage3_clock.BLOCKER_MARKER, record, actor="NONE")

    assert stage3_clock.blocker_suspends_stall([valid], today=date(2026, 7, 16))
    assert not stage3_clock.blocker_suspends_stall([valid], today=date(2026, 7, 20))
    assert not stage3_clock.blocker_suspends_stall([outsider], today=date(2026, 7, 16))


def test_successor_gate_uses_only_live_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep clock gating free of deleted workflow path requirements."""
    assert ".github/workflows/vn2-evidence.yml" in stage3_clock.SUCCESSOR_PATTERNS
    assert all("gate-" + "a.yml" not in path for path in stage3_clock.SUCCESSOR_PATTERNS)
    assert all("oracle-" + "capture.yml" not in path for path in stage3_clock.SUCCESSOR_PATTERNS)

    monkeypatch.setattr(stage3_clock, "pr_touches_successor", lambda _number: True)
    monkeypatch.setattr(stage3_clock, "repo", lambda: "Vzlentin/calibre")
    monkeypatch.setattr(stage3_clock, "gh_api", lambda _path: {"base": {"sha": "a" * 40}})
    monkeypatch.setattr(stage3_clock, "base_has_successor", lambda _sha: True)
    monkeypatch.setattr(stage3_clock, "find_activation_record", lambda: None)
    assert stage3_clock.cmd_gate(390) == 1


def test_heartbeat_is_idempotent_for_actions_marker(monkeypatch: pytest.MonkeyPatch) -> None:
    """Do not append a duplicate heartbeat for the same ISO week."""
    marker = f"{stage3_clock.HEARTBEAT_MARKER} 2026-W29 -->"
    existing = {
        "body": marker,
        "user": {"login": "github-actions[bot]", "type": "Bot"},
        "performed_via_github_app": {"slug": "github-actions"},
    }
    posts: list[object] = []

    class FixedDateTime(datetime):
        @classmethod
        def now(cls, tz: object = None) -> FixedDateTime:
            del tz
            return cls(2026, 7, 16, tzinfo=UTC)

    monkeypatch.setattr(stage3_clock, "datetime", FixedDateTime)
    monkeypatch.setattr(stage3_clock, "issue_comments", lambda _issue: [existing])
    monkeypatch.setattr(stage3_clock, "post_issue_comment", lambda *args: posts.append(args))

    assert stage3_clock.cmd_heartbeat() == 0
    assert posts == []
