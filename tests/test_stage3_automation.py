"""Exercise the Stage 3 tracker and Gate automation contracts."""

from __future__ import annotations

import importlib.util
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType

import pytest

SCRIPTS_DIR = Path(__file__).parents[1] / ".github" / "scripts"

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
    monkeypatch.setattr(
        stage3_clock,
        "issue_comments",
        lambda issue: [{"body": "unrelated"}, {"body": ACTIVATION_BODY}],
    )

    record = stage3_gate_report.find_activation_record()

    assert record == ACTIVATION_RECORD
    assert stage3_gate_report.record_is_schema_complete(record)
    assert stage3_gate_report.find_activation_record is stage3_clock.find_activation_record


def test_deadline_check_uses_one_gate_comment_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    class FixedDateTime(datetime):
        @classmethod
        def now(cls, tz: object = None) -> FixedDateTime:
            return cls(2026, 7, 30, 6, 30, tzinfo=UTC)

    reads: list[int] = []

    def read_comments(issue: int) -> list[dict[str, str]]:
        reads.append(issue)
        return [{"body": ACTIVATION_BODY}]

    monkeypatch.setattr(stage3_clock, "datetime", FixedDateTime)
    monkeypatch.setattr(stage3_clock, "issue_comments", read_comments)
    monkeypatch.setattr(stage3_clock, "milestone_issues", lambda: [])
    monkeypatch.setattr(
        stage3_clock,
        "post_issue_comment",
        lambda *args: pytest.fail(f"unexpected comment write: {args}"),
    )

    assert stage3_clock.cmd_check_deadline() == 0
    assert reads == [stage3_clock.GATE_ISSUE]


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
    records = [{"body": body} for body in comments]

    assert stage3_clock.blocker_suspends_stall(records, today=NOW.date()) is expected


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
            [] if number == stage3_clock.GATE_ISSUE else [{"body": body} for body in comments]
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
