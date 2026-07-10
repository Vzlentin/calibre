"""Stage 3 Gate A clock automation: activation, deadline, stall, heartbeat, PR gate.

Bootstrap-only root tooling (never imported by ``newcalibre``). The Gate
tracking issue is the authoritative record: activation writes an immutable
machine-readable comment there, and every other command only reads or appends.
All GitHub access goes through the ``gh`` CLI with the workflow token.
Control records use an exact first line and GitHub's immutable comment metadata.
An active transition starts with ``s3-started-at: YYYY-MM-DDTHH:MM:SSZ``. A
current blocker starts with ``<!-- s3-blocker -->`` and carries one fenced JSON
object with exactly ``description`` and ``next_review`` (``YYYY-MM-DD``); its
stall exemption expires on that UTC review date. A Gate decision starts with
``<!-- s3-gate-decision -->`` and carries one of the closed JSON records
validated by :func:`decision_record_is_complete`. Operator-written records
must come from an owner/member/collaborator; Gate decisions are owner-only.

Copy-paste templates (replace angle-bracket values)::

    s3-started-at: <YYYY-MM-DDTHH:MM:SSZ>

    <!-- s3-blocker -->
    ```json
    {"description": "<why work is blocked>", "next_review": "<YYYY-MM-DD>"}
    ```

    <!-- s3-gate-decision -->
    ```json
    {"kind": "gate", "decision": "go", "disposition": "mint-b1-b5",
     "candidate_sha": "<40-lower-hex>", "report_sha256": "<64-lower-hex>",
     "report_emitted_at": "<YYYY-MM-DDTHH:MM:SSZ>"}
    ```

    <!-- s3-gate-decision -->
    ```json
    {"kind": "early-halt", "decision": "no-go", "disposition": "abandon"}
    ```

For a normal Gate result, ``go`` pairs only with ``mint-b1-b5`` and ``no-go``
pairs with exactly one of ``rescope-once-max-3-weeks``, ``respec``, or
``abandon``. The report must predate the unedited decision comment, whose
GitHub creation time must not exceed the immutable deadline.
Use ``gh api repos/OWNER/REPO/dispatches -f event_type=stage3-check-deadline``
or ``event_type=stage3-heartbeat`` for a default-branch manual run.

Commands:
    activate --pr N   Record the immutable clock start from a merged PR event.
    gate --pr N       Required-check logic: fail successor PRs while the Gate
                      issue lacks a schema-complete activation record.
    check-deadline    Daily: escalate at the deadline; flag stalled units.
    heartbeat         Weekly: record sublanding states on the Gate issue.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import UTC, date, datetime, timedelta

GATE_ISSUE = 301
S3_U1_ISSUE = 302
MILESTONE_TITLE = "Stage 3 — successor build"
WINDOW_DAYS = 42
STALL_DAYS = 14
CLOCK_START_LABEL = "s3:clock-start"
ACTIVATION_MARKER = "<!-- s3-clock-activation -->"
BLOCKER_MARKER = "<!-- s3-blocker -->"
EXPIRY_MARKER = "<!-- s3-clock-expired -->"
DECISION_MARKER = "<!-- s3-gate-decision -->"
HEARTBEAT_MARKER = "<!-- s3-heartbeat"
STARTED_AT_RE = re.compile(r"s3-started-at:\s*(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z)")
UTC_TIMESTAMP_RE = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z")
TRACKER_OPERATOR_ASSOCIATIONS = frozenset({"OWNER", "MEMBER", "COLLABORATOR"})
GATE_DISPOSITIONS = {
    "go": frozenset({"mint-b1-b5"}),
    "no-go": frozenset({"rescope-once-max-3-weeks", "respec", "abandon"}),
}
EARLY_HALT_DISPOSITIONS = GATE_DISPOSITIONS["no-go"]
SUCCESSOR_PATTERNS = (
    "newcalibre/",
    "stage3/",
    ".github/workflows/newcalibre.yml",
    ".github/workflows/oracle-capture.yml",
    ".github/workflows/gate-a.yml",
    ".github/workflows/stage3-clock.yml",
    ".github/scripts/stage3_",
)


def repo() -> str:
    """Return the owner/name slug the workflow runs in."""
    return os.environ["GITHUB_REPOSITORY"]


def gh_api(path: str, *args: str) -> object:
    """Call ``gh api`` and parse the JSON response."""
    out = subprocess.run(
        ["gh", "api", path, *args], check=True, capture_output=True, text=True
    ).stdout
    return json.loads(out)


def gh_api_paginated(path: str) -> list:
    """Call ``gh api --paginate`` for list endpoints."""
    out = subprocess.run(
        ["gh", "api", "--paginate", path, "--slurp"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    pages = json.loads(out)
    return [item for page in pages for item in page]


def post_issue_comment(issue: int, body: str) -> None:
    """Append a comment to an issue."""
    subprocess.run(
        ["gh", "api", f"repos/{repo()}/issues/{issue}/comments", "-f", f"body={body}"],
        check=True,
        capture_output=True,
        text=True,
    )


def issue_comments(issue: int) -> list:
    """Return every comment on an issue, oldest first."""
    return gh_api_paginated(f"repos/{repo()}/issues/{issue}/comments")


def comment_from_github_actions(comment: dict) -> bool:
    """Accept a machine record only from this repository's Actions application."""
    user = comment.get("user")
    app = comment.get("performed_via_github_app")
    return (
        isinstance(user, dict)
        and user.get("login") == "github-actions[bot]"
        and user.get("type") == "Bot"
        and isinstance(app, dict)
        and app.get("slug") == "github-actions"
    )


def comment_from_tracker_operator(comment: dict) -> bool:
    """Accept human tracker metadata only from repository operators."""
    return comment.get("author_association") in TRACKER_OPERATOR_ASSOCIATIONS


def comment_has_leading_marker(comment: dict, marker: str) -> bool:
    """Require a control marker to be the comment's exact first line."""
    body = comment.get("body")
    return isinstance(body, str) and bool(body.splitlines()) and body.splitlines()[0] == marker


def single_fenced_json_object(comment: dict) -> dict | None:
    """Parse exactly one fenced JSON object from a control comment."""
    body = comment.get("body")
    if not isinstance(body, str):
        return None
    matches = re.findall(r"```json\s*(.*?)\s*```", body, re.DOTALL)
    if len(matches) != 1:
        return None
    try:
        record = json.loads(matches[0])
    except (json.JSONDecodeError, RecursionError):
        return None
    return record if isinstance(record, dict) else None


def parse_utc_timestamp(value: object) -> datetime | None:
    """Parse the exact second-resolution UTC timestamp used by tracker records."""
    if not isinstance(value, str) or not UTC_TIMESTAMP_RE.fullmatch(value):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def find_activation_record(comments: list | None = None) -> dict | None:
    """Find the first schema-complete record backed by immutable PR events."""
    comments_to_read = issue_comments(GATE_ISSUE) if comments is None else comments
    for comment in comments_to_read:
        if not comment_from_github_actions(comment) or not comment_has_leading_marker(
            comment, ACTIVATION_MARKER
        ):
            continue
        record = single_fenced_json_object(comment)
        if record_is_schema_complete(record) and activation_record_matches_pull_request(record):
            return record
    return None


def record_is_schema_complete(record: dict | None) -> bool:
    """Check the activation record against the published schema."""
    if not isinstance(record, dict) or set(record) != {"merge_sha", "merged_at", "deadline", "pr"}:
        return False
    merge_sha = record["merge_sha"]
    if not isinstance(merge_sha, str) or not re.fullmatch(r"[0-9a-f]{40}", merge_sha):
        return False
    merged_at = parse_utc_timestamp(record["merged_at"])
    deadline = parse_utc_timestamp(record["deadline"])
    pr_number = record["pr"]
    try:
        expected_deadline = (
            merged_at + timedelta(days=WINDOW_DAYS) if merged_at is not None else None
        )
    except OverflowError:
        return False
    return (
        merged_at is not None
        and deadline is not None
        and deadline == expected_deadline
        and type(pr_number) is int
        and pr_number > 0
    )


def activation_record_matches_pull_request(record: dict) -> bool:
    """Bind an activation record to immutable merge, label, and U1-link events."""
    if not record_is_schema_complete(record):
        return False
    pr_number = record["pr"]
    pr = gh_api(f"repos/{repo()}/pulls/{pr_number}")
    if not isinstance(pr, dict):
        return False
    if (
        not pr.get("merged")
        or pr.get("merge_commit_sha") != record["merge_sha"]
        or pr.get("merged_at") != record["merged_at"]
        or pr.get("base", {}).get("ref") != "main"
    ):
        return False

    merged_at = parse_utc_timestamp(record["merged_at"])
    if merged_at is None:
        return False
    pr_events = gh_api_paginated(f"repos/{repo()}/issues/{pr_number}/events?per_page=100")
    matching_merge = any(
        event.get("event") == "merged"
        and event.get("commit_id") == record["merge_sha"]
        and parse_utc_timestamp(event.get("created_at")) == merged_at
        for event in pr_events
    )
    label_events = []
    for position, event in enumerate(pr_events):
        if event.get("event") not in {"labeled", "unlabeled"}:
            continue
        label = event.get("label")
        event_at = parse_utc_timestamp(event.get("created_at"))
        if (
            isinstance(label, dict)
            and label.get("name") == CLOCK_START_LABEL
            and event_at is not None
            and event_at <= merged_at
        ):
            label_events.append((event_at, position, event.get("event")))
    labeled_at_merge = bool(label_events) and max(label_events)[2] == "labeled"

    u1_timeline = gh_api_paginated(f"repos/{repo()}/issues/{S3_U1_ISSUE}/timeline?per_page=100")
    linked_before_merge = any(
        event.get("event") == "cross-referenced"
        and parse_utc_timestamp(event.get("created_at")) is not None
        and parse_utc_timestamp(event.get("created_at")) <= merged_at
        and isinstance(event.get("source"), dict)
        and isinstance(event["source"].get("issue"), dict)
        and event["source"]["issue"].get("number") == pr_number
        and event["source"]["issue"].get("repository_url")
        == f"https://api.github.com/repos/{repo()}"
        and "pull_request" in event["source"]["issue"]
        for event in u1_timeline
    )
    return matching_merge and labeled_at_merge and linked_before_merge


def cmd_activate(pr_number: int) -> int:
    """Record the immutable clock start from the merged clock-start PR."""
    pr = gh_api(f"repos/{repo()}/pulls/{pr_number}")
    labels = {label["name"] for label in pr.get("labels", [])}
    if CLOCK_START_LABEL not in labels:
        print(f"PR #{pr_number} lacks {CLOCK_START_LABEL}; refusing activation.")
        return 0  # not the activation PR; nothing to do
    if not pr.get("merged"):
        print(f"PR #{pr_number} carries {CLOCK_START_LABEL} but is not merged; refusing.")
        return 1
    body = pr.get("body") or ""
    title = pr.get("title") or ""
    if f"#{S3_U1_ISSUE}" not in body and "S3-U1" not in f"{title} {body}":
        print(
            f"PR #{pr_number} carries {CLOCK_START_LABEL} but does not link "
            f"S3-U1 (#{S3_U1_ISSUE}); refusing activation."
        )
        return 1
    if find_activation_record() is not None:
        print("An activation record already exists on the Gate issue; refusing a second clock.")
        return 1

    merged_at = parse_utc_timestamp(pr.get("merged_at"))
    if merged_at is None:
        print(f"PR #{pr_number} has no exact UTC merged_at timestamp; refusing activation.")
        return 1
    try:
        deadline = merged_at + timedelta(days=WINDOW_DAYS)
    except OverflowError:
        print(f"PR #{pr_number} has an out-of-range merged_at timestamp; refusing activation.")
        return 1
    record = {
        "merge_sha": pr["merge_commit_sha"],
        "merged_at": merged_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "deadline": deadline.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "pr": pr_number,
    }
    if not record_is_schema_complete(record):
        print(f"PR #{pr_number} produced an invalid activation record; refusing activation.")
        return 1
    if not activation_record_matches_pull_request(record):
        print(
            f"PR #{pr_number} lacks immutable merge-time clock-label/U1-link proof; "
            "refusing activation."
        )
        return 1
    comment = (
        f"{ACTIVATION_MARKER}\n## Clock activation record (immutable)\n\n"
        f"Written by `stage3-clock` from GitHub's merge event for PR #{pr_number} "
        f"(S3-U1 first landing). This record — not the milestone — is authoritative.\n\n"
        f"```json\n{json.dumps(record, indent=2)}\n```\n"
    )
    post_issue_comment(GATE_ISSUE, comment)

    milestones = gh_api_paginated(f"repos/{repo()}/milestones?state=open")
    for milestone in milestones:
        if milestone["title"] == MILESTONE_TITLE:
            gh_api(
                f"repos/{repo()}/milestones/{milestone['number']}",
                "-X",
                "PATCH",
                "-f",
                f"due_on={record['deadline']}",
            )
            break
    print(f"Clock activated: deadline {record['deadline']}.")
    return 0


def pr_touches_successor(pr_number: int) -> bool:
    """Check whether a PR changes any successor-scoped path."""
    files = gh_api_paginated(f"repos/{repo()}/pulls/{pr_number}/files")
    return any(f["filename"].startswith(SUCCESSOR_PATTERNS) for f in files)


def base_has_successor(base_sha: str) -> bool:
    """Check whether the PR base tree already contains the successor project."""
    try:
        gh_api(f"repos/{repo()}/contents/newcalibre/pyproject.toml?ref={base_sha}")
        return True
    except subprocess.CalledProcessError:
        return False


def cmd_gate(pr_number: int) -> int:
    """Fail successor PRs while the activation record is missing post-U1a."""
    if not pr_touches_successor(pr_number):
        print("Not a Stage 3 PR; activation gate passes.")
        return 0
    base_sha = gh_api(f"repos/{repo()}/pulls/{pr_number}")["base"]["sha"]
    if not base_has_successor(base_sha):
        print("Pre-clock Stage 3 work (base has no successor project); gate passes.")
        return 0
    if record_is_schema_complete(find_activation_record()):
        print("Schema-complete activation record present; gate passes.")
        return 0
    print(
        "BLOCKED: the successor project exists on main but the Gate issue "
        f"(#{GATE_ISSUE}) carries no schema-complete clock activation record. "
        "U1a's merge event must be recorded before further Stage 3 work lands."
    )
    return 1


def milestone_issues() -> list:
    """Return every issue on the Stage 3 milestone."""
    milestones = gh_api_paginated(f"repos/{repo()}/milestones?state=open")
    number = next((m["number"] for m in milestones if m["title"] == MILESTONE_TITLE), None)
    if number is None:
        return []
    return gh_api_paginated(f"repos/{repo()}/issues?milestone={number}&state=all&per_page=100")


def blocker_suspends_stall(comments: list, *, today: date) -> bool:
    """Suspend only when the newest blocker comment is valid and unexpired."""
    for comment in reversed(comments):
        if not comment_from_tracker_operator(comment) or not comment_has_leading_marker(
            comment, BLOCKER_MARKER
        ):
            continue
        record = single_fenced_json_object(comment)
        if record is None or set(record) != {"description", "next_review"}:
            return False
        description = record["description"]
        next_review = record["next_review"]
        if not isinstance(description, str) or not description.strip():
            return False
        if not isinstance(next_review, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", next_review):
            return False
        try:
            review_date = date.fromisoformat(next_review)
        except ValueError:
            return False
        return today < review_date
    return False


def started_at(comments: list, *, now: datetime) -> datetime | None:
    """Read the latest s3-started-at timestamp from complete issue comments."""
    for comment in reversed(comments):
        if not comment_from_tracker_operator(comment):
            continue
        body = comment.get("body")
        first_line = body.splitlines()[0] if isinstance(body, str) and body.splitlines() else ""
        match = STARTED_AT_RE.fullmatch(first_line)
        if match:
            parsed = parse_utc_timestamp(match.group(1))
            created_at = parse_utc_timestamp(comment.get("created_at"))
            if (
                parsed is not None
                and created_at is not None
                and parsed <= created_at
                and parsed <= now
            ):
                return parsed
    return None


def decision_record_is_complete(
    record: dict | None,
    *,
    comment_created_at: datetime,
) -> bool:
    """Validate either a report-bound Gate result or an explicit early halt."""
    if not isinstance(record, dict):
        return False
    kind = record.get("kind")
    if not isinstance(kind, str) or kind not in {"gate", "early-halt"}:
        return False
    if kind == "early-halt":
        disposition = record.get("disposition")
        return (
            set(record) == {"kind", "decision", "disposition"}
            and record.get("decision") == "no-go"
            and isinstance(disposition, str)
            and disposition in EARLY_HALT_DISPOSITIONS
        )
    if set(record) != {
        "kind",
        "decision",
        "disposition",
        "candidate_sha",
        "report_sha256",
        "report_emitted_at",
    }:
        return False
    decision = record.get("decision")
    disposition = record.get("disposition")
    candidate_sha = record.get("candidate_sha")
    report_sha256 = record.get("report_sha256")
    report_emitted_at = parse_utc_timestamp(record.get("report_emitted_at"))
    return (
        isinstance(decision, str)
        and decision in GATE_DISPOSITIONS
        and isinstance(disposition, str)
        and disposition in GATE_DISPOSITIONS[decision]
        and isinstance(candidate_sha, str)
        and re.fullmatch(r"[0-9a-f]{40}", candidate_sha) is not None
        and isinstance(report_sha256, str)
        and re.fullmatch(r"[0-9a-f]{64}", report_sha256) is not None
        and report_emitted_at is not None
        and report_emitted_at <= comment_created_at
    )


def comment_is_timely_owner_decision(comment: dict, *, deadline: datetime) -> bool:
    """Accept a closed, unedited owner decision created by the Gate deadline."""
    if comment.get("author_association") != "OWNER" or not comment_has_leading_marker(
        comment, DECISION_MARKER
    ):
        return False
    created_at = parse_utc_timestamp(comment.get("created_at"))
    updated_at = parse_utc_timestamp(comment.get("updated_at"))
    record = single_fenced_json_object(comment)
    return (
        created_at is not None
        and updated_at == created_at
        and created_at <= deadline
        and decision_record_is_complete(record, comment_created_at=created_at)
    )


def comment_is_valid_expiry(
    comment: dict,
    *,
    activation_record: dict,
    deadline: datetime,
) -> bool:
    """Accept only an immutable post-deadline expiry for this activation."""
    if not comment_from_github_actions(comment) or not comment_has_leading_marker(
        comment, EXPIRY_MARKER
    ):
        return False
    created_at = parse_utc_timestamp(comment.get("created_at"))
    updated_at = parse_utc_timestamp(comment.get("updated_at"))
    expiry_record = single_fenced_json_object(comment)
    return (
        created_at is not None
        and updated_at == created_at
        and created_at > deadline
        and expiry_record
        == {
            "activation_merge_sha": activation_record["merge_sha"],
            "deadline": activation_record["deadline"],
        }
    )


def expiry_comment(record: dict) -> str:
    """Render the machine-bound default no-go record."""
    expiry_record = {
        "activation_merge_sha": record["merge_sha"],
        "deadline": record["deadline"],
    }
    return (
        f"{EXPIRY_MARKER}\n## Gate A window expired without a recorded decision\n\n"
        f"The immutable deadline `{record['deadline']}` has passed and no "
        f"`{DECISION_MARKER}` comment exists. Per the ratified abort criterion "
        "this records **no-go by default**: successor work halts pending the "
        "owner disposition (re-scope once ≤ 3 weeks / re-spec / abandon).\n\n"
        f"```json\n{json.dumps(expiry_record, indent=2)}\n```"
    )


def cmd_check_deadline() -> int:
    """Escalate at the immutable deadline; flag stalled active units."""
    now = datetime.now(UTC)
    gate_comments = issue_comments(GATE_ISSUE)
    record = find_activation_record(gate_comments)

    if record_is_schema_complete(record):
        deadline = parse_utc_timestamp(record["deadline"])
        if deadline is None:
            raise RuntimeError("schema-complete activation record has no UTC deadline")
        has_decision = any(
            comment_is_timely_owner_decision(comment, deadline=deadline)
            for comment in gate_comments
        )
        already_escalated = any(
            comment_is_valid_expiry(
                comment,
                activation_record=record,
                deadline=deadline,
            )
            for comment in gate_comments
        )
        if now > deadline and not has_decision and not already_escalated:
            post_issue_comment(GATE_ISSUE, expiry_comment(record))
            print("Deadline escalation recorded.")
    else:
        print("No activation record yet; pre-clock phase — deadline check idle.")

    # Stall tripwire: only actively started U1-U8 issues can stall.
    for issue in milestone_issues():
        labels = {label["name"] for label in issue.get("labels", [])}
        if (
            issue.get("state") != "open"
            or "pull_request" in issue
            or "s3:in-progress" not in labels
            or "s3:queued" in labels
            or "s3:stalled?" in labels
        ):
            continue
        match = re.match(r"S3-U(\d+):", issue.get("title", ""))
        if not match or not 1 <= int(match.group(1)) <= 8:
            continue
        unit_comments = issue_comments(issue["number"])
        if "s3:blocked" in labels and blocker_suspends_stall(unit_comments, today=now.date()):
            continue
        begun = started_at(unit_comments, now=now)
        if begun and now - begun > timedelta(days=STALL_DAYS):
            gh_api(
                f"repos/{repo()}/issues/{issue['number']}/labels",
                "-f",
                "labels[]=s3:stalled?",
            )
            post_issue_comment(
                issue["number"],
                f"`s3:stalled?` — active since {begun.strftime('%Y-%m-%dT%H:%M:%SZ')} "
                f"(> {STALL_DAYS} days) without a current schema-complete blocker. "
                "Owner review required before the gate date.",
            )
            print(f"Stall flagged on #{issue['number']}.")
    return 0


def cmd_heartbeat() -> int:
    """Record the weekly sublanding-state heartbeat on the Gate issue."""
    now = datetime.now(UTC)
    week = now.strftime("%G-W%V")
    marker = f"{HEARTBEAT_MARKER} {week} -->"
    if any(
        comment_from_github_actions(comment) and comment_has_leading_marker(comment, marker)
        for comment in issue_comments(GATE_ISSUE)
    ):
        print(f"Heartbeat for {week} already recorded.")
        return 0
    lines = [f"{marker}\n## Heartbeat {week} (non-gating)\n"]
    for issue in sorted(milestone_issues(), key=lambda i: i["number"]):
        labels = ", ".join(sorted(label["name"] for label in issue.get("labels", [])))
        lines.append(
            f"- #{issue['number']} {issue['title']} — {issue['state']}"
            + (f" [{labels}]" if labels else "")
        )
    post_issue_comment(GATE_ISSUE, "\n".join(lines))
    print(f"Heartbeat {week} recorded.")
    return 0


def main() -> int:
    """Dispatch the requested clock command."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)
    activate = sub.add_parser("activate")
    activate.add_argument("--pr", type=int, required=True)
    gate = sub.add_parser("gate")
    gate.add_argument("--pr", type=int, required=True)
    sub.add_parser("check-deadline")
    sub.add_parser("heartbeat")
    args = parser.parse_args()
    if args.command == "activate":
        return cmd_activate(args.pr)
    if args.command == "gate":
        return cmd_gate(args.pr)
    if args.command == "check-deadline":
        return cmd_check_deadline()
    return cmd_heartbeat()


if __name__ == "__main__":
    sys.exit(main())
