"""Stage 3 Gate A clock automation: activation, deadline, stall, heartbeat, PR gate.

Bootstrap-only root tooling (never imported by ``newcalibre``). The Gate
tracking issue is the authoritative record: activation writes an immutable
machine-readable comment there, and every other command only reads or appends.
All GitHub access goes through the ``gh`` CLI with the workflow token.

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
from datetime import UTC, datetime, timedelta

GATE_ISSUE = 301
S3_U1_ISSUE = 302
MILESTONE_TITLE = "Stage 3 — successor build"
WINDOW_DAYS = 42
STALL_DAYS = 14
CLOCK_START_LABEL = "s3:clock-start"
ACTIVATION_MARKER = "<!-- s3-clock-activation -->"
EXPIRY_MARKER = "<!-- s3-clock-expired -->"
DECISION_MARKER = "<!-- s3-gate-decision -->"
HEARTBEAT_MARKER = "<!-- s3-heartbeat"
STARTED_AT_RE = re.compile(r"s3-started-at:\s*(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z)")
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


def find_activation_record() -> dict | None:
    """Parse the Gate issue's activation record, if one exists."""
    for comment in issue_comments(GATE_ISSUE):
        body = comment.get("body", "")
        if ACTIVATION_MARKER in body:
            match = re.search(r"```json\s*(\{.*?\})\s*```", body, re.DOTALL)
            if not match:
                return {"malformed": True}
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                return {"malformed": True}
    return None


def record_is_schema_complete(record: dict | None) -> bool:
    """Check the activation record against the published schema."""
    if not record or record.get("malformed"):
        return False
    if not re.fullmatch(r"[0-9a-f]{40}", str(record.get("merge_sha", ""))):
        return False
    for field in ("merged_at", "deadline"):
        try:
            datetime.fromisoformat(str(record.get(field, "")).replace("Z", "+00:00"))
        except ValueError:
            return False
    return isinstance(record.get("pr"), int)


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

    merged_at = datetime.fromisoformat(pr["merged_at"].replace("Z", "+00:00"))
    deadline = merged_at + timedelta(days=WINDOW_DAYS)
    record = {
        "merge_sha": pr["merge_commit_sha"],
        "merged_at": merged_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "deadline": deadline.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "pr": pr_number,
    }
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


def started_at(issue: dict) -> datetime | None:
    """Read the s3-started-at timestamp recorded on an active issue."""
    for comment in issue_comments(issue["number"]):
        match = STARTED_AT_RE.search(comment.get("body", ""))
        if match:
            return datetime.fromisoformat(match.group(1).replace("Z", "+00:00"))
    return None


def cmd_check_deadline() -> int:
    """Escalate at the immutable deadline; flag stalled active units."""
    now = datetime.now(UTC)
    record = find_activation_record()
    comments = issue_comments(GATE_ISSUE)
    all_bodies = "\n".join(c.get("body", "") for c in comments)

    if record_is_schema_complete(record):
        deadline = datetime.fromisoformat(record["deadline"].replace("Z", "+00:00"))
        has_decision = DECISION_MARKER in all_bodies
        already_escalated = EXPIRY_MARKER in all_bodies
        if now > deadline and not has_decision and not already_escalated:
            post_issue_comment(
                GATE_ISSUE,
                f"{EXPIRY_MARKER}\n## Gate A window expired without a recorded decision\n\n"
                f"The immutable deadline `{record['deadline']}` has passed and no "
                f"`{DECISION_MARKER}` comment exists. Per the ratified abort criterion "
                "this records **no-go by default**: successor work halts pending the "
                "owner disposition (re-scope once ≤ 3 weeks / re-spec / abandon).",
            )
            print("Deadline escalation recorded.")
    else:
        print("No activation record yet; pre-clock phase — deadline check idle.")

    # Stall tripwire: only actively started U1-U8 issues can stall.
    for issue in milestone_issues():
        labels = {label["name"] for label in issue.get("labels", [])}
        if "s3:in-progress" not in labels or "s3:blocked" in labels:
            continue
        match = re.match(r"S3-U(\d+):", issue.get("title", ""))
        if not match or int(match.group(1)) > 8:
            continue
        begun = started_at(issue)
        if begun and now - begun > timedelta(days=STALL_DAYS) and "s3:stalled?" not in labels:
            gh_api(
                f"repos/{repo()}/issues/{issue['number']}/labels",
                "-f",
                "labels[]=s3:stalled?",
            )
            post_issue_comment(
                issue["number"],
                f"`s3:stalled?` — active since {begun.strftime('%Y-%m-%dT%H:%M:%SZ')} "
                f"(> {STALL_DAYS} days) without a machine-readable blocker. "
                "Owner review required before the gate date.",
            )
            print(f"Stall flagged on #{issue['number']}.")
    return 0


def cmd_heartbeat() -> int:
    """Record the weekly sublanding-state heartbeat on the Gate issue."""
    now = datetime.now(UTC)
    week = now.strftime("%G-W%V")
    marker = f"{HEARTBEAT_MARKER} {week} -->"
    if any(marker in c.get("body", "") for c in issue_comments(GATE_ISSUE)):
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
    parser = argparse.ArgumentParser(description=__doc__)
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
