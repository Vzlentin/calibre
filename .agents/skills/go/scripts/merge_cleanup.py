"""Squash-merge a /go PR and run merge-gated cleanup by execution mode.

``merge <pr> --mode {direct,worktree} --branch <type>/<slug>`` verifies the PR
body carries a ``closes #N`` handle (refusing to merge otherwise), squash-merges
with ``gh pr merge --squash --delete-branch``, and only then cleans up:

- **direct** — the main checkout is on the PR branch: return to ``main``,
  fast-forward, force-delete the local branch.
- **worktree** — the main checkout never left the user's branch/dirty tree:
  remove ``.worktrees/<slug>``, force-delete the branch, prune, and
  fast-forward the local ``main`` ref via ``git fetch origin main:main`` —
  never ``git checkout``/``git pull`` in the caller's tree. The ref update is
  skipped when the caller has ``main`` checked out (``fetch`` refuses to move
  a checked-out branch).

Preserving is the default: ``--no-merge`` and every failure path leave the
branch, worktree, and PR intact; cleanup runs only after the merge command
confirms success. A squash-merged branch never shows as merged to git, so the
local delete is always ``git branch -D``.

Exit codes: 0 merged (and cleaned up), 1 refused or failed (nothing deleted).
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

WORKTREES_DIR = ".worktrees"

_CLOSES_RE = re.compile(r"\bclose[sd]?\s+#\d+", re.IGNORECASE)


def _run(args: list[str], cwd: str | Path | None = None) -> subprocess.CompletedProcess[str]:
    """Run a command, capturing text output without raising on failure."""
    return subprocess.run(args, cwd=cwd, capture_output=True, text=True, check=False)


def has_close_handle(body: str) -> bool:
    """Report whether a PR body carries a ``closes #N`` issue handle."""
    return bool(_CLOSES_RE.search(body))


def cleanup_commands(mode: str, branch: str, main_on_main: bool) -> list[list[str]]:
    """Build the post-merge cleanup command sequence for one execution mode.

    Args:
        mode: ``"direct"`` or ``"worktree"``.
        branch: The PR's ``<type>/<slug>`` branch name.
        main_on_main: Whether the main checkout currently has ``main`` checked
            out; a checked-out branch cannot be moved by ``fetch main:main``.

    Returns:
        Git command argv lists to run, in order, from the main checkout.
    """
    slug = branch.split("/", 1)[1] if "/" in branch else branch
    if mode == "direct":
        return [
            ["git", "checkout", "main"],
            ["git", "pull", "--ff-only"],
            ["git", "branch", "-D", branch],
        ]
    commands = [
        ["git", "worktree", "remove", "--force", f"{WORKTREES_DIR}/{slug}"],
        ["git", "branch", "-D", branch],
        ["git", "worktree", "prune"],
    ]
    if not main_on_main:
        commands.append(["git", "fetch", "origin", "main:main"])
    return commands


def cmd_merge(pr: str, mode: str, branch: str, no_merge: bool) -> int:
    """Verify the close handle, squash-merge, then run merge-gated cleanup."""
    body_proc = _run(["gh", "pr", "view", pr, "--json", "body", "--jq", ".body"])
    if body_proc.returncode != 0:
        print(f"gh pr view failed: {body_proc.stderr.strip()}", file=sys.stderr)
        return 1
    if not has_close_handle(body_proc.stdout):
        print(f"refusing to merge PR {pr}: body carries no `closes #N` handle", file=sys.stderr)
        return 1

    if no_merge:
        print(json.dumps({"pr": pr, "merged": False, "reason": "--no-merge"}, indent=2))
        return 0

    merge_proc = _run(["gh", "pr", "merge", pr, "--squash", "--delete-branch"])
    if merge_proc.returncode != 0:
        print(
            f"merge failed, preserving branch/worktree: {merge_proc.stderr.strip()}",
            file=sys.stderr,
        )
        return 1

    current = _run(["git", "branch", "--show-current"])
    main_on_main = current.stdout.strip() == "main"
    for command in cleanup_commands(mode, branch, main_on_main):
        proc = _run(command)
        if proc.returncode != 0:
            print(
                f"cleanup step failed: {' '.join(command)}: {proc.stderr.strip()}", file=sys.stderr
            )
            return 1

    print(json.dumps({"pr": pr, "merged": True, "cleanup": mode}, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and run the merge + cleanup flow."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_merge = sub.add_parser("merge", help="squash-merge a PR and clean up by mode")
    p_merge.add_argument("pr", help="PR number or URL")
    p_merge.add_argument("--mode", choices=["direct", "worktree"], required=True)
    p_merge.add_argument("--branch", required=True, help="<type>/<slug> branch name")
    p_merge.add_argument(
        "--no-merge",
        action="store_true",
        help="verify the close handle only; preserve branch, worktree, and PR",
    )

    args = parser.parse_args(argv)
    return cmd_merge(args.pr, args.mode, args.branch, args.no_merge)


if __name__ == "__main__":
    sys.exit(main())
