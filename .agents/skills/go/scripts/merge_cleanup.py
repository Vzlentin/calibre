"""Squash-merge a /go PR and run merge-gated cleanup by execution mode.

``merge <pr> --mode {direct,worktree} --branch <type>/<slug> --head-sha <sha>``
verifies the PR body carries a ``closes #N`` handle (refusing to merge
otherwise; ``--issue N`` pins the handle to one issue), squash-merges with
``gh pr merge --squash --match-head-commit <sha>`` — GitHub refuses the merge
if the branch head moved after the green CI verdict — and only then cleans up:

- **direct** — the main checkout is on the PR branch: return to ``main``,
  fast-forward, force-delete the local branch.
- **worktree** — the main checkout never left the user's branch/dirty tree:
  remove ``.worktrees/<slug>`` (without ``--force``, so uncommitted work in the
  worktree refuses the removal instead of being destroyed), force-delete the
  branch, prune, and fast-forward the local ``main`` ref via
  ``git fetch origin main:main`` — never ``git checkout``/``git pull`` in the
  caller's tree. The ref update is skipped when the caller has ``main`` checked
  out (``fetch`` refuses to move a checked-out branch).

Both modes finish by deleting the remote branch (best-effort: a leftover
remote ref is litter, not a safety problem, so its failure only warns).

Retries are idempotent: an already-merged PR skips the merge and proceeds
straight to cleanup, and a merge-command error is re-checked against the PR's
actual state before being reported as a failure (``gh pr merge`` can exit
non-zero after the API merge succeeded).

Preserving is the default: ``--no-merge`` and every pre-merge failure path
leave the branch, worktree, and PR intact. A squash-merged branch never shows
as merged to git, so the local delete is always ``git branch -D``.

Exit codes: 0 merged and cleaned up, 1 refused or not merged (nothing
deleted), 2 merged but cleanup incomplete (the failing step is printed;
finish it manually — do not treat the PR as unmerged).
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

WORKTREES_DIR = ".worktrees"

EXIT_MERGED = 0
EXIT_NOT_MERGED = 1
EXIT_CLEANUP_INCOMPLETE = 2

_CLOSES_RE = re.compile(r"\bclose[sd]?\s+#\d+", re.IGNORECASE)


def _run(args: list[str], cwd: str | Path | None = None) -> subprocess.CompletedProcess[str]:
    """Run a command, capturing text output without raising on failure."""
    return subprocess.run(args, cwd=cwd, capture_output=True, text=True, check=False)


def has_close_handle(body: str, issue: int | None = None) -> bool:
    """Report whether a PR body carries a ``closes #N`` issue handle.

    Args:
        body: The PR body text.
        issue: When given, only a handle closing exactly this issue counts —
            a stale ``closes #12`` from a template must not satisfy the gate.
    """
    if issue is None:
        return bool(_CLOSES_RE.search(body))
    return bool(re.search(rf"\bclose[sd]?\s+#{issue}\b", body, re.IGNORECASE))


def cleanup_commands(mode: str, branch: str, main_on_main: bool) -> list[list[str]]:
    """Build the post-merge cleanup command sequence for one execution mode.

    Args:
        mode: ``"direct"`` or ``"worktree"``.
        branch: The PR's ``<type>/<slug>`` branch name.
        main_on_main: Whether the main checkout currently has ``main`` checked
            out; a checked-out branch cannot be moved by ``fetch main:main``.

    Returns:
        Git command argv lists to run, in order, from the main checkout. The
        worktree removal deliberately omits ``--force``: a dirty worktree must
        refuse the removal (surfacing the uncommitted work) rather than lose it.
    """
    slug = branch.split("/", 1)[1] if "/" in branch else branch
    if mode == "direct":
        return [
            ["git", "checkout", "main"],
            ["git", "pull", "--ff-only"],
            ["git", "branch", "-D", branch],
        ]
    commands = [
        ["git", "worktree", "remove", f"{WORKTREES_DIR}/{slug}"],
        ["git", "branch", "-D", branch],
        ["git", "worktree", "prune"],
    ]
    if not main_on_main:
        commands.append(["git", "fetch", "origin", "main:main"])
    return commands


def _pr_state(pr: str) -> tuple[str, str] | None:
    """Fetch a PR's state and body; ``None`` when ``gh pr view`` fails."""
    proc = _run(["gh", "pr", "view", pr, "--json", "state,body"])
    if proc.returncode != 0:
        print(f"gh pr view failed: {proc.stderr.strip()}", file=sys.stderr)
        return None
    try:
        payload = json.loads(proc.stdout)
        return str(payload["state"]), str(payload["body"])
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        print(f"gh pr view returned malformed JSON: {exc}", file=sys.stderr)
        return None


def _cleanup(pr: str, mode: str, branch: str) -> int:
    """Run the post-merge cleanup sequence; the merge is already confirmed."""
    main_on_main = False
    if mode == "worktree":
        current = _run(["git", "branch", "--show-current"])
        main_on_main = current.stdout.strip() == "main"
    for command in cleanup_commands(mode, branch, main_on_main):
        proc = _run(command)
        if proc.returncode != 0:
            print(
                f"PR {pr} is merged but cleanup failed at: {' '.join(command)}: "
                f"{proc.stderr.strip()}\n"
                "Finish the remaining cleanup manually; do not treat the PR as unmerged.",
                file=sys.stderr,
            )
            return EXIT_CLEANUP_INCOMPLETE

    # Best-effort: without gh's --delete-branch (dropped because its local
    # delete breaks on worktree checkouts) the remote ref lingers; a failed
    # delete is litter, not a safety problem.
    remote_delete = _run(["git", "push", "origin", "--delete", branch])
    if remote_delete.returncode != 0:
        print(
            f"warning: could not delete remote branch {branch}: {remote_delete.stderr.strip()}",
            file=sys.stderr,
        )

    print(json.dumps({"pr": pr, "merged": True, "cleanup": mode}, indent=2))
    return EXIT_MERGED


def cmd_merge(
    pr: str,
    mode: str,
    branch: str,
    head_sha: str,
    no_merge: bool,
    issue: int | None = None,
) -> int:
    """Verify the close handle, squash-merge pinned to the verified SHA, clean up."""
    state = _pr_state(pr)
    if state is None:
        return EXIT_NOT_MERGED
    pr_status, body = state

    if pr_status == "MERGED":
        print(f"PR {pr} is already merged; running cleanup only.", file=sys.stderr)
        return _cleanup(pr, mode, branch)

    if not has_close_handle(body, issue):
        expected = f" for issue #{issue}" if issue is not None else ""
        print(
            f"refusing to merge PR {pr}: body carries no `closes #N` handle{expected}",
            file=sys.stderr,
        )
        return EXIT_NOT_MERGED

    if no_merge:
        print(json.dumps({"pr": pr, "merged": False, "reason": "--no-merge"}, indent=2))
        return EXIT_MERGED

    merge_proc = _run(["gh", "pr", "merge", pr, "--squash", "--match-head-commit", head_sha])
    if merge_proc.returncode != 0:
        # gh can exit non-zero after the API merge succeeded; trust the PR
        # state, not gh's exit code, before declaring failure.
        recheck = _pr_state(pr)
        if recheck is not None and recheck[0] == "MERGED":
            print(
                f"gh pr merge exited non-zero but PR {pr} is merged; continuing to cleanup.",
                file=sys.stderr,
            )
            return _cleanup(pr, mode, branch)
        print(
            f"merge failed, preserving branch/worktree: {merge_proc.stderr.strip()}",
            file=sys.stderr,
        )
        return EXIT_NOT_MERGED

    return _cleanup(pr, mode, branch)


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and run the merge + cleanup flow."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_merge = sub.add_parser("merge", help="squash-merge a PR and clean up by mode")
    p_merge.add_argument("pr", help="PR number or URL")
    p_merge.add_argument("--mode", choices=["direct", "worktree"], required=True)
    p_merge.add_argument("--branch", required=True, help="<type>/<slug> branch name")
    p_merge.add_argument(
        "--head-sha",
        required=True,
        help="commit SHA the CI verdict was computed for; the merge refuses a moved head",
    )
    p_merge.add_argument(
        "--issue",
        type=int,
        help="require the closes handle to reference exactly this issue number",
    )
    p_merge.add_argument(
        "--no-merge",
        action="store_true",
        help="verify the close handle only; preserve branch, worktree, and PR",
    )

    args = parser.parse_args(argv)
    return cmd_merge(args.pr, args.mode, args.branch, args.head_sha, args.no_merge, args.issue)


if __name__ == "__main__":
    sys.exit(main())
