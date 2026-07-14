"""Per-run JSON state for /go runs, stored under the git common dir.

Each run keeps one flat JSON dict at ``<git-common-dir>/go-runs/<slug>.json``.
The common dir is shared between the main checkout and every linked worktree,
so both resolve the same file; and because it lives inside ``.git`` the state
is untracked and private by construction.

Commands: ``init <slug> [--force]``, ``set <slug> <key> <value>``,
``get <slug> [key]``, ``list``.
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

_SLUG_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def _run(args: list[str], cwd: str | None = None) -> subprocess.CompletedProcess[str]:
    """Run a command, capturing text output without raising on failure."""
    return subprocess.run(args, cwd=cwd, capture_output=True, text=True, check=False)


def state_dir() -> Path:
    """Resolve the run-state directory under the git common dir.

    Returns:
        Absolute path of ``<git-common-dir>/go-runs`` for the repository
        containing the current working directory.
    """
    proc = _run(["git", "rev-parse", "--git-common-dir"])
    if proc.returncode != 0:
        raise SystemExit(f"not a git repository: {proc.stderr.strip()}")
    common = Path(proc.stdout.strip())
    if not common.is_absolute():
        common = Path.cwd() / common
    return common.resolve() / "go-runs"


def state_path(slug: str) -> Path:
    """Resolve the state-file path for one run slug."""
    if not _SLUG_RE.fullmatch(slug):
        raise SystemExit(f"invalid slug: {slug!r}")
    return state_dir() / f"{slug}.json"


def load_state(path: Path) -> dict[str, str]:
    """Load a run-state file, failing clearly when it does not exist."""
    if not path.is_file():
        raise SystemExit(f"no run state at {path} (run `init` first)")
    return json.loads(path.read_text(encoding="utf-8"))


def save_state(path: Path, state: dict[str, str]) -> None:
    """Write a run-state dict as pretty-printed JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def cmd_init(slug: str, force: bool) -> int:
    """Create a fresh state file for a run, refusing to clobber without --force."""
    path = state_path(slug)
    if path.exists() and not force:
        print(f"refusing to overwrite existing run state: {path} (use --force)", file=sys.stderr)
        return 1
    save_state(path, {"slug": slug})
    print(path)
    return 0


def cmd_set(slug: str, key: str, value: str) -> int:
    """Set one key in an existing run's state."""
    path = state_path(slug)
    state = load_state(path)
    state[key] = value
    save_state(path, state)
    return 0


def cmd_get(slug: str, key: str | None) -> int:
    """Print one key's value, or the whole state dict when no key is given."""
    state = load_state(state_path(slug))
    if key is None:
        print(json.dumps(state, indent=2, sort_keys=True))
        return 0
    if key not in state:
        print(f"no key {key!r} in run state for {slug!r}", file=sys.stderr)
        return 1
    print(state[key])
    return 0


def cmd_list() -> int:
    """List the slugs of all recorded runs."""
    directory = state_dir()
    if directory.is_dir():
        for path in sorted(directory.glob("*.json")):
            print(path.stem)
    return 0


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and dispatch to a run-state command."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init", help="create a fresh state file for a run")
    p_init.add_argument("slug")
    p_init.add_argument("--force", action="store_true", help="overwrite an existing state file")

    p_set = sub.add_parser("set", help="set one key in a run's state")
    p_set.add_argument("slug")
    p_set.add_argument("key")
    p_set.add_argument("value")

    p_get = sub.add_parser("get", help="print one key, or the whole state")
    p_get.add_argument("slug")
    p_get.add_argument("key", nargs="?")

    sub.add_parser("list", help="list recorded run slugs")

    args = parser.parse_args(argv)
    if args.command == "init":
        return cmd_init(args.slug, args.force)
    if args.command == "set":
        return cmd_set(args.slug, args.key, args.value)
    if args.command == "get":
        return cmd_get(args.slug, args.key)
    return cmd_list()


if __name__ == "__main__":
    raise SystemExit(main())
