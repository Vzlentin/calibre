"""Unit tests for the /go skill scripts in .agents/skills/go/scripts.

Fixture-driven — no test here touches the network. Verdict/decision logic is
exercised as pure functions over canned payloads; the subprocess-facing
commands are exercised with a recording fake runner. One scratch-repo test
covers the git-facing pieces of run-state path resolution and provisioning
collision refusal.
"""

import json
import subprocess
from pathlib import Path

import pytest

from tests.infra import load_script_module

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / ".agents" / "skills" / "go" / "scripts"
FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures" / "go"

run_state = load_script_module(SCRIPTS_DIR / "run_state.py")
ci_verdict = load_script_module(SCRIPTS_DIR / "ci_verdict.py")
provision_worktree = load_script_module(SCRIPTS_DIR / "provision_worktree.py")
merge_cleanup = load_script_module(SCRIPTS_DIR / "merge_cleanup.py")


def _fixture(name: str) -> str:
    return (FIXTURES_DIR / name).read_text(encoding="utf-8")


def _git(args: list[str], cwd: Path) -> str:
    proc = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True)
    return proc.stdout.strip()


@pytest.fixture
def scratch_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(["init", "-b", "main"], cwd=repo)
    _git(["config", "user.email", "test@example.com"], cwd=repo)
    _git(["config", "user.name", "test"], cwd=repo)
    (repo / "README.md").write_text("scratch\n", encoding="utf-8")
    _git(["add", "README.md"], cwd=repo)
    _git(["commit", "-m", "init"], cwd=repo)
    return repo


class FakeRunner:
    """Record commands and answer them from a scripted (prefix -> result) table."""

    def __init__(self, script: list[tuple[list[str], int, str]]):
        self.script = script
        self.calls: list[list[str]] = []

    def __call__(self, args: list[str], cwd: object = None) -> subprocess.CompletedProcess[str]:
        self.calls.append(list(args))
        for prefix, returncode, stdout in self.script:
            if args[: len(prefix)] == prefix:
                return subprocess.CompletedProcess(args, returncode, stdout=stdout, stderr="")
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")


# --- run_state ---------------------------------------------------------------


def test_run_state_init_set_get_roundtrip(
    scratch_repo: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(scratch_repo)
    assert run_state.main(["init", "my-slug"]) == 0
    assert run_state.main(["set", "my-slug", "pr", "42"]) == 0
    assert run_state.main(["set", "my-slug", "stage", "5"]) == 0
    capsys.readouterr()

    assert run_state.main(["get", "my-slug", "pr"]) == 0
    assert capsys.readouterr().out.strip() == "42"

    assert run_state.main(["get", "my-slug"]) == 0
    state = json.loads(capsys.readouterr().out)
    assert state == {"slug": "my-slug", "pr": "42", "stage": "5"}

    assert run_state.main(["list"]) == 0
    assert capsys.readouterr().out.split() == ["my-slug"]


def test_run_state_init_refuses_collision_without_force(
    scratch_repo: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(scratch_repo)
    assert run_state.main(["init", "dup"]) == 0
    assert run_state.main(["set", "dup", "pr", "7"]) == 0
    assert run_state.main(["init", "dup"]) == 1

    capsys.readouterr()
    assert run_state.main(["get", "dup", "pr"]) == 0
    assert capsys.readouterr().out.strip() == "7", "failed init must not clobber state"

    assert run_state.main(["init", "dup", "--force"]) == 0
    capsys.readouterr()
    assert run_state.main(["get", "dup"]) == 0
    assert json.loads(capsys.readouterr().out) == {"slug": "dup"}


def test_run_state_resolves_same_path_from_main_and_worktree(
    scratch_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    worktree = scratch_repo.parent / "wt"
    _git(["worktree", "add", str(worktree), "-b", "feat/wt"], cwd=scratch_repo)

    monkeypatch.chdir(scratch_repo)
    from_main = run_state.state_path("shared-slug")
    monkeypatch.chdir(worktree)
    from_worktree = run_state.state_path("shared-slug")

    assert from_main == from_worktree


def test_run_state_rejects_path_traversal_slug(
    scratch_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(scratch_repo)
    with pytest.raises(SystemExit):
        run_state.state_path("../escape")


# --- ci_verdict --------------------------------------------------------------


def test_verdict_green() -> None:
    code, report = ci_verdict.evaluate(_fixture("check_runs_green.json"))
    assert code == ci_verdict.EXIT_GREEN
    assert report["verdict"] == "green"


def test_verdict_green_with_skipped_and_neutral_conditional_jobs() -> None:
    payload = json.dumps(
        {
            "check_runs": [
                {"name": "tests", "status": "completed", "conclusion": "success"},
                {"name": "vn2-acceptance", "status": "completed", "conclusion": "skipped"},
                {"name": "oracle", "status": "completed", "conclusion": "neutral"},
            ]
        }
    )
    code, report = ci_verdict.evaluate(payload)
    assert code == ci_verdict.EXIT_GREEN
    assert report["verdict"] == "green"


def test_verdict_pending_when_any_run_incomplete() -> None:
    code, report = ci_verdict.evaluate(_fixture("check_runs_pending.json"))
    assert code == ci_verdict.EXIT_PENDING
    assert report["verdict"] == "pending"


def test_verdict_failure_collects_failed_runs_and_ids() -> None:
    code, report = ci_verdict.evaluate(_fixture("check_runs_failure.json"))
    assert code == ci_verdict.EXIT_FAILURE
    assert report["verdict"] == "failure"
    failed = {entry["name"]: entry for entry in report["failed"]}
    assert set(failed) == {"tests", "typecheck"}
    assert failed["tests"]["conclusion"] == "failure"
    assert failed["tests"]["run_id"] == "311"
    assert failed["typecheck"]["conclusion"] == "cancelled"
    assert report["signature"]


@pytest.mark.parametrize(
    "conclusion", ["failure", "cancelled", "timed_out", "action_required", "totally_new_state"]
)
def test_verdict_every_non_success_conclusion_is_failure(conclusion: str) -> None:
    payload = json.dumps(
        {
            "check_runs": [
                {
                    "name": "tests",
                    "status": "completed",
                    "conclusion": conclusion,
                    "details_url": "https://github.com/x/y/actions/runs/9/job/1",
                    "output": {"title": "boom", "summary": ""},
                }
            ]
        }
    )
    code, report = ci_verdict.evaluate(payload)
    assert code == ci_verdict.EXIT_FAILURE
    assert report["verdict"] == "failure"


def test_verdict_empty_check_set_is_non_verdict_never_green() -> None:
    code, report = ci_verdict.evaluate(_fixture("check_runs_empty.json"))
    assert code == ci_verdict.EXIT_NON_VERDICT
    assert report["verdict"] == "non-verdict"


@pytest.mark.parametrize("payload", ["not json at all", "{}", '{"check_runs": 3}'])
def test_verdict_malformed_payload_is_non_verdict(payload: str) -> None:
    code, report = ci_verdict.evaluate(payload)
    assert code == ci_verdict.EXIT_NON_VERDICT
    assert report["verdict"] == "non-verdict"


def test_failure_signature_stable_across_identical_failures() -> None:
    text = _fixture("check_runs_failure.json")
    _, first = ci_verdict.evaluate(text)
    _, second = ci_verdict.evaluate(text)
    assert first["signature"] == second["signature"]

    changed = text.replace("Process completed with exit code 1.", "Different first error line")
    _, third = ci_verdict.evaluate(changed)
    assert third["signature"] != first["signature"]


def test_parse_run_id() -> None:
    url = "https://github.com/Vzlentin/calibre/actions/runs/16543/job/98"
    assert ci_verdict.parse_run_id(url) == "16543"
    assert ci_verdict.parse_run_id("https://example.com/no-run-here") is None


# --- provision_worktree ------------------------------------------------------


@pytest.mark.parametrize(
    ("branch", "porcelain", "expected"),
    [
        ("main", "", "direct"),
        ("main", " M calibre/core/frame.py", "worktree"),
        ("feat/other", "", "worktree"),
        ("", "", "worktree"),  # detached HEAD
    ],
)
def test_decide_mode_matrix(branch: str, porcelain: str, expected: str) -> None:
    assert provision_worktree.decide_mode(branch, porcelain) == expected


def test_read_setup_steps_substitutes_root_path() -> None:
    config = json.dumps(
        {
            "setup-worktree-unix": [
                'cp "$ROOT_WORKTREE_PATH/.env" .env',
                "uv sync --frozen",
            ]
        }
    )
    steps = provision_worktree.read_setup_steps(config, "/c/main")
    assert steps == ['cp "/c/main/.env" .env', "uv sync --frozen"]


def test_read_setup_steps_rejects_non_list() -> None:
    with pytest.raises(SystemExit):
        provision_worktree.read_setup_steps('{"setup-worktree-unix": "oops"}', "/m")


def test_provision_refuses_existing_worktree_path(
    scratch_repo: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    (scratch_repo / ".worktrees" / "taken").mkdir(parents=True)
    monkeypatch.chdir(scratch_repo)
    assert provision_worktree.cmd_provision("feat/taken", require_m5=False) == 1
    assert "already exists" in capsys.readouterr().err


def test_provision_refuses_existing_branch(
    scratch_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _git(["branch", "feat/exists"], cwd=scratch_repo)
    monkeypatch.chdir(scratch_repo)
    assert provision_worktree.cmd_provision("feat/exists", require_m5=False) == 1
    assert not (scratch_repo / ".worktrees" / "exists").exists()


def test_provision_rejects_branch_without_type_prefix(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert provision_worktree.cmd_provision("noslash", require_m5=False) == 1
    assert "<type>/<slug>" in capsys.readouterr().err


def test_provision_aborts_on_first_failed_setup_step(
    scratch_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cursor_dir = scratch_repo / ".cursor"
    cursor_dir.mkdir()
    (cursor_dir / "worktrees.json").write_text(
        json.dumps({"setup-worktree-unix": ["step-one", "step-two"]}), encoding="utf-8"
    )
    monkeypatch.chdir(scratch_repo)

    fake = FakeRunner(
        [
            (
                ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
                0,
                f"{(scratch_repo / '.git').as_posix()}\n",
            ),
            (["git", "rev-parse", "--verify"], 1, ""),
            (["git", "fetch", "origin"], 0, ""),
            (["git", "worktree", "add"], 0, ""),
        ]
    )
    monkeypatch.setattr(provision_worktree, "_run", fake)

    executed: list[str] = []

    def fake_run_step(step: str, cwd: Path) -> int:
        executed.append(step)
        return 1 if step == "step-one" else 0

    monkeypatch.setattr(provision_worktree, "_run_step", fake_run_step)

    assert provision_worktree.cmd_provision("feat/fresh", require_m5=False) == 1
    assert executed == ["step-one"], "a failed step must abort before later steps run"


# --- merge_cleanup -----------------------------------------------------------


def _merge_script(body: str, merge_rc: int = 0, current_branch: str = "feat/x") -> FakeRunner:
    return FakeRunner(
        [
            (["gh", "pr", "view"], 0, body),
            (["gh", "pr", "merge"], merge_rc, ""),
            (["git", "branch", "--show-current"], 0, current_branch + "\n"),
        ]
    )


def test_merge_refuses_without_closes_handle(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _merge_script("A PR body that mentions #12 but carries no close handle.")
    monkeypatch.setattr(merge_cleanup, "_run", fake)
    assert merge_cleanup.cmd_merge("12", "direct", "feat/x", no_merge=False) == 1
    assert ["gh", "pr", "merge", "12", "--squash", "--delete-branch"] not in fake.calls


def test_cleanup_skipped_when_merge_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _merge_script("Body.\n\ncloses #12", merge_rc=1)
    monkeypatch.setattr(merge_cleanup, "_run", fake)
    assert merge_cleanup.cmd_merge("12", "worktree", "feat/x", no_merge=False) == 1
    git_calls = [c for c in fake.calls if c[0] == "git"]
    assert git_calls == [], "no cleanup command may run after a failed merge"


def test_no_merge_preserves_everything(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _merge_script("closes #12")
    monkeypatch.setattr(merge_cleanup, "_run", fake)
    assert merge_cleanup.cmd_merge("12", "worktree", "feat/x", no_merge=True) == 0
    assert fake.calls == [["gh", "pr", "view", "12", "--json", "body", "--jq", ".body"]], (
        "--no-merge must stop after the close-handle check"
    )


def test_worktree_cleanup_never_touches_main_checkout_tree(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _merge_script("closes #12", current_branch="docs/dirty-branch")
    monkeypatch.setattr(merge_cleanup, "_run", fake)
    assert merge_cleanup.cmd_merge("12", "worktree", "feat/x", no_merge=False) == 0

    git_calls = [c for c in fake.calls if c[0] == "git" and c[1] != "branch"]
    commands = {tuple(c[:2]) for c in git_calls}
    assert ("git", "checkout") not in commands
    assert ("git", "pull") not in commands
    assert ["git", "worktree", "remove", "--force", ".worktrees/x"] in fake.calls
    assert ["git", "branch", "-D", "feat/x"] in fake.calls, "squash merge requires -D"
    assert ["git", "fetch", "origin", "main:main"] in fake.calls


def test_worktree_cleanup_skips_main_ref_update_when_user_on_main(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _merge_script("closes #12", current_branch="main")
    monkeypatch.setattr(merge_cleanup, "_run", fake)
    assert merge_cleanup.cmd_merge("12", "worktree", "feat/x", no_merge=False) == 0
    assert ["git", "fetch", "origin", "main:main"] not in fake.calls


def test_direct_cleanup_returns_to_main_and_force_deletes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _merge_script("Closes #34")
    monkeypatch.setattr(merge_cleanup, "_run", fake)
    assert merge_cleanup.cmd_merge("34", "direct", "feat/y", no_merge=False) == 0
    git_calls = [
        c for c in fake.calls if c[0] == "git" and c[:3] != ["git", "branch", "--show-current"]
    ]
    assert git_calls == [
        ["git", "checkout", "main"],
        ["git", "pull", "--ff-only"],
        ["git", "branch", "-D", "feat/y"],
    ]


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ("closes #12", True),
        ("Closes #12", True),
        ("close #7", True),
        ("closed #7", True),
        ("fixes #12", False),
        ("closes 12", False),
        ("disclose #12", False),
    ],
)
def test_has_close_handle(body: str, expected: bool) -> None:
    assert merge_cleanup.has_close_handle(body) is expected
