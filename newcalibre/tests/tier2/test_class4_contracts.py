"""Exercise instantiated class-4 contracts and keep later legs visibly pending."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.tier2


_RUNNER = Path(__file__).with_name("_seeded_time_loop_run.py")


def _run_fixture(
    *,
    mode: str,
    seed: int,
    hash_seed: str,
    output: Path,
) -> bytes:
    environment = {**os.environ, "PYTHONHASHSEED": hash_seed}
    completed = subprocess.run(
        (
            sys.executable,
            str(_RUNNER),
            "--mode",
            mode,
            "--seed",
            str(seed),
            "--output",
            str(output),
        ),
        check=False,
        cwd=Path(__file__).parents[2],
        env=environment,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    result = output.read_bytes()
    assert result
    return result


def _artifact_path(tmp_path: Path, name: str) -> Path:
    configured = os.environ.get("NEWCALIBRE_TIER2_OUT")
    output_directory = tmp_path if configured is None else Path(configured)
    output_directory.mkdir(parents=True, exist_ok=True)
    return output_directory / name


def _point_forecasts(payload: bytes) -> tuple[str, ...]:
    ledger = json.loads(payload)
    return tuple(
        value
        for forecast in ledger["forecasts"]
        for name, value in forecast["values"]
        if name == "point_forecast"
    )


def test_resumed_runs_match_uninterrupted_run(tmp_path: Path) -> None:
    seed = 611
    uninterrupted = _run_fixture(
        mode="uninterrupted",
        seed=seed,
        hash_seed="17",
        output=tmp_path / "uninterrupted.bin",
    )
    resumed_before_commit = _run_fixture(
        mode="resumed-before-commit",
        seed=seed,
        hash_seed="8675309",
        output=tmp_path / "resumed-before-commit.bin",
    )
    resumed_after_commit = _run_fixture(
        mode="resumed-after-commit",
        seed=seed,
        hash_seed="314159",
        output=tmp_path / "resumed-after-commit.bin",
    )

    assert resumed_before_commit == uninterrupted
    assert resumed_after_commit == uninterrupted
    _artifact_path(tmp_path, "resumed-ledger.bin").write_bytes(resumed_after_commit)


@pytest.mark.xfail(
    strict=True,
    reason="Pending U16: distribution invariance needs the dispatch substrate.",
)
def test_distributed_run_matches_sequential_run() -> None:
    pytest.fail("U16 must replace this placeholder with a biting distribution contract.")


@pytest.mark.xfail(
    strict=True,
    reason="Pending U10: state equality needs serializable calibration state.",
)
def test_serialized_state_matches_never_serialized_state() -> None:
    pytest.fail("U10 must replace this placeholder with a biting state round-trip contract.")


def test_same_seed_produces_same_bytes(tmp_path: Path) -> None:
    seed = 72617
    first = _run_fixture(
        mode="uninterrupted",
        seed=seed,
        hash_seed="23",
        output=tmp_path / "same-seed-first.bin",
    )
    second = _run_fixture(
        mode="uninterrupted",
        seed=seed,
        hash_seed="99991",
        output=tmp_path / "same-seed-second.bin",
    )
    different_seed = _run_fixture(
        mode="uninterrupted",
        seed=seed + 1,
        hash_seed="31",
        output=tmp_path / "different-seed.bin",
    )

    assert first == second
    assert first != different_seed
    assert _point_forecasts(first) != _point_forecasts(different_seed)
    _artifact_path(tmp_path, "same-seed-ledger.bin").write_bytes(first)
