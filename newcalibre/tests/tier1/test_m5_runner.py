"""Run tiny strict M5 releases through deterministic Ray placement."""

from __future__ import annotations

import ast
import inspect
import shutil
from dataclasses import FrozenInstanceError, fields
from pathlib import Path
from typing import Any, cast

import pytest

import newcalibre.protocols.m5.runner as runner
from newcalibre.forecasting import resolve_adapter
from newcalibre.protocols.m5 import M5RunResult, run_m5

_FIXTURE = Path(__file__).parents[1] / "fixtures" / "m5" / "tiny"
_LEVELS = frozenset({"bottom", "item", "department", "category", "store", "state", "total"})
_ARTIFACTS = frozenset({"coverage-summary.json", "coverage-by-node.parquet", "report.md"})


def _isolated_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    digest_rank: bool = False,
) -> Path:
    relative = Path("tests/fixtures/m5/tiny")
    target = tmp_path / relative
    target.parent.mkdir(parents=True)
    shutil.copytree(_FIXTURE, target)
    monkeypatch.setattr(runner, "_PROJECT_ROOT", tmp_path)
    return target / ("tiny-digest.yaml" if digest_rank else "tiny.yaml")


def test_tiny_strict_release_runs_end_to_end_and_returns_only_compact_facts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    original_config = runner.load_m5_config
    original_load = runner.load_m5_dataset
    original_compile = runner.compile_m5_protocol
    original_engine = runner.Engine
    original_loop = runner.TimeLoop
    original_reader = runner.InMemoryLedgerReader
    original_score = runner.score_m5

    def record_config(path: Path):
        events.append("config")
        return original_config(path)

    def record_load(project_root: Path, config: object):
        events.append("verified-load")
        return original_load(project_root, cast(Any, config))

    def record_compile(dataset: object, config: object):
        events.append("compile")
        return original_compile(cast(Any, dataset), cast(Any, config))

    def record_engine(**kwargs: object):
        events.append("engine")
        return original_engine(**cast(Any, kwargs))

    def record_loop(**kwargs: object):
        events.append("time-loop")
        return original_loop(**cast(Any, kwargs))

    def record_reader(sink: object):
        events.append("reader")
        return original_reader(cast(Any, sink))

    def record_score(config: object, ledger: object, *, output_dir: Path):
        events.append("score")
        return original_score(cast(Any, config), cast(Any, ledger), output_dir=output_dir)

    monkeypatch.setattr(runner, "load_m5_config", record_config)
    monkeypatch.setattr(runner, "load_m5_dataset", record_load)
    monkeypatch.setattr(runner, "compile_m5_protocol", record_compile)
    monkeypatch.setattr(runner, "Engine", record_engine)
    monkeypatch.setattr(runner, "TimeLoop", record_loop)
    monkeypatch.setattr(runner, "InMemoryLedgerReader", record_reader)
    monkeypatch.setattr(runner, "score_m5", record_score)

    result = run_m5(_isolated_config(tmp_path, monkeypatch))

    assert events == [
        "config",
        "verified-load",
        "compile",
        "engine",
        "time-loop",
        "reader",
        "score",
    ]
    assert isinstance(result, M5RunResult)
    assert result.forecast_origin_count == 64
    assert result.commit_count == 65
    assert result.node_count == 7
    assert result.expected_row_count == 7 * 64 * 28
    assert result.resolved_row_count == 7 * 1414
    assert result.pending_row_count == 7 * 378
    assert result.eligible_row_count == result.diagnostics.population.counts.eligible
    assert result.scored_row_count == result.diagnostics.population.counts.scored
    assert result.diagnostics.status == "VALID"
    assert result.diagnostics.context.reconciler == "wls_struct"
    assert result.diagnostics.context.conformal_method == "split-per-step"
    assert result.diagnostics.context.conformal_partition == "series-horizon"
    assert set(result.diagnostics.levels) == _LEVELS
    assert {path.name for path in result.diagnostics.paths} == _ARTIFACTS
    assert {path.name for path in result.diagnostics.summary_path.parent.iterdir()} == _ARTIFACTS
    assert {field.name for field in fields(result)} == {
        "session",
        "input_inventory_sha256",
        "forecast_origin_count",
        "commit_count",
        "node_count",
        "expected_row_count",
        "resolved_row_count",
        "eligible_row_count",
        "scored_row_count",
        "pending_row_count",
        "diagnostics",
    }
    assert not any(field.name.endswith("rows") for field in fields(result))
    with pytest.raises(FrozenInstanceError):
        cast(Any, result).node_count = 0


def test_full_and_digest_rank_fixture_configs_use_the_same_composition(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class CompositionReached(RuntimeError):
        pass

    compositions: list[tuple[object, ...]] = []

    def capture_engine(**kwargs: object) -> None:
        hierarchy = cast(Any, kwargs["hierarchy"])
        store = cast(Any, kwargs["run_store"])
        compositions.append(
            (
                len(hierarchy.node_labels),
                store.session.series_keys,
                type(kwargs["panel_source"]),
                type(store),
                type(kwargs["dispatch_backend"]),
                kwargs["adapter_resolver"] is resolve_adapter,
                kwargs["reconciliation_strategy"],
                kwargs["orderer"],
            )
        )
        raise CompositionReached

    monkeypatch.setattr(runner, "Engine", capture_engine)
    for name, digest_rank in (("full", False), ("digest", True)):
        root = tmp_path / name
        with pytest.raises(CompositionReached):
            run_m5(_isolated_config(root, monkeypatch, digest_rank=digest_rank))

    assert len(compositions) == 2
    assert compositions[0] == compositions[1]
    assert compositions[0][-3:] == (True, "wls_struct", None)


@pytest.mark.parametrize(
    ("seam", "expected"),
    [
        ("config", ["config"]),
        ("verified-load", ["config", "verified-load"]),
        ("compile", ["config", "verified-load", "compile"]),
        ("engine", ["config", "verified-load", "compile", "engine"]),
        ("time-loop", ["config", "verified-load", "compile", "engine", "time-loop"]),
        (
            "reader",
            ["config", "verified-load", "compile", "engine", "time-loop", "run", "reader"],
        ),
        (
            "score",
            [
                "config",
                "verified-load",
                "compile",
                "engine",
                "time-loop",
                "run",
                "reader",
                "score",
            ],
        ),
    ],
)
def test_failure_at_each_owning_seam_stops_later_work_and_emits_no_diagnostics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    seam: str,
    expected: list[str],
) -> None:
    events: list[str] = []
    originals = {
        "config": runner.load_m5_config,
        "verified-load": runner.load_m5_dataset,
        "compile": runner.compile_m5_protocol,
    }

    def stop_or_call(name: str, *args: object, **kwargs: object):
        events.append(name)
        if seam == name:
            raise RuntimeError(f"{name} failed")
        return originals[name](*args, **kwargs)

    class LoopProxy:
        def run(self):
            events.append("run")
            return object()

    monkeypatch.setattr(
        runner,
        "load_m5_config",
        lambda *args, **kwargs: stop_or_call("config", *args, **kwargs),
    )
    monkeypatch.setattr(
        runner,
        "load_m5_dataset",
        lambda *args, **kwargs: stop_or_call("verified-load", *args, **kwargs),
    )
    monkeypatch.setattr(
        runner,
        "compile_m5_protocol",
        lambda *args, **kwargs: stop_or_call("compile", *args, **kwargs),
    )

    def engine(*_args: object, **_kwargs: object) -> object:
        events.append("engine")
        if seam == "engine":
            raise RuntimeError("engine failed")
        return object()

    def time_loop(*_args: object, **_kwargs: object) -> LoopProxy:
        events.append("time-loop")
        if seam == "time-loop":
            raise RuntimeError("time-loop failed")
        return LoopProxy()

    def reader(*_args: object, **_kwargs: object) -> object:
        events.append("reader")
        if seam == "reader":
            raise RuntimeError("reader failed")
        return object()

    def score(*_args: object, **_kwargs: object) -> object:
        events.append("score")
        raise RuntimeError("score failed")

    monkeypatch.setattr(runner, "Engine", engine)
    monkeypatch.setattr(runner, "TimeLoop", time_loop)
    monkeypatch.setattr(runner, "InMemoryLedgerReader", reader)
    monkeypatch.setattr(runner, "score_m5", score)
    config = _isolated_config(tmp_path, monkeypatch)

    with pytest.raises(RuntimeError, match=f"{seam} failed"):
        run_m5(config)

    assert events == expected
    assert not (tmp_path / "results/m5/tier1-tiny").exists()


def test_runner_source_keeps_one_generic_composition_and_no_extra_surface() -> None:
    source = inspect.getsource(runner)
    tree = ast.parse(source)
    engine_runs = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "run"
    ]
    forbidden = (
        "newcalibre.oracle",
        "frozen-",
        "M5EngineAdapter",
        "load_result_bundle",
        "profiling",
        "custody",
        "receipt_chain",
        "run_origin",
    )

    assert len(engine_runs) == 1
    assert all(term not in source for term in forbidden)
    assert "newcalibre.ledger" not in source
    assert ".forecasts" not in source
    assert "RayDispatch" in source
    assert "InProcessDispatch" not in source
