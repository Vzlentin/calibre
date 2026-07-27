"""Prove the M5 protocol configuration is strict, immutable, and registered."""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest
import yaml

from newcalibre.protocols.m5 import load_m5_config
from newcalibre.protocols.m5.config import M5ConfigError, M5ProtocolConfig

_PROJECT_ROOT = Path(__file__).parents[2]
_GATE_C = _PROJECT_ROOT / "benchmarks" / "m5" / "gate-c.yaml"


def _payload() -> dict[str, object]:
    value = yaml.safe_load(_GATE_C.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _write(tmp_path: Path, payload: object) -> Path:
    path = tmp_path / "m5.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def test_gate_c_configuration_pins_registered_intent() -> None:
    config = load_m5_config(_GATE_C)

    assert config.dataset == "m5"
    assert config.phase == "evaluation"
    assert config.data_dir == Path("data/m5")
    assert config.inventory_path == Path("benchmarks/m5/m5-inputs.json")
    assert config.horizon == 28
    assert config.origin_count == 64
    assert config.population.kind == "full"
    assert config.population.bottom_count is None
    assert config.population.salt is None
    assert config.model_scope == "global"
    assert config.model_config == {
        "backend": "seasonal-naive",
        "censoring_aware": False,
        "m": 7,
        "model_name": "seasonal-naive",
        "quantile_levels": [],
    }
    assert config.reconciliation_strategy == "wls_struct"
    assert config.conformal_partition == "series-horizon"
    assert config.conformal_config == {
        "calibration_window": 10,
        "coverage": 0.9,
        "method": "split-per-step",
        "partition_by": "series-horizon",
        "upper_cap": None,
        "upper_floor": None,
    }
    assert config.minimum_calibration_scores == 10
    assert config.execution.backend == "ray"
    assert config.execution.logical_shards == 16
    assert config.execution.workers == 16
    assert config.execution.numeric_threads_per_worker == 1
    assert config.execution.retries == 0
    assert config.output_dir == Path("results/m5/gate-c")


def test_configuration_snapshots_mutable_values(tmp_path: Path) -> None:
    config = load_m5_config(_write(tmp_path, _payload()))
    model = config.model_config
    model["quantile_levels"] = [0.5]
    conformal = config.conformal_config
    conformal["coverage"] = 0.5

    assert config.model_config["quantile_levels"] == []
    assert config.conformal_config["coverage"] == 0.9


def test_configuration_constructor_and_overrides_are_closed(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="load_m5_config"):
        M5ProtocolConfig()
    assert tuple(inspect.signature(load_m5_config).parameters) == ("path",)
    with pytest.raises(TypeError):
        load_m5_config(_write(tmp_path, _payload()), phase="validation")  # type: ignore[call-arg]


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda value: value.update(extra=True), "exact keys"),
        (lambda value: value.pop("execution"), "exact keys"),
        (lambda value: value["dataset"].update(extra=True), "exact keys"),
        (lambda value: value["protocol"].update(extra=True), "exact keys"),
        (lambda value: value["pipeline"].update(extra=True), "exact keys"),
        (lambda value: value["execution"].update(extra=True), "exact keys"),
        (lambda value: value.update(schema=2), "schema"),
        (lambda value: value["dataset"].update(phase="validation"), "evaluation"),
        (lambda value: value["dataset"].update(phase="auto"), "evaluation"),
        (lambda value: value["dataset"].update(data_dir=["data/m5", "fallback"]), "path"),
        (lambda value: value["dataset"].update(data_dir="../m5"), "path"),
        (lambda value: value["protocol"].update(horizon=27), "horizon"),
        (lambda value: value["pipeline"]["model"].update(scope="local"), "global"),
        (lambda value: value["pipeline"]["model"].update(backend="unknown"), "backend"),
        (
            lambda value: value["pipeline"]["reconciliation"].update(strategy="unknown"),
            "reconciliation",
        ),
        (lambda value: value["pipeline"]["conformal"].update(method="unknown"), "method"),
        (lambda value: value["pipeline"]["conformal"].update(coverage=0.8), "coverage"),
        (lambda value: value["pipeline"]["conformal"].update(upper_cap=10), "clamps"),
        (lambda value: value["execution"].update(retries=1), "retries"),
        (lambda value: value["execution"].update(workers=0), "workers"),
        (lambda value: value.update(output_dir="/tmp/m5"), "path"),
    ],
)
def test_configuration_rejects_drift(
    tmp_path: Path,
    mutate: object,
    match: str,
) -> None:
    payload = _payload()
    mutate(payload)  # type: ignore[operator]
    with pytest.raises(M5ConfigError, match=match):
        load_m5_config(_write(tmp_path, payload))


@pytest.mark.parametrize(
    "population",
    [
        {},
        {"kind": "other"},
        {"kind": "full", "bottom_count": 2},
        {"kind": "digest_rank", "bottom_count": 0, "salt": "test"},
        {"kind": "digest_rank", "bottom_count": 2, "salt": ""},
        {"kind": "digest_rank", "bottom_count": 2, "salt": "test", "extra": 1},
    ],
)
def test_configuration_rejects_malformed_population(
    tmp_path: Path,
    population: dict[str, object],
) -> None:
    payload = _payload()
    payload["protocol"]["population"] = population  # type: ignore[index]
    with pytest.raises(M5ConfigError):
        load_m5_config(_write(tmp_path, payload))


def test_configuration_accepts_digest_rank_population(tmp_path: Path) -> None:
    payload = _payload()
    payload["protocol"]["population"] = {  # type: ignore[index]
        "kind": "digest_rank",
        "bottom_count": 100,
        "salt": "public test salt",
    }
    config = load_m5_config(_write(tmp_path, payload))
    assert config.population.kind == "digest_rank"
    assert config.population.bottom_count == 100
    assert config.population.salt == "public test salt"


@pytest.mark.parametrize("origin_count", [1, 63])
def test_configuration_rejects_origin_windows_below_derived_readiness(
    tmp_path: Path,
    origin_count: int,
) -> None:
    payload = _payload()
    payload["protocol"]["origin_count"] = origin_count  # type: ignore[index]
    with pytest.raises(M5ConfigError, match="readiness"):
        load_m5_config(_write(tmp_path, payload))


def test_configuration_accepts_exact_derived_readiness_boundary(tmp_path: Path) -> None:
    payload = _payload()
    payload["protocol"]["origin_count"] = 64  # type: ignore[index]
    assert load_m5_config(_write(tmp_path, payload)).origin_count == 64


def test_configuration_rejects_duplicate_yaml_keys_recursively(tmp_path: Path) -> None:
    text = _GATE_C.read_text(encoding="utf-8")
    text = text.replace("  phase: evaluation", "  phase: evaluation\n  phase: evaluation")
    path = tmp_path / "duplicate.yaml"
    path.write_text(text, encoding="utf-8")

    with pytest.raises(M5ConfigError, match="duplicate YAML key 'phase'"):
        load_m5_config(path)
