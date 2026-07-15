"""Apply suite-wide numeric gate/witness collection invariants."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from oracle.witnesses import (
    WitnessDeclaration,
    WitnessPairingError,
    require_exact_inventory,
    require_exact_witnesses,
)

_TIERS = tuple(f"tier{number}" for number in range(5))
_PROJECT_ROOT = Path(__file__).parent.parent
_TIER3_ROOT = Path(__file__).parent / "tier3"
_TIER3_INVENTORY = _TIER3_ROOT / "oracle_inventory.json"
_REQUIRED_TIER3_ID = "vn2-conditional-replay"


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Fail collection for every orphaned, cross-tier, or self-owned witness."""
    gates = _marker_declarations(items, "oracle_gate")
    witnesses = _marker_declarations(items, "oracle_witness")
    try:
        require_exact_witnesses(gates, witnesses)
        if _tier3_inventory_is_required(config, items):
            required_gates, required_witnesses = _required_tier3_inventory()
            require_exact_inventory(
                (item for item in gates if item.tier == "tier3"),
                (item for item in witnesses if item.tier == "tier3"),
                required_gates=required_gates,
                required_witnesses=required_witnesses,
            )
    except WitnessPairingError as error:
        raise pytest.UsageError(str(error)) from error


def _marker_declarations(
    items: list[pytest.Item], marker_name: str
) -> tuple[WitnessDeclaration, ...]:
    declarations: list[WitnessDeclaration] = []
    for item in items:
        markers = tuple(item.iter_markers(marker_name))
        if len(markers) > 1:
            raise pytest.UsageError(f"{item.nodeid} declares {marker_name} more than once")
        if not markers:
            continue
        tiers = tuple(tier for tier in _TIERS if item.get_closest_marker(tier) is not None)
        if len(tiers) != 1:
            raise pytest.UsageError(
                f"{item.nodeid} declares {marker_name} without exactly one tier marker"
            )
        marker = markers[0]
        if len(marker.args) != 1 or marker.kwargs or not isinstance(marker.args[0], str):
            raise pytest.UsageError(
                f"{item.nodeid} must declare {marker_name} with one string gate ID"
            )
        identifier = marker.args[0]
        if not identifier or identifier != identifier.strip():
            raise pytest.UsageError(f"{item.nodeid} declares an invalid {marker_name} gate ID")
        declarations.append(
            WitnessDeclaration(
                tiers[0],
                identifier,
                _stable_nodeid(Path(item.path), item.nodeid),
            )
        )
    return tuple(declarations)


def _stable_nodeid(path: Path, nodeid: str) -> str:
    try:
        relative_path = path.resolve().relative_to(_PROJECT_ROOT.resolve())
    except ValueError:
        relative_path = Path(nodeid.partition("::")[0])
    _, separator, remainder = nodeid.partition("::")
    module = ".".join(relative_path.with_suffix("").parts)
    return f"{module}{separator}{remainder}"


def _tier3_inventory_is_required(config: pytest.Config, items: list[pytest.Item]) -> bool:
    tier3_root = _TIER3_ROOT.resolve()
    if any(Path(item.path).resolve().is_relative_to(tier3_root) for item in items):
        return True
    for argument in config.args:
        target_text = str(argument).partition("::")[0]
        target = Path(target_text)
        if not target.is_absolute():
            target = Path(config.rootpath) / target
        resolved = target.resolve()
        if resolved == tier3_root:
            return True
        if resolved.is_relative_to(tier3_root) or tier3_root.is_relative_to(resolved):
            return True
    return False


def _required_tier3_inventory(
    path: Path = _TIER3_INVENTORY,
) -> tuple[tuple[WitnessDeclaration, ...], tuple[WitnessDeclaration, ...]]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise WitnessPairingError(f"invalid required Tier 3 oracle inventory: {error}") from error
    if not isinstance(value, dict) or set(value) != {"named", "schema", "tier"}:
        raise WitnessPairingError("required Tier 3 oracle inventory has an invalid root schema")
    if value["schema"] != 1 or value["tier"] != "tier3":
        raise WitnessPairingError("required Tier 3 oracle inventory has an invalid identity")
    named = value["named"]
    if not isinstance(named, dict) or set(named) != {"gate", "witness"}:
        raise WitnessPairingError(
            "required Tier 3 oracle inventory must name exactly one gate and witness"
        )

    declarations: dict[str, WitnessDeclaration] = {}
    for role in ("gate", "witness"):
        item = named[role]
        if not isinstance(item, dict) or set(item) != {"id", "node"}:
            raise WitnessPairingError(
                f"required Tier 3 oracle inventory {role} has an invalid schema"
            )
        identifier = item["id"]
        nodeid = item["node"]
        if identifier != _REQUIRED_TIER3_ID:
            raise WitnessPairingError(
                f"required Tier 3 oracle inventory {role} must use {_REQUIRED_TIER3_ID}"
            )
        if (
            not isinstance(nodeid, str)
            or not nodeid.startswith("tests.tier3.")
            or "::" not in nodeid
        ):
            raise WitnessPairingError(
                f"required Tier 3 oracle inventory {role} must use a stable Tier 3 node ID"
            )
        declarations[role] = WitnessDeclaration("tier3", identifier, nodeid)
    if declarations["gate"].nodeid == declarations["witness"].nodeid:
        raise WitnessPairingError("required Tier 3 oracle gate and witness must use distinct nodes")
    return (declarations["gate"],), (declarations["witness"],)
