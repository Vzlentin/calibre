"""Apply suite-wide numeric gate/witness collection invariants."""

from __future__ import annotations

import json
from dataclasses import dataclass
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


@dataclass(frozen=True, slots=True)
class _Tier3Protocol:
    """Bind one Tier-3 protocol directory to its exact oracle inventory."""

    name: str
    identifier: str

    @property
    def root(self) -> Path:
        """Return the protocol-owned Tier-3 directory."""
        return _TIER3_ROOT / self.name

    @property
    def inventory(self) -> Path:
        """Return the protocol-owned exact oracle inventory."""
        return self.root / "oracle_inventory.json"

    @property
    def node_prefix(self) -> str:
        """Return the stable module prefix owned by this protocol."""
        return f"tests.tier3.{self.name}."


_TIER3_PROTOCOLS = (
    _Tier3Protocol("vn2", "vn2-conditional-replay"),
    _Tier3Protocol("m5", "m5-frozen-scorer-parity"),
)


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Fail collection for every orphaned, cross-tier, or self-owned witness."""
    gates = _marker_declarations(items, "oracle_gate")
    witnesses = _marker_declarations(items, "oracle_witness")
    try:
        require_exact_witnesses(gates, witnesses)
        for protocol in _required_tier3_protocols(config, items):
            required_gates, required_witnesses = _required_tier3_inventory(protocol)
            require_exact_inventory(
                (
                    item
                    for item in gates
                    if item.tier == "tier3" and item.nodeid.startswith(protocol.node_prefix)
                ),
                (
                    item
                    for item in witnesses
                    if item.tier == "tier3" and item.nodeid.startswith(protocol.node_prefix)
                ),
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


def _required_tier3_protocols(
    config: pytest.Config,
    items: list[pytest.Item],
) -> tuple[_Tier3Protocol, ...]:
    required: list[_Tier3Protocol] = []
    item_paths = tuple(Path(item.path).resolve() for item in items)
    argument_paths: list[Path] = []
    for argument in config.args:
        target_text = str(argument).partition("::")[0]
        target = Path(target_text)
        if not target.is_absolute():
            target = Path(config.rootpath) / target
        argument_paths.append(target.resolve())
    for protocol in _TIER3_PROTOCOLS:
        protocol_root = protocol.root.resolve()
        selected_by_item = any(path.is_relative_to(protocol_root) for path in item_paths)
        selected_by_argument = any(
            path == protocol_root
            or path.is_relative_to(protocol_root)
            or protocol_root.is_relative_to(path)
            for path in argument_paths
        )
        if selected_by_item or selected_by_argument:
            required.append(protocol)
    return tuple(required)


def _required_tier3_inventory(
    protocol: _Tier3Protocol,
) -> tuple[tuple[WitnessDeclaration, ...], tuple[WitnessDeclaration, ...]]:
    path = protocol.inventory
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
        if identifier != protocol.identifier:
            raise WitnessPairingError(
                f"required Tier 3 {protocol.name} oracle inventory {role} "
                f"must use {protocol.identifier}"
            )
        if (
            not isinstance(nodeid, str)
            or not nodeid.startswith(protocol.node_prefix)
            or "::" not in nodeid
        ):
            raise WitnessPairingError(
                f"required Tier 3 {protocol.name} oracle inventory {role} "
                "must use a stable protocol-owned node ID"
            )
        declarations[role] = WitnessDeclaration("tier3", identifier, nodeid)
    if declarations["gate"].nodeid == declarations["witness"].nodeid:
        raise WitnessPairingError("required Tier 3 oracle gate and witness must use distinct nodes")
    return (declarations["gate"],), (declarations["witness"],)
