"""Exercise the VN2 protocol adapter through the generic engine time loop."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import pytest
from tests.import_inspection import imported_modules
from tests.vn2_fixtures import (
    BASE_WEEKS,
    KEY_COLUMNS,
    REVEAL_WEEKS,
    SALES_FILES,
    STATIC_FILES,
    calibrated_config_payload,
    refresh_inventory,
    synthetic_config_payload,
    write_config,
    write_dataset,
)

from newcalibre.domain import ActualsSemantics, GuaranteeClaim, interval_columns, quantile_column
from newcalibre.protocols.vn2 import (
    VN2RunResult,
    load_vn2_config,
    load_vn2_dataset,
    run_vn2,
)

pytestmark = pytest.mark.tier1

FORBIDDEN_VN2_MODULES = frozenset(
    {
        "newcalibre.engine.settlement",
        "newcalibre.ordering.simulation",
    }
)


def _dataset(
    root: Path,
    *,
    base_value: float = 1.0,
    future_value: float = 20.0,
    payload: dict[str, object] | None = None,
):
    data, inventory, config_path = write_dataset(
        root,
        base_value=base_value,
        future_value=future_value,
    )
    config_payload = synthetic_config_payload() if payload is None else payload
    config_payload["model_config"]["m"] = len(BASE_WEEKS)  # type: ignore[index]
    write_config(config_path, config_payload)
    config = load_vn2_config(config_path)
    return load_vn2_dataset(data, inventory, config), data, inventory


def _first_origin_orders(result: VN2RunResult) -> tuple[object, ...]:
    origin = result.time_loop.decision_origins[0]
    return tuple(order for order in result.orders if order.origin == origin)


def _forbidden_vn2_imports(modules: set[str]) -> set[str]:
    return {
        module
        for module in modules
        if any(
            module == forbidden or module.startswith(f"{forbidden}.")
            for forbidden in FORBIDDEN_VN2_MODULES
        )
    }


def test_run_vn2_returns_raw_row_exact_engine_facts_with_hand_checked_costs(
    tmp_path: Path,
) -> None:
    dataset, _data, inventory = _dataset(tmp_path)

    result = run_vn2(dataset)

    assert isinstance(result, VN2RunResult)
    assert result.input_inventory_sha256 == hashlib.sha256(inventory.read_bytes()).hexdigest()
    assert result.time_loop.decision_origins == dataset.config.decision_origins
    assert result.time_loop.settlement_periods == dataset.config.realized_periods
    assert len(result.orders) == dataset.config.series_count * dataset.config.round_count
    assert len(result.settlements) == (
        dataset.config.series_count * len(dataset.config.realized_periods)
    )
    assert result.series_identities == {
        "0_126": (0, 126),
        "1_127": (1, 127),
    }
    with pytest.raises(TypeError):
        result.series_identities["new"] = (9, 9)  # type: ignore[index]

    first_order = next(
        order
        for order in result.orders
        if order.series_key == "0_126" and order.origin == dataset.config.decision_origins[0]
    )
    assert first_order.quantity == 0.0
    assert first_order.evidence is not None
    assert first_order.evidence.source_columns == (quantile_column(0.5),)
    assert first_order.evidence.source_descriptor.type.claim is GuaranteeClaim.NONE
    assert any(order.quantity == 0.0 for order in result.orders)

    first = next(
        record
        for record in result.settlements
        if record.series_key == "0_126" and record.period == dataset.config.realized_periods[0]
    )
    assert first.arrivals == 5.0
    assert first.transition.demand == 4.0
    assert first.transition.fulfilled_demand == 4.0
    assert first.transition.unmet_demand == 0.0
    assert first.transition.closing_on_hand == 4.0
    assert first.inventory_position.on_order == 7.0
    assert first.holding.rate == 0.2
    assert first.holding.amount == 0.8
    assert first.shortage.amount == 0.0
    assert first.realized_cost == first.holding.amount + first.shortage.amount

    second = next(
        record
        for record in result.settlements
        if record.series_key == "0_126" and record.period == dataset.config.realized_periods[1]
    )
    assert second.arrivals == 7.0
    assert second.transition.demand == 20.0
    assert second.transition.fulfilled_demand == 11.0
    assert second.transition.unmet_demand == 9.0
    assert second.transition.closing_on_hand == 0.0
    assert second.shortage.amount == 9.0
    assert all(
        record.actuals_semantics is ActualsSemantics.CENSORED_SALES_SURROGATE
        for record in result.settlements
    )
    assert result.time_loop.inventory_positions["0_126"].on_order == 0.0


def test_calibrated_run_exposes_ordinary_forecasts_coverage_and_cold_start_orders(
    tmp_path: Path,
) -> None:
    payload = calibrated_config_payload()
    dataset, *_ = _dataset(tmp_path, payload=payload)

    result = run_vn2(dataset)

    coverage = dataset.config.cost_structure.critical_ratio
    upper = interval_columns(coverage)[1]
    calibrated = [
        outcome for outcome in result.coverage_report.outcomes if outcome.bound_key == (upper,)
    ]
    native = [
        outcome
        for outcome in result.coverage_report.outcomes
        if outcome.bound_key == (quantile_column(0.5),)
    ]
    assert len(result.forecasts) == (
        dataset.config.series_count * dataset.config.round_count * dataset.config.task_horizon
    )
    assert len(calibrated) == len(result.forecasts)
    assert not native
    assert all(quantile_column(0.5) in row.values for row in result.forecasts)
    assert any(outcome.scored for outcome in calibrated)

    ready_orders = [order for order in result.orders if order.evidence is not None]
    cold_orders = [order for order in result.orders if order.evidence is None]
    assert cold_orders
    assert all(order.quantity == 0.0 for order in cold_orders)
    assert ready_orders
    assert all(order.evidence.source_columns == (upper,) for order in ready_orders)


def test_run_vn2_obeys_alternate_review_and_lead_cadence(tmp_path: Path) -> None:
    data, inventory, config_path = write_dataset(tmp_path)
    sales_names = SALES_FILES[:7]
    for name in set(SALES_FILES) - set(sales_names):
        (data / name).unlink()
    in_stock = pd.read_csv(data / STATIC_FILES[1])[["Store", "Product", *BASE_WEEKS]]
    in_stock.to_csv(data / STATIC_FILES[1], index=False, lineterminator="\n")
    initial = pd.read_csv(data / STATIC_FILES[2]).drop(columns=["In Transit W+2"])
    initial.to_csv(data / STATIC_FILES[2], index=False, lineterminator="\n")

    payload = synthetic_config_payload()
    payload["decision"] = {
        "round_count": 2,
        "lead_time": 1,
        "review_period": 2,
        "protection_period": 3,
        "task_horizon": 3,
        "drain_periods": 1,
        "origins": ["2024-04-22", "2024-05-06"],
    }
    payload["model_config"]["m"] = len(BASE_WEEKS)
    payload["files"]["sales_reveals"] = list(sales_names)
    payload["columns"]["initial_state_columns"].remove("In Transit W+2")
    payload["columns"]["initial_pipeline"] = ["In Transit W+1"]
    write_config(config_path, payload)
    refresh_inventory(data, inventory, names=(*STATIC_FILES, *sales_names))
    dataset = load_vn2_dataset(data, inventory, load_vn2_config(config_path))

    result = run_vn2(dataset)

    assert result.time_loop.decision_origins == dataset.config.decision_origins
    assert result.time_loop.settlement_periods == dataset.config.realized_periods
    assert len(result.orders) == 4
    assert len(result.settlements) == 10
    assert {order.origin for order in result.orders} == set(dataset.config.decision_origins)
    assert all(
        order.arrival_period == dataset.config.calendar.advance(order.origin, 1)
        for order in result.orders
    )


def test_run_vn2_cannot_see_later_reveals_but_uses_prior_history(tmp_path: Path) -> None:
    baseline, *_ = _dataset(tmp_path / "baseline", base_value=10.0, future_value=20.0)
    changed_future, *_ = _dataset(
        tmp_path / "future",
        base_value=10.0,
        future_value=999.0,
    )
    changed_prior, *_ = _dataset(
        tmp_path / "prior",
        base_value=11.0,
        future_value=20.0,
    )

    baseline_orders = _first_origin_orders(run_vn2(baseline))
    future_orders = _first_origin_orders(run_vn2(changed_future))
    prior_orders = _first_origin_orders(run_vn2(changed_prior))

    assert future_orders == baseline_orders
    assert prior_orders != baseline_orders


def test_run_vn2_settles_missing_revealed_sales_as_zero(tmp_path: Path) -> None:
    dataset, data, inventory = _dataset(tmp_path)
    for filename in SALES_FILES[1:]:
        frame = pd.read_csv(data / filename)
        frame.loc[0, REVEAL_WEEKS[0]] = None
        frame.to_csv(data / filename, index=False, lineterminator="\n")
    refresh_inventory(data, inventory)
    dataset = load_vn2_dataset(data, inventory, dataset.config)

    result = run_vn2(dataset)
    first = next(
        record
        for record in result.settlements
        if record.series_key == "0_126" and record.period == dataset.config.realized_periods[0]
    )

    assert first.transition.demand == 0.0
    assert first.shortage.amount == 0.0


def test_run_vn2_uses_configured_cost_rates(tmp_path: Path) -> None:
    payload = synthetic_config_payload()
    payload["cost"] = {
        "currency": "EUR",
        "underage_rate": 2.0,
        "overage_rate": 0.7,
        "holding_rate": 0.7,
        "shortage_rate": 2.0,
    }
    dataset, *_ = _dataset(tmp_path, payload=payload)

    result = run_vn2(dataset)

    assert {record.holding.rate for record in result.settlements} == {0.7}
    assert {record.shortage.rate for record in result.settlements} == {2.0}
    assert all(
        record.holding.amount == 0.7 * record.transition.closing_on_hand
        for record in result.settlements
    )
    assert all(
        record.shortage.amount == 2.0 * record.transition.unmet_demand
        for record in result.settlements
    )


def test_run_vn2_uses_configured_series_keys_when_initial_columns_are_reordered(
    tmp_path: Path,
) -> None:
    data, inventory, config_path = write_dataset(tmp_path)
    initial_path = data / STATIC_FILES[2]
    initial = pd.read_csv(initial_path)
    reordered = [*reversed(KEY_COLUMNS), *initial.columns[2:]]
    initial[reordered].to_csv(initial_path, index=False, lineterminator="\n")
    payload = synthetic_config_payload()
    payload["model_config"]["m"] = len(BASE_WEEKS)  # type: ignore[index]
    payload["columns"]["initial_state_columns"] = reordered  # type: ignore[index]
    write_config(config_path, payload)
    refresh_inventory(data, inventory)
    dataset = load_vn2_dataset(data, inventory, load_vn2_config(config_path))

    result = run_vn2(dataset)

    first_period = dataset.config.realized_periods[0]
    arrivals = {
        record.series_key: record.arrivals
        for record in result.settlements
        if record.period == first_period
    }
    assert arrivals == {"0_126": 5.0, "1_127": 6.0}
    assert result.series_identities == {"0_126": (0, 126), "1_127": (1, 127)}


def test_vn2_adapter_contains_no_protocol_local_transition_or_accounting() -> None:
    import newcalibre.protocols.vn2.adapter as adapter_module

    source = Path(adapter_module.__file__).read_text(encoding="utf-8")
    modules = {
        module for _line, module in imported_modules(source, package="newcalibre.protocols.vn2")
    }

    assert not _forbidden_vn2_imports(modules)
    assert "lost_sales_transition" not in source
    assert "BookedCost" not in source


def test_vn2_layering_detector_bites_on_supported_import_forms() -> None:
    modules = {
        module
        for _line, module in imported_modules(
            "import newcalibre.engine.settlement\n"
            "import newcalibre.engine.settlement.replay\n"
            "from newcalibre.engine.settlement import settle\n"
            "from newcalibre.engine import settlement\n"
            "from ...engine import settlement as relative_settlement\n"
            "import importlib\n"
            "importlib.import_module(name='newcalibre.ordering.simulation')\n"
            "__import__('newcalibre.engine.settlement')\n",
            package="newcalibre.protocols.vn2",
        )
    }

    assert _forbidden_vn2_imports(modules) == {
        "newcalibre.engine.settlement",
        "newcalibre.engine.settlement.replay",
        "newcalibre.engine.settlement.settle",
        "newcalibre.ordering.simulation",
    }
