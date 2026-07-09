"""Tag-side oracle order extraction (KTD-A6): run_config + settle_on_round.

Executed *inside the oracle worktree's frozen venv* by the oracle-capture and
bootstrap-preflight workflows: the interpreter comes from the pinned tag's
lockfile, so the ``calibre`` imported here is the frozen engine at
``oracle-freeze-2026-07-06``, never the checkout's code. It drives the VN2
winning-loop config through ``run_config(..., settle_on_round=...)`` and
serializes each :class:`RoundResult`'s per-series orders mapping in
deterministic key order — the settle CLI's absent order ledger and stdout
totals are not fallbacks. The expected shape (6 decision rounds x 599
series) is hard-asserted; anything else fails the capture.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

WINNING_CONFIG = "benchmarks/vn2/config/vn2-winning-loop.yaml"
EXPECTED_ROUNDS = 6
EXPECTED_SERIES = 599


def main() -> int:
    """Run the pinned winning loop and serialize per-round orders."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worktree", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    worktree = args.worktree.resolve()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    # The frozen engine resolves dataset and config paths relative to its own
    # checkout; run from the worktree root exactly as `calibre run` would.
    os.chdir(worktree)

    from calibre.cli.commands import load_config, run_config

    captured: list[dict] = []

    def on_round(round_result) -> None:  # noqa: ANN001 — frozen type, tag-side
        captured.append(
            {
                "round_num": int(round_result.round_num),
                "origin": str(round_result.origin),
                "orders": {
                    str(key): float(value)
                    for key, value in sorted(round_result.orders.items())
                },
            }
        )

    config = load_config(WINNING_CONFIG)
    run_config(config, settle_on_round=on_round)

    if len(captured) != EXPECTED_ROUNDS:
        raise SystemExit(
            f"expected {EXPECTED_ROUNDS} decision rounds, observer fired {len(captured)}"
        )
    for entry in captured:
        if len(entry["orders"]) != EXPECTED_SERIES:
            raise SystemExit(
                f"round {entry['round_num']}: {len(entry['orders'])} series, "
                f"expected {EXPECTED_SERIES}"
            )

    report_files = {}
    for entry in captured:
        path = out / f"round-{entry['round_num']}.json"
        payload = json.dumps(entry, indent=2, sort_keys=True) + "\n"
        path.write_text(payload, encoding="utf-8")
        report_files[path.name] = hashlib.sha256(payload.encode("utf-8")).hexdigest()

    report = {
        "config": WINNING_CONFIG,
        "rounds": EXPECTED_ROUNDS,
        "series_per_round": EXPECTED_SERIES,
        "files": report_files,
    }
    (out / "extraction-report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report_files, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
