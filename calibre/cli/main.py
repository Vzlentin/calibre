"""CLI entry point: argparse wiring and the ``app`` dispatcher."""

from __future__ import annotations

import argparse
import json
import sys

from calibre.cli import commands
from calibre.core.logging import setup_logging
from calibre.evaluation.m5_coverage import PASS


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="calibre")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--log-format", choices=["json", "text"], default="json")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--config", required=True)
    run_parser.add_argument("--metrics-port", type=int)
    run_parser.add_argument(
        "--tune",
        action="store_true",
        help="Run the config's hpo search instead of a backtest (requires an hpo block).",
    )

    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--config", required=True)

    subparsers.add_parser("health")

    sweep_parser = subparsers.add_parser("run-sweep")
    sweep_parser.add_argument("--configs", required=True)

    m5_coverage_parser = subparsers.add_parser("score-m5-coverage")
    m5_coverage_parser.add_argument("--ledger", required=True)
    m5_coverage_parser.add_argument("--coverage", type=float, default=0.9)
    m5_coverage_parser.add_argument("--output-dir")
    m5_coverage_parser.add_argument(
        "--report-only",
        action="store_true",
        help="Write coverage artifacts without failing on a non-PASS acceptance gate.",
    )

    return parser


def app(argv: list[str] | None = None) -> int:
    """Parse arguments, dispatch the chosen subcommand, and return an exit code."""
    parser = _parser()
    args = parser.parse_args(argv)
    setup_logging(args.log_level, format=args.log_format)

    if args.command == "run":
        result = commands.run(args.config, metrics_port=args.metrics_port, tune=args.tune)
        if args.tune:
            # A tune run writes no ledger — the discovered config is its only output,
            # so surface it as JSON (default=str for numpy floats / lag transforms).
            print(json.dumps(result, sort_keys=True, default=str))
    elif args.command == "validate":
        commands.validate(args.config)
    elif args.command == "health":
        print(json.dumps(commands.health(), sort_keys=True))
    elif args.command == "run-sweep":
        commands.run_sweep(args.configs)
    elif args.command == "score-m5-coverage":
        artifacts = commands.score_m5_coverage(
            args.ledger,
            coverage=args.coverage,
            output_dir=args.output_dir,
        )
        print(
            json.dumps(
                {
                    "coverage_by_node": str(artifacts.coverage_by_node_path),
                    "report": str(artifacts.report_path),
                    "summary": str(artifacts.summary_path),
                    "acceptance_status": artifacts.acceptance_status,
                    "population_status": artifacts.summary["population_status"],
                    "level_status": artifacts.summary["level_status"],
                    "completeness_status": artifacts.summary["completeness_status"],
                },
                sort_keys=True,
            )
        )
        if not args.report_only and artifacts.acceptance_status != PASS:
            return 1
    else:
        parser.error(f"Unknown command: {args.command}")
    return 0


if __name__ == "__main__":
    sys.exit(app())
