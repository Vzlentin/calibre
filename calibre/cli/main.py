from __future__ import annotations

import argparse
import json
import sys

from calibre.cli import commands
from calibre.core.logging import setup_logging


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="calibre")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--log-format", choices=["json", "text"], default="json")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--config", required=True)
    run_parser.add_argument("--metrics-port", type=int)

    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--config", required=True)

    subparsers.add_parser("health")

    sweep_parser = subparsers.add_parser("run-sweep")
    sweep_parser.add_argument("--configs", required=True)

    return parser


def app(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    setup_logging(args.log_level, format=args.log_format)

    if args.command == "run":
        commands.run(args.config, metrics_port=args.metrics_port)
    elif args.command == "validate":
        commands.validate(args.config)
    elif args.command == "health":
        print(json.dumps(commands.health(), sort_keys=True))
    elif args.command == "run-sweep":
        commands.run_sweep(args.configs)
    else:
        parser.error(f"Unknown command: {args.command}")
    return 0


if __name__ == "__main__":
    sys.exit(app())
