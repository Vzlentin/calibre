"""Config-driven entrypoint: ``python -m benchmarks.vn2 --config <cfg>``."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from benchmarks.common.tracking import load_dotenv
from benchmarks.vn2.run_benchmark import run_from_config
from calibre.cli.config import load_config


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the VN2 benchmark harness")
    parser.add_argument(
        "--config",
        default="benchmarks/vn2/config/winning.yaml",
        help="Path to a BackendConfig YAML file",
    )
    args = parser.parse_args(argv)
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    config = load_config(args.config)
    run_from_config(config)


if __name__ == "__main__":
    main()
