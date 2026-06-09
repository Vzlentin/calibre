"""Download the M5 dataset into data/m5/ via datasetsforecast."""

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

OUT_DIR = Path("data/m5")
logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default=str(OUT_DIR), help="Target directory")
    args = parser.parse_args(argv)
    target = Path(args.target)

    if (target / "sales_train_evaluation.csv").exists():
        logger.info("%s already has M5 data, skipping download.", target)
        return

    from datasetsforecast.m5 import M5

    # datasetsforecast extracts into <parent>/m5/datasets/; flatten the CSVs
    # into the target directory the M5 dataset adapter reads.
    parent = target.parent if target.name == "m5" else target
    M5.download(str(parent))
    staging = parent / "m5" / "datasets"
    target.mkdir(parents=True, exist_ok=True)
    for csv in sorted(staging.glob("*.csv")):
        dest = target / csv.name
        if dest.exists():
            dest.unlink()
        shutil.move(str(csv), str(dest))
    shutil.rmtree(staging, ignore_errors=True)
    for path in sorted(target.iterdir()):
        logger.info("%s (%.1f MB)", path.name, path.stat().st_size / 1e6)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
