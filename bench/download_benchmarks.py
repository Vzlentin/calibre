"""Download the nine additional forecasting benchmarks; never commit their outputs.

Run `uv run --no-dev python bench/download_benchmarks.py --help` for the short
preparation process. M5 and VN2 already have loaders and are deliberately excluded.
"""

import argparse
import hashlib
import json
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.util import find_spec
from pathlib import Path
from urllib.request import Request, urlopen

DATA = Path(__file__).resolve().parents[1] / "data"
ETT = (
    "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/1d16c8f4f943005d613b5bc962e9eeb06058cf07"
)
LST = "https://raw.githubusercontent.com/laiguokun/multivariate-time-series-data/7f402f185cc2435b5e66aed13a3b560ed142e023"
FAVORITA = "https://huggingface.co/datasets/t4tiana/store-sales-time-series-forecasting/resolve/1ac138dfdfafc520bc29831e73373954b1b7bc05"
FRESH = "https://huggingface.co/datasets/Dingdong-Inc/FreshRetailNet-50K/resolve/c374e85975c82816c495acfa904cc55ab14a47d3"


@dataclass(frozen=True)
class Source:
    """One exact upstream file, checked before use (also when already downloaded)."""

    name: str
    url: str
    sha256: str


# File hashes from the audited selection, 2026-10-13; stores.csv checked separately.
SOURCES = {
    "etth1": (
        Source(
            "ETTh1.csv",
            f"{ETT}/ETT-small/ETTh1.csv",
            "f18de3ad269cef59bb07b5438d79bb3042d3be49bdeecf01c1cd6d29695ee066",
        ),
    ),
    "electricity": (
        Source(
            "electricity.txt.gz",
            f"{LST}/electricity/electricity.txt.gz",
            "3c4c069588198c1fcc95cace7bb69c99922129edfd673b7286661dad20badefa",
        ),
    ),
    "traffic": (
        Source(
            "traffic.txt.gz",
            f"{LST}/traffic/traffic.txt.gz",
            "c7be5a00519d344a5ec0eabdbfec5ea0c7dd1eed5f9b1a3843a93bb88086a56d",
        ),
    ),
    "solar": (
        Source(
            "solar_AL.txt.gz",
            f"{LST}/solar-energy/solar_AL.txt.gz",
            "1ae9329c3e1032acbb8dd080bfc397c44507006812a715d2788f172820489a1c",
        ),
    ),
    "exchange": (
        Source(
            "exchange_rate.txt.gz",
            f"{LST}/exchange_rate/exchange_rate.txt.gz",
            "6a4fbb561dcdee43f071a83badbeec9223a335552278348656738b556f1c2e9d",
        ),
    ),
    "beijing": (
        Source(
            "beijing.zip",
            "https://archive.ics.uci.edu/ml/machine-learning-databases/00501/PRSA2017_Data_20130301-20170228.zip",
            "d1b9261c54132f04c374f762f1e5e512af19f95c95fd6bfa1e8ac7e927e3b0b8",
        ),
    ),
    "bike": (
        Source(
            "bike.zip",
            "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip",
            "b70182d0d0508e9abbb79306ce5c0cec34869000f8220175ac83d11dbe845401",
        ),
    ),
    "favorita": (
        Source(
            "train.csv",
            f"{FAVORITA}/train.csv",
            "99a1b7f4241821df10fa71f23abdd71c634f71f901e3cab1400c0f0c583f862c",
        ),
        Source(
            "stores.csv",
            f"{FAVORITA}/stores.csv",
            "af503b2bce11d7906d249f81cc0598f10e2addcc9f6c59aa2d95c9f2652c296b",
        ),
    ),
    "freshretail": (
        Source(
            "train.parquet",
            f"{FRESH}/data/train.parquet",
            "6706832db892bbae4969c19d87e07975d2543d2ba7d7d4756360654785de5a3d",
        ),
        Source(
            "eval.parquet",
            f"{FRESH}/data/eval.parquet",
            "1b118840664280c6b88bffc84c80ee1f54c05d911e354b7599e5da10995e960e",
        ),
    ),
}


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def download(source: Source, directory: Path) -> dict:
    """Stream to a temporary file, verify, then atomically publish. Never extract ZIPs."""
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / source.name
    if destination.exists():
        actual = _sha256(destination)
        if actual != source.sha256:
            raise ValueError(f"SHA-256 mismatch: {destination}; remove it to download again")
        print(f"Verified {destination}", flush=True)
    else:
        print(f"Downloading {destination}", flush=True)
        temporary = None
        try:
            request = Request(source.url, headers={"User-Agent": "calibre-benchmarks/1"})
            with tempfile.NamedTemporaryFile(dir=directory, delete=False) as output:
                temporary = Path(output.name)
                with urlopen(request, timeout=120) as response:
                    while block := response.read(1024 * 1024):
                        output.write(block)
            actual = _sha256(temporary)
            if actual != source.sha256:
                raise ValueError(
                    f"SHA-256 mismatch for {source.url}: {actual} != {source.sha256}; "
                    "upstream changed or the download is corrupt"
                )
            temporary.replace(destination)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    return {
        "file": source.name,
        "url": source.url,
        "sha256": actual,
        "bytes": destination.stat().st_size,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "Process: 1. Download. 2. Add --prepare to build a Panel. "
            "For calendar gaps explicitly choose --missing ffill; for undated matrices "
            "acknowledge --positional-time. FreshRetailNet needs `uv run --with pyarrow`. "
            "3. load_prepared('data/<name>') returns panel/hierarchy/observed. "
            "See bench/README.md for a runnable forecaster example and data-use conditions."
        ),
    )
    parser.add_argument("datasets", nargs="*", help=f"all, or: {', '.join(SOURCES)}")
    parser.add_argument("--list", action="store_true", help="list without downloading")
    parser.add_argument(
        "--data-dir", type=Path, default=DATA, help="output root (default: repo/data)"
    )
    parser.add_argument(
        "--prepare", action="store_true", help="also save panel.npz and preparation.json"
    )
    parser.add_argument("--missing", choices=["raise", "ffill"], default="raise")
    parser.add_argument(
        "--positional-time", action="store_true", help="allow synthetic matrix dates"
    )
    parser.add_argument(
        "--limit-series", type=int, help="first K sorted identifiers, not a volume sample"
    )
    args = parser.parse_args()
    if args.list:
        print("\n".join(SOURCES))
        return
    names = list(SOURCES) if args.datasets == ["all"] else args.datasets
    if not names or any(name not in SOURCES for name in names):
        parser.error("choose 'all' alone or one or more names from --list (M5/VN2 excluded)")
    if args.limit_series is not None and args.limit_series < 1:
        parser.error("--limit-series must be positive")
    # Fail before downloading a large file if optional preparation dependencies are absent.
    if args.prepare:
        from load_benchmarks import MATRICES, load, save_prepared

        if any(name in MATRICES for name in names) and not args.positional_time:
            parser.error(
                "undated matrix preparation requires --positional-time (see bench/README.md)"
            )
        if "freshretail" in names and find_spec("pyarrow") is None:
            parser.error("FreshRetailNet preparation needs: uv run --no-dev --with pyarrow ...")
    for name in dict.fromkeys(names):
        directory = args.data_dir / name
        records = [download(source, directory) for source in SOURCES[name]]
        manifest = {"dataset": name, "verified_at": datetime.now(UTC).isoformat(), "files": records}
        (directory / "download.json").write_text(json.dumps(manifest, indent=2) + "\n")
        if args.prepare:
            data = load(
                name,
                directory,
                missing=args.missing,
                positional_time=args.positional_time,
                limit_series=args.limit_series,
            )
            save_prepared(data, directory)
            print(f"Prepared {name}: {data.panel.values.shape} [series, periods]", flush=True)
    print("\nNext: --prepare [--missing ffill] [--positional-time], then load_prepared(path).")
    print("Process + runnable Calibre forecast: bench/README.md. Outputs belong in ignored data/.")


if __name__ == "__main__":
    main()
