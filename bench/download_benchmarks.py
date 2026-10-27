"""Download and verify nine forecasting benchmarks; never commit their outputs.

Run uv run --no-dev python bench/download_benchmarks.py --help for usage.
M5 and VN2 already have loaders and are deliberately excluded.
"""

import argparse
import tempfile
from pathlib import Path
from urllib.request import Request, urlopen

import beijing
import bike
import ett
import favorita
import freshretail
import lstnet
from sources import Source

DATA = Path(__file__).resolve().parents[1] / "data"
SOURCES = {
    "etth1": ett.SOURCES,
    **{name: (source,) for name, source in lstnet.SOURCES.items()},
    "beijing": beijing.SOURCES,
    "bike": bike.SOURCES,
    "favorita": favorita.SOURCES,
    "freshretail": freshretail.SOURCES,
}


def download(source: Source, directory: Path) -> None:
    """Stream to a temporary file, verify, then atomically publish. Never extract ZIPs."""
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / source.name
    if destination.exists():
        source.verify(destination)
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
            source.verify(temporary)
            temporary.replace(destination)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="Load raw files with the dataset's load(path). See bench/README.md for an example.",
    )
    parser.add_argument("datasets", nargs="*", help=f"all, or: {', '.join(SOURCES)}")
    parser.add_argument("--list", action="store_true", help="list without downloading")
    parser.add_argument(
        "--data-dir", type=Path, default=DATA, help="output root (default: repo/data)"
    )
    args = parser.parse_args()
    if args.list:
        print("\n".join(SOURCES))
        return
    names = list(SOURCES) if args.datasets == ["all"] else args.datasets
    if not names or any(name not in SOURCES for name in names):
        parser.error("choose 'all' alone or one or more names from --list (M5/VN2 excluded)")
    for name in dict.fromkeys(names):
        for source in SOURCES[name]:
            download(source, args.data_dir / name)


if __name__ == "__main__":
    main()
