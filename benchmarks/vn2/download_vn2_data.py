import argparse
import json
import logging
from pathlib import Path

import fsspec
import requests

SCRIPT_DIR = Path(__file__).resolve().parent
LINKS_PATH = SCRIPT_DIR / "vn2_file_links.json"
OUT_DIR = Path("data/vn2")
logger = logging.getLogger(__name__)


def _join_uri(base: str, filename: str) -> str:
    if "://" not in base:
        return str(Path(base) / filename)
    return f"{base.rstrip('/')}/{filename}"


def download_file(session: requests.Session, url: str, destination: str) -> None:
    with session.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        with fsspec.open(destination, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default=str(OUT_DIR), help="Target directory URI")
    args = parser.parse_args(argv)
    target = str(args.target)

    fs, target_path = fsspec.core.url_to_fs(target)
    if fs.exists(target_path):
        logger.info("%s already exists, skipping download.", target)
        return

    with LINKS_PATH.open(encoding="utf-8") as f:
        file_entries = json.load(f)["files"]

    fs.mkdirs(target_path, exist_ok=False)

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0",
        }
    )

    for entry in file_entries:
        filename = entry["file_name"]
        url = entry["url"]
        destination = _join_uri(target, filename)
        logger.info("Downloading %s...", filename)
        download_file(session, url, destination)

    logger.info("Done. Downloaded %s files into %s", len(file_entries), target)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
