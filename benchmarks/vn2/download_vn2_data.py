from pathlib import Path
import requests
import json

SCRIPT_DIR = Path(__file__).resolve().parent
LINKS_PATH = SCRIPT_DIR / "vn2_file_links.json"
OUT_DIR = Path("data/vn2")


def download_file(session: requests.Session, url: str, destination: Path) -> None:
    with session.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        with destination.open("wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)


def main() -> None:
    if OUT_DIR.exists():
        print(f"{OUT_DIR} already exists, skipping download.")
        return

    with LINKS_PATH.open(encoding="utf-8") as f:
        file_entries = json.load(f)["files"]

    OUT_DIR.mkdir(parents=True, exist_ok=False)

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0",
        }
    )

    for entry in file_entries:
        filename = entry["file_name"]
        url = entry["url"]
        destination = OUT_DIR / filename
        print(f"Downloading {filename}...")
        download_file(session, url, destination)

    print(f"Done. Downloaded {len(file_entries)} files into {OUT_DIR}")


if __name__ == "__main__":
    main()
