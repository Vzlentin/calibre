from __future__ import annotations

import io
import shutil
import tempfile
import zipfile
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from typing import TypeVar

T = TypeVar("T")


def pack_directory(path: Path) -> bytes:
    """Package a native save directory into transport bytes."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for item in sorted(path.rglob("*")):
            if item.is_file():
                archive.write(item, item.relative_to(path).as_posix())
    return buffer.getvalue()


def unpack_directory(blob: bytes, path: Path) -> None:
    """Restore transport bytes created by ``pack_directory``."""
    path.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(blob), mode="r") as archive:
        for member in archive.infolist():
            if member.is_dir():
                continue
            relative = PurePosixPath(member.filename)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"Unsafe artifact member path: {member.filename!r}")
            target = path.joinpath(*relative.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member, "r") as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)


def save_dir_to_bytes(save: Callable[[Path], object]) -> bytes:
    """Run ``save`` against a temp directory and pack the result to bytes."""
    with tempfile.TemporaryDirectory(prefix="calibre-artifact-") as temp_dir:
        path = Path(temp_dir)
        save(path)
        return pack_directory(path)


def load_dir_from_bytes(blob: bytes, load: Callable[[Path], T]) -> T:
    """Unpack ``blob`` into a temp directory and run ``load`` against it."""
    with tempfile.TemporaryDirectory(prefix="calibre-artifact-") as temp_dir:
        path = Path(temp_dir)
        unpack_directory(blob, path)
        return load(path)
