"""Pinned upstream files and their byte-level verification."""

import hashlib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Source:
    """One exact upstream file, verified before publishing or loading."""

    name: str
    url: str
    sha256: str

    def verify(self, path: str | Path) -> None:
        """Reject bytes that differ from the pinned release."""
        with Path(path).open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != self.sha256:
            raise ValueError(f"SHA-256 mismatch: {path}; {actual} != {self.sha256}")
