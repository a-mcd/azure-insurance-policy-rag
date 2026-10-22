"""Small file helpers shared by the PDF extractors."""

from __future__ import annotations

import hashlib
from pathlib import Path


def calculate_sha256(path: Path) -> str:
    """Calculate the SHA-256 checksum of a file, reading it in 1 MiB blocks."""
    digest = hashlib.sha256()

    with path.open("rb") as source_file:
        for block in iter(lambda: source_file.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()
