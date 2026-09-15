"""Load Atelier's zndx.supervision.v1 instance for EngineHello."""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

DEFAULT_RELATIVE = Path("config/supervision/atelier.textproto")


def instance_path(root: Path | None = None) -> Path:
    base = root or Path(__file__).resolve().parents[3]
    return base / DEFAULT_RELATIVE


@dataclass(frozen=True)
class SpecIdentity:
    spec_version: str
    sha256: str
    path: Path


def load_spec(root: Path | None = None) -> SpecIdentity | None:
    path = instance_path(root)
    if not path.is_file():
        return None
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    version = ""
    text = raw.decode("utf-8", errors="replace")
    m = re.search(r'^\s*spec_version:\s*"([^"]+)"', text, re.MULTILINE)
    if m:
        version = m.group(1)
    return SpecIdentity(spec_version=version, sha256=digest, path=path)
