"""On-disk layout for agent-authored algorithms."""

from __future__ import annotations

import hashlib
import re
from typing import Any, Dict, Optional
from pathlib import Path

USER_ALGS_DIR = Path.home() / ".tuiml" / "user_algorithms"


_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


_VERSION_RE = re.compile(r"^\d+\.\d+\.\d+$")


def _validate_name(name: str) -> Optional[str]:
    """Return an error message if ``name`` is not a valid Python identifier, else None."""
    if not _NAME_RE.match(name):
        return f"name must be a valid Python identifier, got {name!r}"
    return None


def _validate_version(version: str) -> Optional[str]:
    """Return an error message if ``version`` is not MAJOR.MINOR.PATCH semver, else None."""
    if not _VERSION_RE.match(version):
        return f"version must be MAJOR.MINOR.PATCH semver, got {version!r}"
    return None


def _alg_dir(name: str, version: str) -> Path:
    """Return the on-disk directory for one algorithm version."""
    return USER_ALGS_DIR / name / version


def _algorithm_file(name: str, version: str) -> Path:
    """Return the ``algorithm.py`` path for one algorithm version."""
    return _alg_dir(name, version) / "algorithm.py"


def _source_hash(source: str) -> str:
    """Return a short (16-hex-char) SHA-256 hash of the source text."""
    return hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]


def _write_metadata(dir_path: Path, metadata: Dict[str, Any]) -> None:
    """Write ``metadata`` to ``metadata.json`` inside ``dir_path``."""
    import json
    (dir_path / "metadata.json").write_text(json.dumps(metadata, indent=2))


def _read_metadata(dir_path: Path) -> Dict[str, Any]:
    """Read ``metadata.json`` from ``dir_path``; empty dict if missing or invalid."""
    import json
    path = dir_path / "metadata.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}
