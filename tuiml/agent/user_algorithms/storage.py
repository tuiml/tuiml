"""Create, list and delete user algorithm versions on disk."""

from __future__ import annotations

import shutil
from typing import Any, Dict, List, Optional

from ._paths import USER_ALGS_DIR, _algorithm_file, _read_metadata, _source_hash, _validate_name, _validate_version, _write_metadata
from .registration import _import_and_register, _versioned_alias_name
from .validation import _ast_validate

def create(name: str, kind: str, code: str,
           version: str = "1.0.0",
           description: Optional[str] = None,
           force: bool = False) -> Dict[str, Any]:
    """Persist, validate, and register a new user algorithm.

    Parameters
    ----------
    name : str
        Algorithm name; must be a valid Python identifier. Used as the
        directory name under ``USER_ALGS_DIR``.
    kind : str
        Either ``'classifier'`` or ``'regressor'`` (case-insensitive). Must
        match the base class of the decorated class in ``code``.
    code : str
        Full Python source defining one ``@classifier``/``@regressor`` class.
    version : str, default="1.0.0"
        Semver version (MAJOR.MINOR.PATCH).
    description : str, optional
        Human-readable description; defaults to the class docstring's first line.
    force : bool, default=False
        Overwrite an existing ``name``/``version`` on disk.

    Returns
    -------
    result : Dict[str, Any]
        On success: keys ``status``, ``registered_as`` (list of aliases),
        ``name``, ``class_name``, ``kind``, ``version``, ``source_hash``,
        ``path``, and ``usage_hint``. On failure: keys ``status``,
        ``error_type``, ``error`` (and ``path`` for ``AlreadyExists``).
    """
    err = _validate_name(name) or _validate_version(version)
    if err:
        return {"status": "error", "error_type": "ValueError", "error": err}
    kind = kind.lower()
    if kind not in {"classifier", "regressor"}:
        return {"status": "error", "error_type": "ValueError",
                "error": f"kind must be 'classifier' or 'regressor', got {kind!r}"}

    ok, reason = _ast_validate(code)
    if not ok:
        return {"status": "error", "error_type": "UnsafeSource", "error": reason}

    target_file = _algorithm_file(name, version)
    if target_file.exists() and not force:
        return {
            "status": "error", "error_type": "AlreadyExists",
            "error": f"{name} v{version} already exists at {target_file}. "
                     "Pass force=true to overwrite or bump the version.",
            "path": str(target_file),
        }

    target_file.parent.mkdir(parents=True, exist_ok=True)
    target_file.write_text(code)

    try:
        cls, detected_kind = _import_and_register(target_file, name, version)
    except Exception as e:
        # Roll back the file write so we don't keep un-loadable code.
        try:
            target_file.unlink(missing_ok=True)
        except Exception:
            pass
        return {"status": "error", "error_type": "LoadError", "error": str(e)}

    if detected_kind != kind:
        return {
            "status": "error", "error_type": "KindMismatch",
            "error": f"declared kind={kind} but imported class is a {detected_kind}",
        }

    metadata = {
        "name": name,
        "class_name": cls.__name__,
        "kind": kind,
        "version": version,
        "description": description or (cls.__doc__ or "").splitlines()[0] if cls.__doc__ else "",
        "source_hash": _source_hash(code),
    }
    _write_metadata(target_file.parent, metadata)

    return {
        "status": "success",
        "registered_as": [cls.__name__, _versioned_alias_name(cls.__name__, version)],
        "name": name,
        "class_name": cls.__name__,
        "kind": kind,
        "version": version,
        "source_hash": metadata["source_hash"],
        "path": str(target_file),
        "usage_hint": (
            f"Train with: tuiml_train(algorithm='{cls.__name__}', ...) "
            f"or the pinned version: tuiml_train(algorithm='{_versioned_alias_name(cls.__name__, version)}', ...)"
        ),
    }


def list_all() -> Dict[str, Any]:
    """List every user-authored algorithm on disk.

    Returns
    -------
    result : Dict[str, Any]
        Keys ``status``, ``algorithms`` (list of dicts with ``name``,
        ``class_name``, ``kind``, ``version``, ``description``,
        ``source_hash``, ``path``, ``versioned_alias``), and ``count``
        (or ``root`` when the directory does not exist yet).
    """
    if not USER_ALGS_DIR.exists():
        return {"status": "success", "algorithms": [], "root": str(USER_ALGS_DIR)}

    rows: List[Dict[str, Any]] = []
    for name_dir in sorted(USER_ALGS_DIR.iterdir()):
        if not name_dir.is_dir():
            continue
        for ver_dir in sorted(name_dir.iterdir()):
            if not ver_dir.is_dir():
                continue
            meta = _read_metadata(ver_dir)
            rows.append({
                "name": meta.get("name", name_dir.name),
                "class_name": meta.get("class_name", name_dir.name),
                "kind": meta.get("kind"),
                "version": meta.get("version", ver_dir.name),
                "description": meta.get("description"),
                "source_hash": meta.get("source_hash"),
                "path": str(ver_dir / "algorithm.py"),
                "versioned_alias": _versioned_alias_name(
                    meta.get("class_name", name_dir.name),
                    meta.get("version", ver_dir.name),
                ),
            })
    return {"status": "success", "algorithms": rows, "count": len(rows)}


def delete(name: str, version: Optional[str] = None) -> Dict[str, Any]:
    """Delete one version (or every version when ``version`` is None).

    Parameters
    ----------
    name : str
        Algorithm name (directory under ``USER_ALGS_DIR``).
    version : str, optional
        Specific semver version to delete. When None, every version and the
        parent directory are removed.

    Returns
    -------
    result : Dict[str, Any]
        On success: keys ``status``, ``name``, ``removed_versions`` (list of
        version strings), and ``note``. On failure: keys ``status``,
        ``error_type``, ``error``.
    """
    err = _validate_name(name)
    if err:
        return {"status": "error", "error_type": "ValueError", "error": err}

    base = USER_ALGS_DIR / name
    if not base.exists():
        return {"status": "error", "error_type": "NotFound",
                "error": f"no user algorithm named {name!r} on disk"}

    removed: List[str] = []
    if version is None:
        for ver_dir in base.iterdir():
            if ver_dir.is_dir():
                shutil.rmtree(ver_dir)
                removed.append(ver_dir.name)
        shutil.rmtree(base, ignore_errors=True)
    else:
        ver_err = _validate_version(version)
        if ver_err:
            return {"status": "error", "error_type": "ValueError", "error": ver_err}
        ver_dir = base / version
        if not ver_dir.exists():
            return {"status": "error", "error_type": "NotFound",
                    "error": f"{name} v{version} is not installed"}
        shutil.rmtree(ver_dir)
        removed.append(version)
        # Remove empty parent.
        if base.exists() and not any(base.iterdir()):
            base.rmdir()

    return {
        "status": "success",
        "name": name,
        "removed_versions": removed,
        "note": "Registry entries remain until the MCP server restarts.",
    }
