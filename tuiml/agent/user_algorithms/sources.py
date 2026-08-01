"""Read, search and edit algorithm source, user-authored or built-in."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pathlib import Path

from ._paths import USER_ALGS_DIR, _read_metadata, _source_hash, _validate_name, _validate_version, _write_metadata
from .registration import _import_and_register, _versioned_alias_name
from .research_log import _resolve_user_algorithm
from .validation import _ast_validate

def read_source(name: str, version: Optional[str] = None,
                builtin: bool = False) -> Dict[str, Any]:
    """Return the full source code of a user or built-in algorithm.

    For user algorithms, ``name`` is the directory name under USER_ALGS_DIR.
    For built-in algorithms, set ``builtin=True`` and pass the class name or
    file stem (e.g. ``'RandomForestClassifier'`` or ``'random_forest'``).

    Parameters
    ----------
    name : str
        User algorithm name (or versioned alias), or a built-in class
        name / file stem when ``builtin=True``.
    version : str, optional
        Pin a specific semver version of a user algorithm. When None, the
        newest on-disk version is used. Ignored for built-ins.
    builtin : bool, default=False
        Look up a built-in tuiml algorithm instead of a user algorithm.

    Returns
    -------
    result : Dict[str, Any]
        On success: keys ``status``, ``name``, ``version``, ``class_name``,
        ``kind``, ``path``, ``line_count``, ``source``, and
        ``source_with_line_numbers`` (built-ins return ``builtin`` and
        ``note`` instead of version/class/kind). On failure: keys ``status``,
        ``error_type``, ``error``.
    """
    if builtin:
        return _read_builtin_source(name)

    resolved = _resolve_user_algorithm(name) if version is None else None
    if resolved is None and version is not None:
        err = _validate_name(name) or _validate_version(version)
        if err:
            return {"status": "error", "error_type": "ValueError", "error": err}
        resolved = {"name": name, "version": version, "dir": USER_ALGS_DIR / name / version}
    if resolved is None:
        return {"status": "error", "error_type": "NotFound",
                "error": f"no user algorithm named {name!r} found on disk"}

    alg_file = resolved["dir"] / "algorithm.py"
    if not alg_file.exists():
        return {"status": "error", "error_type": "NotFound",
                "error": f"algorithm.py not found at {alg_file}"}

    source = alg_file.read_text(encoding="utf-8")
    meta = _read_metadata(resolved["dir"])
    lines = source.splitlines()
    numbered = "\n".join(f"{i+1:4d} | {line}" for i, line in enumerate(lines))
    return {
        "status": "success",
        "name": name,
        "version": resolved["version"],
        "class_name": meta.get("class_name", name),
        "kind": meta.get("kind"),
        "path": str(alg_file),
        "line_count": len(lines),
        "source": source,
        "source_with_line_numbers": numbered,
    }


def _read_builtin_source(name: str) -> Dict[str, Any]:
    """Locate and return source for a built-in tuiml algorithm file.

    Parameters
    ----------
    name : str
        Class name (e.g. ``'RandomForestClassifier'``) or file stem
        (e.g. ``'random_forest'``) to look up under ``tuiml/algorithms/``.

    Returns
    -------
    result : Dict[str, Any]
        On success: keys ``status``, ``name``, ``builtin``, ``path``,
        ``line_count``, ``source``, ``source_with_line_numbers``, ``note``.
        On failure: keys ``status``, ``error_type``, ``error``.
    """
    import tuiml.algorithms as _alg_pkg
    alg_root = Path(_alg_pkg.__file__).parent

    # Try class name → search all .py files for class definition
    candidates: List[Path] = []
    stem = name.lower().replace(" ", "_")
    for py_file in alg_root.rglob("*.py"):
        if py_file.name.startswith("_"):
            continue
        if py_file.stem == stem or py_file.stem == name:
            candidates.append(py_file)
        elif name in py_file.read_text(encoding="utf-8", errors="ignore"):
            # class name appears in the file
            text = py_file.read_text(encoding="utf-8", errors="ignore")
            if f"class {name}(" in text:
                candidates.insert(0, py_file)
            else:
                candidates.append(py_file)

    if not candidates:
        return {"status": "error", "error_type": "NotFound",
                "error": f"no built-in algorithm file found for {name!r}. "
                         "Use tuiml_list_algorithm_files to see all built-in paths."}

    py_file = candidates[0]
    source = py_file.read_text(encoding="utf-8")
    lines = source.splitlines()
    numbered = "\n".join(f"{i+1:4d} | {line}" for i, line in enumerate(lines))
    return {
        "status": "success",
        "name": name,
        "builtin": True,
        "path": str(py_file),
        "line_count": len(lines),
        "source": source,
        "source_with_line_numbers": numbered,
        "note": "Built-in algorithms are read-only. Fork to user algorithms with tuiml_create_algorithm.",
    }


def list_algorithm_files(builtin: bool = True, user: bool = True) -> Dict[str, Any]:
    """List all algorithm source files, built-in and/or user-authored.

    Parameters
    ----------
    builtin : bool, default=True
        Include built-in algorithm files from ``tuiml/algorithms/``.
    user : bool, default=True
        Include user algorithm files from ``USER_ALGS_DIR``.

    Returns
    -------
    result : Dict[str, Any]
        Keys ``status``, ``count``, and ``files``: a list of dicts with
        ``type`` (``'builtin'``/``'user'``) and ``path``, plus
        ``relative_path``/``category``/``file`` for built-ins and
        ``name``/``version``/``class_name``/``kind`` for user algorithms.
    """
    results: List[Dict[str, Any]] = []

    if builtin:
        import tuiml.algorithms as _alg_pkg
        alg_root = Path(_alg_pkg.__file__).parent
        for py_file in sorted(alg_root.rglob("*.py")):
            if py_file.name.startswith("_"):
                continue
            rel = py_file.relative_to(alg_root)
            results.append({
                "type": "builtin",
                "path": str(py_file),
                "relative_path": str(rel),
                "category": rel.parts[0] if len(rel.parts) > 1 else "root",
                "file": py_file.name,
            })

    if user and USER_ALGS_DIR.exists():
        for name_dir in sorted(USER_ALGS_DIR.iterdir()):
            if not name_dir.is_dir():
                continue
            for ver_dir in sorted(name_dir.iterdir()):
                alg_file = ver_dir / "algorithm.py"
                if alg_file.exists():
                    meta = _read_metadata(ver_dir)
                    results.append({
                        "type": "user",
                        "name": name_dir.name,
                        "version": ver_dir.name,
                        "class_name": meta.get("class_name", name_dir.name),
                        "kind": meta.get("kind"),
                        "path": str(alg_file),
                    })

    return {"status": "success", "count": len(results), "files": results}


def search_source(query: str, name: Optional[str] = None,
                  builtin: bool = True, user: bool = True) -> Dict[str, Any]:
    """Grep for ``query`` across algorithm source files.

    Parameters
    ----------
    query : str
        Regular expression to search for (matched per line).
    name : str, optional
        Scope the search to one user algorithm by name.
    builtin : bool, default=True
        Include built-in algorithm files in the search.
    user : bool, default=True
        Include user algorithm files in the search.

    Returns
    -------
    result : Dict[str, Any]
        On success: keys ``status``, ``query``, ``match_count``, and
        ``matches`` (list of dicts with ``path``, ``line_number``, ``line``).
        On invalid regex: keys ``status``, ``error_type``, ``error``.
    """
    import re as _re
    try:
        pattern = _re.compile(query)
    except _re.error as e:
        return {"status": "error", "error_type": "ValueError",
                "error": f"invalid regex: {e}"}

    files_info = list_algorithm_files(builtin=builtin, user=user)
    search_files: List[Path] = []

    for f in files_info["files"]:
        p = Path(f["path"])
        if name and f.get("type") == "user" and f.get("name") != name:
            continue
        search_files.append(p)

    matches: List[Dict[str, Any]] = []
    for py_file in search_files:
        try:
            lines = py_file.read_text(encoding="utf-8").splitlines()
        except Exception:
            continue
        for i, line in enumerate(lines):
            if pattern.search(line):
                matches.append({
                    "path": str(py_file),
                    "line_number": i + 1,
                    "line": line,
                })

    return {
        "status": "success",
        "query": query,
        "match_count": len(matches),
        "matches": matches,
    }


def edit_algorithm(name: str, old_string: str, new_string: str,
                   version: Optional[str] = None,
                   bump_version: bool = False) -> Dict[str, Any]:
    """Apply a str_replace edit to a user algorithm source.

    Replaces exactly one occurrence of ``old_string`` with ``new_string``.
    Fails if ``old_string`` is not found, or if it appears more than once
    (ambiguous, make the search string more specific in that case).

    After the edit the source is AST-validated and re-registered. Optionally
    bumps the patch version and saves as a new file.

    Parameters
    ----------
    name : str
        User algorithm name (or versioned alias).
    old_string : str
        Exact text to replace; must occur exactly once in the source.
    new_string : str
        Replacement text.
    version : str, optional
        Pin the semver version to edit. When None, the newest on-disk
        version is used.
    bump_version : bool, default=False
        Save the edited source as a new patch version instead of
        overwriting in place.

    Returns
    -------
    result : Dict[str, Any]
        On success: keys ``status``, ``name``, ``version``,
        ``previous_version``, ``class_name``, ``path``, ``source_hash``,
        ``registered_as`` (list of aliases), and ``note``. On failure:
        keys ``status``, ``error_type``, ``error``.
    """
    err = _validate_name(name)
    if err:
        return {"status": "error", "error_type": "ValueError", "error": err}

    resolved = _resolve_user_algorithm(name) if version is None else None
    if resolved is None and version is not None:
        ver_err = _validate_version(version)
        if ver_err:
            return {"status": "error", "error_type": "ValueError", "error": ver_err}
        resolved = {"name": name, "version": version, "dir": USER_ALGS_DIR / name / version}
    if resolved is None:
        return {"status": "error", "error_type": "NotFound",
                "error": f"no user algorithm named {name!r} found on disk"}

    alg_file = resolved["dir"] / "algorithm.py"
    if not alg_file.exists():
        return {"status": "error", "error_type": "NotFound",
                "error": f"algorithm.py not found at {alg_file}"}

    source = alg_file.read_text(encoding="utf-8")

    # Uniqueness check, the heart of the str_replace approach.
    count = source.count(old_string)
    if count == 0:
        return {
            "status": "error", "error_type": "NotFound",
            "error": "old_string not found in the source. Read the file first to verify the exact text.",
        }
    if count > 1:
        return {
            "status": "error", "error_type": "Ambiguous",
            "error": f"old_string appears {count} times, make it more specific by including more surrounding context.",
        }

    new_source = source.replace(old_string, new_string, 1)

    # AST validate the edited source.
    ok, reason = _ast_validate(new_source)
    if not ok:
        return {"status": "error", "error_type": "UnsafeSource", "error": reason}

    current_version = resolved["version"]
    if bump_version:
        parts = current_version.split(".")
        new_ver = f"{parts[0]}.{parts[1]}.{int(parts[2]) + 1}"
        target_dir = USER_ALGS_DIR / name / new_ver
        target_dir.mkdir(parents=True, exist_ok=True)
        target_file = target_dir / "algorithm.py"
        save_version = new_ver
    else:
        target_file = alg_file
        save_version = current_version

    target_file.write_text(new_source, encoding="utf-8")

    try:
        cls, detected_kind = _import_and_register(target_file, name, save_version)
    except Exception as e:
        return {"status": "error", "error_type": "LoadError", "error": str(e)}

    meta = _read_metadata(resolved["dir"])
    new_meta = {
        **meta,
        "version": save_version,
        "source_hash": _source_hash(new_source),
    }
    _write_metadata(target_file.parent, new_meta)

    return {
        "status": "success",
        "name": name,
        "version": save_version,
        "previous_version": current_version,
        "class_name": cls.__name__,
        "path": str(target_file),
        "source_hash": new_meta["source_hash"],
        "registered_as": [cls.__name__, _versioned_alias_name(cls.__name__, save_version)],
        "note": "Algorithm re-registered. Run tuiml_benchmark to validate the change.",
    }
