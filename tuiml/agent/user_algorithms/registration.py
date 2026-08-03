"""Import user algorithm files and register them with the TuiML registry."""

from __future__ import annotations

import importlib.util
import inspect
import sys
from typing import Any, Dict, List, Tuple
from pathlib import Path

from ._paths import USER_ALGS_DIR

def _versioned_alias_name(name: str, version: str) -> str:
    """Return a valid Python identifier encoding the version, e.g. MyGBM_v1_0_0.

    Parameters
    ----------
    name : str
        Bare class name.
    version : str
        Semver version string (dots become underscores).

    Returns
    -------
    alias : str
        Identifier of the form ``<name>_v<major>_<minor>_<patch>``.
    """
    return f"{name}_v{version.replace('.', '_')}"


def _import_and_register(file_path: Path, name: str, version: str) -> Tuple[Any, str]:
    """Import ``algorithm.py`` and register both a versioned alias and a latest alias.

    Parameters
    ----------
    file_path : Path
        Path to the ``algorithm.py`` file to import.
    name : str
        User-facing algorithm name (directory name under ``USER_ALGS_DIR``).
    version : str
        Semver version of the file being loaded.

    Returns
    -------
    class_obj : type
        The imported ``Classifier``/``Regressor`` subclass.
    kind : str
        Either ``'classifier'`` or ``'regressor'``.

    Raises
    ------
    RuntimeError
        If the import spec cannot be built, the module fails to execute, or
        no ``Classifier``/``Regressor`` subclass is defined in the module.
    """
    from tuiml.registry import registry

    module_name = f"_tuiml_user_{name}_v{version.replace('.', '_')}"
    # Remove any previously-imported copy so decorators re-fire.
    sys.modules.pop(module_name, None)

    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not build import spec for {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    # Re-registering a user algorithm (new version, restart, or edit) is
    # intentional, so suppress the registry's "already registered" warning for
    # the whole load, both the user's @classifier/@regressor decorator firing
    # during exec_module and the versioned-alias registration below.
    with registry.suppress_overwrite_warnings():
        try:
            spec.loader.exec_module(module)
        except Exception as e:
            sys.modules.pop(module_name, None)
            raise RuntimeError(f"error while importing user algorithm: {e}") from e

    # Find the decorated class in the imported module.
    from tuiml.base.algorithms import Classifier, Regressor, classifier, regressor
    target_cls = None
    kind = None
    for obj in vars(module).values():
        if not inspect.isclass(obj) or obj.__module__ != module_name:
            continue
        if issubclass(obj, Classifier) and obj is not Classifier:
            target_cls, kind = obj, "classifier"
            break
        if issubclass(obj, Regressor) and obj is not Regressor:
            target_cls, kind = obj, "regressor"
            break
    if target_cls is None:
        raise RuntimeError("imported module defines no Classifier/Regressor subclass")

    # Register a versioned alias so A/B comparisons between versions work.
    alias_name = _versioned_alias_name(target_cls.__name__, version)
    if alias_name != target_cls.__name__:
        alias_cls = type(alias_name, (target_cls,), {
            "__doc__": target_cls.__doc__,
            "__module__": module_name,
        })
        decorator = classifier if kind == "classifier" else regressor
        # Re-apply the decorator to register the alias. We pass empty tags so
        # the decorator doesn't fail looking for metadata. Suppress the
        # overwrite warning: re-registering the alias on reload is intentional.
        with registry.suppress_overwrite_warnings():
            decorator(tags=["custom", f"version={version}"], version=version)(alias_cls)

    return target_cls, kind


def load_all() -> Dict[str, Any]:
    """Scan ``USER_ALGS_DIR`` and register every algorithm found.

    Called once at MCP server startup. Failures on individual files are logged
    but do not abort the whole load. Only the latest version of each
    algorithm is loaded.

    Returns
    -------
    result : Dict[str, Any]
        Keys ``status``, ``loaded`` (count), ``algorithms`` (list of dicts
        with ``name``, ``kind``, ``version``), and ``errors`` (list of dicts
        with ``path``, ``error`` for files that failed to load).
    """
    if not USER_ALGS_DIR.exists():
        return {"status": "success", "loaded": 0, "errors": []}

    loaded: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    for name_dir in sorted(USER_ALGS_DIR.iterdir()):
        if not name_dir.is_dir():
            continue
        ver_dirs = sorted(
            (d for d in name_dir.iterdir() if d.is_dir() and (d / "algorithm.py").exists()),
            key=lambda d: d.name,
        )
        if not ver_dirs:
            continue
        ver_dir = ver_dirs[-1]  # only load the latest version
        alg_file = ver_dir / "algorithm.py"
        try:
            cls, kind = _import_and_register(alg_file, name_dir.name, ver_dir.name)
            loaded.append({"name": cls.__name__, "kind": kind, "version": ver_dir.name})
        except Exception as e:  # keep going on partial failures
            errors.append({"path": str(alg_file), "error": str(e)})

    return {"status": "success", "loaded": len(loaded),
            "algorithms": loaded, "errors": errors}


_LOAD_RESULT: Dict[str, Any] | None = None


def ensure_loaded(verbose: bool = False) -> Dict[str, Any]:
    """Run :func:`load_all` once per process, returning the cached result after.

    Registering user algorithms means executing Python from
    ``USER_ALGS_DIR``, so it must happen before anything reads the registry
    but never merely because a module was imported. Callers that are about to
    touch the registry — the MCP server, a CLI subcommand, ``execute_tool`` —
    call this; ``tuiml --version`` and ``tuiml --help`` never do, and so never
    run user code.

    Parameters
    ----------
    verbose : bool, default=False
        Report the load on stderr. The MCP server passes True, where the
        counts are startup logging; the CLI leaves it False so ordinary
        commands stay quiet.

    Returns
    -------
    result : Dict[str, Any]
        Whatever :func:`load_all` returned on the first call.
    """
    global _LOAD_RESULT
    if _LOAD_RESULT is not None:
        return _LOAD_RESULT

    try:
        _LOAD_RESULT = load_all()
    except Exception as e:  # never block the caller on bootstrap failures
        _LOAD_RESULT = {"status": "error", "loaded": 0, "algorithms": [],
                        "errors": [{"path": str(USER_ALGS_DIR), "error": str(e)}]}
        print(f"[tuiml] user-algorithm bootstrap failed: {e}", file=sys.stderr)
        return _LOAD_RESULT

    if verbose and _LOAD_RESULT.get("loaded"):
        print(f"[tuiml] loaded {_LOAD_RESULT['loaded']} user algorithm(s)",
              file=sys.stderr)
    # Errors surface regardless of verbosity: a user algorithm that failed to
    # register is silently missing from the registry otherwise.
    for err in _LOAD_RESULT.get("errors", []):
        print(f"[tuiml] user algorithm load error: {err}", file=sys.stderr)

    return _LOAD_RESULT
