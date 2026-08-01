"""Conservative AST denylist applied to every submitted source file."""

from __future__ import annotations

import ast
from typing import Tuple

_FORBIDDEN_MODULES = {
    "subprocess", "socket", "os", "shutil", "urllib", "requests", "httpx",
    "http", "ftplib", "smtplib", "paramiko", "telnetlib", "ctypes", "webbrowser",
    "pty", "asyncio.subprocess",
}


_FORBIDDEN_CALLS = {"eval", "exec", "compile", "__import__", "open", "input"}


_FORBIDDEN_ATTRS = {"system", "popen", "spawn", "spawnl", "spawnv"}


def _ast_validate(source: str) -> Tuple[bool, str]:
    """Walk the AST and reject source that uses denylisted modules / calls.

    Parameters
    ----------
    source : str
        Python source code to validate.

    Returns
    -------
    ok : bool
        True if the source passed all checks.
    reason : str
        Rejection reason; empty string on success.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return False, f"syntax error: {e}"

    # Require at least one decorated Classifier / Regressor class
    found_decorator = False

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in _FORBIDDEN_MODULES:
                    return False, f"forbidden import: {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                root = node.module.split(".")[0]
                if root in _FORBIDDEN_MODULES:
                    return False, f"forbidden import from: {node.module}"
        elif isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Name) and f.id in _FORBIDDEN_CALLS:
                return False, f"forbidden call: {f.id}()"
            if isinstance(f, ast.Attribute) and f.attr in _FORBIDDEN_ATTRS:
                return False, f"forbidden attribute call: .{f.attr}()"
        elif isinstance(node, ast.ClassDef):
            for deco in node.decorator_list:
                name = None
                if isinstance(deco, ast.Call) and isinstance(deco.func, ast.Name):
                    name = deco.func.id
                elif isinstance(deco, ast.Name):
                    name = deco.id
                if name in {"classifier", "regressor", "clusterer", "associator"}:
                    found_decorator = True

    if not found_decorator:
        return False, (
            "no @classifier / @regressor decorated class found, did you "
            "forget the decorator? Call tuiml_algorithm_skeleton for a template."
        )

    return True, ""
