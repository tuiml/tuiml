"""Static AST checks applied to every user-authored algorithm source file.

Agent-authored algorithms are executed in-process, so this module is the only
thing standing between a generated source file and the interpreter. Two rules
follow from that.

**Imports are an allowlist, not a denylist.** A denylist has to enumerate every
route to the filesystem and the network, and it always misses one:
``importlib`` reaches anything ``__import__`` does, ``pathlib`` writes files
without ``open``, and blocking attributes by literal name is defeated by
``getattr``. An allowlist inverts the burden -- an import is refused unless it
is known to be needed for writing an estimator.

**Validation runs on every execution, not once at creation.** Checking only at
create time protects nothing if a file can reach the algorithms directory by
another route: whatever lands there is executed unvalidated at the next server
start. :func:`~tuiml.agent.user_algorithms.registration.load_all` therefore
re-runs these checks before importing anything.

This is a static check on source, not a sandbox. It raises the cost of getting
code to run; it does not make execution safe. Running untrusted algorithms in a
subprocess with resource limits is the real fix, and this is the interim.
"""

from __future__ import annotations

import ast
from typing import Tuple

# Modules a numerical estimator legitimately needs. Anything absent is refused
# with a message naming this list, so the failure is actionable rather than
# mysterious.
_ALLOWED_MODULES = {
    # The library being extended, plus the numerics an estimator is written in.
    "tuiml", "numpy", "scipy", "pandas",
    # Modelling libraries TuiML already depends on or ships an extra for.
    # A user algorithm that wraps or ensembles one of these is a normal thing
    # to write, and refusing them breaks existing algorithms.
    "sklearn", "xgboost", "lightgbm", "catboost",
    # Standard-library modules that carry no filesystem, network or process
    # reach: pure data structures, maths and typing support.
    "math", "cmath", "statistics", "random", "decimal", "fractions",
    "typing", "typing_extensions", "dataclasses", "enum", "abc",
    "collections", "itertools", "functools", "operator", "heapq", "bisect",
    "copy", "numbers", "warnings", "textwrap", "re", "json", "time",
    "__future__",
}


# Names that hand back arbitrary code execution, filesystem access, or the
# builtins table, regardless of what was imported.
_FORBIDDEN_CALLS = {
    "eval", "exec", "compile", "__import__", "open", "input", "breakpoint",
    "globals", "locals", "vars",
}


# Attribute access by name. Banning these outright would break
# ``getattr(model, "classes_", None)``, which is ordinary duck-typing and
# common in ensemble code, so they are permitted only with a literal,
# non-dunder attribute name -- see :func:`_check_dynamic_attr`. A computed name
# is refused because it is exactly how an attribute denylist gets bypassed.
_ATTR_BUILTINS = {"getattr", "setattr", "delattr", "hasattr"}


# Dunder attributes reachable from any object that walk back up to the
# interpreter internals -- ``().__class__.__bases__[0].__subclasses__()`` and
# relatives. Blocking the names costs nothing legitimate: an estimator has no
# reason to touch them.
_FORBIDDEN_ATTRS = {
    "__subclasses__", "__bases__", "__mro__", "__globals__", "__code__",
    "__closure__", "__builtins__", "__loader__", "__spec__", "__reduce__",
    "__reduce_ex__", "__getattribute__", "__dict__",
    "system", "popen", "spawn", "spawnl", "spawnv",
}


def _check_dynamic_attr(node: ast.Call) -> str:
    """Check a ``getattr``-family call for a literal, non-dunder attribute name.

    Parameters
    ----------
    node : ast.Call
        A call whose function is one of :data:`_ATTR_BUILTINS`.

    Returns
    -------
    reason : str
        Rejection reason, or an empty string if the call is acceptable.
    """
    fname = node.func.id
    if len(node.args) < 2:
        return f"{fname}() needs a literal attribute name"

    attr = node.args[1]
    if not (isinstance(attr, ast.Constant) and isinstance(attr.value, str)):
        # A computed name defeats any static check on attribute names.
        return f"{fname}() with a computed attribute name is not permitted"
    if attr.value in _FORBIDDEN_ATTRS or attr.value.startswith("__"):
        return f"{fname}() of '{attr.value}' is not permitted"
    return ""


def _ast_validate(source: str) -> Tuple[bool, str]:
    """Walk the AST and reject source outside the permitted subset of Python.

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

    allowed = ", ".join(sorted(_ALLOWED_MODULES))

    # Require at least one decorated Classifier / Regressor class
    found_decorator = False

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root not in _ALLOWED_MODULES:
                    return False, (
                        f"import of '{alias.name}' is not permitted in a user "
                        f"algorithm. Allowed: {allowed}"
                    )
        elif isinstance(node, ast.ImportFrom):
            # A relative import has no module name to check and would resolve
            # against the algorithm's own directory, so refuse it outright.
            if node.level:
                return False, "relative imports are not permitted in a user algorithm"
            root = (node.module or "").split(".")[0]
            if root not in _ALLOWED_MODULES:
                return False, (
                    f"import from '{node.module}' is not permitted in a user "
                    f"algorithm. Allowed: {allowed}"
                )
        elif isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Name):
                if f.id in _FORBIDDEN_CALLS:
                    return False, f"forbidden call: {f.id}()"
                if f.id in _ATTR_BUILTINS:
                    reason = _check_dynamic_attr(node)
                    if reason:
                        return False, reason
            if isinstance(f, ast.Attribute) and f.attr in _FORBIDDEN_ATTRS:
                return False, f"forbidden attribute call: .{f.attr}()"
        elif isinstance(node, ast.Attribute):
            # Checked outside ast.Call as well: the dangerous dunders are read
            # as values (``obj.__class__.__bases__``), not only invoked.
            if node.attr in _FORBIDDEN_ATTRS:
                return False, f"forbidden attribute access: .{node.attr}"
        elif isinstance(node, ast.Name):
            # Catches the bare name used as a value -- ``f = eval`` then
            # ``f(...)``, which never appears as a Call on ``eval`` itself.
            # _ATTR_BUILTINS are deliberately not here: they are checked at
            # their call sites, where the attribute argument is visible.
            if node.id in _FORBIDDEN_CALLS:
                return False, f"forbidden name: {node.id}"
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
