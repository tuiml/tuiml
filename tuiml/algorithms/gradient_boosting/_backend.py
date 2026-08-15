"""Lazy access to the optional gradient-boosting backends.

XGBoost, LightGBM and CatBoost were once hard requirements of TuiML, imported
eagerly by :mod:`tuiml.algorithms`. They are now **optional**, installed with
``pip install 'tuiml[boosting]'``. Three things drove the change:

1. **They are third-party wrappers, not native code.** The project's dependency
   policy confines wrappers to optional backends -- ``tuiml.sklearn``,
   ``tuiml.capymoa``, ``tuiml.weka`` -- precisely so that a core install stays
   small and dependency-light. Three boosting libraries sitting in a core
   namespace as required dependencies contradicted that.
2. **Install weight.** The three together dominate a base TuiML install, for
   algorithms many users never call.
3. **A real crash.** Each library loads its own bundled OpenMP runtime. On
   macOS, importing any of them before PyTorch leaves two OpenMP runtimes in
   one process, and the first torch parallel region segfaults the interpreter
   with no traceback. Because :mod:`tuiml.algorithms` imported all three
   unconditionally, *every* session was in that state, and the neural models
   had to run single-threaded to survive it. Importing lazily means a user who
   never touches boosting never enters the conflicting state at all.

The contract matches the one in :mod:`tuiml.utils.torch_backend`:

- **Import time** -- importing TuiML never imports a boosting library, so the
  classes are still defined, exported and registered. ``list_algorithms()``
  returns the same catalog on every install.
- **Construction time** -- ``XGBoostClassifier()`` succeeds without XGBoost.
  It only records hyperparameters, and refusing would break parameter grids,
  pickling and the generic algorithm contract, none of which need the library.
- **Fit time** -- :func:`require_backend` raises an ``ImportError`` naming the
  class and the exact install command.

Examples
--------
>>> from tuiml.algorithms.gradient_boosting._backend import BACKEND_EXTRA
>>> BACKEND_EXTRA
'boosting'
"""

from __future__ import annotations

from typing import Any

#: pip extra providing all three backends: ``pip install tuiml[boosting]``.
BACKEND_EXTRA = "boosting"

#: Backing distribution name for each importable module, where they differ.
#: All three happen to match today, but naming them keeps the error message
#: correct if that ever stops being true.
_PIP_NAME = {
    "xgboost": "xgboost",
    "lightgbm": "lightgbm",
    "catboost": "catboost",
}


def has_backend(module_name: str) -> bool:
    """Report whether a boosting backend is importable.

    Intended for tests and for callers that want to branch on availability.
    Algorithm code should call :func:`require_backend`, so the user gets an
    actionable message rather than a silent fallback.

    Parameters
    ----------
    module_name : str
        One of ``"xgboost"``, ``"lightgbm"``, ``"catboost"``.

    Returns
    -------
    available : bool
        ``True`` if the module imports.

    Examples
    --------
    >>> from tuiml.algorithms.gradient_boosting._backend import has_backend
    >>> isinstance(has_backend("xgboost"), bool)
    True
    """
    import importlib

    try:
        importlib.import_module(module_name)
    except ImportError:
        return False
    return True


#: Set once, so repeated ``fit`` calls do not re-check.
_openmp_guard_checked = False


def _guard_duplicate_openmp() -> bool:
    """Cap OpenMP threads when torch is already loaded, before importing here.

    The duplicate-``libomp`` conflict on macOS is **symmetric**: whichever of
    torch and a boosting library initialises its OpenMP runtime second can
    crash the interpreter. The two directions do not have the same fix, which
    is why there are two guards:

    - *boosting first, then torch* -- handled by
      :func:`tuiml.utils.torch_backend.guard_duplicate_openmp`, which calls
      ``torch.set_num_threads(1)`` after importing torch.
    - *torch first, then boosting* -- this function. Clamping torch at runtime
      does **not** help here, and neither does ``KMP_DUPLICATE_LIB_OK``;
      measured, both still segfault. Only ``OMP_NUM_THREADS=1`` present in the
      environment when the *second* runtime initialises avoids it, so it has to
      be set before the import rather than after.

    torch has already read the variable by the time we set it, so its own
    thread count is unaffected -- the cost falls on the boosting library, and
    only in a process that genuinely uses both.

    Returns
    -------
    applied : bool
        Whether ``OMP_NUM_THREADS`` was set by this call.
    """
    global _openmp_guard_checked
    if _openmp_guard_checked:
        return False
    _openmp_guard_checked = True

    import os
    import sys

    if sys.platform != "darwin":
        return False
    if "torch" not in sys.modules:
        return False
    if "OMP_NUM_THREADS" in os.environ:
        return False
    os.environ["OMP_NUM_THREADS"] = "1"
    return True


def require_backend(module_name: str, cls_name: str) -> Any:
    """Import a boosting backend, or explain exactly how to install it.

    Call this at the top of ``fit``. It is deliberately *not* called from
    ``__init__``: constructing a wrapper records hyperparameters and must keep
    working on an install without the extra, so that parameter grids, pickling
    and the algorithm catalog behave identically everywhere.

    Parameters
    ----------
    module_name : str
        One of ``"xgboost"``, ``"lightgbm"``, ``"catboost"``.
    cls_name : str
        Name of the calling wrapper class, used in the error message.

    Returns
    -------
    module : module
        The imported backend module.

    Raises
    ------
    ImportError
        If the backend is missing, naming both the single-package install and
        the TuiML extra that pulls in all three.
    """
    import importlib

    _guard_duplicate_openmp()

    try:
        return importlib.import_module(module_name)
    except ImportError as exc:  # pragma: no cover - only without the extra
        pip_name = _PIP_NAME.get(module_name, module_name)
        raise ImportError(
            f"{cls_name} wraps {module_name}, which is not installed. "
            f"Install it with:  pip install 'tuiml[{BACKEND_EXTRA}]'  "
            f"(adds XGBoost, LightGBM and CatBoost), or just "
            f"pip install {pip_name}. TuiML's native algorithms -- including "
            f"GradientBoostingRegressor, RandomForestClassifier and "
            f"NGBoostRegressor -- need no extra install."
        ) from exc


__all__ = ["BACKEND_EXTRA", "has_backend", "require_backend"]
