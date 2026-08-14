"""Shared machinery for the pretrained tabular foundation models.

The rest of :mod:`tuiml.algorithms.tabular_foundation` — FT-Transformer, SAINT,
NODE — is native TuiML code that trains from scratch on your data. The models
built on this module are a different animal: they are **pretrained
checkpoints**, downloaded from a model hub and run in a single forward pass
with no gradient step at all. Nothing is fitted; ``fit`` only memorises the
training set to hand back as in-context examples at predict time.

That difference is why they register under a **namespaced hub key**,
``foundation.<ClassName>``, exactly as :mod:`tuiml.sklearn` registers
``sklearn.SVC``. The namespace is an honest signal that TuiML is delegating to
someone else's artifact rather than running its own implementation, and it
keeps the names from colliding with native algorithms.

Licensing
---------
TuiML is BSD-3-Clause and **never ships or mirrors model weights**. The
upstream package fetches its own checkpoint, so the download is a direct
transaction between the user and whoever publishes it, under that publisher's
license.

This matters because a permissive wrapper does not relicense what it wraps: a
checkpoint restricted to non-commercial use stays restricted no matter how
TuiML is licensed. Only models whose **weights** carry a license compatible
with TuiML's own are integrated here. Today that means TabICL alone, which is
BSD-3-Clause for code *and* weights. Adding a model with restricted weights
would require a consent gate, which deliberately does not exist yet.

Notes
-----
Importing this module never imports torch or any upstream package, and
constructing a wrapper never does either — the dependency is checked in
``fit``, so the algorithm catalog is identical on every install.
"""

from __future__ import annotations

from typing import Any, List, Optional

from tuiml.base.algorithms import classifier, regressor

#: Hub-registry namespace prefix. The hub key for a wrapper named ``Foo`` is
#: ``"foundation.Foo"`` while the Python class stays ``Foo``.
NAMESPACE = "foundation"

#: pip extra providing the backing packages: ``pip install tuiml[foundation]``.
_EXTRA = "foundation"


def _namespaced(base_decorator):
    """Wrap a TuiML registration decorator to register under the namespace.

    Parameters
    ----------
    base_decorator : callable
        One of the TuiML registration decorators (``classifier``,
        ``regressor``).

    Returns
    -------
    factory : callable
        A decorator factory mirroring the base decorator's keyword arguments.
    """

    def factory(tags: Optional[List[str]] = None, version: str = "1.0.0"):
        def decorate(cls):
            key = f"{NAMESPACE}.{cls.__name__}"
            merged_tags = list(tags or [])
            if NAMESPACE not in merged_tags:
                merged_tags.append(NAMESPACE)
            return base_decorator(name=key, tags=merged_tags, version=version)(cls)

        return decorate

    return factory


fd_classifier = _namespaced(classifier)
fd_regressor = _namespaced(regressor)


def require_package(module_name: str, cls_name: str) -> Any:
    """Import an upstream foundation-model package, or explain how to get it.

    Called from ``fit``, never from ``__init__``: constructing a wrapper only
    records hyperparameters, and that must keep working on an install without
    the extra so parameter grids, pickling and the hub catalog behave
    identically everywhere.

    Parameters
    ----------
    module_name : str
        Importable name of the upstream package, e.g. ``"tabicl"``.
    cls_name : str
        Name of the calling wrapper class, used in the error message.

    Returns
    -------
    module : module
        The imported upstream module.

    Raises
    ------
    ImportError
        If the package is missing, naming the exact install command.
    """
    import importlib

    # The upstream packages import torch themselves, which on macOS lands in a
    # process that already holds the boosting libraries' OpenMP runtime and
    # segfaults on the first parallel region. Clamp torch *before* the upstream
    # import triggers any tensor work. See ``guard_duplicate_openmp``.
    from tuiml.utils.torch_backend import guard_duplicate_openmp, has_torch

    if has_torch():
        import torch

        guard_duplicate_openmp(torch)

    try:
        return importlib.import_module(module_name)
    except ImportError as exc:  # pragma: no cover - only without the extra
        raise ImportError(
            f"{cls_name} is a pretrained foundation model backed by the "
            f"'{module_name}' package, which is not installed. Install it "
            f"with:  pip install 'tuiml[{_EXTRA}]'  (this also pulls in "
            f"PyTorch). The model weights are downloaded separately on first "
            f"use, by '{module_name}' itself — TuiML does not ship them."
        ) from exc


__all__ = ["NAMESPACE", "fd_classifier", "fd_regressor", "require_package"]
