"""Lazy access to the optional PyTorch backend.

A handful of TuiML algorithms — the attention-based tabular models and the deep
forecasters — need a tensor library with autograd. PyTorch is therefore an
**optional** dependency, installed with ``pip install 'tuiml[torch]'``.

"Optional" is enforced at three levels, and all three matter:

1. **Import time.** Importing :mod:`tuiml`, or any package containing a
   torch-backed model, never imports torch. The classes are defined, exported
   and registered in the algorithm hub whether or not torch is present, so
   ``list_algorithms()`` and the docs show the same catalog on every install.
2. **Construction time.** ``FTTransformerClassifier()`` succeeds without torch.
   Constructing a model only records hyperparameters, and refusing to do that
   would break parameter grids, serialization round-trips and the generic
   algorithm contract, none of which need a tensor.
3. **Fit time.** :func:`require_torch` is called from ``fit``, which is the
   first operation that genuinely cannot proceed. It raises an ``ImportError``
   naming the class and the exact install command.

The result is that a user without torch sees the model in the catalog, can
inspect its parameters, and gets one clear, actionable message the moment they
try to train it — rather than a bare ``ModuleNotFoundError`` from deep inside a
layer definition.

Examples
--------
>>> from tuiml.utils.torch_backend import has_torch, require_torch
>>> isinstance(has_torch(), bool)
True
"""

from __future__ import annotations

from typing import Any, Tuple

#: pip extra that provides the backing dependency: ``pip install tuiml[torch]``.
_EXTRA = "torch"

#: Lowest version the deep models are written against. 2.2 is the first release
#: where ``torch.nn.functional.scaled_dot_product_attention`` is stable on CPU,
#: which the transformer blocks rely on.
_MIN_VERSION = "2.2.0"


def has_torch() -> bool:
    """Report whether PyTorch is importable.

    Intended for tests and for callers that want to branch on availability.
    Algorithm code should call :func:`require_torch` instead, so the user gets
    an actionable message rather than a silent fallback.

    Returns
    -------
    available : bool
        ``True`` if ``import torch`` succeeds.

    Examples
    --------
    >>> from tuiml.utils.torch_backend import has_torch
    >>> isinstance(has_torch(), bool)
    True
    """
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


def require_torch(cls_name: str) -> Tuple[Any, Any]:
    """Import torch, or raise an error naming the class and the install command.

    Call this at the top of ``fit``. It is deliberately *not* called from
    ``__init__``: constructing a model records hyperparameters and must keep
    working on an install without torch, so that parameter grids, pickling and
    the generic algorithm contract behave identically everywhere.

    Parameters
    ----------
    cls_name : str
        Name of the calling class, used in the error message so the user knows
        which model pulled the dependency in.

    Returns
    -------
    torch : module
        The imported ``torch`` module.
    nn : module
        ``torch.nn``, returned alongside because every caller needs both.

    Raises
    ------
    ImportError
        If torch is not installed, with the exact install command.

    Examples
    --------
    >>> from tuiml.utils.torch_backend import has_torch, require_torch
    >>> if has_torch():
    ...     torch, nn = require_torch("MyModel")
    ...     print(hasattr(nn, "Linear"))
    ... else:
    ...     print(True)
    True
    """
    try:
        import torch
        from torch import nn
    except ImportError as exc:  # pragma: no cover - exercised only without torch
        raise ImportError(
            f"{cls_name} is a neural model and requires PyTorch, which is not "
            f"installed. Install it with:  pip install 'tuiml[{_EXTRA}]'  "
            f"(needs torch >= {_MIN_VERSION}). Every other TuiML algorithm is "
            f"pure NumPy and needs no extra install."
        ) from exc
    return torch, nn


def resolve_device(device: str, torch_module: Any) -> Any:
    """Turn a device string into a ``torch.device``, falling back to CPU.

    ``"auto"`` prefers CUDA, then Apple Silicon's MPS, then CPU. An explicit
    request for an unavailable accelerator falls back to CPU rather than
    raising: the model still trains, only slower, and a hard failure here would
    make a saved parameter grid non-portable between machines.

    Parameters
    ----------
    device : {"auto", "cpu", "cuda", "mps"}
        Requested device.
    torch_module : module
        The ``torch`` module, passed in so this function never imports it.

    Returns
    -------
    device : torch.device
        The device to place tensors on.
    """
    if device == "auto":
        if torch_module.cuda.is_available():
            return torch_module.device("cuda")
        if getattr(torch_module.backends, "mps", None) is not None \
                and torch_module.backends.mps.is_available():
            return torch_module.device("mps")
        return torch_module.device("cpu")

    if device == "cuda" and not torch_module.cuda.is_available():
        return torch_module.device("cpu")
    if device == "mps":
        mps = getattr(torch_module.backends, "mps", None)
        if mps is None or not mps.is_available():
            return torch_module.device("cpu")
    return torch_module.device(device)


__all__ = ["has_torch", "require_torch", "resolve_device"]
