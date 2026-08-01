"""Process-wide random seed management for reproducibility.

Provides a small global registry for a single "current" seed and applies it
to Python's :mod:`random` module and :mod:`numpy.random` in one call. Code
that needs a source of randomness (train/test splitting, weight
initialization, bootstrap resampling, ...) can call :func:`get_global_seed`
to discover whatever the caller last set with :func:`set_global_seed`,
instead of every call site inventing its own seeding convention.

Notes
-----
The seed is stored in a single module-level variable, so it is shared
process-wide and is **not** thread-safe: concurrent calls to
:func:`set_global_seed` from different threads can race.
"""

import random
import numpy as np

_GLOBAL_SEED = None


def set_global_seed(seed: int | None) -> None:
    """Set the process-wide random seed for ``random`` and ``numpy.random``.

    Parameters
    ----------
    seed : int or None
        The integer seed to apply to Python's :mod:`random` module and
        :mod:`numpy.random`. If ``None``, the recorded global seed is
        cleared without reseeding either generator.

    Examples
    --------
    >>> from tuiml.utils.seed import set_global_seed, get_global_seed
    >>> set_global_seed(42)
    >>> get_global_seed()
    42
    """
    global _GLOBAL_SEED
    _GLOBAL_SEED = seed
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)


def get_global_seed() -> int | None:
    """Return the seed most recently set with :func:`set_global_seed`.

    Returns
    -------
    seed : int or None
        The seed currently in effect, or ``None`` if it was never set (or
        was last cleared by passing ``None`` to :func:`set_global_seed`).

    Examples
    --------
    >>> from tuiml.utils.seed import set_global_seed, get_global_seed
    >>> set_global_seed(7)
    >>> get_global_seed()
    7
    >>> set_global_seed(None)
    >>> get_global_seed() is None
    True
    """
    return _GLOBAL_SEED
