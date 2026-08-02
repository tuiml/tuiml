"""Contract checks for cross-validation splitters.

A splitter that leaks a row from train into test inflates every score built on
it, silently and in the direction that looks like success. These checks are
cheap and catch exactly that.
"""

from __future__ import annotations

from typing import Any, Callable, Tuple

import numpy as np

from ._data import make_split_data


def _split(splitter, X, y, groups):
    """Call ``split`` with groups when the splitter accepts them.

    Group-aware splitters raise without ``groups``; the rest do not accept
    the argument at all, so it cannot simply always be passed.

    Parameters
    ----------
    splitter : object
        The splitter to drive.
    X, y, groups : np.ndarray
        Fixture data from :func:`make_split_data`.

    Returns
    -------
    folds : list of tuple
        ``(train_idx, test_idx)`` pairs.
    """
    import inspect

    params = inspect.signature(splitter.split).parameters
    if "groups" in params:
        return list(splitter.split(X, y, groups))
    return list(splitter.split(X, y))


def check_train_and_test_are_disjoint(name: str, splitter) -> None:
    """No index appears in both sides of the same fold.

    This is the leak that makes a model look better than it is.

    Parameters
    ----------
    name : str
        Splitter name, used in failure messages.
    splitter : object
        A constructed splitter.

    Returns
    -------
    None
    """
    X, y, groups = make_split_data()
    for i, (train, test) in enumerate(_split(splitter, X, y, groups)):
        overlap = np.intersect1d(train, test)
        assert overlap.size == 0, (
            f"{name}: fold {i} has {overlap.size} index/indices in both train "
            f"and test ({overlap[:5].tolist()}), which leaks test rows into "
            f"training"
        )


def check_indices_are_in_range(name: str, splitter) -> None:
    """Every emitted index addresses a real row.

    Parameters
    ----------
    name : str
        Splitter name, used in failure messages.
    splitter : object
        A constructed splitter.

    Returns
    -------
    None
    """
    X, y, groups = make_split_data()
    n = len(X)
    for i, (train, test) in enumerate(_split(splitter, X, y, groups)):
        for label, idx in (("train", train), ("test", test)):
            idx = np.asarray(idx)
            assert idx.size == 0 or (idx.min() >= 0 and idx.max() < n), (
                f"{name}: fold {i} {label} indices fall outside [0, {n})"
            )


def check_folds_are_non_empty(name: str, splitter) -> None:
    """Both sides of every fold contain at least one row.

    Parameters
    ----------
    name : str
        Splitter name, used in failure messages.
    splitter : object
        A constructed splitter.

    Returns
    -------
    None
    """
    X, y, groups = make_split_data()
    folds = _split(splitter, X, y, groups)
    assert folds, f"{name}: split() yielded no folds"
    for i, (train, test) in enumerate(folds):
        assert len(train) > 0, f"{name}: fold {i} has an empty training set"
        assert len(test) > 0, f"{name}: fold {i} has an empty test set"


def check_split_is_reproducible(name: str, splitter) -> None:
    """Splitting the same data twice yields the same folds.

    Only meaningful once a seed is pinned: a splitter left at
    ``random_state=None`` is *supposed* to reshuffle on every call. The sweep
    constructs with ``random_state=0`` wherever the splitter accepts it.

    Parameters
    ----------
    name : str
        Splitter name, used in failure messages.
    splitter : object
        A constructed splitter.

    Returns
    -------
    None
    """
    X, y, groups = make_split_data()
    first = [(np.asarray(a).tolist(), np.asarray(b).tolist())
             for a, b in _split(splitter, X, y, groups)]
    second = [(np.asarray(a).tolist(), np.asarray(b).tolist())
              for a, b in _split(splitter, X, y, groups)]
    assert first == second, (
        f"{name}: two calls to split() on identical data produced different "
        f"folds, so any score built on it is not reproducible"
    )


#: Every splitter check, in run order.
ALL_CHECKS: Tuple[Callable[[str, Any], None], ...] = (
    check_train_and_test_are_disjoint,
    check_indices_are_in_range,
    check_folds_are_non_empty,
    check_split_is_reproducible,
)
