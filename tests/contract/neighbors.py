"""Contract checks for nearest-neighbour search structures.

The three backends -- BallTree, KDTree and LinearNNSearch -- had one test
module each, all asserting the same eight properties against their own
implementation in isolation. That is the wrong shape twice over: the
properties are shared, and the invariant that actually matters is that the
three *agree with each other*, which no per-backend module could express.

:func:`check_matches_brute_force` covers the agreement; the rest are the
shared properties, written once.
"""

from __future__ import annotations

from typing import Any, Callable, Tuple

import numpy as np


def _data(n: int = 40, d: int = 3) -> np.ndarray:
    """Return a fixed point cloud to index.

    Parameters
    ----------
    n : int, default=40
        Number of points.
    d : int, default=3
        Dimensionality.

    Returns
    -------
    X : np.ndarray of shape (n, d)
        Deterministic point cloud.
    """
    return np.random.default_rng(42).normal(size=(n, d))


def _built(cls):
    """Construct and build a search structure over the fixture data.

    Parameters
    ----------
    cls : type
        The search-structure class.

    Returns
    -------
    tree : object
        A built search structure.
    X : np.ndarray
        The data it was built over.

    Notes
    -----
    ``query`` returns ``(distances, indices)`` in that order, which is easy to
    get backwards.
    """
    X = _data()
    tree = cls()
    tree.build(X)
    return tree, X


def check_query_before_build_raises(name: str, cls) -> None:
    """Querying an unbuilt structure raises rather than returning junk.

    Parameters
    ----------
    name : str
        Class name, used in failure messages.
    cls : type
        The search-structure class.

    Returns
    -------
    None
    """
    tree = cls()
    try:
        tree.query(np.zeros(3), k=1)
    except Exception:
        return
    raise AssertionError(
        f"{name}: query() before build() returned instead of raising"
    )


def check_radius_results_are_sorted(name: str, cls) -> None:
    """``query_radius`` returns its hits nearest-first.

    Parameters
    ----------
    name : str
        Class name, used in failure messages.
    cls : type
        The search-structure class.

    Returns
    -------
    None
    """
    tree, X = _built(cls)
    dist, _ = tree.query_radius(X[0], 1.5)
    dist = np.asarray(dist, dtype=float)
    assert np.all(np.diff(dist) >= -1e-9), (
        f"{name}: query_radius distances are not ascending: {dist.tolist()}"
    )


def check_build_returns_self(name: str, cls) -> None:
    """``build`` returns the instance, so it chains.

    Parameters
    ----------
    name : str
        Class name, used in failure messages.
    cls : type
        The search-structure class.

    Returns
    -------
    None
    """
    X = _data()
    tree = cls()
    assert tree.build(X) is tree, f"{name}: build() did not return self"


def check_query_returns_k_neighbours(name: str, cls) -> None:
    """A k-query returns exactly k neighbours, clipped to the sample count.

    Parameters
    ----------
    name : str
        Class name, used in failure messages.
    cls : type
        The search-structure class.

    Returns
    -------
    None
    """
    tree, X = _built(cls)
    for k in (1, 3, 7):
        dist, idx = tree.query(X[0], k=k)
        assert len(idx) == k, f"{name}: query(k={k}) returned {len(idx)} neighbours"
        assert len(dist) == k, f"{name}: query(k={k}) returned {len(dist)} distances"

    _, idx = tree.query(X[0], k=len(X) + 10)
    assert len(idx) == len(X), (
        f"{name}: k larger than the dataset returned {len(idx)} of {len(X)} points"
    )


def check_nearest_neighbour_is_self(name: str, cls) -> None:
    """The closest point to a member of the dataset is itself, at distance 0.

    Parameters
    ----------
    name : str
        Class name, used in failure messages.
    cls : type
        The search-structure class.

    Returns
    -------
    None
    """
    tree, X = _built(cls)
    for i in (0, 5, len(X) - 1):
        dist, idx = tree.query(X[i], k=1)
        assert idx[0] == i, f"{name}: nearest to point {i} was {idx[0]}, not itself"
        assert abs(float(dist[0])) < 1e-9, f"{name}: self-distance was {dist[0]}"


def check_distances_are_sorted(name: str, cls) -> None:
    """Returned neighbours are ordered nearest first.

    Parameters
    ----------
    name : str
        Class name, used in failure messages.
    cls : type
        The search-structure class.

    Returns
    -------
    None
    """
    tree, X = _built(cls)
    dist, _ = tree.query(X[3], k=8)
    dist = np.asarray(dist, dtype=float)
    assert np.all(np.diff(dist) >= -1e-9), (
        f"{name}: query distances are not ascending: {dist.tolist()}"
    )


def check_radius_query_is_consistent(name: str, cls) -> None:
    """``query_radius`` returns exactly the points within the radius.

    Parameters
    ----------
    name : str
        Class name, used in failure messages.
    cls : type
        The search-structure class.

    Returns
    -------
    None
    """
    tree, X = _built(cls)
    point = X[0]
    radius = 1.5
    _, within = tree.query_radius(point, radius)
    found = set(np.asarray(within).tolist())

    expected = {i for i, row in enumerate(X)
                if float(np.linalg.norm(row - point)) <= radius}
    assert found == expected, (
        f"{name}: query_radius({radius}) returned {sorted(found)}, expected "
        f"{sorted(expected)}"
    )

    _, all_idx = tree.query_radius(point, 1e6)
    everything = set(np.asarray(all_idx).tolist())
    assert everything == set(range(len(X))), (
        f"{name}: an enormous radius returned {len(everything)} of {len(X)} points"
    )


def check_matches_brute_force(name: str, cls) -> None:
    """The structure agrees with an exact brute-force search.

    This is the property the per-backend modules could not express: an index
    is only useful if it returns what a linear scan would. A tree that is
    self-consistent but wrong passes every other check here.

    Parameters
    ----------
    name : str
        Class name, used in failure messages.
    cls : type
        The search-structure class.

    Returns
    -------
    None
    """
    tree, X = _built(cls)
    for probe in (X[0], X[17], np.zeros(X.shape[1])):
        k = 5
        _, idx = tree.query(probe, k=k)
        exact = np.argsort(np.linalg.norm(X - probe, axis=1))[:k]
        assert set(np.asarray(idx).tolist()) == set(exact.tolist()), (
            f"{name}: returned neighbours {sorted(np.asarray(idx).tolist())} "
            f"but brute force gives {sorted(exact.tolist())}"
        )


#: Every neighbour-search check, in run order.
ALL_CHECKS: Tuple[Callable[[str, Any], None], ...] = (
    check_query_before_build_raises,
    check_build_returns_self,
    check_query_returns_k_neighbours,
    check_nearest_neighbour_is_self,
    check_distances_are_sorted,
    check_radius_query_is_consistent,
    check_radius_results_are_sorted,
    check_matches_brute_force,
)
