"""Contract checks for distance functions.

Five distance modules each hand-rolled their own symmetry and identity tests,
which is how ``test_symmetry`` came to exist five times over. The axioms are
the same for every metric, so they belong in one place; what differs between
metrics is only *which* axioms apply, and that is data rather than code.
"""

from __future__ import annotations

from typing import Any, Callable, Tuple

import numpy as np

#: Distances that satisfy the triangle inequality, and so are true metrics.
#: Cosine distance is deliberately absent: it violates the inequality (it is
#: a dissimilarity, not a metric), and asserting otherwise would be wrong
#: rather than merely strict.
TRUE_METRICS = {"euclidean_distance", "manhattan_distance",
                "chebyshev_distance", "minkowski_distance"}


def _points(n: int = 12, d: int = 4) -> np.ndarray:
    """Return a fixed cloud of points to measure between.

    Parameters
    ----------
    n : int, default=12
        Number of points.
    d : int, default=4
        Dimensionality.

    Returns
    -------
    X : np.ndarray of shape (n, d)
        Deterministic point cloud spanning positive and negative coordinates.
    """
    return np.random.default_rng(0).normal(size=(n, d))


def check_distance_to_self_is_zero(name: str, fn) -> None:
    """``d(x, x) == 0`` for every point.

    Parameters
    ----------
    name : str
        Function name, used in failure messages.
    fn : callable
        The distance function.

    Returns
    -------
    None
    """
    for x in _points():
        value = float(fn(x, x))
        assert abs(value) < 1e-9, f"{name}: d(x, x) = {value}, not 0"


def check_symmetry(name: str, fn) -> None:
    """``d(x, y) == d(y, x)``.

    Parameters
    ----------
    name : str
        Function name, used in failure messages.
    fn : callable
        The distance function.

    Returns
    -------
    None
    """
    X = _points()
    for i in range(len(X)):
        for j in range(i + 1, len(X)):
            a, b = float(fn(X[i], X[j])), float(fn(X[j], X[i]))
            assert np.isclose(a, b), f"{name}: d(x, y) = {a} but d(y, x) = {b}"


def check_non_negative(name: str, fn) -> None:
    """Distances are never negative.

    Parameters
    ----------
    name : str
        Function name, used in failure messages.
    fn : callable
        The distance function.

    Returns
    -------
    None
    """
    X = _points()
    for i in range(len(X)):
        for j in range(len(X)):
            value = float(fn(X[i], X[j]))
            assert value >= -1e-9, f"{name}: returned a negative distance {value}"


def check_triangle_inequality(name: str, fn) -> None:
    """``d(x, z) <= d(x, y) + d(y, z)`` for true metrics.

    Skipped for dissimilarities such as cosine distance, which are not
    metrics and are not expected to satisfy this.

    Parameters
    ----------
    name : str
        Function name, used in failure messages.
    fn : callable
        The distance function.

    Returns
    -------
    None
    """
    if name not in TRUE_METRICS:
        return
    X = _points(8)
    for i in range(len(X)):
        for j in range(len(X)):
            for k in range(len(X)):
                direct = float(fn(X[i], X[k]))
                detour = float(fn(X[i], X[j])) + float(fn(X[j], X[k]))
                assert direct <= detour + 1e-9, (
                    f"{name}: d(x, z) = {direct} exceeds "
                    f"d(x, y) + d(y, z) = {detour}"
                )


def check_handles_single_dimension(name: str, fn) -> None:
    """One-dimensional inputs work and match the absolute difference.

    A 1-D case is where an implementation that assumes a matrix shape falls
    over, and every metric agrees on the answer there.

    Parameters
    ----------
    name : str
        Function name, used in failure messages.
    fn : callable
        The distance function.

    Returns
    -------
    None
    """
    if name == "cosine_distance":
        return  # undefined in 1-D: every non-zero scalar points the same way
    a, b = np.array([3.0]), np.array([-1.0])
    value = float(fn(a, b))
    assert np.isclose(value, 4.0), f"{name}: 1-D distance was {value}, expected 4.0"


#: Every distance check, in run order.
ALL_CHECKS: Tuple[Callable[[str, Any], None], ...] = (
    check_distance_to_self_is_zero,
    check_symmetry,
    check_non_negative,
    check_triangle_inequality,
    check_handles_single_dimension,
)
