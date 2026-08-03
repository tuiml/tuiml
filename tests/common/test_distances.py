"""Contract tests over every distance function.

Replaces five near-identical modules that each re-derived symmetry,
non-negativity and identity for one metric. Adding a metric to
``tuiml.algorithms.clustering.distance`` subscribes it to the axioms
automatically.
"""

import inspect
import warnings

import pytest

from tuiml.algorithms.clustering import distance as distance_module

from ..contract.distances import ALL_CHECKS

XFAIL_CHECKS: dict = {}


def _discover():
    """Find every point-to-point distance function on the public API.

    The ``*_pairwise`` variants take matrices rather than two points and are
    covered by their scalar counterparts, so they are excluded here.

    Returns
    -------
    distances : list of tuple
        ``(name, fn)`` pairs, sorted.
    """
    found = []
    for name in dir(distance_module):
        if not name.endswith("_distance"):
            continue
        fn = getattr(distance_module, name)
        if callable(fn) and not inspect.isclass(fn):
            found.append((name, fn))
    return sorted(found)


DISTANCES = _discover()


def _cases():
    """Build the (distance, check) grid.

    Returns
    -------
    params : list of pytest.param
        One param per (distance, check) pair.
    """
    params = []
    for name, fn in DISTANCES:
        for check in ALL_CHECKS:
            reason = XFAIL_CHECKS.get(name, {}).get(check.__name__)
            marks = [pytest.mark.xfail(reason=reason, strict=False)] if reason else []
            params.append(
                pytest.param(name, fn, check, id=f"{name}-{check.__name__}", marks=marks)
            )
    return params


def test_distances_were_discovered():
    """Guard against the sweep silently covering nothing."""
    assert len(DISTANCES) >= 5, f"only {len(DISTANCES)} distance functions found"


@pytest.mark.parametrize("name, fn, check", _cases())
def test_distance_contract(name, fn, check):
    """Every distance function satisfies the metric axioms that apply to it."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        check(name, fn)
