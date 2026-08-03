"""Contract tests over every nearest-neighbour search backend.

Replaces three near-identical modules -- BallTree, KDTree and LinearNNSearch
each had its own copy of the same eight tests -- and adds the property none of
them could state alone: that all three agree with an exact brute-force search.
"""

import inspect
import warnings

import pytest

from tuiml.algorithms.neighbors import search as search_module

from ..contract.neighbors import ALL_CHECKS

SKIP = {"NearestNeighborSearch"}   # abstract base

XFAIL_CHECKS: dict = {}


def _discover():
    """Find every concrete search backend on the public API.

    Returns
    -------
    backends : list of tuple
        ``(name, cls)`` pairs, sorted.
    """
    found = []
    for name in dir(search_module):
        if name.startswith("_") or name in SKIP:
            continue
        obj = getattr(search_module, name)
        if inspect.isclass(obj) and hasattr(obj, "build") and hasattr(obj, "query"):
            found.append((name, obj))
    return sorted(found)


BACKENDS = _discover()


def _cases():
    """Build the (backend, check) grid.

    Returns
    -------
    params : list of pytest.param
        One param per (backend, check) pair.
    """
    params = []
    for name, cls in BACKENDS:
        for check in ALL_CHECKS:
            reason = XFAIL_CHECKS.get(name, {}).get(check.__name__)
            marks = [pytest.mark.xfail(reason=reason, strict=False)] if reason else []
            params.append(
                pytest.param(name, cls, check, id=f"{name}-{check.__name__}", marks=marks)
            )
    return params


def test_backends_were_discovered():
    """Guard against the sweep silently covering nothing."""
    assert len(BACKENDS) >= 3, f"only {len(BACKENDS)} search backends found"


@pytest.mark.parametrize("name, cls, check", _cases())
def test_neighbor_search_contract(name, cls, check):
    """Every search backend satisfies every contract check."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        check(name, cls)
