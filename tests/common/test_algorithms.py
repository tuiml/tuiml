"""Contract tests run over every algorithm in the registry.

One parametrised test replaces the six invariants that used to be copy-pasted
into every algorithm test module. Coverage now follows the registry rather
than author patience: registering an algorithm subscribes it to the whole
battery, and adding a check in :mod:`tests._contract` applies it to every
algorithm at once.

Algorithm-*specific* behaviour still belongs in the per-algorithm modules
under ``test_algorithms/``. This file only asserts what every algorithm owes
its callers.
"""

import re
import warnings

import pytest

import tuiml
from tuiml.registry import registry

from ..contract.algorithms import ALL_CHECKS

# Algorithms excluded from the sweep entirely, with the reason. Prefer
# XFAIL_CHECKS below: skipping an algorithm drops it from every check, whereas
# an xfail entry keeps the remaining checks honest.
SKIP_ALGORITHMS = {
    "STLDecomposition": "__init__ requires a `period` argument with no default",
    # The sweep constructs every algorithm as `cls()`, and these default to a
    # 60-second search. Eleven checks each running one -- twice, for the
    # reproducibility check -- puts the suite into the tens of minutes on its
    # own. They need coverage that constructs them with a small `time_budget`;
    # see tests for tuiml.automl, which do not exist yet.
    "AutoMLClassifier": "60s default search budget per check; needs dedicated tests",
    "AutoMLRegressor": "60s default search budget per check; needs dedicated tests",
}

# Known contract violations, as {algorithm: {check: reason}}. Every entry is a
# bug to fix, not a permanent exemption -- an empty table is the goal, and it is
# currently met: every registered algorithm satisfies every check. Listing a
# violation here keeps the suite green while making the debt explicit and
# greppable, and `strict=False` means a fix turns the entry into an XPASS
# rather than a failure, so the table degrades safely as things are repaired.
#
# Add an entry only for a violation you are recording to fix. Prefer fixing the
# algorithm; the last eight entries were cleared by finding that three of them
# were not algorithm bugs at all but a fixture that put NaN in the regression
# target, and one advised refitting with a constructor argument that had never
# existed.
XFAIL_CHECKS = {
}


def _registered_algorithms():
    """Yield ``(name, cls)`` for every native algorithm in the registry.

    Third-party wrappers (``sklearn.*``, ``capymoa.*``, ``weka.*``) are excluded: they obey
    their upstream library's contract, not TuiML's. Associators are excluded
    for the same reason, from the other direction: they mine itemsets from a
    transaction matrix, so they take ``fit(X)`` with no target and expose
    ``frequent_itemsets_``/``rules_`` instead of ``predict``, none of which the
    checks below describe. Versioned aliases are
    excluded so each algorithm is checked once, and anything defined outside
    the ``tuiml`` package is excluded so the suite does not depend on which
    user-authored algorithms happen to sit in ``~/.tuiml/user_algorithms``.

    Returns
    -------
    algorithms : list of tuple
        ``(name, cls)`` pairs, sorted by name.
    """
    out, seen = [], set()
    for info in tuiml.list_algorithms():
        name = info["name"]
        if "." in name or "_v" in name or name in SKIP_ALGORITHMS or name in seen:
            continue
        if info.get("type") == "associator":
            continue
        try:
            cls = registry.get(name)
        except Exception:
            continue
        if not getattr(cls, "__module__", "").startswith("tuiml."):
            continue
        seen.add(name)
        out.append((name, cls))
    return sorted(out, key=lambda pair: pair[0])


ALGORITHMS = _registered_algorithms()


def _cases():
    """Build the (algorithm, check) grid, marking known failures xfail.

    Returns
    -------
    params : list of pytest.param
        One param per (algorithm, check) pair.
    """
    params = []
    for name, cls in ALGORITHMS:
        for check in ALL_CHECKS:
            reason = XFAIL_CHECKS.get(name, {}).get(check.__name__)
            marks = [pytest.mark.xfail(reason=reason, strict=False)] if reason else []
            params.append(
                pytest.param(name, cls, check, id=f"{name}-{check.__name__}", marks=marks)
            )
    return params


def test_registry_is_not_empty():
    """Guard against the sweep silently covering nothing."""
    assert len(ALGORITHMS) > 50, (
        f"only {len(ALGORITHMS)} algorithms discovered; the registry filter is "
        f"probably wrong and the contract suite is testing almost nothing"
    )


# TuiML's optional-dependency contract says construction always succeeds and
# only fit() raises, with an ImportError naming the exact install command. An
# algorithm from an uninstalled extra therefore reaches these checks and fails
# in fit -- which is the library working as designed, not a contract violation.
# Matching the message keeps that distinct from a genuine ImportError bug: a
# real one says "cannot import name X", not "install it with pip install ...".
_MISSING_EXTRA = re.compile(r"not installed|pip install", re.IGNORECASE)


@pytest.mark.parametrize("name, cls, check", _cases())
def test_algorithm_contract(name, cls, check):
    """Every registered algorithm satisfies every contract check."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            check(name, cls())
        except ImportError as e:
            if not _MISSING_EXTRA.search(str(e)):
                raise
            pytest.skip(f"{name}: backing library not installed ({e})")
