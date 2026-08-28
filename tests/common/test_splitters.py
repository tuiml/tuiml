"""Contract tests over every cross-validation splitter.

A splitter that leaks rows between train and test inflates every score built
on it, in the direction that looks like success. These run over whatever
``tuiml.evaluation.splitting`` exports.
"""

import inspect
import warnings

import pytest

from tuiml.evaluation import splitting as splitting_module

from ..contract.splitters import ALL_CHECKS

SKIP = {"BaseSplitter"}

# Known contract violations, as {splitter: {check: reason}}. Each is a bug to
# fix, not a permanent exemption. Currently empty: every splitter satisfies
# every check.
XFAIL_CHECKS = {
}


def _discover():
    """Find every concrete splitter on the public API.

    Returns
    -------
    splitters : list of tuple
        ``(name, cls)`` pairs, sorted.
    """
    found = {}
    for name in dir(splitting_module):
        if name.startswith("_") or name in SKIP:
            continue
        obj = getattr(splitting_module, name)
        if inspect.isclass(obj) and hasattr(obj, "split") and not inspect.isabstract(obj):
            found.setdefault(name, obj)
    return sorted(found.items())


SPLITTERS = _discover()


def _cases():
    """Build the (splitter, check) grid.

    Returns
    -------
    params : list of pytest.param
        One param per (splitter, check) pair.
    """
    params = []
    for name, cls in SPLITTERS:
        for check in ALL_CHECKS:
            reason = XFAIL_CHECKS.get(name, {}).get(check.__name__)
            marks = [pytest.mark.xfail(reason=reason, strict=False)] if reason else []
            params.append(
                pytest.param(name, cls, check, id=f"{name}-{check.__name__}", marks=marks)
            )
    return params


def test_splitters_were_discovered():
    """Guard against the sweep silently covering nothing."""
    assert len(SPLITTERS) > 8, f"only {len(SPLITTERS)} splitters discovered"


def _construct(cls):
    """Build a splitter with its seed pinned where it accepts one.

    A splitter left at ``random_state=None`` reshuffles on every call by
    design, so reproducibility is only a meaningful property once the seed is
    fixed -- the same convention scikit-learn's own tests use.

    Parameters
    ----------
    cls : type
        The splitter class.

    Returns
    -------
    splitter : object
        A constructed splitter.
    """
    if "random_state" in inspect.signature(cls.__init__).parameters:
        return cls(random_state=0)
    return cls()


@pytest.mark.parametrize("name, cls, check", _cases())
def test_splitter_contract(name, cls, check):
    """Every splitter satisfies every contract check."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        check(name, _construct(cls))
