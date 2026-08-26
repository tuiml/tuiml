"""Pick and rank the algorithms an AutoML run should try.

A wall-clock budget is spent one candidate at a time, so the order matters
more than the list: whatever is tried first is what a short run gets to keep.
This module queries the registry for every algorithm that fits the task, drops
the ones that cannot run here, and returns them **cheap-and-strong first** --
linear and tree baselines, then kernel and instance methods, then ensembles,
then boosting, then deep models.

Examples
--------
>>> from tuiml.automl.portfolio import build_portfolio
>>> names = [c.name for c in build_portfolio("classification")]
>>> "LogisticRegression" in names
True
>>> names.index("LogisticRegression") < names.index("XGBoostClassifier")
True
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from tuiml.registry import ComponentType, registry

#: Task name to the registry component type holding its algorithms.
_TASK_TYPES = {
    "classification": ComponentType.CLASSIFIER,
    "regression": ComponentType.REGRESSOR,
}

#: Tags that disqualify an algorithm from a general tabular portfolio.
#:
#: ``meta`` / ``stacking`` / ``voting``
#:     Combiners that need a list of base learners; they have nothing to do
#:     until there are trials to combine, which is what
#:     :mod:`tuiml.automl.ensembling` does at the end of the run.
#: ``timeseries``
#:     Expects ordered observations, not i.i.d. rows.
#: ``baseline``
#:     Majority-class / mean predictors. Useful as a floor to compare
#:     against, never as an answer, so they do not earn a slot in the budget.
DEFAULT_EXCLUDED_TAGS = frozenset({
    "meta", "stacking", "voting", "timeseries", "baseline",
})

#: Ordered cost/complexity tiers. An algorithm lands in the **highest** tier
#: any of its tags matches: a model tagged both ``functions`` and
#: ``deep-learning`` costs what the deep half costs, so that half decides.
#: Anything unmatched lands in :data:`_DEFAULT_TIER`. The ordering encodes the
#: standard tabular result: regularised linear models and single trees are
#: within a few points of the best model on most datasets and cost a fraction
#: of the time, so they go first and boosting follows.
_TIERS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("linear", ("linear", "bayes", "bayesian", "probabilistic", "functions")),
    # ``interpretable`` is deliberately absent: it describes what a model
    # tells you, not what it costs, and it is worn by everything from a
    # single stump to a rule ensemble.
    ("tree", ("trees", "glassbox", "rules")),
    ("instance", ("lazy", "instance-based", "knn", "svm", "kernel")),
    ("ensemble", ("ensemble", "ensembles", "bagging", "random")),
    ("boosting", ("gradient-boosting", "boosting", "xgboost", "lightgbm", "catboost")),
    ("deep", ("deep-learning", "neural-network", "transformer", "torch")),
)

#: Tier index used for an algorithm whose tags match no tier.
_DEFAULT_TIER = 3

#: Names pulled to the front of their tier, in this order. These are the
#: workhorses: a short budget should reach them before anything else at the
#: same cost level.
_PREFERRED = (
    "LogisticRegression",
    "LinearRegression",
    "NaiveBayesClassifier",
    "RandomForestClassifier",
    "RandomForestRegressor",
    "DecisionTreeClassifier",
    "DecisionTreeRegressor",
    "KNearestNeighborsClassifier",
    "KNearestNeighborsRegressor",
    "SVC",
    "SVR",
)


@dataclass(frozen=True)
class Candidate:
    """One algorithm an AutoML run may try.

    Parameters
    ----------
    name : str
        Registry name, e.g. ``"RandomForestClassifier"``.
    cls : type
        The algorithm class itself.
    tier : int
        Cost tier index (lower is cheaper); see :data:`_TIERS`.
    tags : tuple of str
        The registry tags the algorithm was registered with.
    """

    name: str
    cls: type
    tier: int
    tags: Tuple[str, ...] = field(default_factory=tuple)


def _tier_of(tags: Sequence[str]) -> int:
    """Return the cost tier index for a set of registry tags.

    Parameters
    ----------
    tags : sequence of str
        The algorithm's registry tags.

    Returns
    -------
    tier : int
        Index of the highest matching tier in :data:`_TIERS`, or
        :data:`_DEFAULT_TIER` if nothing matched.
    """
    lowered = {tag.lower() for tag in tags}
    matched = [
        index
        for index, (_, tier_tags) in enumerate(_TIERS)
        if lowered & set(tier_tags)
    ]
    return max(matched) if matched else _DEFAULT_TIER


def _is_usable(cls: type) -> bool:
    """Return True if the class can be constructed with its defaults here.

    Algorithms that wrap an optional backend raise at construction when the
    backend (or its JVM) is missing. Probing with a default construction is
    cheap and catches that without a hard-coded table of extras.

    Parameters
    ----------
    cls : type
        The candidate algorithm class.

    Returns
    -------
    usable : bool
        Whether ``cls()`` succeeded.
    """
    try:
        cls()
    except Exception:
        return False
    return True


def build_portfolio(
    task: str,
    *,
    candidates: Optional[Iterable[Any]] = None,
    exclude: Optional[Iterable[str]] = None,
    exclude_tags: Optional[Iterable[str]] = None,
    include_wrappers: bool = False,
    max_candidates: Optional[int] = None,
) -> List[Candidate]:
    """Return the ranked candidate algorithms for a task.

    Overview
    --------
    1. Ask the registry for every component of the task's type.
    2. Drop namespaced wrapper keys (``sklearn.SVC``, ``weka.J48``) unless
       ``include_wrappers=True``: they duplicate native algorithms and need an
       extra installed.
    3. Drop anything tagged with :data:`DEFAULT_EXCLUDED_TAGS`, and anything
       that cannot be constructed with its defaults in this environment.
    4. Sort by ``(tier, preference, name)`` -- a total order with no ties, so
       two runs on the same install always try the same models in the same
       sequence.

    Parameters
    ----------
    task : {"classification", "regression"}
        Which kind of algorithm to collect.
    candidates : iterable of str or type, optional
        An explicit list that overrides discovery entirely. Names are resolved
        through the registry; classes are used as given. The order is kept as
        written, since an explicit list is already a statement of preference.
    exclude : iterable of str, optional
        Registry names to leave out.
    exclude_tags : iterable of str, optional
        Tags to leave out, replacing :data:`DEFAULT_EXCLUDED_TAGS`.
    include_wrappers : bool, default=False
        Keep namespaced wrapper entries such as ``"sklearn.SVC"``.
    max_candidates : int, optional
        Truncate the ranked list to this many entries.

    Returns
    -------
    portfolio : list of Candidate
        Ranked candidates, cheapest and most reliable first.

    Raises
    ------
    ValueError
        If ``task`` is not a known task name.

    Examples
    --------
    >>> from tuiml.automl.portfolio import build_portfolio
    >>> portfolio = build_portfolio("regression", max_candidates=3)
    >>> [c.name for c in portfolio]
    ['LinearRegression', 'BayesianLinearRegressor', 'SGDRegressor']

    An explicit list is used verbatim:

    >>> [c.name for c in build_portfolio(
    ...     "classification", candidates=["SVC", "NaiveBayesClassifier"])]
    ['SVC', 'NaiveBayesClassifier']
    """
    if task not in _TASK_TYPES:
        raise ValueError(
            f"Unknown task {task!r}. Expected one of {sorted(_TASK_TYPES)}."
        )

    if candidates is not None:
        return _explicit_portfolio(candidates)

    excluded_names = set(exclude or ())
    blocked_tags = {
        tag.lower()
        for tag in (DEFAULT_EXCLUDED_TAGS if exclude_tags is None else exclude_tags)
    }

    portfolio: List[Candidate] = []
    for info in registry.list(_TASK_TYPES[task]):
        name = info.get("name", "")
        if name in excluded_names:
            continue
        if "." in name and not include_wrappers:
            continue
        tags = tuple(info.get("tags") or ())
        if {tag.lower() for tag in tags} & blocked_tags:
            continue
        try:
            cls = registry.get(name)
        except KeyError:
            continue
        if not _is_usable(cls):
            continue
        portfolio.append(Candidate(name=name, cls=cls, tier=_tier_of(tags), tags=tags))

    portfolio.sort(key=lambda c: (c.tier, _preference(c.name), c.name))
    if max_candidates is not None:
        portfolio = portfolio[:max_candidates]
    return portfolio


def _preference(name: str) -> int:
    """Return the within-tier sort key for a registry name (lower is earlier)."""
    try:
        return _PREFERRED.index(name)
    except ValueError:
        return len(_PREFERRED)


def _explicit_portfolio(candidates: Iterable[Any]) -> List[Candidate]:
    """Turn a user-supplied candidate list into :class:`Candidate` objects.

    Parameters
    ----------
    candidates : iterable of str or type
        Registry names or algorithm classes, in the order to try them.

    Returns
    -------
    portfolio : list of Candidate
        One entry per input, order preserved.

    Raises
    ------
    KeyError
        If a name is not registered.
    """
    portfolio = []
    for position, item in enumerate(candidates):
        if isinstance(item, str):
            cls = registry.get(item)
            name = item
            tags = tuple(registry.get_info(item).get("tags") or ())
        else:
            cls = item
            name = getattr(cls, "_component_name", None) or cls.__name__
            tags = tuple(getattr(cls, "_tags", ()) or ())
        portfolio.append(Candidate(name=name, cls=cls, tier=position, tags=tags))
    return portfolio


def describe_portfolio(portfolio: Sequence[Candidate]) -> List[Dict[str, Any]]:
    """Render a portfolio as plain dicts, for logging or display.

    Parameters
    ----------
    portfolio : sequence of Candidate
        The candidates to describe.

    Returns
    -------
    rows : list of dict
        One ``{"rank", "name", "tier", "tags"}`` dict per candidate.

    Examples
    --------
    >>> from tuiml.automl.portfolio import build_portfolio, describe_portfolio
    >>> describe_portfolio(build_portfolio("regression", max_candidates=1))
    [{'rank': 1, 'name': 'LinearRegression', 'tier': 0, 'tags': ['linear', 'regression', 'interpretable']}]
    """
    return [
        {
            "rank": index + 1,
            "name": candidate.name,
            "tier": candidate.tier,
            "tags": list(candidate.tags),
        }
        for index, candidate in enumerate(portfolio)
    ]


__all__ = [
    "Candidate",
    "DEFAULT_EXCLUDED_TAGS",
    "build_portfolio",
    "describe_portfolio",
]
