"""Turn a component's JSON Schema into a searchable parameter space.

Every TuiML algorithm already publishes
:meth:`~tuiml.base.algorithms.Algorithm.get_parameter_schema`, a JSON Schema
per constructor parameter. This module reads that schema and derives a
:class:`~tuiml.base.tuning.ParameterDistribution` from it, so a tuner can
search any algorithm in the library without a hand-written grid.

The module is deliberately standalone: it depends only on the registry
metadata and :mod:`tuiml.base.tuning`, so
:class:`~tuiml.evaluation.tuning.RandomSearchCV` and friends can use it
directly, with or without :mod:`tuiml.automl`.

Examples
--------
>>> from tuiml.automl.search_space import search_space_for
>>> from tuiml.algorithms.trees import DecisionTreeClassifier
>>> space = search_space_for(DecisionTreeClassifier)
>>> sorted(space.param_distributions)
['criterion', 'max_depth', 'min_samples_leaf', 'min_samples_split']
>>> space.skipped_["random_state"]
'excluded: not a model-behaviour parameter'
"""

import inspect
import math
from typing import Any, Dict, Iterable, List, Optional, Tuple

from tuiml.base.tuning import ParameterDistribution

#: Parameters that must never be searched. They either control reproducibility
#: and reporting rather than model behaviour (``random_state``, ``verbose``),
#: control resource use (``n_jobs``), or are wiring that the caller owns
#: (``base_estimator``, ``callbacks``). Searching them wastes budget at best
#: and makes a run irreproducible at worst.
EXCLUDED_PARAMETERS = frozenset({
    "random_state",
    "random_seed",
    "seed",
    "n_jobs",
    "verbose",
    "verbosity",
    "warm_start",
    "cache_size",
    "device",
    "backend",
    "callbacks",
    "base_estimator",
    "estimator",
    "estimators",
    "classifiers",
    "regressors",
    "meta_classifier",
    "meta_regressor",
    "objective",
    "class_weight",
    "name",
    "memory",
})

#: Cost knobs: the iteration counts and ensemble sizes whose only effect is
#: to trade a roughly linear increase in fit time for marginal accuracy. Under
#: a wall-clock budget that trade is usually bad -- one 1000-tree draw can eat
#: a whole run -- so :mod:`tuiml.automl` excludes these on top of
#: :data:`EXCLUDED_PARAMETERS` and leaves each algorithm at the size its
#: author chose. They are *not* excluded by default here, so a caller with
#: time to spend can still search them.
BUDGET_PARAMETERS = frozenset({
    "max_iter",
    "max_iterations",
    "max_epochs",
    "n_iter",
    "n_epochs",
    "epochs",
    "n_restarts",
    "n_init",
    "n_estimators",
    "n_trees",
    "iterations",
    "n_rounds",
    "num_boost_round",
})

#: A bounded numeric range is searched on a log scale when
#: ``high / low >= LOG_SCALE_RATIO`` and ``low > 0``. Spanning two or more
#: orders of magnitude is the signature of a scale parameter -- a learning
#: rate, a regularisation strength -- where uniform sampling would spend
#: almost all draws in the top decade.
LOG_SCALE_RATIO = 100.0

#: Half-width, as a multiplicative factor, of the range derived for an
#: unbounded continuous parameter: the search runs over
#: ``[default / UNBOUNDED_SPAN, default * UNBOUNDED_SPAN]``, log-scaled.
UNBOUNDED_SPAN = 100.0

#: The same half-width for unbounded **integer** parameters, which are far
#: more often cost knobs (``n_estimators``, ``leaf_size``) than scales. One
#: decade either side of the default keeps the expensive end of the range
#: reachable without making it the common case.
UNBOUNDED_INT_SPAN = 10.0

#: Multiplier used to bound a nullable integer whose default is ``None``
#: ("no limit", as in ``max_depth``). The levels run geometrically from the
#: declared minimum up to ``minimum * NULL_DEFAULT_SPAN``, which for the usual
#: ``minimum=1`` gives ``1, 2, 4, 8, 16, 32`` -- depths that matter, without
#: pretending an unlimited tree has a numeric upper bound.
NULL_DEFAULT_SPAN = 32

#: Number of discrete levels generated when a bounded integer has to be
#: expressed as a choice list (because ``None`` is also a legal value).
N_LEVELS = 6

#: JSON Schema type names, plus the Python type objects and their ``repr``
#: strings, that some components emit instead of the JSON name.
_TYPE_ALIASES = {
    "int": "integer",
    "float": "number",
    "double": "number",
    "str": "string",
    "bool": "boolean",
    "list": "array",
    "tuple": "array",
    "dict": "object",
    "nonetype": "null",
}


def _normalize_types(spec: Dict[str, Any]) -> List[str]:
    """Return the schema's declared types as a list of JSON Schema names.

    Components in the wild declare ``type`` in three ways: the JSON name
    (``"integer"``), a list of names for a union (``["integer", "null"]``),
    or a Python type object / its ``repr`` (``<class 'int'>``). All three
    are normalized here so the rest of the module sees one form.

    Parameters
    ----------
    spec : dict
        One parameter's JSON Schema fragment.

    Returns
    -------
    types : list of str
        Lowercase JSON Schema type names, possibly empty.
    """
    raw = spec.get("type")
    if raw is None:
        return []
    items = raw if isinstance(raw, (list, tuple)) else [raw]

    names = []
    for item in items:
        if isinstance(item, type):
            text = item.__name__
        else:
            text = str(item)
            if text.startswith("<class "):
                text = text.split("'")[1]
        text = text.rsplit(".", 1)[-1].lower()
        names.append(_TYPE_ALIASES.get(text, text))
    return names


def _bounds(spec: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    """Return the ``(minimum, maximum)`` declared for a parameter.

    Both the JSON Schema keywords (``minimum``/``maximum``) and the
    ``"range": [low, high]`` shorthand used by some components are read.

    Parameters
    ----------
    spec : dict
        One parameter's JSON Schema fragment.

    Returns
    -------
    low : float or None
        Lower bound, or None if unbounded below.
    high : float or None
        Upper bound, or None if unbounded above.
    """
    low = spec.get("minimum", spec.get("exclusiveMinimum"))
    high = spec.get("maximum", spec.get("exclusiveMaximum"))
    rng = spec.get("range")
    if isinstance(rng, (list, tuple)) and len(rng) == 2:
        low = rng[0] if low is None else low
        high = rng[1] if high is None else high
    if not isinstance(low, (int, float)) or isinstance(low, bool):
        low = None
    if not isinstance(high, (int, float)) or isinstance(high, bool):
        high = None
    return low, high


def _is_number(value: Any) -> bool:
    """Return True if ``value`` is a real number (and not a bool)."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _int_levels(low: int, high: int, n: int = N_LEVELS) -> List[int]:
    """Return up to ``n`` distinct integers spread geometrically over a range.

    Geometric rather than linear spacing: for parameters such as
    ``max_depth`` the interesting variation sits at the small end, so
    ``[2, 4, 8, 16, 32]`` explores the space better than ``[10, 20, 30, 40]``.

    Parameters
    ----------
    low : int
        Lower bound, inclusive.
    high : int
        Upper bound, inclusive.
    n : int, default=6
        Maximum number of levels to return.

    Returns
    -------
    levels : list of int
        Sorted, de-duplicated integer levels.
    """
    low = max(1, int(low))
    high = max(low, int(high))
    if high - low + 1 <= n:
        return list(range(low, high + 1))
    ratio = (high / low) ** (1.0 / (n - 1))
    levels = sorted({int(round(low * ratio ** i)) for i in range(n)})
    return [v for v in levels if low <= v <= high]


def _numeric_distribution(
    low: Optional[float],
    high: Optional[float],
    default: Any,
    integer: bool,
) -> Tuple[Optional[Any], Optional[str]]:
    """Derive a range spec for one numeric parameter.

    The rules, in order:

    1. **Bounded** (both ``low`` and ``high`` known): use the declared range.
       Sample it log-uniformly when ``low > 0`` and
       ``high / low >= LOG_SCALE_RATIO`` -- a span of two or more orders of
       magnitude means a scale parameter.
    2. **Unbounded** with a positive numeric default ``d``: search
       ``[d / UNBOUNDED_SPAN, d * UNBOUNDED_SPAN]`` on a log scale, clipped
       to whichever bound *is* declared. Two decades either side of the
       author's default covers the useful region without inventing a scale.
    3. Otherwise (no default, a non-positive default such as the ``-1``
       "auto" sentinel, or a default that is not a number): **skip**. A
       negative or zero default is usually a sentinel, and interpolating
       around it produces illegal values.

    Parameters
    ----------
    low : float or None
        Declared lower bound.
    high : float or None
        Declared upper bound.
    default : Any
        The parameter's declared default.
    integer : bool
        Whether the parameter is an integer.

    Returns
    -------
    dist : tuple or None
        A ``ParameterDistribution`` range spec, or None if skipped.
    reason : str or None
        Why the parameter was skipped, or None if it was not.
    """
    if low is not None and high is not None and high > low:
        if integer:
            return (int(low), int(math.floor(high)), "int"), None
        if low > 0 and high / low >= LOG_SCALE_RATIO:
            return (float(low), float(high), "log"), None
        return (float(low), float(high)), None

    if not _is_number(default):
        return None, "unbounded and no numeric default to anchor a range"
    if default <= 0:
        return None, (
            f"unbounded and the default ({default!r}) is not positive, so it "
            f"is most likely a sentinel rather than a searchable scale"
        )

    span = UNBOUNDED_INT_SPAN if integer else UNBOUNDED_SPAN
    derived_low = default / span
    derived_high = default * span
    if low is not None:
        derived_low = max(derived_low, low)
    if high is not None:
        derived_high = min(derived_high, high)
    if derived_high <= derived_low:
        return None, "derived range collapsed against the declared bounds"

    if integer:
        derived_low = max(1, int(round(derived_low)))
        derived_high = max(derived_low + 1, int(round(derived_high)))
        return (derived_low, derived_high, "int"), None
    return (float(derived_low), float(derived_high), "log"), None


def schema_to_distribution(
    schema: Dict[str, Dict[str, Any]],
    *,
    include: Optional[Iterable[str]] = None,
    exclude: Optional[Iterable[str]] = None,
) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """Convert a parameter schema into ``ParameterDistribution`` specs.

    Each parameter is mapped to one of the forms
    :class:`~tuiml.base.tuning.ParameterDistribution` understands:

    ==================================  ================================
    Schema fragment                     Distribution spec
    ==================================  ================================
    ``{"enum": [...]}``                 the list of enum values
    ``{"type": "boolean"}``             ``[True, False]``
    ``{"type": "integer", min, max}``   ``(min, max, 'int')``
    ``{"type": "number", min, max}``    ``(min, max)`` or ``(min, max, 'log')``
    ``{"type": ["integer", "null"]}``   ``[None, level, level, ...]``
    ``{"default": None, min, no max}``  ``[None, level, level, ...]``
    ==================================  ================================

    A union with ``"null"`` keeps ``None`` as an explicit choice, since for
    parameters like ``max_depth`` "no limit" is a real setting rather than a
    missing value. A union of two or more *concrete* types (for example
    ``["number", "string"]`` for a parameter that also accepts ``"auto"``)
    is skipped unless it carries an ``enum``: there is no safe way to guess
    which arm the tuner should explore.

    Parameters
    ----------
    schema : dict
        Mapping of parameter name to its JSON Schema fragment, as returned
        by :meth:`~tuiml.base.algorithms.Algorithm.get_parameter_schema`.
    include : iterable of str, optional
        If given, only these parameter names are considered.
    exclude : iterable of str, optional
        Extra parameter names to drop, on top of :data:`EXCLUDED_PARAMETERS`.

    Returns
    -------
    distributions : dict
        Parameter name to distribution spec, ready for
        :class:`~tuiml.base.tuning.ParameterDistribution`.
    skipped : dict
        Parameter name to a human-readable reason it was left out.

    Examples
    --------
    >>> from tuiml.automl.search_space import schema_to_distribution
    >>> schema = {
    ...     "alpha": {"type": "number", "minimum": 0.0001, "maximum": 1.0},
    ...     "fit_intercept": {"type": "boolean", "default": True},
    ...     "n_jobs": {"type": "integer", "default": 1},
    ... }
    >>> dists, skipped = schema_to_distribution(schema)
    >>> dists["alpha"]
    (0.0001, 1.0, 'log')
    >>> dists["fit_intercept"]
    [True, False]
    >>> skipped["n_jobs"]
    'excluded: not a model-behaviour parameter'
    """
    include = set(include) if include is not None else None
    blocked = set(EXCLUDED_PARAMETERS)
    if exclude is not None:
        blocked |= set(exclude)

    distributions: Dict[str, Any] = {}
    skipped: Dict[str, str] = {}

    for name, spec in (schema or {}).items():
        if include is not None and name not in include:
            skipped[name] = "not in the requested include list"
            continue
        if name in blocked:
            skipped[name] = "excluded: not a model-behaviour parameter"
            continue
        if not isinstance(spec, dict):
            skipped[name] = "schema entry is not a dict"
            continue

        types = _normalize_types(spec)
        nullable = "null" in types
        concrete = [t for t in types if t != "null"]
        default = spec.get("default")

        enum = spec.get("enum")
        if isinstance(enum, (list, tuple)) and len(enum) > 1:
            choices = list(enum)
            if nullable and None not in choices:
                choices.append(None)
            distributions[name] = choices
            continue

        if concrete == ["boolean"]:
            distributions[name] = [True, False]
            continue

        if len(concrete) > 1:
            skipped[name] = (
                f"ambiguous union type {concrete} without an enum; no safe "
                f"way to pick which arm to search"
            )
            continue
        if not concrete:
            skipped[name] = "no usable type declared"
            continue

        kind = concrete[0]
        if kind not in ("integer", "number"):
            skipped[name] = f"type {kind!r} is not searchable without an enum"
            continue

        low, high = _bounds(spec)

        if default is None and kind == "integer" and low is not None and high is None:
            # "No limit" parameters such as ``max_depth``: None is the default
            # and a real setting, so search it alongside a ladder of finite
            # values anchored on the declared minimum.
            distributions[name] = [None] + _int_levels(
                low, low * NULL_DEFAULT_SPAN
            )
            continue

        dist, reason = _numeric_distribution(
            low, high, default, integer=(kind == "integer")
        )
        if dist is None:
            skipped[name] = reason
            continue

        if nullable:
            # ``None`` cannot be mixed into a continuous range, so express the
            # parameter as a choice list: None plus a few levels of the range.
            if kind == "integer":
                levels: List[Any] = [None] + _int_levels(dist[0], dist[1])
            else:
                lo, hi = float(dist[0]), float(dist[1])
                step = (hi - lo) / (N_LEVELS - 1)
                levels = [None] + [lo + step * i for i in range(N_LEVELS)]
            distributions[name] = levels
            continue

        distributions[name] = dist

    return distributions, skipped


def search_space_for(
    algorithm_cls,
    *,
    include: Optional[Iterable[str]] = None,
    exclude: Optional[Iterable[str]] = None,
) -> ParameterDistribution:
    """Build a searchable parameter space for one algorithm class.

    Reads the class's :meth:`get_parameter_schema`, converts it with
    :func:`schema_to_distribution`, and drops anything the constructor does
    not actually accept -- a schema occasionally documents a parameter that
    the current signature no longer takes, and sampling it would raise.

    Parameters
    ----------
    algorithm_cls : type or str
        The algorithm class, or its registry name (``"RandomForestClassifier"``).
    include : iterable of str, optional
        If given, only these parameter names are searched.
    exclude : iterable of str, optional
        Extra parameter names to drop, on top of :data:`EXCLUDED_PARAMETERS`.

    Returns
    -------
    space : ParameterDistribution
        The search space. The attribute ``space.skipped_`` maps every
        parameter that was left out to the reason why, so an empty or
        surprising space can be debugged without re-deriving it.

    Examples
    --------
    >>> from tuiml.automl.search_space import search_space_for
    >>> space = search_space_for("RandomForestClassifier")
    >>> space.param_distributions["criterion"]
    ['gini', 'entropy']
    >>> params = space.sample(random_state=0)
    >>> sorted(params)
    ['bootstrap', 'criterion', 'max_depth', 'min_samples_leaf', 'min_samples_split', 'n_estimators', 'oob_score']
    >>> space.skipped_["max_features"]
    "ambiguous union type ['string', 'integer', 'number'] without an enum; no safe way to pick which arm to search"
    """
    if isinstance(algorithm_cls, str):
        from tuiml.registry import registry
        algorithm_cls = registry.get(algorithm_cls)

    getter = getattr(algorithm_cls, "get_parameter_schema", None)
    schema = getter() if callable(getter) else {}

    distributions, skipped = schema_to_distribution(
        schema, include=include, exclude=exclude
    )

    accepted = _constructor_parameters(algorithm_cls)
    if accepted is not None:
        for name in list(distributions):
            if name not in accepted:
                del distributions[name]
                skipped[name] = "not a parameter of the constructor"

    space = ParameterDistribution(distributions)
    space.skipped_ = skipped
    return space


def _constructor_parameters(algorithm_cls) -> Optional[set]:
    """Return the names ``algorithm_cls.__init__`` accepts, or None.

    Parameters
    ----------
    algorithm_cls : type
        The class to inspect.

    Returns
    -------
    names : set of str or None
        Accepted keyword names, or None when the signature cannot be read or
        the constructor takes ``**kwargs`` (in which case anything goes).
    """
    try:
        signature = inspect.signature(algorithm_cls.__init__)
    except (TypeError, ValueError):
        return None
    names = set()
    for parameter in signature.parameters.values():
        if parameter.kind is inspect.Parameter.VAR_KEYWORD:
            return None
        if parameter.name != "self":
            names.add(parameter.name)
    return names


__all__ = [
    "EXCLUDED_PARAMETERS",
    "BUDGET_PARAMETERS",
    "LOG_SCALE_RATIO",
    "UNBOUNDED_SPAN",
    "UNBOUNDED_INT_SPAN",
    "NULL_DEFAULT_SPAN",
    "N_LEVELS",
    "schema_to_distribution",
    "search_space_for",
]
