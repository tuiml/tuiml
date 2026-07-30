"""Declarative specification layer for the scikit-learn bridge.

The wrapper modules in this package are **generated** from the rows in
:mod:`tuiml.sklearn.specs` by ``scripts/generate_sklearn_wrappers.py``. This
module holds the pieces those generated classes rely on at runtime:

* :class:`SklearnSpec` — the declarative description of one wrapped estimator.
* :func:`build_estimator` — resolves the backing scikit-learn class and
  constructs it, validating parameter names against the real signature.
* :func:`derive_schema` — derives a JSON Schema for an estimator's constructor
  from scikit-learn's own signature and ``_parameter_constraints`` metadata, so
  the schema can never drift from the installed scikit-learn version.

scikit-learn is an optional dependency: nothing here imports it at module scope.
"""

import importlib
import inspect
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

#: Component kinds a spec may declare, mapped to the ``(decorator, mixin, base)``
#: names the generated modules import. Kept as strings so this module stays
#: import-light; the generator resolves them when emitting code.
KIND_IMPORTS: Dict[str, Tuple[str, str, str]] = {
    "classifier": ("sk_classifier", "_SklearnBackedMixin", "Classifier"),
    "regressor": ("sk_regressor", "_SklearnBackedMixin", "Regressor"),
    "clusterer": ("sk_clusterer", "_SklearnClustererMixin", "Clusterer"),
    "transformer": ("sk_transformer", "_SklearnTransformerMixin", "Transformer"),
    "feature_selector": (
        "sk_feature_selector",
        "_SklearnSelectorMixin",
        "FeatureSelector",
    ),
    "feature_extractor": (
        "sk_feature_extractor",
        "_SklearnExtractorMixin",
        "FeatureExtractor",
    ),
}


@dataclass(frozen=True)
class SklearnSpec:
    """Declarative description of one wrapped scikit-learn estimator.

    Parameters
    ----------
    name : str
        Wrapper class name. The hub key becomes ``sklearn.<name>``. Usually the
        scikit-learn class name, but may differ where a TuiML name is already
        established (e.g. ``PCAExtractor`` for ``sklearn.decomposition.PCA``).
    target : str
        Backing estimator as ``"module:ClassName"``, resolved lazily.
    kind : str
        One of the keys of :data:`KIND_IMPORTS`; selects base class and decorator.
    tags : tuple of str
        Hub search tags. ``"sklearn"`` is appended automatically at registration.
    capabilities : tuple of str
        Values returned by the wrapper's ``get_capabilities()``.
    defaults : dict
        TuiML-opinionated parameter defaults applied before user parameters.
    highlight : tuple of str
        Parameters surfaced first in the derived schema. Also the offline
        fallback when scikit-learn is not installed.
    exclude : tuple of str
        Parameters omitted from the derived schema — either meaningless through
        TuiML (``n_jobs``, ``verbose``) or unsupported (a ``precomputed`` kernel
        needs a kernel matrix rather than a feature array).
    tier : int
        ``1`` for the curated, hand-reviewed set surfaced by default; ``2`` for
        the bulk-generated remainder, additionally tagged ``sklearn-extended``.
    builder : str, optional
        Name of a function in :mod:`tuiml.sklearn._overrides` used instead of the
        default construction path, for the rare estimator needing extra assembly.
    """

    name: str
    target: str
    kind: str
    tags: Tuple[str, ...] = ()
    capabilities: Tuple[str, ...] = ()
    defaults: Dict[str, Any] = field(default_factory=dict)
    highlight: Tuple[str, ...] = ()
    exclude: Tuple[str, ...] = ()
    tier: int = 2
    builder: Optional[str] = None


#: Parameters excluded from every derived schema: execution/plumbing knobs that
#: are not modelling choices.
COMMON_EXCLUDE: Tuple[str, ...] = (
    "n_jobs",
    "verbose",
    "warm_start",
    "copy",
    "copy_X",
    "memory",
)


def resolve_target(target: str) -> type:
    """Import and return the backing scikit-learn class.

    Parameters
    ----------
    target : str
        Estimator reference as ``"module:ClassName"``.

    Returns
    -------
    cls : type
        The scikit-learn estimator class.
    """
    module_path, _, class_name = target.partition(":")
    return getattr(importlib.import_module(module_path), class_name)


def _valid_param_names(est_cls: type) -> set:
    """Return the constructor parameter names accepted by ``est_cls``.

    Parameters
    ----------
    est_cls : type
        A scikit-learn estimator class.

    Returns
    -------
    names : set of str
        Accepted keyword parameter names.
    """
    sig = inspect.signature(est_cls.__init__)
    return {
        name
        for name, p in sig.parameters.items()
        if name != "self"
        and p.kind not in (p.VAR_KEYWORD, p.VAR_POSITIONAL)
    }


def build_estimator(target: str, params: Dict[str, Any], wrapper_name: str) -> Any:
    """Construct the backing estimator, validating parameter names.

    Parameters
    ----------
    target : str
        Estimator reference as ``"module:ClassName"``.
    params : dict
        Parameters to pass to the estimator constructor.
    wrapper_name : str
        Wrapper class name, used in the error message.

    Returns
    -------
    estimator : object
        A fresh, configured scikit-learn estimator.

    Raises
    ------
    TypeError
        If ``params`` contains a name the estimator does not accept.
    """
    est_cls = resolve_target(target)
    valid = _valid_param_names(est_cls)
    unknown = sorted(set(params) - valid)
    if unknown:
        raise TypeError(
            f"sklearn.{wrapper_name}: unknown parameter(s) {unknown}. "
            f"Valid parameters: {sorted(valid)}"
        )
    return est_cls(**params)


def _json_type_from_default(default: Any) -> Optional[str]:
    """Infer a JSON Schema type from a default value.

    Parameters
    ----------
    default : Any
        The parameter's default value.

    Returns
    -------
    json_type : str or None
        A JSON Schema type name, or None when undeterminable.
    """
    # bool before int: bool is a subclass of int in Python.
    if isinstance(default, bool):
        return "boolean"
    if isinstance(default, int):
        return "integer"
    if isinstance(default, float):
        return "number"
    if isinstance(default, str):
        return "string"
    if isinstance(default, (list, tuple)):
        return "array"
    return None


def _fragment_from_constraints(constraints: Any) -> Dict[str, Any]:
    """Translate a scikit-learn ``_parameter_constraints`` entry to JSON Schema.

    Parameters
    ----------
    constraints : list
        The constraint list scikit-learn declares for one parameter.

    Returns
    -------
    fragment : dict
        Partial JSON Schema (``type``, and ``enum`` / ``minimum`` / ``maximum``
        where scikit-learn declares them).
    """
    import numbers

    from sklearn.utils._param_validation import Interval, Options, StrOptions

    fragment: Dict[str, Any] = {}
    for constraint in constraints:
        if isinstance(constraint, StrOptions):
            fragment.setdefault("type", "string")
            fragment["enum"] = sorted(constraint.options)
        elif isinstance(constraint, Options):
            values = [v for v in constraint.options if isinstance(v, (str, int, float))]
            if values:
                fragment["enum"] = sorted(values, key=str)
        elif isinstance(constraint, Interval):
            fragment.setdefault(
                "type",
                "integer" if constraint.type is numbers.Integral else "number",
            )
            if constraint.left is not None:
                fragment["minimum"] = constraint.left
            if constraint.right is not None:
                fragment["maximum"] = constraint.right
        elif constraint == "boolean":
            fragment.setdefault("type", "boolean")
    return fragment


def derive_schema(
    target: str,
    highlight: Tuple[str, ...] = (),
    exclude: Tuple[str, ...] = (),
) -> Dict[str, Any]:
    """Derive a JSON Schema for an estimator's constructor parameters.

    Reads the live scikit-learn signature and, when available, the estimator's
    ``_parameter_constraints`` metadata — which supplies real enums and numeric
    bounds. Because it is derived rather than hand-written, the schema cannot
    drift from the installed scikit-learn version.

    Parameters
    ----------
    target : str
        Estimator reference as ``"module:ClassName"``.
    highlight : tuple of str, optional
        Parameters to place first in the result.
    exclude : tuple of str, optional
        Parameters to omit, in addition to :data:`COMMON_EXCLUDE`.

    Returns
    -------
    schema : dict
        Mapping of parameter name to JSON Schema fragment. Falls back to the
        highlighted names alone when scikit-learn is not installed.
    """
    try:
        est_cls = resolve_target(target)
    except ImportError:
        # scikit-learn absent: registration and introspection must still work.
        return {name: {} for name in highlight}

    omit = set(COMMON_EXCLUDE) | set(exclude)
    # scikit-learn declares these privately; absence must degrade, not raise.
    constraints = getattr(est_cls, "_parameter_constraints", {}) or {}
    signature = inspect.signature(est_cls.__init__)

    schema: Dict[str, Any] = {}
    for name, parameter in signature.parameters.items():
        if name == "self" or name in omit:
            continue
        if parameter.kind in (parameter.VAR_KEYWORD, parameter.VAR_POSITIONAL):
            continue
        default = (
            None
            if parameter.default is inspect.Parameter.empty
            else parameter.default
        )
        entry: Dict[str, Any] = {}
        if name in constraints:
            try:
                entry.update(_fragment_from_constraints(constraints[name]))
            except Exception:
                # Private scikit-learn API: fall back to default-based typing.
                pass
        if "type" not in entry:
            json_type = _json_type_from_default(default)
            if json_type:
                entry["type"] = json_type
        entry["default"] = default
        schema[name] = entry

    # Highlighted parameters first, remaining ones in signature order.
    ordered = {name: schema[name] for name in highlight if name in schema}
    ordered.update({k: v for k, v in schema.items() if k not in ordered})
    return ordered
