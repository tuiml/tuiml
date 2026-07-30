"""Pipelines — an ordered list of steps ending in a model.

A :class:`Workflow` chains transformation steps and a final model into one
object that behaves like a model itself: ``fit``, ``predict``, ``score``,
``save``. Because the whole pipeline is a single estimator, the fitted
transformations always travel with the model, so inference can never
accidentally skip them.

Steps may be written three ways — a class name, a spec dict, or a configured
instance — and the three can be mixed freely::

    Workflow(["StandardScaler", "NaiveBayesClassifier"])
    Workflow([{"name": "PCAExtractor", "params": {"n_components": 5}}, "SVC"])
    Workflow([StandardScaler(), RandomForestClassifier(n_estimators=100)])

This is what lets :func:`tuiml.train` accept a JSON spec and this module accept
imported classes while sharing one execution path.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from tuiml.base.algorithms import Algorithm, call_metric

# Tabular file formats whose loaders accept a target column and otherwise fall
# back to the *last* column. ARFF is intentionally excluded: it declares its
# class attribute in the file metadata, so no target guessing (or warning) is
# needed for it.
_TABULAR_TARGET_FORMATS = {
    ".csv", ".tsv", ".parquet", ".pq", ".xlsx", ".xls",
    ".json", ".jsonl", ".ndjson",
}


# =============================================================================
# Helpers
# =============================================================================

def _clone_estimator(prototype):
    """Return a fresh, unfitted copy of an estimator instance.

    Every fit gets its own copy, so a cross-validation fold can never inherit
    state from a previous one and the instance the caller passed in is never
    itself mutated.

    Parameters
    ----------
    prototype : object
        An estimator or transformer instance to clone.

    Returns
    -------
    object
        A fresh copy in the same configuration as ``prototype``.

    Raises
    ------
    TypeError
        If the copy cannot be built. This is deliberately loud: silently
        returning the original would make every fold share one fitted
        object, quietly invalidating the scores.
    """
    if not hasattr(prototype, "get_params"):
        # Custom components need only fit/transform; without get_params there
        # is nothing to copy from, so reuse it and let the caller's own
        # convention govern state.
        return prototype

    import inspect

    try:
        # Composites (a nested Workflow, On) expose nested ``a__b`` keys in
        # their deep params, which are not constructor arguments.
        params = prototype.get_params(deep=False)
    except TypeError:
        params = prototype.get_params()

    cls = type(prototype)
    try:
        signature = inspect.signature(cls.__init__)
        accepts_kwargs = any(
            p.kind is inspect.Parameter.VAR_KEYWORD
            for p in signature.parameters.values()
        )
        if not accepts_kwargs:
            params = {k: v for k, v in params.items() if k in signature.parameters}
    except (TypeError, ValueError):
        pass

    try:
        return cls(**params)
    except Exception as exc:
        raise TypeError(
            f"Could not create a fresh copy of {cls.__name__}: {exc}. Its "
            f"__init__ must accept the parameters reported by get_params(), so "
            f"each fit can start from a clean instance."
        ) from exc


def _inject_seed(model_cls, params: dict, seed: Optional[int]) -> dict:
    """Add ``random_seed``/``random_state`` to ``params`` when the class takes one.

    Parameters
    ----------
    model_cls : type
        The class about to be instantiated.
    params : dict
        Constructor parameters collected so far.
    seed : int or None
        Seed to inject. ``None`` leaves ``params`` untouched.

    Returns
    -------
    dict
        Parameters, with a seed added only if the constructor accepts one and
        the caller did not already choose one.

    Notes
    -----
    A parameter present but set to ``None`` counts as unset — that is the
    default for seed arguments, and ``get_params()`` reports every constructor
    parameter, not only the ones the caller passed.
    """
    if seed is None:
        return params

    import inspect
    params = dict(params)
    try:
        accepted = inspect.signature(model_cls.__init__).parameters
    except Exception:
        return params

    for key in ("random_seed", "random_state"):
        if key in accepted:
            if params.get(key) is None:
                params[key] = seed
            break
    return params


_MISSING = object()


def _same_value(a, b) -> bool:
    """Compare two parameter values, tolerating arrays.

    ``==`` on a numpy array yields an array rather than a bool, so a plain
    comparison would raise when a component takes an array parameter.

    Parameters
    ----------
    a, b : object
        Values to compare.

    Returns
    -------
    bool
        Whether the two values are equal.
    """
    if b is _MISSING:
        return False
    try:
        return bool(np.asarray(a == b).all())
    except Exception:
        return a is b


def _is_serializable(value) -> bool:
    """Whether a parameter value survives a round-trip through JSON.

    :meth:`Workflow.to_config` promises a spec that can be written to disk and
    replayed, so callables, class objects, and estimator instances are dropped
    rather than embedded.

    Parameters
    ----------
    value : object
        A parameter value.

    Returns
    -------
    bool
        Whether the value is JSON-serializable.
    """
    import json

    try:
        json.dumps(value)
    except (TypeError, ValueError):
        return False
    return True


def _resolve_component(name: str):
    """Look up a component class by name.

    Parameters
    ----------
    name : str
        Class name, e.g. ``"StandardScaler"`` or ``"sklearn.SVC"``.

    Returns
    -------
    type
        The component class.

    Raises
    ------
    ValueError
        If no component with that name is registered or importable.
    """
    from tuiml.hub import registry

    try:
        return registry.get(name)
    except KeyError:
        pass

    # Modules that register lazily may not have been imported yet.
    for module_name in ("tuiml.algorithms", "tuiml.preprocessing", "tuiml.features"):
        __import__(module_name)
    try:
        return registry.get(name)
    except KeyError:
        pass

    from tuiml import preprocessing
    from tuiml.features import selection, extraction, generation
    for module in (preprocessing, selection, extraction, generation):
        cls = getattr(module, name, None)
        if cls is not None:
            return cls

    raise ValueError(
        f"Unknown component '{name}'. Use tuiml.search_algorithms('{name}') to "
        f"find the right name, or pass the class/instance directly."
    )


def _build_step(spec: Any, seed: Optional[int] = None):
    """Turn a step spec into a component instance.

    Parameters
    ----------
    spec : str, dict, type, or object
        ``"StandardScaler"``, ``{"name": ..., "params": {...}}``, a class, or
        an already-configured instance (returned as-is).
    seed : int, optional
        Seed to inject when the component accepts one.

    Returns
    -------
    object
        A component instance.

    Raises
    ------
    ValueError
        If a dict spec has no ``"name"``, or carries unexpected keys.
    """
    if isinstance(spec, str):
        cls, params = _resolve_component(spec), {}
    elif isinstance(spec, dict):
        name = spec.get("name")
        if not name:
            raise ValueError(
                'A step spec dict needs a "name" key, e.g. '
                '{"name": "StandardScaler", "params": {...}}.'
            )
        extra = set(spec) - {"name", "params"}
        if extra:
            raise ValueError(
                f"Unexpected keys {sorted(extra)} in the spec for '{name}'. "
                f'Parameters belong inside "params": '
                f'{{"name": "{name}", "params": {{...}}}}.'
            )
        params = spec.get("params") or {}
        if not isinstance(params, dict):
            raise ValueError(f'The "params" value for \'{name}\' must be a dict.')
        cls = _resolve_component(name)
    elif isinstance(spec, type):
        cls, params = spec, {}
    else:
        return spec  # already an instance

    return cls(**_inject_seed(cls, params, seed))


def _fit_transform(step, X, y):
    """Fit a transformer on ``(X, y)`` and return the transformed data.

    Components may implement ``fit_transform`` directly, or the plain
    ``fit`` + ``transform`` pair — both are accepted as steps.

    Parameters
    ----------
    step : object
        The transformer to fit.
    X : array-like of shape (n_samples, n_features)
        Input data.
    y : array-like of shape (n_samples,) or None
        Target values, forwarded when available.

    Returns
    -------
    array-like
        The transformed data (or an ``(X, y)`` tuple for steps that reshape
        both).
    """
    if hasattr(step, "fit_transform"):
        return step.fit_transform(X, y)
    step.fit(X, y)
    return step.transform(X)


def _auto_name(component: Any, taken: Dict[str, int]) -> str:
    """Derive a step name from a component's class name.

    Follows the ``make_pipeline`` convention — the lowercased class name, with
    a numeric suffix when the same class appears more than once.

    Parameters
    ----------
    component : object
        The component to name.
    taken : dict
        Counter of names already used, mutated in place.

    Returns
    -------
    str
        A unique step name, e.g. ``"standardscaler"`` or ``"on-2"``.
    """
    base = type(component).__name__.lower()
    count = taken.get(base, 0)
    taken[base] = count + 1
    return base if count == 0 else f"{base}-{count + 1}"


def _numeric_mask(X: np.ndarray) -> np.ndarray:
    """Return a boolean mask marking which columns hold numbers.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Feature matrix, possibly of ``object`` dtype with mixed columns.

    Returns
    -------
    np.ndarray of shape (n_features,)
        ``True`` where the column is numeric.
    """
    X = np.asarray(X)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if np.issubdtype(X.dtype, np.number):
        return np.ones(X.shape[1], dtype=bool)

    mask = np.zeros(X.shape[1], dtype=bool)
    for j in range(X.shape[1]):
        column = X[:, j]
        try:
            # A column counts as numeric only if every non-missing value casts.
            values = column[~pd.isna(column)]
            values.astype(float)
            mask[j] = True
        except (TypeError, ValueError):
            mask[j] = False
    return mask


# =============================================================================
# Column routing
# =============================================================================

class On:
    """Apply a transformer to a subset of columns.

    Mixed-type tables need different treatment per column — scale the numbers,
    encode the categories. ``On`` wraps a transformer so it sees only the
    columns you name, and is itself an ordinary pipeline step::

        Workflow([
            On("number", StandardScaler()),
            On("category", OneHotEncoder()),
            RandomForestClassifier(),
        ])

    Transformed columns come first in the output, followed by any columns not
    selected (unless ``remainder="drop"``).

    Parameters
    ----------
    columns : str, list of str, list of int, or callable
        Which columns to route:

        - ``"number"`` — every numeric column, detected from the data
        - ``"category"`` — every non-numeric column
        - ``["age", "income"]`` — these named columns
        - ``[0, 3]`` — these column positions
        - a callable taking the feature-name list and returning names/indices

    transformer : object
        Any component with ``fit_transform``/``transform``, given as an
        instance, class name, or spec dict.
    remainder : {"passthrough", "drop"}, default="passthrough"
        What to do with the columns that were not selected.

    Attributes
    ----------
    columns_ : list of int
        Positions of the selected columns, resolved at fit time.
    transformer_ : object
        The fitted transformer.

    Notes
    -----
    Selecting by *name* requires the incoming column names, which are known
    only for the pipeline's original input. Put name-based ``On`` steps first,
    before anything that changes the column layout; positional and
    type-based selection work anywhere.

    Examples
    --------
    >>> On("number", StandardScaler())                      # doctest: +SKIP
    >>> On(["age", "fare"], "SimpleImputer")                # doctest: +SKIP
    """

    def __init__(self, columns, transformer, remainder: str = "passthrough"):
        if remainder not in ("passthrough", "drop"):
            raise ValueError(
                f'remainder must be "passthrough" or "drop", got {remainder!r}.'
            )
        self.columns = columns
        self.transformer = transformer
        self.remainder = remainder
        self._feature_names: Optional[List[str]] = None

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Return the constructor parameters.

        Parameters
        ----------
        deep : bool, default=True
            Included for API symmetry; ``On`` has no nested params to expose.

        Returns
        -------
        dict
            The ``columns``, ``transformer``, and ``remainder`` parameters.
        """
        return {
            "columns": self.columns,
            "transformer": self.transformer,
            "remainder": self.remainder,
        }

    def _bind_feature_names(self, names: Optional[List[str]]) -> None:
        """Record the incoming column names so names can be resolved to positions.

        Parameters
        ----------
        names : list of str or None
            Column names of the data this step will receive.
        """
        self._feature_names = list(names) if names is not None else None

    def _resolve_columns(self, X: np.ndarray) -> List[int]:
        """Resolve the ``columns`` specification to column positions.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            The data this step is about to transform.

        Returns
        -------
        list of int
            Positions of the selected columns.

        Raises
        ------
        ValueError
            If names are requested but unknown, or a name is not present.
        """
        columns = self.columns
        if callable(columns):
            columns = columns(self._feature_names)

        if isinstance(columns, str):
            if columns in ("number", "numeric"):
                return list(np.flatnonzero(_numeric_mask(X)))
            if columns in ("category", "categorical"):
                return list(np.flatnonzero(~_numeric_mask(X)))
            columns = [columns]

        resolved = []
        for column in columns:
            if isinstance(column, str):
                if self._feature_names is None:
                    raise ValueError(
                        f"Cannot select column {column!r} by name: the column "
                        f"names of this step's input are unknown. Put "
                        f"name-based On steps first in the pipeline, or select "
                        f'by position or by "number"/"category".'
                    )
                if column not in self._feature_names:
                    raise ValueError(
                        f"Column {column!r} not found in {self._feature_names}."
                    )
                resolved.append(self._feature_names.index(column))
            else:
                resolved.append(int(column))
        return resolved

    def fit_transform(self, X, y=None):
        """Fit the wrapped transformer on the selected columns and transform.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.
        y : array-like of shape (n_samples,), optional
            Target values, forwarded to the wrapped transformer.

        Returns
        -------
        np.ndarray
            Transformed selected columns, followed by the passthrough columns.
        """
        X = np.asarray(X)
        self.columns_ = self._resolve_columns(X)
        self.remainder_ = (
            [j for j in range(X.shape[1]) if j not in set(self.columns_)]
            if self.remainder == "passthrough" else []
        )
        # Always fit a copy, so repeated fits stay independent of each other.
        self.transformer_ = _clone_estimator(_build_step(self.transformer))
        transformed = _fit_transform(self.transformer_, X[:, self.columns_], y)
        return self._concat(transformed, X)

    def transform(self, X):
        """Transform new data with the already-fitted transformer.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        np.ndarray
            Transformed selected columns, followed by the passthrough columns.
        """
        if not hasattr(self, "transformer_"):
            raise RuntimeError("This On step is not fitted yet. Call fit_transform first.")
        X = np.asarray(X)
        return self._concat(self.transformer_.transform(X[:, self.columns_]), X)

    def _concat(self, transformed, X):
        """Join transformed columns with the untouched passthrough columns.

        Parameters
        ----------
        transformed : array-like
            Output of the wrapped transformer.
        X : np.ndarray
            The original input, source of the passthrough columns.

        Returns
        -------
        np.ndarray
            The combined matrix, cast to float when every value is numeric.
        """
        transformed = np.asarray(transformed)
        if transformed.ndim == 1:
            transformed = transformed.reshape(-1, 1)
        if self.remainder_:
            transformed = np.hstack([transformed, X[:, self.remainder_]])
        if not np.issubdtype(transformed.dtype, np.number):
            try:
                transformed = transformed.astype(float)
            except (TypeError, ValueError):
                pass
        return transformed

    def _tuiml_visual_block_(self):
        """Return the diagram layout: one branch per routed column group."""
        from tuiml.utils.html_repr import VisualBlock

        transformer = getattr(self, "transformer_", None) or _build_step(self.transformer)
        # Each branch is captioned by the columns it receives, so the diagram
        # reads as "these columns go through this transformer".
        branches = [transformer]
        names = [repr(self.columns)]
        details = [f"columns: {self.columns!r}"]
        if self.remainder == "passthrough":
            branches.append(None)
            names.append("remainder")
            details.append("every other column, unchanged")
        return VisualBlock("parallel", branches, names=names, details=details,
                           title="On", framed=True)

    def __repr__(self):
        return (
            f"On({self.columns!r}, {self.transformer!r}"
            + (f", remainder={self.remainder!r})" if self.remainder != "passthrough" else ")")
        )


# =============================================================================
# Workflow
# =============================================================================

class Workflow(Algorithm):
    """An ordered pipeline of steps ending in a model.

    A ``Workflow`` *is* a model: it exposes the same ``fit``/``predict``/
    ``score``/``save`` interface as any single algorithm, so it can be used
    anywhere one can — including as a step inside another ``Workflow``, or as
    an entry in :func:`tuiml.experiment`.

    Every step except the last transforms the data; the last one is the model.

    Parameters
    ----------
    steps : list, optional
        The pipeline, in execution order. Each element is a component — a
        class name, a ``{"name": ..., "params": {...}}`` spec dict, a class,
        or a configured instance — or an explicit ``(name, component)`` tuple
        when you want to choose the step's name. Names are otherwise derived
        from the class name, lowercased.

    Attributes
    ----------
    steps_ : list
        The fitted transformation steps. Set by :meth:`fit`.
    model_ : object
        The fitted final model. Set by :meth:`fit`.
    metrics_ : dict or None
        Held-out scores, when :meth:`fit` was given ``cv`` or ``test_size``.
    cv_results_ : dict or None
        Per-fold scores from cross-validation.
    predictions_ : np.ndarray or None
        Predictions on the held-out split.
    feature_names_in_ : list of str or None
        Column names of the training data, when known.
    metadata_ : dict
        Details of the run — algorithm name, step names, evaluation method.

    Notes
    -----
    :meth:`fit` always leaves the pipeline fitted on **all** the data it was
    given. Passing ``cv`` or ``test_size`` additionally measures held-out
    performance into :attr:`metrics_` before that final fit.

    See Also
    --------
    :func:`tuiml.train` : The same engine driven by a single spec dict.
    :class:`On` : Apply a step to a subset of columns.

    Examples
    --------
    Strings — no imports needed:

    >>> wf = Workflow(["StandardScaler", "NaiveBayesClassifier"])
    >>> wf = wf.fit("iris", test_size=0.2)              # doctest: +SKIP
    >>> wf.metrics_                                      # doctest: +SKIP
    {'accuracy_score': 0.967, 'f1_score': 0.947}

    Instances — full editor autocomplete:

    >>> from tuiml.preprocessing import StandardScaler          # doctest: +SKIP
    >>> from tuiml.algorithms.trees import RandomForestClassifier   # doctest: +SKIP
    >>> wf = Workflow([StandardScaler(), RandomForestClassifier(n_estimators=100)])
    ...                                                  # doctest: +SKIP
    >>> wf.fit(X_train, y_train).predict(X_test)         # doctest: +SKIP

    Mixed-type table, cross-validated:

    >>> wf = Workflow([                                  # doctest: +SKIP
    ...     On("number", "SimpleImputer"),
    ...     On("category", "OneHotEncoder"),
    ...     "RandomForestClassifier",
    ... ])
    >>> wf.fit("sales.csv", target="label", cv=10)       # doctest: +SKIP
    """

    def __init__(self, steps: Optional[List[Any]] = None):
        self.steps = list(steps) if steps else []
        self._named_steps = self._normalize(self.steps)
        if self._named_steps:
            self._validate()

    # ----- construction -------------------------------------------------

    @staticmethod
    def _normalize(steps: List[Any]) -> List[Tuple[str, Any]]:
        """Turn the ``steps`` argument into a list of ``(name, instance)`` pairs.

        Parameters
        ----------
        steps : list
            Raw steps as passed to the constructor.

        Returns
        -------
        list of tuple
            ``(name, component_instance)`` in execution order.
        """
        normalized: List[Tuple[str, Any]] = []
        taken: Dict[str, int] = {}
        for step in steps:
            if isinstance(step, tuple) and len(step) == 2 and isinstance(step[0], str):
                name, component = step[0], _build_step(step[1])
                taken[name] = taken.get(name, 0) + 1
            else:
                component = _build_step(step)
                name = _auto_name(component, taken)
            normalized.append((name, component))
        return normalized

    def _validate(self) -> None:
        """Check that every step can transform and the final step can predict.

        Raises
        ------
        TypeError
            If a step cannot transform, or the final step cannot predict.
        """
        *transformers, (final_name, final) = self._named_steps
        for name, component in transformers:
            transforms = hasattr(component, "fit_transform") or (
                hasattr(component, "fit") and hasattr(component, "transform")
            )
            if not (transforms or hasattr(component, "fit_resample")):
                raise TypeError(
                    f"Step '{name}' ({type(component).__name__}) cannot transform "
                    f"data. Every step but the last needs fit_transform() (or "
                    f"fit_resample()); the model goes last."
                )
        if not (hasattr(final, "fit") and hasattr(final, "predict")):
            raise TypeError(
                f"The last step '{final_name}' ({type(final).__name__}) must be a "
                f"model with fit() and predict(). Add the model at the end of the "
                f"steps list."
            )

    # ----- introspection ------------------------------------------------

    @property
    def named_steps(self) -> Dict[str, Any]:
        """Steps by name, e.g. ``wf.named_steps["standardscaler"]``."""
        return dict(self._named_steps)

    @property
    def model(self) -> Any:
        """The final step — the model, unfitted unless :meth:`fit` has run."""
        if not self._named_steps:
            return None
        return self._named_steps[-1][1]

    @property
    def transformers(self) -> List[Tuple[str, Any]]:
        """The ``(name, component)`` pairs before the model."""
        return list(self._named_steps[:-1])

    @property
    def _estimator_type(self) -> Optional[str]:
        """The final model's estimator type — a pipeline's task is its model's.

        Tooling (``experiment()``, wrapper adapters) uses this to tell
        supervised models from clusterers. Without it a pipeline would be
        judged by its own ``fit`` signature, where ``y`` is optional, and be
        mistaken for a clusterer.
        """
        from tuiml.hub import ComponentType

        final = self.model
        if final is None:
            return None
        declared = getattr(final, "_estimator_type", None)
        if declared:
            return declared
        return {
            ComponentType.CLASSIFIER: "classifier",
            ComponentType.REGRESSOR: "regressor",
            ComponentType.CLUSTERER: "clusterer",
        }.get(getattr(final, "_component_type", None), "classifier")

    def __len__(self):
        return len(self._named_steps)

    def __getitem__(self, key):
        """Index by position (``wf[0]``, ``wf[-1]``) or by step name."""
        if isinstance(key, str):
            return self.named_steps[key]
        if isinstance(key, slice):
            return Workflow(self._named_steps[key])
        return self._named_steps[key][1]

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Return the pipeline's parameters.

        Parameters
        ----------
        deep : bool, default=True
            When True, also return each step's parameters under
            ``"<step_name>__<param>"`` keys.

        Returns
        -------
        dict
            The parameters.
        """
        params: Dict[str, Any] = {"steps": self.steps}
        if deep:
            for name, component in self._named_steps:
                if hasattr(component, "get_params"):
                    try:
                        nested = component.get_params(deep=False)
                    except TypeError:
                        nested = component.get_params()
                    for key, value in nested.items():
                        params[f"{name}__{key}"] = value
        return params

    def set_params(self, **params) -> "Workflow":
        """Set parameters, including a step's own via ``step__param``.

        Parameters
        ----------
        **params
            ``steps=[...]`` to replace the pipeline, or
            ``standardscaler__with_mean=False`` to reach into a step.

        Returns
        -------
        self : Workflow

        Raises
        ------
        ValueError
            If a step name is unknown or a key is not a pipeline parameter.

        Examples
        --------
        >>> wf = Workflow(["PCAExtractor", "NaiveBayesClassifier"])
        >>> _ = wf.set_params(pcaextractor__n_components=3)
        """
        if "steps" in params:
            self.steps = list(params.pop("steps"))
            self._named_steps = self._normalize(self.steps)
            if self._named_steps:
                self._validate()

        named = self.named_steps
        for key, value in params.items():
            name, sep, param = key.partition("__")
            if not sep:
                raise ValueError(
                    f"'{key}' is not a Workflow parameter. Use "
                    f"'<step>__<param>' to set a step's parameter, e.g. "
                    f"'{next(iter(named), 'step')}__some_param'."
                )
            if name not in named:
                raise ValueError(
                    f"Unknown step '{name}'. Available steps: {sorted(named)}."
                )
            named[name].set_params(**{param: value})
        return self

    # ----- fitting ------------------------------------------------------

    def fit(
        self,
        data=None,
        y=None,
        *,
        target=None,
        features: Optional[List[str]] = None,
        cv: Optional[int] = None,
        test_size: Optional[float] = None,
        stratify: bool = True,
        metrics: Union[str, List[str]] = "auto",
        random_seed: Optional[int] = None,
    ) -> "Workflow":
        """Fit the pipeline, optionally measuring held-out performance first.

        Parameters
        ----------
        data : str, DataFrame, ndarray, or Dataset, optional
            Training data. A string is a file path (csv, arff, parquet, json,
            excel — auto-detected) or a builtin dataset name such as
            ``"iris"``. Arrays and DataFrames are used directly.
        y : array-like of shape (n_samples,), optional
            Target values, for the ``fit(X, y)`` form.
        target : str or array-like, optional
            Target column name when ``data`` is a file or DataFrame, or a
            separate target array.
        features : list of str, optional
            Restrict the feature matrix to these named columns.
        cv : int, optional
            Number of cross-validation folds. When given, scores are collected
            per fold into :attr:`metrics_` and :attr:`cv_results_`.
        test_size : float, optional
            Hold out this fraction to score into :attr:`metrics_`. Ignored
            when ``cv`` is given.
        stratify : bool, default=True
            Preserve class balance in the holdout split.
        metrics : str or list of str, default="auto"
            Metric function names from ``tuiml.evaluation.metrics``.
            ``"auto"`` picks metrics that suit the model's task.
        random_seed : int, optional
            Seed for splits, folds, and any step that accepts one. Falls back
            to the global seed, then 42.

        Returns
        -------
        self : Workflow
            Fitted on all the data given, with results on :attr:`metrics_`.

        Raises
        ------
        ValueError
            If no data or no steps were provided.

        Examples
        --------
        >>> wf = Workflow(["StandardScaler", "NaiveBayesClassifier"])
        >>> _ = wf.fit("iris", cv=5)                      # doctest: +SKIP
        >>> wf.metrics_["cv_accuracy_score_mean"]         # doctest: +SKIP
        0.953
        """
        if not self._named_steps:
            raise ValueError(
                "This Workflow has no steps. Pass them to the constructor: "
                'Workflow(["StandardScaler", "RandomForestClassifier"]).'
            )

        X, y, feature_names = self._load(data, y, target, features)
        seed = self._resolve_seed(random_seed)
        task = self._task()
        requested = self._requested_metrics(metrics, task)

        self.feature_names_in_ = feature_names
        self.metrics_ = None
        self.cv_results_ = None
        self.predictions_ = None
        evaluation = None

        if task in ("clusterer", "anomaly", "timeseries"):
            # These tasks score on the data they were fitted on: clusterers and
            # anomaly detectors have no held-out notion of correctness, and a
            # time series must stay in order.
            self.metrics_ = self._fit_unsupervised(X, y, task, requested, seed)
            evaluation = task
        elif cv:
            self.metrics_, self.cv_results_ = self._cross_validate(
                X, y, cv, requested, seed
            )
            evaluation = "cross_validate"
        elif test_size:
            self.metrics_, self.predictions_ = self._holdout(
                X, y, test_size, stratify, requested, seed, feature_names
            )
            evaluation = "holdout"

        if task not in ("clusterer", "anomaly", "timeseries"):
            self.steps_, self.model_ = self._fit_steps(X, y, seed, feature_names)

        self.metadata_ = {
            "algorithm": type(self.model).__name__,
            "steps": [name for name, _ in self._named_steps],
            "evaluation_method": evaluation,
            "n_samples": len(X),
        }
        return self

    def _fit_steps(self, X, y, seed, feature_names=None):
        """Fit every transformation step, then the model, on the given data.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Features.
        y : np.ndarray of shape (n_samples,) or None
            Targets.
        seed : int or None
            Seed injected into components that accept one.
        feature_names : list of str, optional
            Column names, bound to column-aware steps such as :class:`On`.

        Returns
        -------
        fitted_steps : list
            The fitted transformation steps, in order.
        model : object
            The fitted final model.
        """
        X_current, y_current = X, y
        fitted_steps = []

        for _, prototype in self.transformers:
            step = _clone_estimator(_inject_seed_into(prototype, seed))
            if hasattr(step, "_bind_feature_names"):
                step._bind_feature_names(feature_names)
            if hasattr(step, "fit_resample") and y_current is not None:
                X_current, y_current = step.fit_resample(X_current, y_current)
            else:
                transformed = _fit_transform(step, X_current, y_current)
                if isinstance(transformed, tuple):
                    X_current, y_current = transformed
                else:
                    X_current = transformed
            fitted_steps.append(step)

        model = _clone_estimator(_inject_seed_into(self.model, seed))
        if y_current is None:
            model.fit(X_current)
        else:
            model.fit(X_current, y_current)
        return fitted_steps, model

    @staticmethod
    def _apply_steps(X, fitted_steps):
        """Transform data through already-fitted steps.

        Resamplers are skipped: they reshape the training set only and must
        never touch validation or inference data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to transform.
        fitted_steps : list
            Steps previously fitted by :meth:`_fit_steps`.

        Returns
        -------
        np.ndarray
            The transformed data.
        """
        X_current = X
        for step in fitted_steps:
            if hasattr(step, "fit_resample"):
                continue
            transformed = step.transform(X_current)
            X_current = transformed[0] if isinstance(transformed, tuple) else transformed
        return X_current

    # ----- evaluation paths ---------------------------------------------

    def _cross_validate(self, X, y, cv, requested, seed):
        """Score the pipeline with k-fold cross-validation.

        Each fold refits the whole pipeline from scratch, so no transformation
        ever sees its validation fold.

        Parameters
        ----------
        X, y : np.ndarray
            The full dataset.
        cv : int
            Number of folds.
        requested : list of str
            Metric function names.
        seed : int or None
            Seed for the fold split.

        Returns
        -------
        metrics : dict
            ``cv_<metric>_mean`` / ``cv_<metric>_std`` per metric.
        cv_results : dict
            Raw per-fold scores.
        """
        from tuiml.evaluation.splitting import KFold

        kfold = KFold(n_splits=cv, shuffle=True, random_state=seed)
        scores = {m: [] for m in requested if self._metric_func(m) is not None}

        for train_idx, val_idx in kfold.split(X, y):
            fitted_steps, model = self._fit_steps(
                X[train_idx], y[train_idx], seed, self.feature_names_in_
            )
            predictions = model.predict(self._apply_steps(X[val_idx], fitted_steps))
            for name in scores:
                try:
                    scores[name].append(
                        call_metric(self._metric_func(name), y[val_idx], predictions)
                    )
                except Exception:
                    pass

        metrics = {}
        for name, fold_scores in scores.items():
            if fold_scores:
                metrics[f"cv_{name}_mean"] = float(np.mean(fold_scores))
                metrics[f"cv_{name}_std"] = float(np.std(fold_scores))
        return metrics, {"scores": scores}

    def _holdout(self, X, y, test_size, stratify, requested, seed, feature_names):
        """Score the pipeline on a single held-out split.

        Parameters
        ----------
        X, y : np.ndarray
            The full dataset.
        test_size : float
            Fraction held out.
        stratify : bool
            Preserve class balance in the split.
        requested : list of str
            Metric function names.
        seed : int or None
            Seed for the split.
        feature_names : list of str or None
            Column names for column-aware steps.

        Returns
        -------
        metrics : dict
            Metric name to value.
        predictions : np.ndarray
            Predictions on the held-out portion.
        """
        from tuiml.evaluation.splitting import train_test_split

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=seed,
            stratify=y if (stratify and y is not None) else None,
        )
        fitted_steps, model = self._fit_steps(X_train, y_train, seed, feature_names)
        predictions = model.predict(self._apply_steps(X_test, fitted_steps))

        metrics = {}
        for name in requested:
            func = self._metric_func(name)
            if func is None:
                continue
            try:
                metrics[name] = float(call_metric(func, y_test, predictions))
            except Exception as exc:
                metrics[f"{name}_error"] = str(exc)
        return metrics, predictions

    def _fit_unsupervised(self, X, y, task, requested, seed):
        """Fit and score a clusterer, anomaly detector, or forecaster.

        Parameters
        ----------
        X, y : np.ndarray
            The dataset. ``y`` is unused for clustering and anomaly detection.
        task : {"clusterer", "anomaly", "timeseries"}
            Which path to take.
        requested : list of str
            Metric function names.
        seed : int or None
            Seed injected into components that accept one.

        Returns
        -------
        dict
            Metrics for the task — cluster quality scores, anomaly counts, or
            forecast errors.
        """
        metrics: Dict[str, Any] = {}

        if task == "timeseries":
            # Forecasters take a 1-D series and predict a number of steps
            # ahead, so the "split" is the tail of the series.
            series = y
            if series is None or np.unique(series).size <= 1:
                series = X[:, 0] if X.ndim == 2 else X
            split_at = max(1, int(len(series) * 0.8))
            train, test = series[:split_at], series[split_at:]

            # Forecasters consume the raw 1-D series, so transformation steps
            # (which are column-oriented) do not apply on this path.
            model = _clone_estimator(_inject_seed_into(self.model, seed))
            model.fit(train)
            if len(test):
                predictions = model.predict(len(test))
                for name in requested:
                    func = self._metric_func(name)
                    if func is None:
                        continue
                    try:
                        metrics[name] = float(func(test[:len(predictions)], predictions))
                    except Exception as exc:
                        metrics[f"{name}_error"] = str(exc)
            # Refit on the whole series so the delivered model is complete.
            self.steps_ = []
            self.model_ = _clone_estimator(_inject_seed_into(self.model, seed))
            self.model_.fit(series)
            return metrics

        self.steps_, self.model_ = self._fit_steps(X, None, seed, self.feature_names_in_)
        X_transformed = self._apply_steps(X, self.steps_)
        labels = (
            self.model_.predict(X_transformed)
            if hasattr(self.model_, "predict")
            else self.model_.labels_
        )

        if task == "anomaly":
            # Detectors label -1 for anomalies and 1 for normal points.
            metrics = {
                "n_anomalies": int((labels == -1).sum()),
                "n_normal": int((labels == 1).sum()),
                "anomaly_ratio": float((labels == -1).mean()),
            }
            if hasattr(self.model_, "decision_function"):
                scores = self.model_.decision_function(X_transformed)
                metrics["score_mean"] = float(np.mean(scores))
                metrics["score_std"] = float(np.std(scores))
            return metrics

        for name in requested:  # clustering metrics take (X, labels)
            func = self._metric_func(name)
            if func is None:
                continue
            try:
                metrics[name] = float(func(X_transformed, labels))
            except Exception as exc:
                metrics[f"{name}_error"] = str(exc)
        return metrics

    # ----- inference ----------------------------------------------------

    def predict(self, X) -> np.ndarray:
        """Predict, applying the fitted transformations first.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Raw input, in the same shape as the training data.

        Returns
        -------
        np.ndarray
            Predicted labels or values.
        """
        self._check_fitted()
        return self.model_.predict(self._apply_steps(X, self.steps_))

    def predict_proba(self, X) -> np.ndarray:
        """Predict class probabilities, applying the fitted transformations first.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Raw input.

        Returns
        -------
        np.ndarray of shape (n_samples, n_classes)
            Class probabilities.

        Raises
        ------
        AttributeError
            If the final model has no ``predict_proba``.
        """
        self._check_fitted()
        if not hasattr(self.model_, "predict_proba"):
            raise AttributeError(
                f"{type(self.model_).__name__} does not support predict_proba()."
            )
        return self.model_.predict_proba(self._apply_steps(X, self.steps_))

    def score(self, X, y) -> float:
        """Return the final model's default score on transformed data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Raw input.
        y : array-like of shape (n_samples,)
            True labels or values.

        Returns
        -------
        float
            Accuracy for classifiers, R² for regressors.
        """
        self._check_fitted()
        return self.model_.score(self._apply_steps(X, self.steps_), y)

    def evaluate(self, X, y, metrics: Union[str, List[str]] = "auto") -> Dict[str, float]:
        """Score the fitted pipeline on new data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Raw input.
        y : array-like of shape (n_samples,)
            True labels or values.
        metrics : str or list of str, default="auto"
            Metric function names, or ``"auto"`` to pick by task.

        Returns
        -------
        dict
            Metric name to value.
        """
        self._check_fitted()
        return self.model_.evaluate(self._apply_steps(X, self.steps_), y, metrics=metrics)

    def _check_fitted(self) -> None:
        """Raise a clear error if :meth:`fit` has not run yet."""
        if not hasattr(self, "model_"):
            raise RuntimeError(
                "This Workflow is not fitted yet. Call fit() before predicting."
            )

    @property
    def _is_fitted(self) -> bool:
        """Whether :meth:`fit` has completed."""
        return hasattr(self, "model_")

    # ----- data loading and task detection ------------------------------

    def _load(self, data, y, target, features):
        """Resolve the ``data`` argument into ``(X, y, feature_names)``.

        Parameters
        ----------
        data : str, DataFrame, ndarray, Dataset, or None
            The data source.
        y : array-like or None
            Targets passed positionally.
        target : str, array-like, or None
            Target column name, or a target array.
        features : list of str or None
            Column subset to keep.

        Returns
        -------
        X : np.ndarray
            Feature matrix.
        y : np.ndarray or None
            Targets.
        feature_names : list of str or None
            Column names, when known.

        Raises
        ------
        ValueError
            If no data was given, or a requested feature is missing.
        """
        import os
        import warnings
        from tuiml.datasets.loaders.arff import Dataset

        if data is None:
            raise ValueError(
                'No data provided. Pass a file path, a builtin name such as '
                '"iris", or X and y arrays.'
            )
        if y is None and target is not None and not isinstance(target, str):
            y = target

        feature_names = None

        if isinstance(data, Dataset):
            X, y_loaded, feature_names = data.X, data.y, data.feature_names
            y = y if y is not None else y_loaded

        elif isinstance(data, str):
            from tuiml.datasets import load, load_dataset

            if os.path.exists(data):
                load_kwargs = self._loader_target_kwargs(data, target)
                suffix = os.path.splitext(data)[1].lower()
                if "target_column" not in load_kwargs and suffix in _TABULAR_TARGET_FORMATS:
                    warnings.warn(
                        f"No target column specified for {data!r}; defaulting to "
                        f"the last column. Pass target=... to be explicit.",
                        stacklevel=3,
                    )
                dataset = load(data, **load_kwargs)
            else:
                dataset = load_dataset(data)
            X, feature_names = dataset.X, dataset.feature_names
            y = y if y is not None else dataset.y

        elif isinstance(data, pd.DataFrame):
            from tuiml.datasets import from_pandas

            dataset = from_pandas(
                data, target_column=target if isinstance(target, str) else None
            )
            X, feature_names = dataset.X, dataset.feature_names
            y = y if y is not None else dataset.y

        else:
            X = data if isinstance(data, np.ndarray) else np.asarray(data)
            y = np.asarray(y) if y is not None else None

        if features is not None:
            if feature_names is None:
                raise ValueError(
                    "Cannot restrict to named features: the column names of "
                    "this data are unknown."
                )
            feature_names = list(feature_names)
            missing = [f for f in features if f not in feature_names]
            if missing:
                raise ValueError(
                    f"features not found in data columns {feature_names}: {missing}"
                )
            indices = [feature_names.index(f) for f in features]
            X, feature_names = np.asarray(X)[:, indices], list(features)

        return X, y, feature_names

    @staticmethod
    def _loader_target_kwargs(path: str, target) -> Dict[str, Any]:
        """Build loader keyword arguments for a file path.

        ARFF files declare their class column in the file metadata, so an
        explicit target is rejected there rather than silently ignored.

        Parameters
        ----------
        path : str
            Path to the data file.
        target : str or None
            Requested target column.

        Returns
        -------
        dict
            Keyword arguments for :func:`tuiml.datasets.load`.

        Raises
        ------
        ValueError
            If a target column is given for an ARFF file.
        """
        import os

        if not isinstance(target, str):
            return {}
        if os.path.splitext(path)[1].lower() == ".arff":
            raise ValueError(
                f"ARFF files declare their class column in the file metadata; "
                f"remove target={target!r} when loading {os.path.basename(path)}."
            )
        return {"target_column": target}

    def _task(self) -> str:
        """Classify the final model's task, to pick metrics and a fit strategy.

        Anomaly detectors and forecasters are registered as classifiers and
        regressors, so their registry *tags* — not their component type — are
        what distinguishes them.

        Returns
        -------
        {"classifier", "regressor", "clusterer", "anomaly", "timeseries"}
            The detected task.
        """
        from tuiml.hub import ComponentType

        model = self.model
        tags = []
        try:
            tags = type(model).get_component_info().get("tags") or []
        except Exception:
            pass
        if "anomaly-detection" in tags:
            return "anomaly"
        if "timeseries" in tags:
            return "timeseries"

        component_type = getattr(model, "_component_type", None)
        estimator_type = getattr(model, "_estimator_type", None)
        if component_type == ComponentType.CLUSTERER or estimator_type == "clusterer":
            return "clusterer"
        if component_type == ComponentType.REGRESSOR or estimator_type == "regressor":
            return "regressor"
        return "classifier"

    @staticmethod
    def _requested_metrics(metrics, task: str) -> List[str]:
        """Resolve the ``metrics`` argument to a list of metric function names.

        Parameters
        ----------
        metrics : str or list of str
            Explicit names, or ``"auto"``.
        task : str
            Task from :meth:`_task`, used to choose sensible defaults.

        Returns
        -------
        list of str
            Metric function names.
        """
        if metrics is not None and metrics != "auto":
            return [metrics] if isinstance(metrics, str) else list(metrics)
        return {
            "clusterer": ["silhouette_score", "calinski_harabasz_score"],
            "anomaly": [],
            "timeseries": ["r2_score", "root_mean_squared_error", "mean_absolute_error"],
            "regressor": ["r2_score", "mean_squared_error", "mean_absolute_error"],
            "classifier": ["accuracy_score", "f1_score"],
        }[task]

    @staticmethod
    def _metric_func(name: str):
        """Look up a metric function by name.

        Parameters
        ----------
        name : str
            Function name from ``tuiml.evaluation.metrics``.

        Returns
        -------
        callable or None
            The metric function, or None when the name is unknown.
        """
        from tuiml.evaluation import metrics as metrics_module

        return getattr(metrics_module, name, None)

    @staticmethod
    def _resolve_seed(random_seed: Optional[int]) -> int:
        """Resolve the effective seed: explicit, then global, then 42.

        Parameters
        ----------
        random_seed : int or None
            Explicitly requested seed.

        Returns
        -------
        int
            The seed to use.
        """
        if random_seed is not None:
            return random_seed
        from tuiml.utils.seed import get_global_seed

        return get_global_seed() or 42

    # ----- export and display -------------------------------------------

    def to_config(self) -> Dict[str, Any]:
        """Export the pipeline as a :func:`tuiml.train` spec.

        Returns
        -------
        dict
            A spec with ``model`` and, when there are transformation steps,
            ``pipeline`` — each component as ``{"name": ..., "params": {...}}``.

        Notes
        -----
        The result is JSON-writable, which means parameters that cannot survive
        that round trip are **omitted**: a callable passed as a parameter (a
        custom ``score_func``, say) is dropped, and replaying the spec gets the
        component's default instead. Parameters left at their default are also
        omitted, to keep the spec small.

        Examples
        --------
        >>> Workflow(["StandardScaler", "NaiveBayesClassifier"]).to_config()
        {'model': {'name': 'NaiveBayesClassifier'}, 'pipeline': [{'name': 'StandardScaler'}]}
        """
        def spec(component):
            entry: Dict[str, Any] = {"name": type(component).__name__}
            if hasattr(component, "get_params"):
                try:
                    params = component.get_params(deep=False)
                except TypeError:
                    params = component.get_params()
                defaults = self._default_params(type(component))
                accepted = self._accepted_params(type(component))
                params = {
                    k: v for k, v in params.items()
                    if not k.endswith("_")
                    and (accepted is None or k in accepted)
                    and not _same_value(v, defaults.get(k, _MISSING))
                    and _is_serializable(v)
                }
                if params:
                    entry["params"] = params
            return entry

        config: Dict[str, Any] = {"model": spec(self.model)}
        if self.transformers:
            config["pipeline"] = [spec(component) for _, component in self.transformers]
        return config

    @staticmethod
    def _accepted_params(cls) -> Optional[set]:
        """Return the parameter names a class's constructor accepts.

        ``get_params()`` may report fitted state or derived values that the
        constructor would reject, which would make :meth:`to_config` emit a
        spec that cannot be replayed.

        Parameters
        ----------
        cls : type
            The component class.

        Returns
        -------
        set of str or None
            Accepted keyword names, or ``None`` when the constructor takes
            ``**kwargs`` (so anything is allowed) or cannot be inspected.
        """
        import inspect

        try:
            signature = inspect.signature(cls.__init__)
        except Exception:
            return None
        names = set()
        for name, param in signature.parameters.items():
            if param.kind is inspect.Parameter.VAR_KEYWORD:
                return None
            if name not in ("self",) and param.kind is not inspect.Parameter.VAR_POSITIONAL:
                names.add(name)
        return names

    @staticmethod
    def _default_params(cls) -> Dict[str, Any]:
        """Return a class's default constructor arguments.

        Used to keep :meth:`to_config` output small by omitting values the
        constructor would have chosen anyway.

        Parameters
        ----------
        cls : type
            The component class.

        Returns
        -------
        dict
            Parameter name to default value.
        """
        import inspect

        try:
            signature = inspect.signature(cls.__init__)
        except Exception:
            return {}
        return {
            name: param.default
            for name, param in signature.parameters.items()
            if param.default is not inspect.Parameter.empty
        }

    def _tuiml_visual_block_(self):
        """Return the diagram layout: the steps, stacked in execution order."""
        from tuiml.utils.html_repr import VisualBlock

        components = [component for _, component in self._named_steps]
        if self._is_fitted:
            components = list(self.steps_) + [self.model_]
        return VisualBlock(
            "serial",
            components,
            names=[name for name, _ in self._named_steps],
            details=[repr(component) for component in components],
            title="Workflow",
        )

    def _repr_html_(self) -> str:
        """Render the pipeline as an HTML diagram (used by Jupyter)."""
        from tuiml.utils.html_repr import component_html_repr

        return component_html_repr(self)

    def __repr__(self):
        if not self._named_steps:
            return "Workflow([])"
        components = ",\n    ".join(
            repr(component) for _, component in self._named_steps
        )
        return f"Workflow([\n    {components},\n])"


def _inject_seed_into(prototype, seed: Optional[int]):
    """Return a copy of ``prototype`` carrying ``seed``, when it accepts one.

    Parameters
    ----------
    prototype : object
        Component instance to seed.
    seed : int or None
        Seed to apply.

    Returns
    -------
    object
        ``prototype`` itself when no seed applies, otherwise a seeded copy.
    """
    if seed is None:
        return prototype
    try:
        params = prototype.get_params(deep=False)
    except TypeError:
        params = prototype.get_params()
    except Exception:
        return prototype

    seeded = _inject_seed(type(prototype), params, seed)
    changed = any(
        not _same_value(seeded.get(key, _MISSING), params.get(key, _MISSING))
        for key in ("random_seed", "random_state")
    )
    if not changed:
        return prototype
    try:
        return type(prototype)(**seeded)
    except Exception:
        return prototype
