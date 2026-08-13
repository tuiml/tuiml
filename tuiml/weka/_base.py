"""Shared machinery for the Weka bridge.

This module centralizes everything the ``tuiml.weka`` wrappers need:

* **Namespaced registration decorators** (``weka_classifier`` etc.) that register
  each wrapper into the hub under a ``weka.<ClassName>`` key, so a wrapper never
  collides with the native TuiML algorithm of the same name.
* **JVM lifecycle management**. Weka is Java, reached through
  ``python-weka-wrapper3`` and JPype. Exactly one JVM is started per process,
  on first use, and it is deliberately never stopped — see :func:`ensure_jvm`.
* **Wrapper mixins** that implement ``fit`` / ``predict`` by delegating to a
  Weka object. A concrete wrapper only declares its Weka classname, its
  constructor parameters, and how those map to Weka's command-line options.

Weka is an *optional* dependency: importing this module never requires it.
The dependency is checked at ``fit`` time and raises a clear, actionable
``ImportError`` naming the extra to install.

Notes
-----
Unlike the scikit-learn and CapyMOA bridges, this one has to bridge two type
systems. Weka works on ``Instances`` — a table that carries an ARFF header
declaring each attribute numeric or nominal — while TuiML passes plain numpy
arrays. :func:`to_instances` performs that conversion, and ``nominal_features``
is how a caller declares which columns are categorical rather than continuous.

See Also
--------
:mod:`tuiml.sklearn` : The same bridge pattern for scikit-learn estimators.
:mod:`tuiml.capymoa` : The same bridge pattern for CapyMOA streaming learners.
"""

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from tuiml.base.algorithms import classifier, regressor, clusterer

#: Hub-registry namespace prefix. The hub key for a wrapper named ``Foo`` is
#: ``"weka.Foo"`` while the Python class stays ``Foo``.
NAMESPACE = "weka"

#: pip extra that provides the backing dependency: ``pip install tuiml[weka]``.
_EXTRA = "weka"

#: Process-wide flag: the JVM has been started. Weka's JVM cannot be restarted
#: once stopped, so this is a one-way latch.
_JVM_STARTED = False


def _ensure_weka(cls_name: str) -> None:
    """Raise an actionable ``ImportError`` if python-weka-wrapper3 is missing.

    Parameters
    ----------
    cls_name : str
        Name of the wrapper class being used, shown in the error message so the
        user knows which wrapper prompted the check.
    """
    try:
        import weka.core.jvm  # noqa: F401
    except ImportError as exc:  # pragma: no cover - exercised only without weka
        raise ImportError(
            f"{cls_name} requires python-weka-wrapper3, which is not installed. "
            f"Install it with:  pip install 'tuiml[{_EXTRA}]'\n"
            "It also needs a Java runtime (11+) on PATH: check with 'java -version'."
        ) from exc


def ensure_jvm(max_heap_size: Optional[str] = None, packages: bool = False) -> None:
    """Start the Weka JVM once per process, if it is not already running.

    Parameters
    ----------
    max_heap_size : str or None, default=None
        JVM maximum heap, e.g. ``"2048m"``. ``None`` uses the JVM default.
    packages : bool, default=False
        Whether to load Weka's external package manager packages. Off by
        default: loading packages is slow and only needed for algorithms that
        live outside weka-core.

    Notes
    -----
    The JVM is started lazily and **never stopped**. This is deliberate, not an
    oversight: JPype cannot restart a JVM inside the same process once it has
    been shut down, so stopping it would make any later Weka call fail for the
    remainder of the session. The JVM is reclaimed when the process exits.

    Callers who need a non-default heap must call this **before** the first
    ``fit``, because the first call is what fixes the heap for the process.

    Examples
    --------
    >>> from tuiml.weka import ensure_jvm
    >>> ensure_jvm(max_heap_size="4096m")   # doctest: +SKIP
    """
    global _JVM_STARTED
    if _JVM_STARTED:
        return
    _ensure_weka("The Weka bridge")
    import weka.core.jvm as jvm

    if not jvm.started:
        kwargs: Dict[str, Any] = {"packages": packages}
        if max_heap_size is not None:
            kwargs["max_heap_size"] = max_heap_size
        jvm.start(**kwargs)
    _JVM_STARTED = True


def fmt_num(value: Any) -> str:
    """Format a number for a Weka command-line option.

    Parameters
    ----------
    value : int or float
        The value to render.

    Returns
    -------
    text : str
        The value as a string, with whole floats rendered without a trailing
        ``.0``.

    Notes
    -----
    Weka parses some options with ``Integer.parseInt``, which rejects ``"2.0"``
    outright (``REPTree``'s ``-M`` is one such option). Emitting ``"2"`` for any
    whole number keeps both the integer- and double-parsed options happy.

    Examples
    --------
    >>> from tuiml.weka._base import fmt_num
    >>> fmt_num(2.0), fmt_num(2), fmt_num(0.25)
    ('2', '2', '0.25')
    """
    if isinstance(value, bool):  # bool is an int subclass; guard it explicitly
        return str(int(value))
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _namespaced(base_decorator):
    """Wrap a base decorator to register under the ``weka.<ClassName>`` namespace.

    Parameters
    ----------
    base_decorator : callable
        The TuiML base decorator (:func:`tuiml.base.algorithms.classifier`,
        :func:`~tuiml.base.algorithms.regressor` or
        :func:`~tuiml.base.algorithms.clusterer`).

    Returns
    -------
    factory : callable
        A callable ``(tags=None, version="1.0.0")`` returning a decorator that
        registers the class under ``weka.<ClassName>``.
    """

    def factory(tags: Optional[List[str]] = None, version: str = "1.0.0"):
        def decorate(cls):
            key = f"{NAMESPACE}.{cls.__name__}"
            merged = list(tags or [])
            if NAMESPACE not in merged:
                merged.append(NAMESPACE)
            return base_decorator(name=key, tags=merged, version=version)(cls)

        return decorate

    return factory


weka_classifier = _namespaced(classifier)
weka_regressor = _namespaced(regressor)
weka_clusterer = _namespaced(clusterer)


def to_instances(
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    nominal_features: Optional[Sequence[int]] = None,
    nominal_target: bool = False,
    name: str = "tuiml",
):
    """Convert numpy arrays into a Weka ``Instances`` table.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Feature matrix. Nominal columns must already hold integer codes.
    y : np.ndarray of shape (n_samples,) or None, default=None
        Target values. When None the table is built with no class attribute at
        all, which is the form the clusterers need.
    nominal_features : sequence of int or None, default=None
        0-based indices of columns to declare **nominal** rather than numeric.
        Weka's tree and rule learners use their native categorical handling for
        these; leaving a categorical column numeric makes Weka treat its codes
        as an ordered scale.
    nominal_target : bool, default=False
        Declare the class attribute nominal. True for classification, False for
        regression.
    name : str, default="tuiml"
        Relation name recorded in the ARFF header.

    Returns
    -------
    data : weka.core.dataset.Instances
        Table with the class attribute set to the last column when ``y`` is
        given, and no class attribute at all when it is not.

    Notes
    -----
    When ``y`` is None no placeholder class column is added, which is what the
    clusterers need — Weka refuses to delete an attribute that is set as the
    class.
    """
    from weka.core.dataset import create_instances_from_matrices

    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    nominal_x = list(nominal_features) if nominal_features else None

    if y is None:
        data = create_instances_from_matrices(X, name=name, nominal_x=nominal_x)
        return data

    y_arr = np.asarray(y, dtype=np.float64)
    data = create_instances_from_matrices(
        X, y_arr, name=name, nominal_x=nominal_x, nominal_y=nominal_target
    )
    data.class_is_last()
    return data


def rows_to_instances(header, X: np.ndarray):
    """Build a table of rows for ``X`` that reuses an existing header.

    Parameters
    ----------
    header : weka.core.dataset.Instances
        A previously built table whose header (attribute declarations and
        nominal label sets) the new rows must conform to.
    X : np.ndarray of shape (n_samples, n_features)
        Feature values to convert.

    Returns
    -------
    data : weka.core.dataset.Instances
        Table holding one row per row of ``X``, sharing ``header``'s header.

    Notes
    -----
    Reusing the training header is what keeps train and test consistent: a
    nominal attribute keeps exactly the label set it was trained with, so
    Weka never sees a column whose categories shifted between fit and predict.

    A nominal value that did not occur during training has no index in the
    header and is therefore written as **missing** (``NaN``) rather than being
    silently coerced to some other category. Weka's learners handle missing
    values natively.
    """
    from weka.core.dataset import Instance, Instances

    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    out = Instances.template_instances(header, 0)
    n_attr = header.num_attributes
    has_class = header.class_index >= 0
    n_feat = n_attr - 1 if has_class else n_attr

    # Pre-resolve which feature columns are nominal, and their label->index maps.
    #
    # The labels are matched on their *numeric* value, not their string form.
    # Weka stores a nominal label built from a float column as "3.0" while the
    # caller passes 3.0, and ``str(int(3.0))`` is "3" — comparing strings makes
    # every lookup miss, turning the whole column into missing values and
    # silently wrecking the model. Labels that are not numeric fall back to an
    # exact string match.
    lookups: Dict[int, Dict[Any, int]] = {}
    for j in range(n_feat):
        att = header.attribute(j)
        if not att.is_nominal:
            continue
        table: Dict[Any, int] = {}
        for k in range(att.num_values):
            label = att.value(k)
            try:
                table[float(label)] = k
            except (TypeError, ValueError):
                table[label] = k
        lookups[j] = table

    for row in X:
        values = np.empty(n_attr, dtype=np.float64)
        for j in range(n_feat):
            v = row[j]
            if j in lookups:
                # Nominal: translate the raw category value to its header index.
                idx = lookups[j].get(float(v))
                values[j] = np.nan if idx is None else idx
            else:
                values[j] = v
        if has_class:
            # The true class of a prediction row is unknown; mark it missing.
            values[header.class_index] = np.nan
        out.add_instance(Instance.create_instance(values))

    if has_class:
        out.class_is_last()
    return out


class _WekaMixin:
    """Common plumbing shared by the classifier, regressor and clusterer mixins.

    Subclasses declare ``_weka_classname`` and implement :meth:`_options`.
    """

    #: Fully-qualified Weka class, e.g. ``"weka.classifiers.trees.J48"``.
    _weka_classname: str = ""
    _requires = "python-weka-wrapper3"
    _extra = _EXTRA

    def _options(self) -> List[str]:
        """Return the Weka command-line options for this configuration.

        Returns
        -------
        options : list of str
            Option tokens, e.g. ``["-C", "0.25", "-M", "2"]``.
        """
        return []

    def _resolved_options(self) -> List[str]:
        """Return :meth:`_options` plus any user-supplied ``options`` override.

        Returns
        -------
        options : list of str
            The final option list handed to Weka.
        """
        opts = list(self._options())
        extra = getattr(self, "options", None)
        if extra:
            opts.extend(list(extra))
        return opts

    def _nominal_features(self) -> Optional[Sequence[int]]:
        """Return the 0-based indices of columns to treat as nominal."""
        return getattr(self, "nominal_features", None)

    @classmethod
    def get_references(cls) -> List[str]:
        """Return the citation for the backing Weka toolkit."""
        return [
            "Hall, M., Frank, E., Holmes, G., Pfahringer, B., Reutemann, P., & "
            "Witten, I.H. (2009). The WEKA Data Mining Software: An Update. "
            "SIGKDD Explorations, 11(1), 10-18."
        ]

    def __repr__(self) -> str:
        """Return a short representation naming the backing Weka class."""
        state = "fitted" if getattr(self, "_is_fitted", False) else "not fitted"
        return f"{type(self).__name__}({self._weka_classname}, {state})"


class _WekaSupervisedMixin(_WekaMixin):
    """Adapt a Weka ``Classifier`` to TuiML's batch ``fit`` / ``predict``.

    Handles converting arrays to ``Instances``, building the Weka classifier
    with the resolved options, and decoding Weka's output back into the label
    space the caller passed in.

    Attributes
    ----------
    model_ : weka.classifiers.Classifier
        The fitted backing Weka classifier.
    header_ : weka.core.dataset.Instances
        Zero-row table holding the training header, reused so that predictions
        are made against the same attribute declarations as training.
    classes_ : np.ndarray
        Original class labels, in the order Weka assigned them. Classification
        wrappers only.
    """

    #: True for classification (nominal class), False for regression.
    _is_classification: bool = True

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_WekaSupervisedMixin":
        """Fit the backing Weka model.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data.
        y : np.ndarray of shape (n_samples,)
            Target values. For classification these may be any hashable labels;
            they are encoded to integer codes for Weka and decoded on predict.

        Returns
        -------
        self : _WekaSupervisedMixin
            Fitted instance.
        """
        _ensure_weka(type(self).__name__)
        ensure_jvm()
        from weka.classifiers import Classifier as _WekaClf

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)

        if self._is_classification:
            self.classes_, y_codes = np.unique(y, return_inverse=True)
            y_num = y_codes.astype(np.float64)
        else:
            self.classes_ = None
            y_num = y.astype(np.float64)

        data = to_instances(
            X,
            y_num,
            nominal_features=self._nominal_features(),
            nominal_target=self._is_classification,
        )
        # Keep only the header: predictions are built against it, and holding
        # the training rows would double the memory the model costs.
        from weka.core.dataset import Instances
        self.header_ = Instances.template_instances(data, 0)
        self.model_ = _WekaClf(
            classname=self._weka_classname, options=self._resolved_options()
        )
        self.model_.build_classifier(data)
        self._is_fitted = True
        return self

    def _test_instances(self, X: np.ndarray):
        """Build a test table that reuses the training header.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to convert.

        Returns
        -------
        test : weka.core.dataset.Instances
            Test table using the training header.
        """
        return rows_to_instances(self.header_, X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels or values for ``X``.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        predictions : np.ndarray of shape (n_samples,)
            Predicted labels (classification, in the caller's original label
            space) or values (regression).
        """
        self._check_is_fitted()
        test = self._test_instances(X)
        dists = self.model_.distributions_for_instances(test)
        if dists is None:  # pragma: no cover - AbstractClassifier is a BatchPredictor
            dists = np.asarray(
                [
                    self.model_.distribution_for_instance(test.get_instance(i))
                    for i in range(test.num_instances)
                ]
            )
        dists = np.asarray(dists)
        if not self._is_classification:
            return dists[:, 0].astype(float)
        idx = dists.argmax(axis=1)
        # Nominal class values are the string forms of the encoded integer codes.
        codes = np.asarray(
            [
                int(float(test.class_attribute.value(i)))
                for i in range(test.class_attribute.num_values)
            ]
        )
        return self.classes_[codes[idx]]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        proba : np.ndarray of shape (n_samples, n_classes)
            Class membership probabilities, with columns ordered to match
            ``classes_``.

        Raises
        ------
        AttributeError
            If called on a regression wrapper.
        """
        if not self._is_classification:
            raise AttributeError(
                f"{type(self).__name__} is a regressor and has no predict_proba."
            )
        self._check_is_fitted()
        test = self._test_instances(X)
        dists = np.asarray(self.model_.distributions_for_instances(test))
        # Reorder Weka's columns into classes_ order.
        codes = np.asarray(
            [
                int(float(test.class_attribute.value(i)))
                for i in range(test.class_attribute.num_values)
            ]
        )
        proba = np.zeros((dists.shape[0], len(self.classes_)), dtype=float)
        proba[:, codes] = dists
        return proba

    def to_weka_string(self) -> str:
        """Return Weka's own textual description of the fitted model.

        This is the human-readable model dump Weka prints — the tree, the rule
        set, the coefficients — which is often the reason to reach for Weka in
        the first place.

        Returns
        -------
        description : str
            Weka's model description.
        """
        self._check_is_fitted()
        return str(self.model_)


class _WekaClustererMixin(_WekaMixin):
    """Adapt a Weka ``Clusterer`` to TuiML's ``fit`` / ``predict`` interface.

    Attributes
    ----------
    model_ : weka.clusterers.Clusterer
        The fitted backing Weka clusterer.
    labels_ : np.ndarray
        Cluster assignment for each training row.
    n_clusters_ : int
        Number of clusters Weka produced.
    """

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "_WekaClustererMixin":
        """Build the clustering model.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data.
        y : None
            Ignored (unsupervised).

        Returns
        -------
        self : _WekaClustererMixin
            Fitted instance.
        """
        _ensure_weka(type(self).__name__)
        ensure_jvm()
        from weka.clusterers import Clusterer as _WekaClusterer
        from weka.core.dataset import Instances

        X = np.asarray(X, dtype=np.float64)
        # Passing y=None builds a table with no class attribute at all, which is
        # what a clusterer needs.
        data = to_instances(X, None, nominal_features=self._nominal_features())
        self.header_ = Instances.template_instances(data, 0)
        self.model_ = _WekaClusterer(
            classname=self._weka_classname, options=self._resolved_options()
        )
        self.model_.build_clusterer(data)
        self.labels_ = np.asarray(
            [self.model_.cluster_instance(data.get_instance(i))
             for i in range(data.num_instances)],
            dtype=int,
        )
        self.n_clusters_ = int(self.model_.number_of_clusters)
        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Assign each row of ``X`` to a cluster.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        labels : np.ndarray of shape (n_samples,)
            Cluster index for each row.
        """
        self._check_is_fitted()
        data = rows_to_instances(self.header_, X)
        return np.asarray(
            [self.model_.cluster_instance(data.get_instance(i))
             for i in range(data.num_instances)],
            dtype=int,
        )

    def to_weka_string(self) -> str:
        """Return Weka's own textual description of the fitted clusterer.

        Returns
        -------
        description : str
            Weka's model description.
        """
        self._check_is_fitted()
        return str(self.model_)
