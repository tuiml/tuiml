"""Shared machinery for the CapyMOA (streaming) bridge.

CapyMOA exposes **incremental** (instance-at-a-time) learners backed by the
MOA JVM runtime. These wrappers adapt that streaming API to TuiML's batch
``fit`` / ``predict`` interface by iterating the rows of the input arrays,
while still registering each learner into the hub under a ``capymoa.<ClassName>``
key.

CapyMOA is an optional dependency (it pulls in a JVM via ``moa``). Importing
this module never requires it; the dependency is checked at ``fit`` time with
a clear, actionable error message pointing at ``pip install 'tuiml[capymoa]'``.

Examples
--------
>>> from tuiml.capymoa import NaiveBayes
>>> from tuiml.datasets.generators import Agrawal
>>> data = Agrawal(n_samples=300, function=1, random_state=42).generate()
>>> nb = NaiveBayes().fit(data.X, data.y)
>>> nb.predict(data.X[:3]).shape
(3,)
"""

from typing import List, Optional

import numpy as np

from tuiml.base.algorithms import classifier, regressor

#: Hub-registry namespace prefix for CapyMOA wrappers.
NAMESPACE = "capymoa"
_EXTRA = "capymoa"


def _ensure_capymoa(cls_name: str) -> None:
    """Raise an actionable ``ImportError`` if CapyMOA is not installed.

    Parameters
    ----------
    cls_name : str
        Name of the wrapper class being instantiated, shown in the error
        message so the user knows exactly which learner prompted the check.
    """
    try:
        import capymoa  # noqa: F401
    except ImportError as exc:  # pragma: no cover - exercised only without capymoa
        raise ImportError(
            f"{cls_name} requires CapyMOA, which is not installed. "
            f"Install it with:  pip install 'tuiml[{_EXTRA}]'"
        ) from exc


def _namespaced(base_decorator):
    """Wrap a base ``@classifier`` / ``@regressor`` decorator to register under
    the ``capymoa.<ClassName>`` hub namespace.

    The returned factory merges ``capymoa`` and ``streaming`` tags into
    whatever tags the caller supplies, then delegates to the TuiML base
    decorator with the namespaced key.

    Parameters
    ----------
    base_decorator : callable
        The TuiML base decorator (:func:`tuiml.base.algorithms.classifier` or
        :func:`tuiml.base.algorithms.regressor`).

    Returns
    -------
    factory : callable
        A callable ``(tags=None, version="1.0.0")`` returning a decorator
        that registers the class under ``capymoa.<ClassName>``.
    """

    def factory(tags: Optional[List[str]] = None, version: str = "1.0.0"):
        def decorate(cls):
            key = f"{NAMESPACE}.{cls.__name__}"
            merged = list(tags or [])
            for t in (NAMESPACE, "streaming"):
                if t not in merged:
                    merged.append(t)
            return base_decorator(name=key, tags=merged, version=version)(cls)

        return decorate

    return factory


capymoa_classifier = _namespaced(classifier)
capymoa_regressor = _namespaced(regressor)


class _CapyMOAStreamMixin:
    """Adapt a CapyMOA incremental learner to TuiML's batch interface.

    Subclasses implement :meth:`_build_learner`, returning a CapyMOA learner
    constructed against the provided stream schema. The mixin handles
    building a ``NumpyStream`` from arrays, streaming instances through the
    learner one at a time, and collecting predictions.

    Attributes
    ----------
    schema_ : capymoa.stream.Schema
        Stream schema captured during ``fit``.
    learner_ : capymoa learner
        The fitted CapyMOA learner instance.
    """

    _requires = "capymoa"
    _extra = _EXTRA

    def _build_learner(self, schema):
        """Return a CapyMOA learner configured for ``schema``.

        Parameters
        ----------
        schema : capymoa.stream.Schema
            Schema that describes the feature types and target of the stream.

        Returns
        -------
        learner
            A fresh, untrained CapyMOA learner instance.
        """
        raise NotImplementedError  # pragma: no cover

    def _make_stream(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """Build a CapyMOA ``NumpyStream`` (and schema) from arrays.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Feature matrix.
        y : np.ndarray of shape (n_samples,) or None
            Target values. When None, a dummy zero vector is used (for
            prediction-only streams where the schema is already known).

        Returns
        -------
        stream : capymoa.stream.NumpyStream
            Stream wrapping the input arrays.
        """
        from capymoa.stream import NumpyStream
        X = np.asarray(X, dtype=float)
        if y is None:
            y = np.zeros(X.shape[0])
        return NumpyStream(X, np.asarray(y))

    def fit(self, X: np.ndarray, y: np.ndarray = None) -> "_CapyMOAStreamMixin":
        """Train incrementally over every instance in ``X`` / ``y``.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data.
        y : np.ndarray of shape (n_samples,)
            Target values.

        Returns
        -------
        self : _CapyMOAStreamMixin
            Fitted instance.
        """
        _ensure_capymoa(type(self).__name__)
        stream = self._make_stream(X, y)
        self.schema_ = stream.get_schema()
        self.learner_ = self._build_learner(self.schema_)
        while stream.has_more_instances():
            self.learner_.train(stream.next_instance())
        self._is_fitted = True
        return self

    def partial_fit(
        self,
        X: np.ndarray,
        y: np.ndarray = None,
        classes: Optional[np.ndarray] = None,
    ) -> "_CapyMOAStreamMixin":
        """Incrementally train on a new batch of instances.

        If the learner has not been fitted yet, the schema and learner are
        built from this first batch (same as :meth:`fit`). Otherwise the
        existing learner continues training on the new data without
        reinitialisation — this is true online incremental learning.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Batch of training features.
        y : np.ndarray of shape (n_samples,), optional
            Batch of target values.
        classes : np.ndarray, optional
            Full array of possible class labels. Accepted for compatibility
            with the base ``partial_fit`` signature; CapyMOA discovers
            classes from the stream schema automatically.

        Returns
        -------
        self : _CapyMOAStreamMixin
            The updated instance.

        Notes
        -----
        This method mirrors the base
        :meth:`~tuiml.base.algorithms.Algorithm.partial_fit` contract while
        providing a true streaming implementation: each row in ``X`` is
        passed to the learner one at a time via ``learner.train()``.

        Examples
        --------
        >>> from tuiml.capymoa import HoeffdingTree
        >>> from tuiml.datasets.generators import Agrawal
        >>> data = Agrawal(n_samples=500, function=1, random_state=42).generate()
        >>> # Initial fit on first 300 samples
        >>> model = HoeffdingTree().partial_fit(data.X[:300], data.y[:300])
        >>> model.predict(data.X[:3]).shape
        (3,)
        >>> # Continue learning on remaining 200 (returns self, for chaining)
        >>> model = model.partial_fit(data.X[300:], data.y[300:])
        >>> model.predict(data.X[:3]).shape
        (3,)
        """
        _ensure_capymoa(type(self).__name__)
        if not self._is_fitted:
            stream = self._make_stream(X, y)
            self.schema_ = stream.get_schema()
            self.learner_ = self._build_learner(self.schema_)
            while stream.has_more_instances():
                self.learner_.train(stream.next_instance())
            self._is_fitted = True
            return self

        stream = self._make_stream(X, y)
        while stream.has_more_instances():
            self.learner_.train(stream.next_instance())
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict by streaming each row through the fitted learner.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        predictions : np.ndarray of shape (n_samples,)
            Predicted labels or values.
        """
        self._check_is_fitted()
        stream = self._make_stream(X)
        preds = []
        while stream.has_more_instances():
            preds.append(self.learner_.predict(stream.next_instance()))
        return np.asarray(preds)
