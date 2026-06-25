"""Shared machinery for the CapyMOA (streaming) bridge.

CapyMOA exposes *incremental* (instance-at-a-time) learners. These wrappers
adapt that streaming API to TuiML's batch ``fit`` / ``predict`` interface by
iterating the rows of the input arrays, while still registering each learner
into the hub under a ``capymoa.<ClassName>`` key.

CapyMOA is an optional dependency (it pulls in a JVM via ``moa``). Importing this
module never requires it; the dependency is checked at ``fit`` time with a clear,
actionable error.
"""

from typing import List, Optional

import numpy as np

from tuiml.base.algorithms import classifier, regressor

#: Hub-registry namespace prefix for CapyMOA wrappers.
NAMESPACE = "capymoa"
_EXTRA = "capymoa"


def _ensure_capymoa(cls_name: str) -> None:
    """Raise an actionable error if CapyMOA is not importable."""
    try:
        import capymoa  # noqa: F401
    except ImportError as exc:  # pragma: no cover - exercised only without capymoa
        raise ImportError(
            f"{cls_name} requires CapyMOA, which is not installed. "
            f"Install it with:  pip install 'tuiml[{_EXTRA}]'"
        ) from exc


def _namespaced(base_decorator):
    """Wrap a base tuiml decorator to register under ``capymoa.<ClassName>``."""

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

    Subclasses implement :meth:`_build_learner(schema)`, returning a CapyMOA
    learner constructed against the provided stream schema.
    """

    _requires = "capymoa"
    _extra = _EXTRA

    def _build_learner(self, schema):
        """Return a CapyMOA learner configured for ``schema``."""
        raise NotImplementedError

    def _make_stream(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """Build a CapyMOA ``NumpyStream`` (and schema) from arrays."""
        from capymoa.stream import NumpyStream
        X = np.asarray(X, dtype=float)
        if y is None:
            y = np.zeros(X.shape[0])
        return NumpyStream(X, np.asarray(y))

    def fit(self, X: np.ndarray, y: np.ndarray = None) -> "_CapyMOAStreamMixin":
        """Train incrementally over every instance in ``X`` / ``y``."""
        _ensure_capymoa(type(self).__name__)
        stream = self._make_stream(X, y)
        self.schema_ = stream.get_schema()
        self.learner_ = self._build_learner(self.schema_)
        while stream.has_more_instances():
            self.learner_.train(stream.next_instance())
        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict by streaming each row through the fitted learner."""
        self._check_is_fitted()
        stream = self._make_stream(X)
        preds = []
        while stream.has_more_instances():
            preds.append(self.learner_.predict(stream.next_instance()))
        return np.asarray(preds)
