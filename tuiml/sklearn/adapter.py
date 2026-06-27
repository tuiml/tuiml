"""Generic adapter that turns any scikit-learn-compatible estimator into a
first-class TuiML algorithm.

Where the curated wrappers in :mod:`tuiml.sklearn.algorithms` expose a small,
discoverable set of named estimators, :class:`SklearnAdapter` covers the entire
ecosystem with zero per-estimator code. It wraps *any* object implementing the
scikit-learn estimator protocol (``fit`` / ``predict``) — including pipelines,
``GridSearchCV`` objects, and third-party estimators (imbalanced-learn, a user's
own class) — so they can be passed straight into ``tuiml.train`` /
``tuiml.experiment``.

Examples
--------
>>> from sklearn.svm import SVC
>>> import tuiml
>>> tuiml.train(SVC(C=2.0), "iris", "class", cv=10)   # auto-wrapped, no ceremony

>>> from tuiml.sklearn import wrap_sklearn
>>> model = wrap_sklearn(SVC()).fit(X_train, y_train)
"""

from typing import Any

import numpy as np

from tuiml.base.algorithms import Algorithm, Classifier, Regressor


def is_sklearn_estimator(obj: Any) -> bool:
    """Return True if ``obj`` looks like a (non-TuiML) scikit-learn estimator.

    A duck-typed check: the object exposes ``fit`` and ``predict`` but is not
    already a TuiML :class:`~tuiml.base.algorithms.Algorithm`.

    Parameters
    ----------
    obj : Any
        Candidate object.

    Returns
    -------
    bool
        Whether ``obj`` should be wrapped by :class:`SklearnAdapter`.
    """
    if isinstance(obj, Algorithm):
        return False
    return callable(getattr(obj, "fit", None)) and callable(getattr(obj, "predict", None))


class SklearnAdapter(Classifier):
    """Adapt any scikit-learn-compatible estimator to the TuiML interface.

    The adapter forwards ``fit`` / ``predict`` / ``predict_proba`` to the wrapped
    estimator. It subclasses :class:`~tuiml.base.algorithms.Classifier` for a
    concrete base, but works for regressors too — :func:`wrap_sklearn` selects
    the right reported estimator type based on the wrapped object.

    Parameters
    ----------
    estimator : object
        A scikit-learn-compatible estimator (anything with ``fit`` / ``predict``).
    """

    def __init__(self, estimator: Any):
        super().__init__()
        self.estimator = estimator

    def fit(self, X: np.ndarray, y: np.ndarray = None) -> "SklearnAdapter":
        """Fit the wrapped estimator."""
        self.estimator.fit(np.asarray(X, dtype=float), y)
        for attr in ("classes_", "n_features_in_"):
            if hasattr(self.estimator, attr):
                setattr(self, attr, getattr(self.estimator, attr))
        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict with the wrapped estimator."""
        self._check_is_fitted()
        return self.estimator.predict(np.asarray(X, dtype=float))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Delegate to the wrapped estimator's ``predict_proba`` if available."""
        self._check_is_fitted()
        if hasattr(self.estimator, "predict_proba"):
            return self.estimator.predict_proba(np.asarray(X, dtype=float))
        return super().predict_proba(X)

    def get_params(self, deep: bool = True) -> dict:
        """Forward to the wrapped estimator's ``get_params`` when available."""
        if hasattr(self.estimator, "get_params"):
            return self.estimator.get_params(deep=deep)
        return {"estimator": self.estimator}


def wrap_sklearn(estimator: Any) -> SklearnAdapter:
    """Wrap a scikit-learn estimator as a TuiML algorithm.

    Reports the correct estimator type (``"regressor"`` vs ``"classifier"``) so
    downstream metric auto-selection behaves correctly.

    Parameters
    ----------
    estimator : object
        A scikit-learn-compatible estimator.

    Returns
    -------
    SklearnAdapter
        The wrapped estimator.
    """
    adapter = SklearnAdapter(estimator)
    # Mirror the wrapped estimator's task type for metric auto-selection.
    est_type = getattr(estimator, "_estimator_type", None)
    if est_type == "regressor" or isinstance(estimator, Regressor):
        adapter._estimator_type = "regressor"
    else:
        adapter._estimator_type = "classifier"
    return adapter
