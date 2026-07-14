"""Scikit-Learn Naive Bayes wrappers."""

import numpy as np
from typing import Dict, List, Any, Optional

from tuiml.base.algorithms import Classifier, classifier

try:
    from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
    SKLEARN_AVAILABLE = True
except ImportError:
    GaussianNB = MultinomialNB = BernoulliNB = None
    SKLEARN_AVAILABLE = False


@classifier(tags=["bayesian", "sklearn"], version="1.0.0")
class SklearnGaussianNB(Classifier):
    def __init__(self, var_smoothing: float = 1e-9):
        super().__init__()
        if not SKLEARN_AVAILABLE: raise ImportError("scikit-learn is not installed.")
        self.var_smoothing = var_smoothing
        self.model_ = None
        self.classes_ = None

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        return {"var_smoothing": {"type": "number", "default": 1e-9}}

    @classmethod
    def get_capabilities(cls) -> List[str]: return ["numeric", "nominal", "multi_class"]

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SklearnGaussianNB":
        X, y = np.asarray(X, dtype=float), np.asarray(y)
        if X.ndim == 1: X = X.reshape(-1, 1)
        self.model_ = GaussianNB(var_smoothing=self.var_smoothing)
        self.model_.fit(X, y)
        self.classes_ = self.model_.classes_
        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._check_is_fitted()
        X = np.asarray(X, dtype=float)
        if X.ndim == 1: X = X.reshape(-1, 1)
        return self.model_.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self._check_is_fitted()
        X = np.asarray(X, dtype=float)
        if X.ndim == 1: X = X.reshape(-1, 1)
        return self.model_.predict_proba(X)


@classifier(tags=["bayesian", "sklearn"], version="1.0.0")
class SklearnMultinomialNB(Classifier):
    def __init__(self, alpha: float = 1.0, fit_prior: bool = True):
        super().__init__()
        if not SKLEARN_AVAILABLE: raise ImportError("scikit-learn is not installed.")
        self.alpha = alpha
        self.fit_prior = fit_prior
        self.model_ = None
        self.classes_ = None

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "alpha": {"type": "number", "default": 1.0},
            "fit_prior": {"type": "boolean", "default": True}
        }

    @classmethod
    def get_capabilities(cls) -> List[str]: return ["numeric", "nominal", "multi_class"]

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SklearnMultinomialNB":
        X, y = np.asarray(X, dtype=float), np.asarray(y)
        if X.ndim == 1: X = X.reshape(-1, 1)
        self.model_ = MultinomialNB(alpha=self.alpha, fit_prior=self.fit_prior)
        self.model_.fit(X, y)
        self.classes_ = self.model_.classes_
        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._check_is_fitted()
        X = np.asarray(X, dtype=float)
        if X.ndim == 1: X = X.reshape(-1, 1)
        return self.model_.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self._check_is_fitted()
        X = np.asarray(X, dtype=float)
        if X.ndim == 1: X = X.reshape(-1, 1)
        return self.model_.predict_proba(X)
