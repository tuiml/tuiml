"""Curated scikit-learn algorithm wrappers.

A small, discoverable set of popular scikit-learn estimators exposed as native
TuiML algorithms. They register into the hub under ``sklearn.<ClassName>`` keys
(e.g. ``sklearn.RandomForestClassifier``) so they never collide with the native
TuiML algorithms of the same name.

For anything not curated here, pass the estimator object directly to
``tuiml.train`` / ``tuiml.experiment`` — it is auto-wrapped via
:class:`tuiml.sklearn.adapter.SklearnAdapter`.
"""

from typing import Any, Dict, List, Optional

from tuiml.base.algorithms import Classifier, Regressor, Clusterer
from tuiml.sklearn._base import (
    _SklearnBackedMixin,
    _SklearnClustererMixin,
    sk_classifier,
    sk_regressor,
    sk_clusterer,
)


# =============================================================================
# Classifiers
# =============================================================================

@sk_classifier(tags=["ensemble", "tree"])
class RandomForestClassifier(_SklearnBackedMixin, Classifier):
    """scikit-learn RandomForestClassifier (hub key ``sklearn.RandomForestClassifier``)."""

    def __init__(self, n_estimators: int = 100, max_depth: Optional[int] = None,
                 random_state: Optional[int] = None):
        super().__init__()
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state

    def _build_estimator(self):
        from sklearn.ensemble import RandomForestClassifier as _Est
        return _Est(n_estimators=self.n_estimators, max_depth=self.max_depth,
                    random_state=self.random_state)

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Any]:
        return {"n_estimators": {"type": "integer", "default": 100},
                "max_depth": {"type": "integer", "default": None}}

    @classmethod
    def get_capabilities(cls) -> List[str]:
        return ["numeric", "multiclass", "probabilities"]


@sk_classifier(tags=["gradient-boosting", "ensemble"])
class HistGradientBoostingClassifier(_SklearnBackedMixin, Classifier):
    """scikit-learn HistGradientBoostingClassifier."""

    def __init__(self, learning_rate: float = 0.1, max_iter: int = 100,
                 max_depth: Optional[int] = None, random_state: Optional[int] = None):
        super().__init__()
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.max_depth = max_depth
        self.random_state = random_state

    def _build_estimator(self):
        from sklearn.ensemble import HistGradientBoostingClassifier as _Est
        return _Est(learning_rate=self.learning_rate, max_iter=self.max_iter,
                    max_depth=self.max_depth, random_state=self.random_state)

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Any]:
        return {"learning_rate": {"type": "number", "default": 0.1},
                "max_iter": {"type": "integer", "default": 100}}

    @classmethod
    def get_capabilities(cls) -> List[str]:
        return ["numeric", "multiclass", "probabilities"]


@sk_classifier(tags=["svm", "kernel"])
class SVC(_SklearnBackedMixin, Classifier):
    """scikit-learn Support Vector Classifier."""

    def __init__(self, C: float = 1.0, kernel: str = "rbf",
                 probability: bool = True, random_state: Optional[int] = None):
        super().__init__()
        self.C = C
        self.kernel = kernel
        self.probability = probability
        self.random_state = random_state

    def _build_estimator(self):
        from sklearn.svm import SVC as _Est
        return _Est(C=self.C, kernel=self.kernel, probability=self.probability,
                    random_state=self.random_state)

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Any]:
        return {"C": {"type": "number", "default": 1.0},
                "kernel": {"type": "string", "default": "rbf",
                           "enum": ["linear", "poly", "rbf", "sigmoid"]}}

    @classmethod
    def get_capabilities(cls) -> List[str]:
        return ["numeric", "multiclass", "probabilities"]


@sk_classifier(tags=["linear"])
class LogisticRegression(_SklearnBackedMixin, Classifier):
    """scikit-learn LogisticRegression."""

    def __init__(self, C: float = 1.0, max_iter: int = 1000,
                 random_state: Optional[int] = None):
        super().__init__()
        self.C = C
        self.max_iter = max_iter
        self.random_state = random_state

    def _build_estimator(self):
        from sklearn.linear_model import LogisticRegression as _Est
        return _Est(C=self.C, max_iter=self.max_iter, random_state=self.random_state)

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Any]:
        return {"C": {"type": "number", "default": 1.0},
                "max_iter": {"type": "integer", "default": 1000}}

    @classmethod
    def get_capabilities(cls) -> List[str]:
        return ["numeric", "multiclass", "probabilities"]


@sk_classifier(tags=["neighbors", "instance-based"])
class KNeighborsClassifier(_SklearnBackedMixin, Classifier):
    """scikit-learn KNeighborsClassifier."""

    def __init__(self, n_neighbors: int = 5, weights: str = "uniform"):
        super().__init__()
        self.n_neighbors = n_neighbors
        self.weights = weights

    def _build_estimator(self):
        from sklearn.neighbors import KNeighborsClassifier as _Est
        return _Est(n_neighbors=self.n_neighbors, weights=self.weights)

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Any]:
        return {"n_neighbors": {"type": "integer", "default": 5},
                "weights": {"type": "string", "default": "uniform",
                            "enum": ["uniform", "distance"]}}

    @classmethod
    def get_capabilities(cls) -> List[str]:
        return ["numeric", "multiclass", "probabilities"]


@sk_classifier(tags=["neural", "mlp"])
class MLPClassifier(_SklearnBackedMixin, Classifier):
    """scikit-learn Multi-layer Perceptron classifier."""

    def __init__(self, hidden_layer_sizes: tuple = (100,), max_iter: int = 200,
                 random_state: Optional[int] = None):
        super().__init__()
        self.hidden_layer_sizes = hidden_layer_sizes
        self.max_iter = max_iter
        self.random_state = random_state

    def _build_estimator(self):
        from sklearn.neural_network import MLPClassifier as _Est
        return _Est(hidden_layer_sizes=self.hidden_layer_sizes,
                    max_iter=self.max_iter, random_state=self.random_state)

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Any]:
        return {"max_iter": {"type": "integer", "default": 200}}

    @classmethod
    def get_capabilities(cls) -> List[str]:
        return ["numeric", "multiclass", "probabilities"]


@sk_classifier(tags=["bayesian", "naive-bayes"])
class GaussianNB(_SklearnBackedMixin, Classifier):
    """scikit-learn Gaussian Naive Bayes."""

    def __init__(self):
        super().__init__()

    def _build_estimator(self):
        from sklearn.naive_bayes import GaussianNB as _Est
        return _Est()

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Any]:
        return {}

    @classmethod
    def get_capabilities(cls) -> List[str]:
        return ["numeric", "multiclass", "probabilities"]


# =============================================================================
# Regressors
# =============================================================================

@sk_regressor(tags=["ensemble", "tree"])
class RandomForestRegressor(_SklearnBackedMixin, Regressor):
    """scikit-learn RandomForestRegressor."""

    def __init__(self, n_estimators: int = 100, max_depth: Optional[int] = None,
                 random_state: Optional[int] = None):
        super().__init__()
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state

    def _build_estimator(self):
        from sklearn.ensemble import RandomForestRegressor as _Est
        return _Est(n_estimators=self.n_estimators, max_depth=self.max_depth,
                    random_state=self.random_state)

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Any]:
        return {"n_estimators": {"type": "integer", "default": 100},
                "max_depth": {"type": "integer", "default": None}}

    @classmethod
    def get_capabilities(cls) -> List[str]:
        return ["numeric"]


@sk_regressor(tags=["gradient-boosting", "ensemble"])
class HistGradientBoostingRegressor(_SklearnBackedMixin, Regressor):
    """scikit-learn HistGradientBoostingRegressor."""

    def __init__(self, learning_rate: float = 0.1, max_iter: int = 100,
                 max_depth: Optional[int] = None, random_state: Optional[int] = None):
        super().__init__()
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.max_depth = max_depth
        self.random_state = random_state

    def _build_estimator(self):
        from sklearn.ensemble import HistGradientBoostingRegressor as _Est
        return _Est(learning_rate=self.learning_rate, max_iter=self.max_iter,
                    max_depth=self.max_depth, random_state=self.random_state)

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Any]:
        return {"learning_rate": {"type": "number", "default": 0.1},
                "max_iter": {"type": "integer", "default": 100}}

    @classmethod
    def get_capabilities(cls) -> List[str]:
        return ["numeric"]


@sk_regressor(tags=["svm", "kernel"])
class SVR(_SklearnBackedMixin, Regressor):
    """scikit-learn Support Vector Regressor."""

    def __init__(self, C: float = 1.0, kernel: str = "rbf", epsilon: float = 0.1):
        super().__init__()
        self.C = C
        self.kernel = kernel
        self.epsilon = epsilon

    def _build_estimator(self):
        from sklearn.svm import SVR as _Est
        return _Est(C=self.C, kernel=self.kernel, epsilon=self.epsilon)

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Any]:
        return {"C": {"type": "number", "default": 1.0},
                "epsilon": {"type": "number", "default": 0.1}}

    @classmethod
    def get_capabilities(cls) -> List[str]:
        return ["numeric"]


@sk_regressor(tags=["linear", "regularized"])
class Ridge(_SklearnBackedMixin, Regressor):
    """scikit-learn Ridge regression."""

    def __init__(self, alpha: float = 1.0, random_state: Optional[int] = None):
        super().__init__()
        self.alpha = alpha
        self.random_state = random_state

    def _build_estimator(self):
        from sklearn.linear_model import Ridge as _Est
        return _Est(alpha=self.alpha, random_state=self.random_state)

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Any]:
        return {"alpha": {"type": "number", "default": 1.0}}

    @classmethod
    def get_capabilities(cls) -> List[str]:
        return ["numeric"]


@sk_regressor(tags=["linear", "regularized"])
class Lasso(_SklearnBackedMixin, Regressor):
    """scikit-learn Lasso regression."""

    def __init__(self, alpha: float = 1.0, random_state: Optional[int] = None):
        super().__init__()
        self.alpha = alpha
        self.random_state = random_state

    def _build_estimator(self):
        from sklearn.linear_model import Lasso as _Est
        return _Est(alpha=self.alpha, random_state=self.random_state)

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Any]:
        return {"alpha": {"type": "number", "default": 1.0}}

    @classmethod
    def get_capabilities(cls) -> List[str]:
        return ["numeric"]


# =============================================================================
# Clusterers
# =============================================================================

@sk_clusterer(tags=["partitioning"])
class KMeans(_SklearnClustererMixin, Clusterer):
    """scikit-learn KMeans clusterer."""

    def __init__(self, n_clusters: int = 8, random_state: Optional[int] = None):
        super().__init__()
        self.n_clusters = n_clusters
        self.random_state = random_state

    def _build_estimator(self):
        from sklearn.cluster import KMeans as _Est
        return _Est(n_clusters=self.n_clusters, random_state=self.random_state,
                    n_init="auto")

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Any]:
        return {"n_clusters": {"type": "integer", "default": 8}}

    @classmethod
    def get_capabilities(cls) -> List[str]:
        return ["numeric"]


@sk_clusterer(tags=["density"])
class DBSCAN(_SklearnClustererMixin, Clusterer):
    """scikit-learn DBSCAN clusterer (transductive)."""

    def __init__(self, eps: float = 0.5, min_samples: int = 5):
        super().__init__()
        self.eps = eps
        self.min_samples = min_samples

    def _build_estimator(self):
        from sklearn.cluster import DBSCAN as _Est
        return _Est(eps=self.eps, min_samples=self.min_samples)

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Any]:
        return {"eps": {"type": "number", "default": 0.5},
                "min_samples": {"type": "integer", "default": 5}}

    @classmethod
    def get_capabilities(cls) -> List[str]:
        return ["numeric"]


@sk_clusterer(tags=["mixture", "probabilistic"])
class GaussianMixture(_SklearnClustererMixin, Clusterer):
    """scikit-learn GaussianMixture clusterer."""

    def __init__(self, n_components: int = 1, random_state: Optional[int] = None):
        super().__init__()
        self.n_components = n_components
        self.random_state = random_state

    def _build_estimator(self):
        from sklearn.mixture import GaussianMixture as _Est
        return _Est(n_components=self.n_components, random_state=self.random_state)

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Any]:
        return {"n_components": {"type": "integer", "default": 1}}

    @classmethod
    def get_capabilities(cls) -> List[str]:
        return ["numeric"]


__all__ = [
    "RandomForestClassifier", "HistGradientBoostingClassifier", "SVC",
    "LogisticRegression", "KNeighborsClassifier", "MLPClassifier", "GaussianNB",
    "RandomForestRegressor", "HistGradientBoostingRegressor", "SVR", "Ridge", "Lasso",
    "KMeans", "DBSCAN", "GaussianMixture",
]
