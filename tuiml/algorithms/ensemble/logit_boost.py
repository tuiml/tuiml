"""LogitBoostClassifier classifier implementation."""

import numpy as np
from typing import Dict, List, Any, Optional, Type
from collections import Counter

from tuiml.base.algorithms import Classifier, classifier
from tuiml.hub import registry

@classifier(tags=["ensemble", "boosting", "logistic"], version="1.0.0")
class LogitBoostClassifier(Classifier):
    """LogitBoostClassifier meta-classifier.

    LogitBoostClassifier performs additive logistic regression using boosting. It 
    iteratively fits regression functions to working responses derived 
    from the logistic loss function.

    Parameters
    ----------
    base_classifier : str or class, default='DecisionStumpClassifier'
        The base classifier to use.
    n_iterations : int, default=10
        The number of boosting iterations.
    shrinkage : float, default=1.0
        Learning rate (shrinkage parameter).
    use_resampling : bool, default=False
        Whether to use resampling instead of weighting.
    weight_threshold : float, default=100
        Weight mass percentage for resampling.
    random_state : int, optional
        Random seed for reproducibility.

    Attributes
    ----------
    estimators_ : list of list
        The collection of fitted estimators per class.
    classes_ : np.ndarray
        The unique class labels.
    n_classes_ : int
        The number of classes.

    Examples
    --------
    >>> from tuiml.algorithms.ensemble import LogitBoostClassifier
    >>> clf = LogitBoostClassifier(n_iterations=100)
    >>> clf.fit(X_train, y_train)
    LogitBoostClassifier(...)
    >>> predictions = clf.predict(X_test)

    References
    ----------
    .. [1] Friedman, J., Hastie, T., & Tibshirani, R. (2000). Additive 
           Logistic Regression: A Statistical View of Boosting. 
           Annals of Statistics, 28(2), 337-407.
    """

    def __init__(self, base_classifier: Any = 'DecisionStumpClassifier',
                 n_iterations: int = 10,
                 shrinkage: float = 1.0,
                 use_resampling: bool = False,
                 weight_threshold: float = 100,
                 random_state: Optional[int] = None):
        super().__init__()
        self.base_classifier = base_classifier
        self.n_iterations = n_iterations
        self.shrinkage = shrinkage
        self.use_resampling = use_resampling
        self.weight_threshold = weight_threshold
        self.random_state = random_state
        self.estimators_ = None
        self.classes_ = None
        self.n_classes_ = None
        self._base_class = None

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "base_classifier": {
                "type": "string", "default": "DecisionStump",
                "description": "Base classifier name"
            },
            "n_iterations": {
                "type": "integer", "default": 10, "minimum": 1,
                "description": "Number of boosting iterations"
            },
            "shrinkage": {
                "type": "number", "default": 1.0, "minimum": 0.01, "maximum": 1.0,
                "description": "Learning rate (shrinkage)"
            },
            "use_resampling": {
                "type": "boolean", "default": False,
                "description": "Use resampling instead of weighting"
            },
            "weight_threshold": {
                "type": "number", "default": 100, "minimum": 1,
                "description": "Weight mass percentage for resampling"
            },
            "random_state": {
                "type": "integer", "default": None,
                "description": "Random seed"
            }
        }

    @classmethod
    def get_capabilities(cls) -> List[str]:
        return ["numeric", "nominal", "missing_values", "binary_class", "multiclass"]

    @classmethod
    def get_complexity(cls) -> str:
        return "O(n * n_iterations * base_complexity)"

    @classmethod
    def get_references(cls) -> List[str]:
        return [
            "Friedman, J., Hastie, T., & Tibshirani, R. (2000). "
            "Additive Logistic Regression: A Statistical View of Boosting. "
            "Annals of Statistics, 28(2), 337-407."
        ]

    def _get_base_class(self) -> Type[Classifier]:
        """Get the base classifier class."""
        if isinstance(self.base_classifier, str):
            return registry.get(self.base_classifier)
        elif isinstance(self.base_classifier, type):
            return self.base_classifier
        elif isinstance(self.base_classifier, Classifier):
            # Handle classifier instances by extracting their class
            return type(self.base_classifier)
        else:
            raise ValueError(f"Invalid base_classifier: {self.base_classifier}")

    def _softmax(self, F: np.ndarray) -> np.ndarray:
        """Compute softmax probabilities.

        Parameters
        ----------
        F : np.ndarray of shape (n_samples, n_classes)
            Log-odds values.

        Returns
        -------
        proba : np.ndarray of shape (n_samples, n_classes)
            Softmax probabilities.
        """
        exp_F = np.exp(F - np.max(F, axis=1, keepdims=True))
        return exp_F / (np.sum(exp_F, axis=1, keepdims=True) + 1e-10)

    def _fit_regression_stump(self, X: np.ndarray, z: np.ndarray,
                               weights: np.ndarray) -> dict:
        """Fit a weighted regression stump to the working response.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training features.
        z : np.ndarray of shape (n_samples,)
            Working response values.
        weights : np.ndarray of shape (n_samples,)
            Sample weights for weighted least squares.

        Returns
        -------
        stump : dict
            Dictionary with keys 'feature', 'threshold', 'left_val', 'right_val'.
        """
        n_samples, n_features = X.shape
        best_loss = np.inf
        best_stump = {'feature': 0, 'threshold': 0.0,
                      'left_val': np.average(z, weights=weights),
                      'right_val': np.average(z, weights=weights)}

        total_wz = np.sum(weights * z)
        total_w = np.sum(weights)

        for feat in range(n_features):
            order = np.argsort(X[:, feat])
            x_sorted = X[order, feat]
            z_sorted = z[order]
            w_sorted = weights[order]

            left_wz = 0.0
            left_w = 0.0

            for i in range(n_samples - 1):
                left_wz += w_sorted[i] * z_sorted[i]
                left_w += w_sorted[i]

                # Skip if same feature value as next point
                if x_sorted[i] == x_sorted[i + 1]:
                    continue

                right_w = total_w - left_w
                right_wz = total_wz - left_wz

                if left_w < 1e-10 or right_w < 1e-10:
                    continue

                left_val = left_wz / left_w
                right_val = right_wz / right_w

                # Weighted MSE reduction
                loss = -(left_wz ** 2 / left_w + right_wz ** 2 / right_w)

                if loss < best_loss:
                    best_loss = loss
                    best_stump = {
                        'feature': feat,
                        'threshold': (x_sorted[i] + x_sorted[i + 1]) / 2.0,
                        'left_val': left_val,
                        'right_val': right_val,
                    }

        return best_stump

    def _predict_regression_stump(self, stump: dict,
                                   X: np.ndarray) -> np.ndarray:
        """Predict using a regression stump.

        Parameters
        ----------
        stump : dict
            Fitted stump with 'feature', 'threshold', 'left_val', 'right_val'.
        X : np.ndarray of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        predictions : np.ndarray of shape (n_samples,)
            Predicted regression values.
        """
        mask = X[:, stump['feature']] <= stump['threshold']
        predictions = np.where(mask, stump['left_val'], stump['right_val'])
        return predictions

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogitBoostClassifier":
        """Fit the LogitBoostClassifier classifier.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data.
        y : np.ndarray of shape (n_samples,)
            Target labels.

        Returns
        -------
        self : LogitBoostClassifier
            Returns the fitted instance.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        n_samples = X.shape[0]
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)

        # Convert y to class indices
        class_to_idx = {c: i for i, c in enumerate(self.classes_)}
        y_idx = np.array([class_to_idx[c] for c in y])

        # One-hot encode targets
        Y = np.zeros((n_samples, self.n_classes_))
        Y[np.arange(n_samples), y_idx] = 1

        # Initialize F (log-odds) to zero
        F = np.zeros((n_samples, self.n_classes_))

        # Initialize estimators list for each class
        self.estimators_ = [[] for _ in range(self.n_classes_)]

        rng = np.random.RandomState(self.random_state)

        for iteration in range(self.n_iterations):
            # Compute probabilities
            P = self._softmax(F)

            # For each class, fit a regression model
            for k in range(self.n_classes_):
                # Working response (Newton-Raphson step)
                p_k = P[:, k]
                y_k = Y[:, k]

                # Weights for weighted least squares
                weights = p_k * (1 - p_k)
                weights = np.clip(weights, 1e-6, None)

                # Working response
                z = (y_k - p_k) / (weights + 1e-6)
                z = np.clip(z, -4, 4)  # Truncate extreme values

                # Fit a weighted regression stump to the working response
                stump = self._fit_regression_stump(X, z, weights)
                self.estimators_[k].append(stump)

                # Update F
                predictions = self._predict_regression_stump(stump, X)
                F[:, k] += self.shrinkage * predictions

        self._is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities for samples.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Test data.

        Returns
        -------
        proba : np.ndarray of shape (n_samples, n_classes)
            The class probabilities of the input samples.
        """
        self._check_is_fitted()
        X = np.asarray(X, dtype=float)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        n_samples = X.shape[0]
        F = np.zeros((n_samples, self.n_classes_))

        # Sum predictions from all estimators
        for k in range(self.n_classes_):
            for stump in self.estimators_[k]:
                predictions = self._predict_regression_stump(stump, X)
                F[:, k] += self.shrinkage * predictions

        return self._softmax(F)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for samples.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Test data.

        Returns
        -------
        y_pred : np.ndarray of shape (n_samples,)
            Predicted class labels.
        """
        proba = self.predict_proba(X)
        indices = np.argmax(proba, axis=1)
        return self.classes_[indices]

    def __repr__(self) -> str:
        if self._is_fitted:
            return f"LogitBoostClassifier(base={self.base_classifier}, n_iter={self.n_iterations})"
        return f"LogitBoostClassifier(base_classifier={self.base_classifier})"
