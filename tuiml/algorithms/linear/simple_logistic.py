"""Linear logistic regression models built using LogitBoost."""

import numpy as np
from typing import Dict, List, Any, Optional

from tuiml.base.algorithms import Classifier, classifier

@classifier(tags=["linear", "logistic", "boosting"], version="1.0.0")
class SimpleLogisticRegression(Classifier):
    r"""SimpleLogisticRegression classifier using **LogitBoost** with simple base learners.

    Builds linear logistic regression models using **LogitBoost** with simple
    regression functions as base learners. It automatically determines the
    optimal number of boosting iterations using **cross-validation**.

    Overview
    --------
    The algorithm builds a logistic model via additive boosting:

    1. Initialize the additive function :math:`F(x)` to zero
    2. At each boosting iteration, fit a weighted least-squares regression
       to the working response derived from the current class probabilities
    3. Update :math:`F(x)` with the new base learner output
    4. Optionally use cross-validation to select the optimal number of iterations
    5. Return the accumulated linear model as the final classifier

    Theory
    ------
    LogitBoost fits the logistic regression model by **additive modeling**:

    .. math::
        F(x) = \sum_{i=1}^{M} f_i(x)

    where each :math:`f_i(x)` is a simple linear regression function. At each
    iteration, the algorithm computes a working response :math:`z` and
    observation weights :math:`w`:

    .. math::
        z_i = \\frac{y_i - p_i}{p_i (1 - p_i)}, \quad w_i = p_i (1 - p_i)

    where :math:`p_i = P(y=1 \mid x_i)` is the current probability estimate.
    A weighted least-squares regression is then fit to :math:`(x_i, z_i)` with
    weights :math:`w_i`.

    Parameters
    ----------
    n_boosting_iterations : int, default=500
        Maximum number of boosting iterations to perform.
    use_cross_validation : bool, default=True
        Whether to use cross-validation to automatically determine the
        optimal number of boosting iterations.
    n_folds : int, default=5
        Number of folds to use for cross-validation.
    weight_trim_beta : float, default=0
        Beta parameter used for weight trimming in LogitBoost.
    use_aic : bool, default=False
        Whether to use Akaike Information Criterion (AIC) as an
        alternative early stopping criterion.

    Attributes
    ----------
    weights_ : np.ndarray
        Learned linear coefficients of shape (n_classes, n_features + 1),
        including an intercept term in the first column.
    n_iterations_ : int
        The actual number of boosting iterations performed.
    classes_ : np.ndarray
        Unique class labels found during training.

    Notes
    -----
    **Complexity:**

    - Training: :math:`O(n \cdot m \cdot \\text{n\_iterations})` where :math:`n`
      is the number of samples and :math:`m` is the number of features.
    - Prediction: :math:`O(m \cdot c)` per sample where :math:`c` is the
      number of classes.

    **When to use SimpleLogisticRegression:**

    - When you want a logistic regression model with built-in regularization via early stopping
    - When automatic selection of model complexity (number of iterations) is desired
    - As a component of logistic model trees (LMT)
    - When the decision boundary is approximately linear but you want boosting-based fitting

    References
    ----------
    .. [Landwehr2005] Landwehr, N., Hall, M. and Frank, E. (2005).
           **Logistic Model Trees.**
           *Machine Learning*, 59(1-2), 161-205.

    .. [Friedman2000] Friedman, J., Hastie, T. and Tibshirani, R. (2000).
           **Additive Logistic Regression: A Statistical View of Boosting.**
           *The Annals of Statistics*, 28(2), 337-407.

    See Also
    --------
    :class:`~tuiml.algorithms.linear.LogisticRegression` : Standard logistic regression with gradient descent and L2 regularization.
    :class:`~tuiml.algorithms.linear.SGDClassifier` : SGD-trained linear classifier for large-scale problems.

    Examples
    --------
    Binary classification using LogitBoost with cross-validated iterations:

    >>> import numpy as np
    >>> from tuiml.algorithms.linear import SimpleLogisticRegression
    >>>
    >>> X = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
    >>> y = np.array([0, 0, 1, 1])
    >>>
    >>> clf = SimpleLogisticRegression(n_boosting_iterations=50)
    >>> clf.fit(X, y)
    >>>
    >>> clf.predict([[1.5, 1.5]])
    array([0])
    """

    def __init__(self, n_boosting_iterations: int = 500,
                 use_cross_validation: bool = True,
                 n_folds: int = 5,
                 weight_trim_beta: float = 0,
                 use_aic: bool = False):
        """Initialize SimpleLogisticRegression classifier.

        Parameters
        ----------
        n_boosting_iterations : int, default=500
            Maximum number of boosting iterations.
        use_cross_validation : bool, default=True
            Whether to use CV for iteration selection.
        n_folds : int, default=5
            Number of cross-validation folds.
        weight_trim_beta : float, default=0
            Weight trimming parameter.
        use_aic : bool, default=False
            Whether to use AIC for early stopping.
        """
        super().__init__()
        self.n_boosting_iterations = n_boosting_iterations
        self.use_cross_validation = use_cross_validation
        self.n_folds = n_folds
        self.weight_trim_beta = weight_trim_beta
        self.use_aic = use_aic
        self.weights_ = None
        self.n_iterations_ = None
        self.classes_ = None
        self._n_classes = None

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "n_boosting_iterations": {
                "type": "integer", "default": 500, "minimum": 1,
                "description": "Maximum boosting iterations"
            },
            "use_cross_validation": {
                "type": "boolean", "default": True,
                "description": "Use CV to select iterations"
            },
            "n_folds": {
                "type": "integer", "default": 5, "minimum": 2,
                "description": "Number of CV folds"
            },
            "weight_trim_beta": {
                "type": "number", "default": 0, "minimum": 0, "maximum": 1,
                "description": "Weight trimming beta"
            },
            "use_aic": {
                "type": "boolean", "default": False,
                "description": "Use AIC for early stopping"
            }
        }

    @classmethod
    def get_capabilities(cls) -> List[str]:
        return ["numeric", "binary_class", "multiclass"]

    @classmethod
    def get_complexity(cls) -> str:
        return "O(n * d * n_iterations)"

    @classmethod
    def get_references(cls) -> List[str]:
        return [
            "Landwehr, N., Hall, M., & Frank, E. (2005). Logistic Model Trees. "
            "Machine Learning, 59(1-2), 161-205."
        ]

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Numerically stable sigmoid function.

        Parameters
        ----------
        x : np.ndarray
            Raw logit values.

        Returns
        -------
        result : np.ndarray
            Sigmoid-transformed values in the range (0, 1).
        """
        return np.where(
            x >= 0,
            1 / (1 + np.exp(-x)),
            np.exp(x) / (1 + np.exp(x))
        )

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Compute softmax probabilities.

        Parameters
        ----------
        x : np.ndarray of shape (n_samples, n_classes)
            Raw logit values.

        Returns
        -------
        result : np.ndarray of shape (n_samples, n_classes)
            Softmax probability distributions per sample.
        """
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / (np.sum(exp_x, axis=1, keepdims=True) + 1e-10)

    def _logit_boost_step(self, X: np.ndarray, y: np.ndarray,
                          F: np.ndarray, learning_rate: float = 0.5) -> np.ndarray:
        """Perform one LogitBoost iteration.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training features.
        y : np.ndarray of shape (n_samples,)
            Integer-encoded target labels.
        F : np.ndarray of shape (n_samples, n_classes) or (n_samples, 1)
            Current additive model output (modified in place).
        learning_rate : float, default=0.5
            Shrinkage factor for the update step.

        Returns
        -------
        betas : np.ndarray
            Coefficient updates of shape (1, n_features + 1) for binary
            or (n_classes, n_features + 1) for multiclass.
        """
        n_samples = len(y)
        n_features = X.shape[1]

        if self._n_classes == 2:
            # Binary case
            p = self._sigmoid(F[:, 0])
            weights = p * (1 - p) + 1e-6
            z = (y - p) / weights

            # Weighted least squares for simple regression
            # Fit f(x) = beta_0 + beta_1 * X
            # Use broadcasting instead of full diagonal matrix (O(n) vs O(n^2) space)
            sqrt_weights = np.sqrt(weights)
            X_aug = np.column_stack([np.ones(n_samples), X])
            X_weighted = X_aug * sqrt_weights[:, np.newaxis]
            z_weighted = z * sqrt_weights

            try:
                beta = np.linalg.lstsq(X_weighted, z_weighted, rcond=None)[0]
            except np.linalg.LinAlgError:
                beta = np.zeros(n_features + 1)

            # Update F
            F[:, 0] += learning_rate * (X_aug @ beta)

            return beta.reshape(1, -1)
        else:
            # Multiclass case
            P = self._softmax(F)
            betas = np.zeros((self._n_classes, n_features + 1))

            X_aug = np.column_stack([np.ones(n_samples), X])

            for k in range(self._n_classes):
                y_k = (y == k).astype(float)
                p_k = P[:, k]
                weights = p_k * (1 - p_k) + 1e-6
                z = (y_k - p_k) / weights

                # Use broadcasting instead of full diagonal matrix
                sqrt_weights = np.sqrt(weights)
                X_weighted = X_aug * sqrt_weights[:, np.newaxis]
                z_weighted = z * sqrt_weights

                try:
                    beta = np.linalg.lstsq(X_weighted, z_weighted, rcond=None)[0]
                except np.linalg.LinAlgError:
                    beta = np.zeros(n_features + 1)

                F[:, k] += learning_rate * (X_aug @ beta)
                betas[k] = beta

            return betas

    def _compute_log_likelihood(self, X: np.ndarray, y: np.ndarray,
                                weights: np.ndarray) -> float:
        """Compute log-likelihood for model.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Feature matrix.
        y : np.ndarray of shape (n_samples,)
            Integer-encoded target labels.
        weights : np.ndarray
            Current model weight matrix.

        Returns
        -------
        ll : float
            Log-likelihood value.
        """
        X_aug = np.column_stack([np.ones(len(X)), X])

        if self._n_classes == 2:
            logits = X_aug @ weights[0]
            p = self._sigmoid(logits)
            ll = np.sum(y * np.log(p + 1e-10) + (1 - y) * np.log(1 - p + 1e-10))
        else:
            logits = X_aug @ weights.T
            P = self._softmax(logits)
            ll = np.sum(np.log(P[np.arange(len(y)), y.astype(int)] + 1e-10))

        return ll

    def _cross_validate_iterations(self, X: np.ndarray, y: np.ndarray) -> int:
        """Find optimal number of iterations using cross-validation.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training features.
        y : np.ndarray of shape (n_samples,)
            Integer-encoded target labels.

        Returns
        -------
        best_iter : int
            Optimal number of boosting iterations.
        """
        n_samples = len(y)
        fold_size = n_samples // self.n_folds
        indices = np.arange(n_samples)
        np.random.shuffle(indices)

        best_iter = self.n_boosting_iterations
        best_ll = -np.inf

        # Try different iteration counts
        iteration_candidates = [10, 25, 50, 100, 200, 500]
        iteration_candidates = [i for i in iteration_candidates
                               if i <= self.n_boosting_iterations]

        for n_iter in iteration_candidates:
            cv_ll = 0

            for fold in range(self.n_folds):
                val_start = fold * fold_size
                val_end = val_start + fold_size if fold < self.n_folds - 1 else n_samples

                val_idx = indices[val_start:val_end]
                train_idx = np.concatenate([indices[:val_start], indices[val_end:]])

                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                # Train with n_iter iterations
                n_features = X_train.shape[1]
                if self._n_classes == 2:
                    F = np.zeros((len(y_train), 1))
                    weights = np.zeros((1, n_features + 1))
                else:
                    F = np.zeros((len(y_train), self._n_classes))
                    weights = np.zeros((self._n_classes, n_features + 1))

                for _ in range(n_iter):
                    delta = self._logit_boost_step(X_train, y_train, F)
                    weights += delta * 0.5  # Accumulate weights

                cv_ll += self._compute_log_likelihood(X_val, y_val, weights)

            if cv_ll > best_ll:
                best_ll = cv_ll
                best_iter = n_iter

        return best_iter

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SimpleLogisticRegression":
        """Fit the SimpleLogisticRegression classifier.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data.
        y : np.ndarray of shape (n_samples,)
            Target labels.

        Returns
        -------
        self : SimpleLogisticRegression
            Fitted classifier.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        n_samples, n_features = X.shape
        self.classes_ = np.unique(y)
        self._n_classes = len(self.classes_)

        # Convert y to indices
        class_to_idx = {c: i for i, c in enumerate(self.classes_)}
        y_idx = np.array([class_to_idx[c] for c in y])

        # Determine number of iterations
        if self.use_cross_validation and n_samples >= 2 * self.n_folds:
            self.n_iterations_ = self._cross_validate_iterations(X, y_idx)
        else:
            self.n_iterations_ = min(self.n_boosting_iterations, n_samples // 2)

        # Initialize F and weights
        if self._n_classes == 2:
            F = np.zeros((n_samples, 1))
            self.weights_ = np.zeros((1, n_features + 1))
        else:
            F = np.zeros((n_samples, self._n_classes))
            self.weights_ = np.zeros((self._n_classes, n_features + 1))

        # LogitBoost training
        for _ in range(self.n_iterations_):
            delta = self._logit_boost_step(X, y_idx, F)
            self.weights_ += delta * 0.5

        self._is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities for samples.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Test features to predict.

        Returns
        -------
        probabilities : np.ndarray of shape (n_samples, n_classes)
            Predicted probabilities for each class.
        """
        self._check_is_fitted()
        X = np.asarray(X, dtype=float)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        X_aug = np.column_stack([np.ones(len(X)), X])

        if self._n_classes == 2:
            logits = X_aug @ self.weights_[0]
            p = self._sigmoid(logits)
            return np.column_stack([1 - p, p])
        else:
            logits = X_aug @ self.weights_.T
            return self._softmax(logits)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for samples.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Test features.

        Returns
        -------
        predictions : np.ndarray
            Predicted class labels for each sample.
        """
        proba = self.predict_proba(X)
        indices = np.argmax(proba, axis=1)
        return self.classes_[indices]

    def __repr__(self) -> str:
        if self._is_fitted:
            return f"SimpleLogisticRegression(n_iterations={self.n_iterations_})"
        return f"SimpleLogisticRegression(max_iterations={self.n_boosting_iterations})"
