"""Logistic Regression classifier with L2 regularization."""

import numpy as np
from typing import Dict, List, Any, Optional

from tuiml.base.algorithms import Classifier, classifier

@classifier(tags=["functions", "linear", "probabilistic"], version="1.0.0")
class LogisticRegression(Classifier):
    r"""Logistic Regression classifier with **L2 regularization**.

    Logistic regression models the probability of class membership using the
    **logistic (sigmoid) function**. It supports both binary and multiclass
    classification via the softmax generalization.

    Overview
    --------
    The algorithm fits a linear decision boundary by iterative optimization:

    1. Initialize weight matrix and bias to zeros
    2. Compute predicted probabilities using sigmoid (binary) or softmax (multiclass)
    3. Compute the cross-entropy loss with L2 regularization penalty
    4. Update weights via batch gradient descent
    5. Repeat until convergence or ``max_iter`` is reached

    For binary problems the model uses a single sigmoid output; for multiclass
    problems it uses the softmax function over all classes simultaneously.

    Theory
    ------
    For binary classification, the model predicts the probability of the
    positive class using the **sigmoid function**:

    .. math::
        P(y=1 \mid x) = \sigma(w^T x + b) = \\frac{1}{1 + e^{-(w^T x + b)}}

    For multiclass classification with :math:`c` classes, the **softmax function**
    is applied:

    .. math::
        P(y=k \mid x) = \\frac{e^{w_k^T x + b_k}}{\sum_{j=1}^{c} e^{w_j^T x + b_j}}

    The loss function minimized is the **regularized cross-entropy**:

    .. math::
        \mathcal{L}(w) = -\\frac{1}{n}\sum_{i=1}^{n} \sum_{k=1}^{c}
        y_{ik} \log P(y=k \mid x_i) + \\frac{\lambda}{2} \|w\|_2^2

    where :math:`\lambda` is the ridge regularization parameter.

    Parameters
    ----------
    max_iter : int, default=100
        Maximum number of iterations for the gradient descent solver.
    learning_rate : float, default=1.0
        Step size for weight updates.
    ridge : float, default=1e-8
        L2 regularization parameter (penalty) to prevent overfitting.
    tol : float, default=1e-4
        Convergence tolerance. Training stops when the improvement
        in loss is less than this value.

    Attributes
    ----------
    classes_ : np.ndarray
        Unique class labels found during training.
    coef_ : np.ndarray
        Weight matrix of shape (n_classes, n_features).
    intercept_ : np.ndarray
        Bias vector of shape (n_classes,).
    n_iter_ : int
        Actual number of iterations run during training.

    Notes
    -----
    **Complexity:**

    - Training: :math:`O(n \cdot m \cdot c \cdot \\text{iterations})` where :math:`n` is
      the number of samples, :math:`m` features, and :math:`c` classes.
    - Prediction: :math:`O(m \cdot c)` per sample.

    **When to use LogisticRegression:**

    - When you need a probabilistic classifier with calibrated probabilities
    - Binary or multiclass problems with linearly separable (or near-linear) classes
    - When model interpretability via feature coefficients is important
    - As a baseline before trying more complex nonlinear models

    References
    ----------
    .. [LeCessie1992] le Cessie, S. and van Houwelingen, J.C. (1992).
           **Ridge Estimators in Logistic Regression.**
           *Applied Statistics*, 41(1), 191-201.

    .. [Bishop2006] Bishop, C.M. (2006).
           **Pattern Recognition and Machine Learning.**
           *Springer*, Chapter 4.

    See Also
    --------
    :class:`~tuiml.algorithms.linear.SimpleLogisticRegression` : LogitBoost-based logistic regression with automatic iteration selection.
    :class:`~tuiml.algorithms.linear.SGDClassifier` : SGD-trained linear classifier supporting hinge, log, and modified Huber losses.

    Examples
    --------
    Basic binary classification with logistic regression:

    >>> import numpy as np
    >>> from tuiml.algorithms.linear import LogisticRegression
    >>>
    >>> # Generate binary classification data
    >>> X = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
    >>> y = np.array([0, 0, 1, 1])
    >>>
    >>> # Fit model
    >>> clf = LogisticRegression()
    >>> clf.fit(X, y)
    >>>
    >>> # Predict classes
    >>> clf.predict([[1.5, 1.5], [4, 4]])
    array([0, 1])
    >>>
    >>> # Get probabilities
    >>> clf.predict_proba([[1.5, 1.5]])
    array([[0.5, 0.5]])
    """

    def __init__(self, max_iter: int = 100,
                 learning_rate: float = 1.0,
                 ridge: float = 1e-8,
                 tol: float = 1e-4):
        """Initialize LogisticRegression classifier with optimization parameters.

        Parameters
        ----------
        max_iter : int, default=100
            Maximum number of iterations.
        learning_rate : float, default=1.0
            Step size for gradient descent.
        ridge : float, default=1e-8
            L2 regularization strength.
        tol : float, default=1e-4
            Convergence threshold.
        """
        super().__init__()
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.ridge = ridge
        self.tol = tol
        self.classes_ = None
        self.coef_ = None
        self.intercept_ = None
        self.n_iter_ = None
        self._n_features = None

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        """Return parameter schema."""
        return {
            "max_iter": {
                "type": "integer",
                "default": 100,
                "minimum": 1,
                "description": "Maximum number of iterations"
            },
            "learning_rate": {
                "type": "number",
                "default": 0.1,
                "minimum": 0,
                "description": "Learning rate for gradient descent"
            },
            "ridge": {
                "type": "number",
                "default": 1e-8,
                "minimum": 0,
                "description": "L2 regularization parameter"
            },
            "tol": {
                "type": "number",
                "default": 1e-4,
                "minimum": 0,
                "description": "Convergence tolerance"
            }
        }

    @classmethod
    def get_capabilities(cls) -> List[str]:
        """Return classifier capabilities."""
        return [
            "numeric",
            "binary_class",
            "multiclass"
        ]

    @classmethod
    def get_complexity(cls) -> str:
        """Return time/space complexity."""
        return "O(n * m * c * iterations) training, O(m * c) prediction"

    @classmethod
    def get_references(cls) -> List[str]:
        """Return academic references."""
        return [
            "le Cessie, S., & van Houwelingen, J.C. (1992). Ridge Estimators "
            "in Logistic Regression. Applied Statistics, 41(1), 191-201."
        ]

    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Compute sigmoid function with numerical stability.

        Parameters
        ----------
        z : np.ndarray
            Raw logit values.

        Returns
        -------
        result : np.ndarray
            Sigmoid-transformed values in the range (0, 1).
        """
        # Clip values to avoid overflow
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def _softmax(self, z: np.ndarray) -> np.ndarray:
        """Compute softmax function with numerical stability.

        Parameters
        ----------
        z : np.ndarray of shape (n_samples, n_classes)
            Raw logit values.

        Returns
        -------
        result : np.ndarray of shape (n_samples, n_classes)
            Softmax-transformed probability distributions per sample.
        """
        # Subtract max for numerical stability
        z_shifted = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z_shifted)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def _fit_binary(self, X: np.ndarray, y: np.ndarray):
        """Fit binary logistic regression using batch gradient descent.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Standardized training features.
        y : np.ndarray of shape (n_samples,)
            Binary target labels (0 or 1).

        Returns
        -------
        None
            Updates ``coef_``, ``intercept_``, and ``n_iter_`` in place.
        """
        n_samples, n_features = X.shape

        # Initialize weights
        self.coef_ = np.zeros((1, n_features))
        self.intercept_ = np.zeros(1)

        prev_loss = np.inf

        for iteration in range(self.max_iter):
            # Forward pass
            z = X @ self.coef_.T + self.intercept_
            proba = self._sigmoid(z.ravel())

            # Compute loss with regularization
            loss = -np.mean(y * np.log(proba + 1e-10) +
                           (1 - y) * np.log(1 - proba + 1e-10))
            loss += 0.5 * self.ridge * np.sum(self.coef_ ** 2)

            # Check convergence
            if abs(prev_loss - loss) < self.tol:
                self.n_iter_ = iteration + 1
                break
            prev_loss = loss

            # Compute gradients
            error = proba - y
            grad_w = (X.T @ error) / n_samples + self.ridge * self.coef_.ravel()
            grad_b = np.mean(error)

            # Update weights
            self.coef_ -= self.learning_rate * grad_w.reshape(1, -1)
            self.intercept_ -= self.learning_rate * grad_b
        else:
            self.n_iter_ = self.max_iter

    def _fit_multiclass(self, X: np.ndarray, y: np.ndarray):
        """Fit multiclass logistic regression using softmax and gradient descent.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Standardized training features.
        y : np.ndarray of shape (n_samples,)
            Integer-encoded target labels.

        Returns
        -------
        None
            Updates ``coef_``, ``intercept_``, and ``n_iter_`` in place.
        """
        n_samples, n_features = X.shape
        n_classes = len(self.classes_)

        # Initialize weights
        self.coef_ = np.zeros((n_classes, n_features))
        self.intercept_ = np.zeros(n_classes)

        # One-hot encode y (vectorized)
        y_onehot = np.eye(n_classes)[y]

        prev_loss = np.inf

        for iteration in range(self.max_iter):
            # Forward pass
            z = X @ self.coef_.T + self.intercept_
            proba = self._softmax(z)

            # Compute cross-entropy loss with regularization
            loss = -np.mean(np.sum(y_onehot * np.log(proba + 1e-10), axis=1))
            loss += 0.5 * self.ridge * np.sum(self.coef_ ** 2)

            # Check convergence
            if abs(prev_loss - loss) < self.tol:
                self.n_iter_ = iteration + 1
                break
            prev_loss = loss

            # Compute gradients
            error = proba - y_onehot
            grad_w = (X.T @ error).T / n_samples + self.ridge * self.coef_
            grad_b = np.mean(error, axis=0)

            # Update weights
            self.coef_ -= self.learning_rate * grad_w
            self.intercept_ -= self.learning_rate * grad_b
        else:
            self.n_iter_ = self.max_iter

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegression":
        """Fit the Logistic Regression classifier.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training features.
        y : np.ndarray of shape (n_samples,)
            Target labels.

        Returns
        -------
        self : Logistic
            Fitted classifier.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self._n_features = X.shape[1]
        self.classes_ = np.unique(y)

        # Convert y to integer indices
        class_to_idx = {c: i for i, c in enumerate(self.classes_)}
        y_idx = np.array([class_to_idx[c] for c in y])

        # Standardize features for better convergence
        self._mean = np.mean(X, axis=0)
        self._std = np.std(X, axis=0)
        self._std[self._std == 0] = 1  # Avoid division by zero
        X_scaled = (X - self._mean) / self._std

        # Fit model
        if len(self.classes_) == 2:
            self._fit_binary(X_scaled, y_idx)
        else:
            self._fit_multiclass(X_scaled, y_idx)

        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for samples.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Test features to predict.

        Returns
        -------
        predictions : np.ndarray
            Predicted class labels for each sample.
        """
        self._check_is_fitted()
        X = np.asarray(X, dtype=float)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        # Scale features
        X_scaled = (X - self._mean) / self._std

        if len(self.classes_) == 2:
            # Binary classification
            z = X_scaled @ self.coef_.T + self.intercept_
            proba = self._sigmoid(z.ravel())
            pred_idx = (proba >= 0.5).astype(int)
        else:
            # Multiclass classification
            z = X_scaled @ self.coef_.T + self.intercept_
            pred_idx = np.argmax(z, axis=1)

        return self.classes_[pred_idx]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities for samples.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Test features.

        Returns
        -------
        probabilities : np.ndarray of shape (n_samples, n_classes)
            Predicted probabilities for each class.
        """
        self._check_is_fitted()
        X = np.asarray(X, dtype=float)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        # Scale features
        X_scaled = (X - self._mean) / self._std

        if len(self.classes_) == 2:
            # Binary classification
            z = X_scaled @ self.coef_.T + self.intercept_
            proba_1 = self._sigmoid(z.ravel())
            proba = np.column_stack([1 - proba_1, proba_1])
        else:
            # Multiclass classification
            z = X_scaled @ self.coef_.T + self.intercept_
            proba = self._softmax(z)

        return proba

    def __repr__(self) -> str:
        """String representation."""
        if self._is_fitted:
            return (f"LogisticRegression(n_classes={len(self.classes_)}, "
                   f"n_iter={self.n_iter_})")
        return f"LogisticRegression(max_iter={self.max_iter}, ridge={self.ridge})"
