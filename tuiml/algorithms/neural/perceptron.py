"""PerceptronClassifier classifier implementation."""

import numpy as np
from typing import Dict, List, Any, Optional

from tuiml.base.algorithms import Classifier, classifier

@classifier(tags=["functions", "neural-network", "linear"], version="1.0.0")
class PerceptronClassifier(Classifier):
    """PerceptronClassifier classifier.

    A single-layer **neural network** that learns a **linear decision boundary**
    for classification. Supports binary and multiclass classification using a
    one-vs-all strategy.

    Overview
    --------
    The Perceptron trains by iterating over samples and adjusting weights when
    a misclassification occurs:

    1. Initialize weight vectors and bias terms to zero for each class
    2. For each training sample, compute scores across all classes
    3. Predict the class with the highest score
    4. If the prediction is incorrect, update weights: increase weights for the
       correct class and decrease weights for the predicted class
    5. Repeat for multiple epochs until convergence or early stopping

    Theory
    ------
    The Perceptron uses a linear activation with the following update rule.
    Given input :math:`\\mathbf{x}` with true label :math:`y` and predicted
    label :math:`\\hat{y}`:

    .. math::
        \\mathbf{w}_y \\leftarrow \\mathbf{w}_y + \\eta \\, \\mathbf{x}

    .. math::
        \\mathbf{w}_{\\hat{y}} \\leftarrow \\mathbf{w}_{\\hat{y}} - \\eta \\, \\mathbf{x}

    where :math:`\\eta` is the learning rate. The decision function for class
    :math:`k` is:

    .. math::
        f_k(\\mathbf{x}) = \\mathbf{w}_k \\cdot \\mathbf{x} + b_k

    The predicted label is :math:`\\hat{y} = \\arg\\max_k f_k(\\mathbf{x})`.

    Parameters
    ----------
    learning_rate : float, default=1.0
        Learning rate for weight updates.
    max_iter : int, default=1000
        Maximum number of passes over the training data (epochs).
    tol : float, default=1e-3
        Tolerance for stopping criterion based on error rate.
    shuffle : bool, default=True
        Whether to shuffle training data after each epoch.
    random_state : int or None, default=None
        Seed used by the random number generator if ``shuffle`` is True.
    early_stopping : bool, default=True
        Whether to stop training if zero mistakes are made in an epoch.

    Attributes
    ----------
    weights_ : np.ndarray of shape (n_classes, n_features)
        Weight vectors for each class.
    bias_ : np.ndarray of shape (n_classes,)
        Bias terms for each class.
    classes_ : np.ndarray of shape (n_classes,)
        Unique class labels discovered during :meth:`fit`.
    n_iter_ : int
        Number of iterations run during training.

    Notes
    -----
    **Complexity:**

    - Training: :math:`O(n \\cdot d \\cdot T)` where :math:`n` = number of
      samples, :math:`d` = number of features, :math:`T` = max_iter
    - Prediction: :math:`O(n \\cdot d \\cdot K)` where :math:`K` = number of
      classes

    **When to use PerceptronClassifier:**

    - When the data is linearly separable or nearly so
    - When a fast, simple baseline classifier is needed
    - When interpretability of the weight vector is important
    - As a building block before moving to more complex neural models

    References
    ----------
    .. [Rosenblatt1958] Rosenblatt, F. (1958).
           **The Perceptron: A Probabilistic Model for Information Storage and
           Organization in the Brain.**
           *Psychological Review*, 65(6), 386-408.

    .. [Novikoff1963] Novikoff, A.B. (1963).
           **On Convergence Proofs for Perceptrons.**
           *Symposium on the Mathematical Theory of Automata*, 12, 615-622.

    See Also
    --------
    :class:`~tuiml.algorithms.neural.MultilayerPerceptronClassifier` : Multi-layer neural network with backpropagation.

    Examples
    --------
    Train a Perceptron on a simple binary classification task:

    >>> from tuiml.algorithms.neural import PerceptronClassifier
    >>> import numpy as np
    >>> X = np.array([[1, 2], [2, 3], [4, 5], [5, 6]])
    >>> y = np.array([0, 0, 1, 1])
    >>> clf = PerceptronClassifier(learning_rate=0.1, max_iter=100)
    >>> clf.fit(X, y)
    PerceptronClassifier(n_iter=..., n_classes=2)
    >>> clf.predict([[3, 4]])
    array([0])
    """

    def __init__(
        self,
        learning_rate: float = 1.0,
        max_iter: int = 1000,
        tol: float = 1e-3,
        shuffle: bool = True,
        random_state: Optional[int] = None,
        early_stopping: bool = True
    ):
        super().__init__()
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.shuffle = shuffle
        self.random_state = random_state
        self.early_stopping = early_stopping
        self.weights_ = None
        self.bias_ = None
        self.classes_ = None
        self.n_iter_ = 0
        self._n_features = None

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "learning_rate": {
                "type": "number",
                "default": 1.0,
                "minimum": 0,
                "description": "Learning rate for weight updates"
            },
            "max_iter": {
                "type": "integer",
                "default": 1000,
                "minimum": 1,
                "description": "Maximum number of passes over training data"
            },
            "tol": {
                "type": "number",
                "default": 1e-3,
                "minimum": 0,
                "description": "Tolerance for stopping criterion"
            },
            "shuffle": {
                "type": "boolean",
                "default": True,
                "description": "Whether to shuffle data each epoch"
            },
            "early_stopping": {
                "type": "boolean",
                "default": True,
                "description": "Stop if no mistakes in an epoch"
            },
            "random_state": {
                "type": ["integer", "null"],
                "default": None,
                "description": "Random seed for reproducibility",
            },
        }

    @classmethod
    def get_capabilities(cls) -> List[str]:
        return ["numeric", "binary_class", "multiclass"]

    @classmethod
    def get_complexity(cls) -> str:
        return "O(n * n_features * max_iter)"

    @classmethod
    def get_references(cls) -> List[str]:
        return [
            "Rosenblatt, F. (1958). The perceptron: A probabilistic model for "
            "information storage and organization in the brain. "
            "Psychological Review, 65(6), 386-408."
        ]

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PerceptronClassifier":
        """Fit the PerceptronClassifier classifier.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training features.
        y : np.ndarray of shape (n_samples,)
            Target labels.

        Returns
        -------
        self : PerceptronClassifier
            Returns the fitted instance.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        n_samples, self._n_features = X.shape
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)

        # Create class to index mapping
        self._class_to_idx = {c: i for i, c in enumerate(self.classes_)}
        y_idx = np.array([self._class_to_idx[c] for c in y])

        # Initialize random state
        rng = np.random.RandomState(self.random_state)

        # Initialize weights and bias
        # For multiclass, we use one-vs-all approach
        self.weights_ = np.zeros((n_classes, self._n_features))
        self.bias_ = np.zeros(n_classes)

        # Training loop
        self.n_iter_ = 0

        for epoch in range(self.max_iter):
            self.n_iter_ = epoch + 1

            # Shuffle data if requested
            if self.shuffle:
                indices = rng.permutation(n_samples)
                X_shuffled = X[indices]
                y_shuffled = y_idx[indices]
            else:
                X_shuffled = X
                y_shuffled = y_idx

            n_mistakes = 0

            # Process each sample
            for i in range(n_samples):
                xi = X_shuffled[i]
                yi = y_shuffled[i]

                # Compute scores for all classes
                scores = self.weights_ @ xi + self.bias_

                # Predict class with highest score
                y_pred = np.argmax(scores)

                # Update if prediction is wrong
                if y_pred != yi:
                    n_mistakes += 1

                    # Update weights for correct class (increase)
                    self.weights_[yi] += self.learning_rate * xi
                    self.bias_[yi] += self.learning_rate

                    # Update weights for predicted class (decrease)
                    self.weights_[y_pred] -= self.learning_rate * xi
                    self.bias_[y_pred] -= self.learning_rate

            # Early stopping if no mistakes
            if self.early_stopping and n_mistakes == 0:
                break

            # Check for convergence based on error rate
            error_rate = n_mistakes / n_samples
            if error_rate < self.tol:
                break

        self._is_fitted = True
        return self

    def partial_fit(self, X: np.ndarray, y: np.ndarray, classes: Optional[np.ndarray] = None) -> "PerceptronClassifier":
        """Incrementally fit the PerceptronClassifier classifier on a batch of samples.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Incremental training features.
        y : np.ndarray of shape (n_samples,)
            Incremental target labels.
        classes : np.ndarray of shape (n_classes,), default=None
            List of all classes expected. Must be provided at the first call,
            can be omitted afterwards.

        Returns
        -------
        self : PerceptronClassifier
            Returns the updated instance.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        n_samples, n_features = X.shape

        if not self._is_fitted:
            self._n_features = n_features
            if classes is not None:
                self.classes_ = np.asarray(classes)
            else:
                self.classes_ = np.unique(y)
            n_classes = len(self.classes_)

            self._class_to_idx = {c: i for i, c in enumerate(self.classes_)}
            
            # Initialize weights and bias
            self.weights_ = np.zeros((n_classes, self._n_features))
            self.bias_ = np.zeros(n_classes)
            self.n_iter_ = 0
            self._is_fitted = True
        else:
            self._n_features = n_features

        y_idx = np.array([self._class_to_idx[c] for c in y])
        rng = np.random.RandomState(self.random_state)

        # Shuffle if requested
        if self.shuffle:
            indices = rng.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y_idx[indices]
        else:
            X_shuffled = X
            y_shuffled = y_idx

        # Process each sample (single pass/epoch over the batch)
        for i in range(n_samples):
            xi = X_shuffled[i]
            yi = y_shuffled[i]

            scores = self.weights_ @ xi + self.bias_
            y_pred = np.argmax(scores)

            if y_pred != yi:
                self.weights_[yi] += self.learning_rate * xi
                self.bias_[yi] += self.learning_rate
                self.weights_[y_pred] -= self.learning_rate * xi
                self.bias_[y_pred] -= self.learning_rate

        self.n_iter_ += 1
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Compute decision scores for samples.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input features.

        Returns
        -------
        scores : np.ndarray of shape (n_samples, n_classes)
            Confidence scores for each class.
        """
        self._check_is_fitted()
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        return X @ self.weights_.T + self.bias_

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for samples.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input features.

        Returns
        -------
        y_pred : np.ndarray of shape (n_samples,)
            Predicted class labels.
        """
        scores = self.decision_function(X)
        indices = np.argmax(scores, axis=1)
        return self.classes_[indices]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Estimate class probabilities for samples.

        PerceptronClassifier doesn't naturally produce probabilities. This method 
        applies a softmax function to decision scores to generate 
        pseudo-probabilities.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input features.

        Returns
        -------
        probabilities : np.ndarray of shape (n_samples, n_classes)
            Predicted class probabilities.
        """
        scores = self.decision_function(X)
        # Apply softmax
        scores_shifted = scores - np.max(scores, axis=1, keepdims=True)
        exp_scores = np.exp(scores_shifted)
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    def __repr__(self) -> str:
        if self._is_fitted:
            return f"PerceptronClassifier(n_iter={self.n_iter_}, n_classes={len(self.classes_)})"
        return f"PerceptronClassifier(learning_rate={self.learning_rate}, max_iter={self.max_iter})"

