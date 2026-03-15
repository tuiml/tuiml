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
    :class:`~tuiml.algorithms.neural.VotedPerceptronClassifier` : Voted variant with weighted voting for improved robustness.
    :class:`~tuiml.algorithms.neural.AveragedPerceptronClassifier` : Averaged variant for better generalization.
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
            }
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

@classifier(tags=["functions", "neural-network", "linear", "kernel"], version="1.0.0")
class VotedPerceptronClassifier(Classifier):
    """Voted Perceptron classifier.

    A variant of the Perceptron that keeps track of all **weight vectors**
    generated during training and uses **weighted voting** for prediction based
    on their survival time (how many samples they correctly classified before
    an update). This is generally more robust than the standard perceptron.

    Overview
    --------
    The Voted Perceptron maintains a history of weight vectors during training:

    1. Initialize a weight vector and bias to zero
    2. For each training sample, compute the prediction using the current weights
    3. If the prediction is correct, increment the survival counter
    4. If the prediction is incorrect, store the current weight vector along with
       its survival count, then update the weights using the perceptron rule
    5. At prediction time, each stored weight vector casts a vote weighted by
       its survival time

    Theory
    ------
    The Voted Perceptron stores a sequence of weight vectors
    :math:`(\\mathbf{w}_1, c_1), (\\mathbf{w}_2, c_2), \\ldots, (\\mathbf{w}_k, c_k)`
    where :math:`c_i` is the survival time of :math:`\\mathbf{w}_i`. The
    prediction for a new instance :math:`\\mathbf{x}` is:

    .. math::
        \\hat{y} = \\text{sign}\\left(\\sum_{i=1}^{k} c_i \\, \\text{sign}(\\mathbf{w}_i \\cdot \\mathbf{x} + b_i)\\right)

    With a polynomial kernel of exponent :math:`p`, the dot product is replaced
    by:

    .. math::
        K(\\mathbf{x}_1, \\mathbf{x}_2) = (\\mathbf{x}_1 \\cdot \\mathbf{x}_2 + 1)^p

    Parameters
    ----------
    max_iter : int, default=1
        Maximum number of passes over the training data (epochs).
    exponent : float, default=1.0
        Exponent for the polynomial kernel. Use 1.0 for a linear model.
    random_state : int or None, default=None
        Seed used by the random number generator.

    Attributes
    ----------
    weight_vectors_ : list of list of (np.ndarray, float)
        Stored weight vectors and biases for each one-vs-all classifier.
    survival_times_ : list of list of int
        Survival times corresponding to each stored weight vector.
    classes_ : np.ndarray of shape (n_classes,)
        Unique class labels discovered during :meth:`fit`.

    Notes
    -----
    **Complexity:**

    - Training: :math:`O(n \\cdot d \\cdot T)` where :math:`n` = number of
      samples, :math:`d` = number of features, :math:`T` = max_iter
    - Prediction: :math:`O(V \\cdot d)` where :math:`V` = total number of
      stored weight vectors

    **When to use VotedPerceptronClassifier:**

    - When the standard perceptron overfits or is unstable
    - When improved generalization is desired without additional tuning
    - When the data is linearly separable but noisy
    - When a theoretically motivated margin-based classifier is needed

    References
    ----------
    .. [FreundSchapire1999] Freund, Y. and Schapire, R.E. (1999).
           **Large Margin Classification Using the Perceptron Algorithm.**
           *Machine Learning*, 37(3), 277-296.
           DOI: `10.1023/A:1007662407062 <https://doi.org/10.1023/A:1007662407062>`_

    See Also
    --------
    :class:`~tuiml.algorithms.neural.PerceptronClassifier` : Standard single-layer perceptron classifier.
    :class:`~tuiml.algorithms.neural.AveragedPerceptronClassifier` : Averaged variant for better generalization.
    :class:`~tuiml.algorithms.neural.MultilayerPerceptronClassifier` : Multi-layer neural network with backpropagation.

    Examples
    --------
    Train a Voted Perceptron on a simple binary classification task:

    >>> from tuiml.algorithms.neural import VotedPerceptronClassifier
    >>> import numpy as np
    >>> X = np.array([[1, 2], [2, 3], [4, 5], [5, 6]])
    >>> y = np.array([0, 0, 1, 1])
    >>> clf = VotedPerceptronClassifier(max_iter=3)
    >>> clf.fit(X, y)
    VotedPerceptronClassifier(n_vectors=...)
    >>> clf.predict([[3, 4]])
    array([0])
    """

    def __init__(
        self,
        max_iter: int = 1,
        exponent: float = 1.0,
        random_state: Optional[int] = None
    ):
        super().__init__()
        self.max_iter = max_iter
        self.exponent = exponent
        self.random_state = random_state
        self.weight_vectors_ = None
        self.bias_vectors_ = None
        self.survival_times_ = None
        self.classes_ = None
        self._n_features = None

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "max_iter": {
                "type": "integer",
                "default": 1,
                "minimum": 1,
                "description": "Maximum passes over training data"
            },
            "exponent": {
                "type": "number",
                "default": 1.0,
                "minimum": 0,
                "description": "Exponent for polynomial kernel"
            }
        }

    @classmethod
    def get_capabilities(cls) -> List[str]:
        return ["numeric", "binary_class", "multiclass"]

    @classmethod
    def get_complexity(cls) -> str:
        return "O(n * n_features * max_iter) training, O(n_vectors * n_features) prediction"

    @classmethod
    def get_references(cls) -> List[str]:
        return [
            "Freund, Y., & Schapire, R.E. (1999). Large margin classification "
            "using the perceptron algorithm. Machine Learning, 37(3), 277-296."
        ]

    def _kernel(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Polynomial kernel function.

        Computes :math:`K(\\mathbf{x}_1, \\mathbf{x}_2) = (\\mathbf{x}_1 \\cdot \\mathbf{x}_2 + 1)^p`
        where :math:`p` is the exponent.

        Parameters
        ----------
        x1 : np.ndarray of shape (n_features,)
            First input vector.
        x2 : np.ndarray of shape (n_features,)
            Second input vector.

        Returns
        -------
        result : float
            The polynomial kernel value.
        """
        return (np.dot(x1, x2) + 1) ** self.exponent

    def fit(self, X: np.ndarray, y: np.ndarray) -> "VotedPerceptronClassifier":
        """Fit the Voted PerceptronClassifier classifier.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training features.
        y : np.ndarray of shape (n_samples,)
            Target labels.

        Returns
        -------
        self : VotedPerceptronClassifier
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

        # Initialize
        rng = np.random.RandomState(self.random_state)

        # For multiclass, one-vs-all
        self.weight_vectors_ = []  # List of (weights, bias) tuples per class
        self.survival_times_ = []

        for cls_idx in range(n_classes):
            # Binary labels for this class
            y_binary = (y_idx == cls_idx).astype(int) * 2 - 1  # +1 or -1

            # Initialize weight and bias
            w = np.zeros(self._n_features)
            b = 0.0

            class_weights = []
            class_survivals = []
            survival = 0

            for epoch in range(self.max_iter):
                indices = rng.permutation(n_samples)

                for i in indices:
                    xi = X[i]
                    yi = y_binary[i]

                    # Compute prediction
                    if self.exponent == 1.0:
                        score = np.dot(w, xi) + b
                    else:
                        score = self._kernel(w, xi) + b

                    y_pred = 1 if score >= 0 else -1

                    if y_pred == yi:
                        survival += 1
                    else:
                        # Save current weight vector with its survival time
                        if survival > 0:
                            class_weights.append((w.copy(), b))
                            class_survivals.append(survival)

                        # Update weights
                        w = w + yi * xi
                        b = b + yi
                        survival = 1

            # Save final weight vector
            if survival > 0:
                class_weights.append((w.copy(), b))
                class_survivals.append(survival)

            self.weight_vectors_.append(class_weights)
            self.survival_times_.append(class_survivals)

        self._is_fitted = True
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Compute weighted decision scores.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input features.

        Returns
        -------
        scores : np.ndarray of shape (n_samples, n_classes)
            Decision scores calculated as the weighted sum of votes.
        """
        self._check_is_fitted()
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        scores = np.zeros((n_samples, n_classes))

        for cls_idx in range(n_classes):
            class_weights = self.weight_vectors_[cls_idx]
            class_survivals = self.survival_times_[cls_idx]

            for (w, b), c in zip(class_weights, class_survivals):
                if self.exponent == 1.0:
                    preds = X @ w + b
                else:
                    preds = np.array([self._kernel(w, xi) + b for xi in X])

                votes = np.sign(preds)
                scores[:, cls_idx] += c * votes

        return scores

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels using weighted voting.

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
        """Estimate class probabilities using softmax on voting scores.

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
        # Normalize scores
        scores_shifted = scores - np.max(scores, axis=1, keepdims=True)
        exp_scores = np.exp(scores_shifted / 10.0)  # Temperature scaling
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    def __repr__(self) -> str:
        if self._is_fitted:
            n_vectors = sum(len(w) for w in self.weight_vectors_)
            return f"VotedPerceptronClassifier(n_vectors={n_vectors})"
        return f"VotedPerceptronClassifier(max_iter={self.max_iter})"

@classifier(tags=["functions", "neural-network", "linear", "averaged"], version="1.0.0")
class AveragedPerceptronClassifier(Classifier):
    """Averaged Perceptron classifier.

    An enhanced Perceptron that **averages all weight vectors** seen during
    training to compute the final model. This usually leads to better
    **generalization** and **stability** compared to the standard perceptron.

    Overview
    --------
    The Averaged Perceptron accumulates weight snapshots during training:

    1. Initialize current weight vectors and bias terms to zero
    2. Initialize running sum accumulators for weights and biases
    3. For each training sample, compute the prediction using current weights
    4. If the prediction is incorrect, update the current weights using the
       standard perceptron rule
    5. After each sample (correct or not), add the current weights to the
       running sum
    6. After all epochs, divide the accumulated sums by the total number of
       update steps to obtain the averaged weights

    Theory
    ------
    The Averaged Perceptron maintains both current weights :math:`\\mathbf{w}_t`
    and a cumulative sum. After :math:`T` total steps, the final averaged
    weight vector is:

    .. math::
        \\bar{\\mathbf{w}} = \\frac{1}{T} \\sum_{t=1}^{T} \\mathbf{w}_t

    The update rule at each step is identical to the standard perceptron:

    .. math::
        \\mathbf{w}_{t+1} = \\mathbf{w}_t + \\eta \\, (y - \\hat{y}) \\, \\mathbf{x}

    Averaging acts as a form of **implicit regularization**, reducing the
    variance of the final model.

    Parameters
    ----------
    learning_rate : float, default=1.0
        Learning rate for weight updates.
    max_iter : int, default=5
        Maximum number of passes over the training data (epochs).
    shuffle : bool, default=True
        Whether to shuffle training data after each epoch.
    random_state : int or None, default=None
        Seed used by the random number generator.

    Attributes
    ----------
    weights_ : np.ndarray of shape (n_classes, n_features)
        Averaged weight vectors for each class.
    bias_ : np.ndarray of shape (n_classes,)
        Averaged bias terms for each class.
    classes_ : np.ndarray of shape (n_classes,)
        Unique class labels discovered during :meth:`fit`.

    Notes
    -----
    **Complexity:**

    - Training: :math:`O(n \\cdot d \\cdot T)` where :math:`n` = number of
      samples, :math:`d` = number of features, :math:`T` = max_iter
    - Prediction: :math:`O(n \\cdot d \\cdot K)` where :math:`K` = number of
      classes

    **When to use AveragedPerceptronClassifier:**

    - When the standard perceptron shows high variance across runs
    - For structured prediction tasks (e.g., POS tagging, NER)
    - When a simple, fast, and well-regularized linear model is needed
    - As a drop-in replacement for the standard perceptron with better
      generalization

    References
    ----------
    .. [Collins2002] Collins, M. (2002).
           **Discriminative Training Methods for Hidden Markov Models: Theory
           and Experiments with Perceptron Algorithms.**
           *Proceedings of the ACL Conference on Empirical Methods in Natural
           Language Processing (EMNLP)*, pp. 1-8.
           DOI: `10.3115/1118693.1118694 <https://doi.org/10.3115/1118693.1118694>`_

    .. [Freund1999] Freund, Y. and Schapire, R.E. (1999).
           **Large Margin Classification Using the Perceptron Algorithm.**
           *Machine Learning*, 37(3), 277-296.

    See Also
    --------
    :class:`~tuiml.algorithms.neural.PerceptronClassifier` : Standard single-layer perceptron classifier.
    :class:`~tuiml.algorithms.neural.VotedPerceptronClassifier` : Voted variant with weighted voting for improved robustness.
    :class:`~tuiml.algorithms.neural.MultilayerPerceptronClassifier` : Multi-layer neural network with backpropagation.

    Examples
    --------
    Train an Averaged Perceptron on a simple binary classification task:

    >>> from tuiml.algorithms.neural import AveragedPerceptronClassifier
    >>> import numpy as np
    >>> X = np.array([[1, 2], [2, 3], [4, 5], [5, 6]])
    >>> y = np.array([0, 0, 1, 1])
    >>> clf = AveragedPerceptronClassifier(max_iter=10)
    >>> clf.fit(X, y)
    AveragedPerceptronClassifier(n_classes=2)
    >>> clf.predict([[3, 4]])
    array([0])
    """

    def __init__(
        self,
        learning_rate: float = 1.0,
        max_iter: int = 5,
        shuffle: bool = True,
        random_state: Optional[int] = None
    ):
        super().__init__()
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.shuffle = shuffle
        self.random_state = random_state
        self.weights_ = None
        self.bias_ = None
        self.classes_ = None
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
                "default": 5,
                "minimum": 1,
                "description": "Maximum passes over training data"
            },
            "shuffle": {
                "type": "boolean",
                "default": True,
                "description": "Whether to shuffle data each epoch"
            }
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
            "Collins, M. (2002). Discriminative training methods for hidden "
            "Markov models: Theory and experiments with perceptron algorithms. "
            "EMNLP."
        ]

    def fit(self, X: np.ndarray, y: np.ndarray) -> "AveragedPerceptronClassifier":
        """Fit the Averaged PerceptronClassifier classifier.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training features.
        y : np.ndarray of shape (n_samples,)
            Target labels.

        Returns
        -------
        self : AveragedPerceptronClassifier
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

        # Initialize
        rng = np.random.RandomState(self.random_state)

        # Current weights
        weights = np.zeros((n_classes, self._n_features))
        bias = np.zeros(n_classes)

        # Accumulated weights for averaging
        weights_sum = np.zeros((n_classes, self._n_features))
        bias_sum = np.zeros(n_classes)

        # Counter for averaging
        counter = 0

        for epoch in range(self.max_iter):
            if self.shuffle:
                indices = rng.permutation(n_samples)
                X_shuffled = X[indices]
                y_shuffled = y_idx[indices]
            else:
                X_shuffled = X
                y_shuffled = y_idx

            for i in range(n_samples):
                xi = X_shuffled[i]
                yi = y_shuffled[i]

                # Predict
                scores = weights @ xi + bias
                y_pred = np.argmax(scores)

                # Update if wrong
                if y_pred != yi:
                    weights[yi] += self.learning_rate * xi
                    bias[yi] += self.learning_rate
                    weights[y_pred] -= self.learning_rate * xi
                    bias[y_pred] -= self.learning_rate

                # Accumulate for averaging
                weights_sum += weights
                bias_sum += bias
                counter += 1

        # Compute averaged weights
        self.weights_ = weights_sum / counter
        self.bias_ = bias_sum / counter

        self._is_fitted = True
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Compute decision scores.

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
        """Predict class labels.

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
        """Estimate class probabilities using softmax.

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
        scores_shifted = scores - np.max(scores, axis=1, keepdims=True)
        exp_scores = np.exp(scores_shifted)
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    def __repr__(self) -> str:
        if self._is_fitted:
            return f"AveragedPerceptronClassifier(n_classes={len(self.classes_)})"
        return f"AveragedPerceptronClassifier(max_iter={self.max_iter})"
