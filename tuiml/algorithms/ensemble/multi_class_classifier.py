"""MultiClassClassifier implementation."""

import numpy as np
from typing import Dict, List, Any, Optional, Type
from itertools import combinations

from tuiml.base.algorithms import Classifier, classifier
from tuiml.hub import registry

@classifier(tags=["ensemble", "multiclass", "meta"], version="1.0.0")
class MultiClassClassifier(Classifier):
    """MultiClassClassifier for **multiclass decomposition** of binary classifiers.

    MultiClassClassifier enables **binary classifiers** to handle multi-class
    problems using various **decomposition strategies** such as One-vs-All,
    One-vs-One, and Error-Correcting Output Codes.

    Overview
    --------
    The algorithm decomposes a :math:`K`-class problem into binary sub-problems:

    1. **One-vs-All (OvA):** Train :math:`K` binary classifiers, each separating
       one class from the rest
    2. **One-vs-One (OvO):** Train :math:`K(K-1)/2` binary classifiers, one for
       each pair of classes, and aggregate via voting
    3. **Error-Correcting Output Codes (ECOC):** Assign a binary codeword to
       each class and train one classifier per code bit

    Theory
    ------
    **One-vs-All (OvA):**

    Train classifiers :math:`h_k` for :math:`k = 1, \\ldots, K`, where :math:`h_k`
    separates class :math:`k` from all other classes. Prediction:

    .. math::
        H(x) = \\arg\\max_{k} \\; f_k(x)

    where :math:`f_k(x)` is the confidence score of classifier :math:`h_k`.

    **One-vs-One (OvO):**

    Train classifiers :math:`h_{ij}` for all pairs :math:`(i, j)` with :math:`i < j`.
    Prediction uses voting:

    .. math::
        H(x) = \\arg\\max_{k} \\sum_{(i,j): i<j} \\mathbb{1}[h_{ij}(x) = k]

    **ECOC:**

    Assign a code matrix :math:`M \\in \\{-1, +1\\}^{K \\times B}` and train
    :math:`B` binary classifiers. Prediction finds the nearest codeword:

    .. math::
        H(x) = \\arg\\min_{k} \\; \\|M_k - f(x)\\|^2

    Parameters
    ----------
    base_classifier : str or class, default='SMO'
        The binary classifier to use as the base learner.

    method : {'ova', 'ovo', 'ecoc'}, default='ova'
        The decomposition method:

        - ``'ova'`` -- One-vs-All (trains :math:`K` classifiers)
        - ``'ovo'`` -- One-vs-One (trains :math:`K(K-1)/2` classifiers)
        - ``'ecoc'`` -- Error-Correcting Output Codes

    random_state : int or None, default=None
        Random seed for reproducibility. Primarily used for ECOC code
        matrix generation.

    Attributes
    ----------
    estimators_ : list
        The collection of fitted binary classifiers.

    classes_ : np.ndarray
        The unique class labels discovered during ``fit()``.

    n_classes_ : int
        The number of distinct classes.

    Notes
    -----
    **Complexity:**

    - Training (OvA): :math:`O(K \\cdot n \\cdot C_{\\text{base}})` where :math:`K` = number of classes
    - Training (OvO): :math:`O\\bigl(\\frac{K(K-1)}{2} \\cdot \\frac{2n}{K} \\cdot C_{\\text{base}}\\bigr)`
    - Training (ECOC): :math:`O(B \\cdot n \\cdot C_{\\text{base}})` where :math:`B` = number of code bits
    - Prediction: :math:`O(M \\cdot C_{\\text{predict}})` per sample, where :math:`M` = number of estimators

    **When to use MultiClassClassifier:**

    - When your base classifier only supports binary classification (e.g., SVM)
    - When you need to extend a strong binary learner to multiclass settings
    - ECOC is preferred when robustness to individual classifier errors is desired
    - OvO is preferred for classifiers with high training cost (smaller subproblems)

    References
    ----------
    .. [Dietterich1995] Dietterich, T.G. and Bakiri, G. (1995).
           **Solving Multiclass Learning Problems via Error-Correcting
           Output Codes.**
           *Journal of Artificial Intelligence Research*, 2, 263-286.
           DOI: `10.1613/jair.105 <https://doi.org/10.1613/jair.105>`_

    .. [Allwein2000] Allwein, E.L., Schapire, R.E. and Singer, Y. (2000).
           **Reducing Multiclass to Binary: A Unifying Approach for Margin
           Classifiers.**
           *Journal of Machine Learning Research*, 1, 113-141.

    .. [Rifkin2004] Rifkin, R. and Klautau, A. (2004).
           **In Defense of One-Vs-All Classification.**
           *Journal of Machine Learning Research*, 5, 101-141.

    See Also
    --------
    :class:`~tuiml.algorithms.ensemble.VotingClassifier` : Combines multiple multiclass classifiers.
    :class:`~tuiml.algorithms.ensemble.StackingClassifier` : Stacked generalization meta-classifier.
    :class:`~tuiml.algorithms.ensemble.AdaBoostClassifier` : Boosting for multiclass problems.

    Examples
    --------
    Basic usage with One-vs-One decomposition:

    >>> from tuiml.algorithms.ensemble import MultiClassClassifier
    >>> import numpy as np
    >>>
    >>> # Create sample multiclass data
    >>> X_train = np.array([[1, 2], [2, 3], [3, 1], [4, 3], [5, 2], [6, 1]])
    >>> y_train = np.array([0, 0, 1, 1, 2, 2])
    >>>
    >>> # Fit with One-vs-One method
    >>> clf = MultiClassClassifier(base_classifier='SMO', method='ovo')
    >>> clf.fit(X_train, y_train)
    MultiClassClassifier(...)
    >>> predictions = clf.predict(X_train)
    """

    def __init__(self, base_classifier: Any = 'SMO',
                 method: str = 'ova',
                 random_state: Optional[int] = None):
        """Initialize MultiClassClassifier.

        Parameters
        ----------
        base_classifier : str or class, default='SMO'
            The binary classifier to use as base learner.
        method : str, default='ova'
            Decomposition method: ``'ova'``, ``'ovo'``, or ``'ecoc'``.
        random_state : int or None, default=None
            Random seed, primarily for ECOC code generation.
        """
        super().__init__()
        self.base_classifier = base_classifier
        self.method = method
        self.random_state = random_state
        self.estimators_ = None
        self.classes_ = None
        self.n_classes_ = None
        self._base_class = None
        self._class_pairs = None  # For OvO
        self._code_matrix = None  # For ECOC

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "base_classifier": {
                "type": "string", "default": "SMO",
                "description": "Base binary classifier"
            },
            "method": {
                "type": "string", "default": "ova",
                "enum": ["ova", "ovo", "ecoc"],
                "description": "Multiclass method (ova/ovo/ecoc)"
            },
            "random_state": {
                "type": "integer", "default": None,
                "description": "Random seed for ECOC"
            }
        }

    @classmethod
    def get_capabilities(cls) -> List[str]:
        return ["numeric", "nominal", "missing_values", "multiclass"]

    @classmethod
    def get_complexity(cls) -> str:
        return "O(n_estimators * base_complexity)"

    @classmethod
    def get_references(cls) -> List[str]:
        return [
            "Dietterich, T. & Bakiri, G. (1995). Solving Multiclass Learning "
            "Problems via Error-Correcting Output Codes. JAIR, 2, 263-286."
        ]

    def _get_base_class(self) -> Type[Classifier]:
        """Resolve the base classifier specification to a class.

        Parameters
        ----------
        None

        Returns
        -------
        base_class : Type[Classifier]
            The resolved classifier class.
        """
        if isinstance(self.base_classifier, str):
            return registry.get(self.base_classifier)
        elif isinstance(self.base_classifier, type):
            return self.base_classifier
        elif isinstance(self.base_classifier, Classifier):
            # Handle classifier instances by extracting their class
            return type(self.base_classifier)
        else:
            raise ValueError(f"Invalid base_classifier: {self.base_classifier}")

    def _fit_ova(self, X: np.ndarray, y_idx: np.ndarray):
        """Fit one-vs-all binary classifiers.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data.
        y_idx : np.ndarray of shape (n_samples,)
            Integer-encoded target labels.

        Returns
        -------
        None
        """
        self.estimators_ = []

        for k in range(self.n_classes_):
            # Create binary labels: class k vs rest
            y_binary = (y_idx == k).astype(int)

            estimator = self._base_class()
            estimator.fit(X, y_binary)
            self.estimators_.append(estimator)

    def _fit_ovo(self, X: np.ndarray, y_idx: np.ndarray):
        """Fit one-vs-one binary classifiers for all class pairs.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data.
        y_idx : np.ndarray of shape (n_samples,)
            Integer-encoded target labels.

        Returns
        -------
        None
        """
        self.estimators_ = []
        self._class_pairs = list(combinations(range(self.n_classes_), 2))

        for i, j in self._class_pairs:
            # Filter samples for classes i and j
            mask = (y_idx == i) | (y_idx == j)
            X_pair = X[mask]
            y_pair = (y_idx[mask] == j).astype(int)

            estimator = self._base_class()
            estimator.fit(X_pair, y_pair)
            self.estimators_.append(estimator)

    def _fit_ecoc(self, X: np.ndarray, y_idx: np.ndarray):
        """Fit binary classifiers using error-correcting output codes.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data.
        y_idx : np.ndarray of shape (n_samples,)
            Integer-encoded target labels.

        Returns
        -------
        None
        """
        rng = np.random.RandomState(self.random_state)

        # Generate random code matrix
        # Each row is a class, each column is a classifier
        n_classifiers = min(15, self.n_classes_ * 2)
        self._code_matrix = rng.choice([-1, 1], size=(self.n_classes_, n_classifiers))

        self.estimators_ = []

        for col in range(n_classifiers):
            # Create binary labels based on code
            codes = self._code_matrix[:, col]
            y_binary = np.array([1 if codes[yi] == 1 else 0 for yi in y_idx])

            # Skip if all same class
            if len(np.unique(y_binary)) < 2:
                continue

            estimator = self._base_class()
            estimator.fit(X, y_binary)
            self.estimators_.append(estimator)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MultiClassClassifier":
        """Fit the MultiClassClassifier.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data.
        y : np.ndarray of shape (n_samples,)
            Target labels.

        Returns
        -------
        self : MultiClassClassifier
            Returns the fitted instance.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        self._base_class = self._get_base_class()

        # Convert y to indices
        class_to_idx = {c: i for i, c in enumerate(self.classes_)}
        y_idx = np.array([class_to_idx[c] for c in y])

        if self.method == 'ova':
            self._fit_ova(X, y_idx)
        elif self.method == 'ovo':
            self._fit_ovo(X, y_idx)
        elif self.method == 'ecoc':
            self._fit_ecoc(X, y_idx)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        self._is_fitted = True
        return self

    def _predict_ova(self, X: np.ndarray) -> np.ndarray:
        """Predict class indices using one-vs-all classifiers.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        indices : np.ndarray of shape (n_samples,)
            Predicted class indices.
        """
        scores = np.zeros((X.shape[0], self.n_classes_))

        for k, estimator in enumerate(self.estimators_):
            try:
                proba = estimator.predict_proba(X)
                if proba.shape[1] >= 2:
                    scores[:, k] = proba[:, 1]
                else:
                    scores[:, k] = proba[:, 0]
            except (NotImplementedError, AttributeError):
                preds = estimator.predict(X)
                scores[:, k] = (preds == 1).astype(float)

        return np.argmax(scores, axis=1)

    def _predict_ovo(self, X: np.ndarray) -> np.ndarray:
        """Predict class indices using one-vs-one voting.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        indices : np.ndarray of shape (n_samples,)
            Predicted class indices.
        """
        votes = np.zeros((X.shape[0], self.n_classes_))

        for idx, (i, j) in enumerate(self._class_pairs):
            preds = self.estimators_[idx].predict(X)
            for sample_idx in range(len(X)):
                if preds[sample_idx] == 0:
                    votes[sample_idx, i] += 1
                else:
                    votes[sample_idx, j] += 1

        return np.argmax(votes, axis=1)

    def _predict_ecoc(self, X: np.ndarray) -> np.ndarray:
        """Predict class indices using error-correcting output codes.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        indices : np.ndarray of shape (n_samples,)
            Predicted class indices based on nearest codeword.
        """
        # Get predictions from all classifiers
        predictions = np.zeros((X.shape[0], len(self.estimators_)))

        for idx, estimator in enumerate(self.estimators_):
            preds = estimator.predict(X)
            predictions[:, idx] = 2 * preds - 1  # Convert to -1/1

        # Find closest codeword for each sample
        result = np.zeros(X.shape[0], dtype=int)
        for i in range(X.shape[0]):
            distances = np.sum((self._code_matrix[:, :len(self.estimators_)] -
                              predictions[i]) ** 2, axis=1)
            result[i] = np.argmin(distances)

        return result

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
        self._check_is_fitted()
        X = np.asarray(X, dtype=float)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if self.method == 'ova':
            indices = self._predict_ova(X)
        elif self.method == 'ovo':
            indices = self._predict_ovo(X)
        elif self.method == 'ecoc':
            indices = self._predict_ecoc(X)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        return self.classes_[indices]

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

        if self.method == 'ova':
            scores = np.zeros((X.shape[0], self.n_classes_))
            for k, estimator in enumerate(self.estimators_):
                try:
                    proba = estimator.predict_proba(X)
                    scores[:, k] = proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]
                except (NotImplementedError, AttributeError):
                    preds = estimator.predict(X)
                    scores[:, k] = preds.astype(float)
            # Normalize
            scores = scores / (np.sum(scores, axis=1, keepdims=True) + 1e-10)
            return scores
        else:
            # For OvO/ECOC, use voting-based pseudo-probabilities
            if self.method == 'ovo':
                votes = np.zeros((X.shape[0], self.n_classes_))
                for idx, (i, j) in enumerate(self._class_pairs):
                    preds = self.estimators_[idx].predict(X)
                    for sample_idx in range(len(X)):
                        if preds[sample_idx] == 0:
                            votes[sample_idx, i] += 1
                        else:
                            votes[sample_idx, j] += 1
                return votes / (np.sum(votes, axis=1, keepdims=True) + 1e-10)
            else:
                # ECOC - compute distances as pseudo-probabilities
                predictions = np.zeros((X.shape[0], len(self.estimators_)))
                for idx, estimator in enumerate(self.estimators_):
                    preds = estimator.predict(X)
                    predictions[:, idx] = 2 * preds - 1

                proba = np.zeros((X.shape[0], self.n_classes_))
                for i in range(X.shape[0]):
                    distances = np.sum((self._code_matrix[:, :len(self.estimators_)] -
                                      predictions[i]) ** 2, axis=1)
                    proba[i] = 1 / (distances + 1e-10)
                return proba / (np.sum(proba, axis=1, keepdims=True) + 1e-10)

    def __repr__(self) -> str:
        if self._is_fitted:
            return f"MultiClassClassifier(base={self.base_classifier}, method={self.method})"
        return f"MultiClassClassifier(base_classifier={self.base_classifier})"
