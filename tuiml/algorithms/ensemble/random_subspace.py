"""Random Subspace ensemble implementations for classification and regression."""

import numpy as np
from typing import Dict, List, Any, Optional, Type
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from tuiml.base.algorithms import Classifier, classifier, Regressor, regressor
from tuiml.hub import registry

@classifier(tags=["ensemble", "subspace", "feature_selection"], version="1.0.0")
class RandomSubspaceClassifier(Classifier):
    """RandomSubspaceClassifier for **feature subspace** ensemble classification.

    The Random Subspace method builds an ensemble of classifiers where each
    classifier is trained on a **random subset of input features**, creating
    diversity through **feature subsampling** rather than data subsampling.

    Overview
    --------
    The algorithm proceeds as follows:

    1. For each of :math:`T` ensemble members:
       a. Randomly select :math:`d` features from the full :math:`D`-dimensional feature space
       b. Train a base classifier using only the selected feature subspace
       c. Store the classifier along with its selected feature indices
    2. To predict, aggregate predictions from all :math:`T` classifiers via majority vote

    Theory
    ------
    Each subspace :math:`\\mathcal{F}_t` is a random subset of the feature set
    :math:`\\{1, 2, \\ldots, D\\}`:

    .. math::
        \\mathcal{F}_t \\subset \\{1, \\ldots, D\\}, \\quad |\\mathcal{F}_t| = d

    where :math:`d = \\lfloor D \\cdot \\text{subspace\\_size} \\rfloor` when
    ``subspace_size`` is a float.

    The final prediction uses majority voting across subspace classifiers:

    .. math::
        H(x) = \\arg\\max_{k} \\sum_{t=1}^{T} \\mathbb{1}[h_t(x_{\\mathcal{F}_t}) = k]

    The ensemble achieves diversity because each classifier operates in a
    different projection of the feature space, reducing the correlation
    :math:`\\rho` between base learners.

    Parameters
    ----------
    base_classifier : str or class, default='C45TreeClassifier'
        The base classifier to use. Works best with classifiers that are
        sensitive to the feature set presented.

    n_estimators : int, default=10
        The number of base classifiers in the ensemble.

    subspace_size : float or int, default=0.5
        Number of features for each subspace:

        - ``float`` in (0, 1] -- proportion of total features
        - ``int`` -- absolute number of features

    random_state : int or None, default=None
        Random seed for reproducibility.

    n_jobs : int, default=1
        The number of jobs to run in parallel. ``-1`` means use all processors.

    Attributes
    ----------
    estimators_ : list of tuple
        The collection of fitted ``(estimator, feature_indices)`` tuples.

    classes_ : np.ndarray
        The unique class labels discovered during ``fit()``.

    n_features_ : int
        The total number of features observed during ``fit()``.

    Notes
    -----
    **Complexity:**

    - Training: :math:`O(T \\cdot n \\cdot C_{\\text{base}}(d))` where :math:`T` = n_estimators,
      :math:`n` = number of samples, :math:`C_{\\text{base}}(d)` = base classifier complexity
      on :math:`d` features
    - Prediction: :math:`O(T \\cdot C_{\\text{predict}}(d))` per sample

    **When to use RandomSubspaceClassifier:**

    - High-dimensional datasets where feature subsets carry discriminative information
    - When you want to reduce overfitting through feature decorrelation
    - As an alternative to bagging when data size is limited
    - When combined with decision trees, provides a Random-Forest-like approach

    References
    ----------
    .. [Ho1998] Ho, T.K. (1998).
           **The Random Subspace Method for Constructing Decision Forests.**
           *IEEE Transactions on Pattern Analysis and Machine Intelligence*,
           20(8), 832-844.
           DOI: `10.1109/34.709601 <https://doi.org/10.1109/34.709601>`_

    .. [Skurichina2002] Skurichina, M. and Duin, R.P.W. (2002).
           **Bagging, Boosting and the Random Subspace Method for Linear
           Classifiers.**
           *Pattern Analysis and Applications*, 5(2), 121-135.

    See Also
    --------
    :class:`~tuiml.algorithms.ensemble.BaggingClassifier` : Bootstrap aggregating (data subsampling).
    :class:`~tuiml.algorithms.ensemble.RandomCommitteeClassifier` : Ensemble using randomizable base classifiers.
    :class:`~tuiml.algorithms.ensemble.VotingClassifier` : Combines heterogeneous classifiers via voting.

    Examples
    --------
    Basic usage for classification with random feature subspaces:

    >>> from tuiml.algorithms.ensemble import RandomSubspaceClassifier
    >>> import numpy as np
    >>>
    >>> # Create sample training data
    >>> X_train = np.array([[1, 2, 3], [2, 3, 4], [3, 1, 2], [4, 3, 1]])
    >>> y_train = np.array([0, 0, 1, 1])
    >>>
    >>> # Fit the Random Subspace classifier
    >>> clf = RandomSubspaceClassifier(subspace_size=0.5, n_estimators=10, random_state=42)
    >>> clf.fit(X_train, y_train)
    RandomSubspaceClassifier(...)
    >>> predictions = clf.predict(X_train)
    """

    def __init__(self, base_classifier: Any = 'C45TreeClassifier',
                 n_estimators: int = 10,
                 subspace_size: float = 0.5,
                 random_state: Optional[int] = None,
                 n_jobs: int = 1):
        """Initialize RandomSubspaceClassifier.

        Parameters
        ----------
        base_classifier : str or class, default='C45TreeClassifier'
            The base classifier to use.
        n_estimators : int, default=10
            Number of base classifiers in the ensemble.
        subspace_size : float or int, default=0.5
            Proportion or absolute count of features per subspace.
        random_state : int or None, default=None
            Random seed for reproducibility.
        n_jobs : int, default=1
            Number of parallel jobs. ``-1`` uses all processors.
        """
        super().__init__()
        self.base_classifier = base_classifier
        self.n_estimators = n_estimators
        self.subspace_size = subspace_size
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.estimators_ = None
        self.classes_ = None
        self.n_features_ = None
        self._base_class = None

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "base_classifier": {
                "type": "string", "default": "C45TreeClassifier",
                "description": "Base classifier name"
            },
            "n_estimators": {
                "type": "integer", "default": 10, "minimum": 1,
                "description": "Number of classifiers"
            },
            "subspace_size": {
                "type": "number", "default": 0.5, "minimum": 0.01, "maximum": 1.0,
                "description": "Proportion of features to use (or int for absolute count)"
            },
            "random_state": {
                "type": "integer", "default": None,
                "description": "Random seed"
            },
            "n_jobs": {
                "type": "integer", "default": 1,
                "description": "Parallel jobs (-1 for all CPUs)"
            }
        }

    @classmethod
    def get_capabilities(cls) -> List[str]:
        return ["numeric", "nominal", "missing_values", "binary_class", "multiclass"]

    @classmethod
    def get_complexity(cls) -> str:
        return "O(n_estimators * base_complexity * subspace_size)"

    @classmethod
    def get_references(cls) -> List[str]:
        return [
            "Ho, T.K. (1998). The Random Subspace Method for Constructing "
            "Decision Forests. IEEE PAMI, 20(8), 832-844."
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

    def _get_n_features_subset(self) -> int:
        """Compute the number of features to select for each subspace.

        Parameters
        ----------
        None

        Returns
        -------
        n_features_subset : int
            Number of features for each random subspace.
        """
        if isinstance(self.subspace_size, float) and 0 < self.subspace_size <= 1:
            return max(1, int(self.n_features_ * self.subspace_size))
        elif isinstance(self.subspace_size, int) and self.subspace_size > 0:
            return min(self.subspace_size, self.n_features_)
        else:
            return max(1, self.n_features_ // 2)

    def _fit_estimator(self, args) -> tuple:
        """Fit a single estimator on a random feature subspace.

        Parameters
        ----------
        args : tuple of (np.ndarray, np.ndarray, int)
            Tuple containing (X, y, seed) where X is the training data,
            y is the target labels, and seed is the random seed.

        Returns
        -------
        result : tuple of (Classifier, np.ndarray)
            Tuple of (fitted estimator, selected feature indices).
        """
        X, y, seed = args
        rng = np.random.RandomState(seed)

        # Select random feature subset
        n_subset = self._get_n_features_subset()
        feature_indices = rng.choice(self.n_features_, size=n_subset, replace=False)
        feature_indices = np.sort(feature_indices)

        X_subset = X[:, feature_indices]

        estimator = self._base_class()
        estimator.fit(X_subset, y)

        return (estimator, feature_indices)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomSubspaceClassifier":
        """Fit the RandomSubspaceClassifier classifier.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data.
        y : np.ndarray of shape (n_samples,)
            Target labels.

        Returns
        -------
        self : RandomSubspaceClassifier
            Returns the fitted instance.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self.n_features_ = X.shape[1]
        self.classes_ = np.unique(y)
        self._base_class = self._get_base_class()

        base_seed = self.random_state if self.random_state is not None else 0
        args = [(X, y, base_seed + i) for i in range(self.n_estimators)]

        if self.n_jobs == 1:
            self.estimators_ = [self._fit_estimator(arg) for arg in args]
        else:
            n_jobs = self.n_jobs if self.n_jobs > 0 else None
            with ThreadPoolExecutor(max_workers=n_jobs) as executor:
                self.estimators_ = list(executor.map(self._fit_estimator, args))

        self._is_fitted = True
        return self

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

        all_predictions = []
        for estimator, feature_indices in self.estimators_:
            X_subset = X[:, feature_indices]
            preds = estimator.predict(X_subset)
            all_predictions.append(preds)

        all_predictions = np.array(all_predictions)
        predictions = [Counter(all_predictions[:, i]).most_common(1)[0][0] for i in range(X.shape[0])]
        return np.array(predictions, dtype=self.classes_.dtype)

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

        all_proba = []
        for estimator, feature_indices in self.estimators_:
            X_subset = X[:, feature_indices]
            try:
                proba = estimator.predict_proba(X_subset)
                all_proba.append(proba)
            except NotImplementedError:
                preds = estimator.predict(X_subset)
                proba = np.zeros((len(X), len(self.classes_)))
                for i, pred in enumerate(preds):
                    idx = np.where(self.classes_ == pred)[0]
                    if len(idx) > 0:
                        proba[i, idx[0]] = 1.0
                all_proba.append(proba)

        return np.mean(all_proba, axis=0)

    def __repr__(self) -> str:
        if self._is_fitted:
            return (f"RandomSubspaceClassifier(base={self.base_classifier}, "
                    f"n_estimators={self.n_estimators}, subspace={self.subspace_size})")
        return f"RandomSubspaceClassifier(base_classifier={self.base_classifier})"


@regressor(tags=["ensemble", "subspace", "regression"], version="1.0.0")
class RandomSubspaceRegressor(Regressor):
    """RandomSubspaceRegressor for **feature subspace** ensemble regression.

    The Random Subspace method builds an ensemble of regressors where each
    regressor is trained on a **random subset of input features**, creating
    diversity through **feature subsampling** rather than data subsampling.

    Overview
    --------
    The algorithm proceeds as follows:

    1. For each of :math:`T` ensemble members:
       a. Randomly select :math:`d` features from the full :math:`D`-dimensional feature space
       b. Train a base regressor using only the selected feature subspace
       c. Store the regressor along with its selected feature indices
    2. To predict, average predictions from all :math:`T` regressors

    Theory
    ------
    Each subspace :math:`\\mathcal{F}_t` is a random subset of the feature set
    :math:`\\{1, 2, \\ldots, D\\}`:

    .. math::
        \\mathcal{F}_t \\subset \\{1, \\ldots, D\\}, \\quad |\\mathcal{F}_t| = d

    where :math:`d = \\lfloor D \\cdot \\text{subspace\\_size} \\rfloor` when
    ``subspace_size`` is a float.

    The final prediction averages across subspace regressors:

    .. math::
        H(x) = \\frac{1}{T} \\sum_{t=1}^{T} h_t(x_{\\mathcal{F}_t})

    Parameters
    ----------
    base_regressor : str or class, default='AdditiveRegression'
        The base regressor to use. Works best with regressors that are
        sensitive to the feature set presented.

    n_estimators : int, default=10
        The number of base regressors in the ensemble.

    subspace_size : float or int, default=0.5
        Number of features for each subspace:

        - ``float`` in (0, 1] -- proportion of total features
        - ``int`` -- absolute number of features

    random_state : int or None, default=None
        Random seed for reproducibility.

    n_jobs : int, default=1
        The number of jobs to run in parallel. ``-1`` means use all processors.

    Attributes
    ----------
    estimators_ : list of tuple
        The collection of fitted ``(estimator, feature_indices)`` tuples.

    n_features_ : int
        The total number of features observed during ``fit()``.

    Notes
    -----
    **Complexity:**

    - Training: :math:`O(T \\cdot n \\cdot C_{\\text{base}}(d))` where :math:`T` = n_estimators,
      :math:`n` = number of samples, :math:`C_{\\text{base}}(d)` = base regressor complexity
      on :math:`d` features
    - Prediction: :math:`O(T \\cdot C_{\\text{predict}}(d))` per sample

    **When to use RandomSubspaceRegressor:**

    - High-dimensional datasets where feature subsets carry predictive information
    - When you want to reduce overfitting through feature decorrelation
    - As an alternative to bagging when data size is limited
    - When combined with tree-based regressors for robust predictions

    References
    ----------
    .. [Ho1998] Ho, T.K. (1998).
           **The Random Subspace Method for Constructing Decision Forests.**
           *IEEE Transactions on Pattern Analysis and Machine Intelligence*,
           20(8), 832-844.
           DOI: `10.1109/34.709601 <https://doi.org/10.1109/34.709601>`_

    See Also
    --------
    :class:`~tuiml.algorithms.ensemble.RandomSubspaceClassifier` : Feature subsampling for classification.
    :class:`~tuiml.algorithms.ensemble.BaggingRegressor` : Bootstrap aggregating (data subsampling) for regression.
    :class:`~tuiml.algorithms.ensemble.RandomCommitteeRegressor` : Ensemble using randomizable base regressors.

    Examples
    --------
    Basic usage for regression with random feature subspaces:

    >>> from tuiml.algorithms.ensemble import RandomSubspaceRegressor
    >>> import numpy as np
    >>>
    >>> # Create sample training data
    >>> X_train = np.array([[1, 2, 3], [2, 3, 4], [3, 1, 2], [4, 3, 1]])
    >>> y_train = np.array([1.5, 2.3, 3.1, 4.2])
    >>>
    >>> # Fit the Random Subspace regressor
    >>> reg = RandomSubspaceRegressor(subspace_size=0.5, n_estimators=10, random_state=42)
    >>> reg.fit(X_train, y_train)
    RandomSubspaceRegressor(...)
    >>> predictions = reg.predict(X_train)
    """

    def __init__(self, base_regressor: Any = 'AdditiveRegression',
                 n_estimators: int = 10,
                 subspace_size: float = 0.5,
                 random_state: Optional[int] = None,
                 n_jobs: int = 1):
        """Initialize RandomSubspaceRegressor.

        Parameters
        ----------
        base_regressor : str or class, default='AdditiveRegression'
            The base regressor to use.
        n_estimators : int, default=10
            Number of base regressors in the ensemble.
        subspace_size : float or int, default=0.5
            Proportion or absolute count of features per subspace.
        random_state : int or None, default=None
            Random seed for reproducibility.
        n_jobs : int, default=1
            Number of parallel jobs. ``-1`` uses all processors.
        """
        super().__init__()
        self.base_regressor = base_regressor
        self.n_estimators = n_estimators
        self.subspace_size = subspace_size
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.estimators_ = None
        self.n_features_ = None
        self._base_class = None

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        """Return JSON Schema for constructor parameters."""
        return {
            "base_regressor": {
                "type": "string", "default": "AdditiveRegression",
                "description": "Base regressor name"
            },
            "n_estimators": {
                "type": "integer", "default": 10, "minimum": 1,
                "description": "Number of regressors"
            },
            "subspace_size": {
                "type": "number", "default": 0.5, "minimum": 0.01, "maximum": 1.0,
                "description": "Proportion of features to use (or int for absolute count)"
            },
            "random_state": {
                "type": "integer", "default": None,
                "description": "Random seed"
            },
            "n_jobs": {
                "type": "integer", "default": 1,
                "description": "Parallel jobs (-1 for all CPUs)"
            }
        }

    @classmethod
    def get_capabilities(cls) -> List[str]:
        """Return algorithm capabilities."""
        return ["numeric", "numeric_class"]

    @classmethod
    def get_complexity(cls) -> str:
        """Return time/space complexity."""
        return "O(n_estimators * base_complexity * subspace_size)"

    @classmethod
    def get_references(cls) -> List[str]:
        """Return academic references."""
        return [
            "Ho, T.K. (1998). The Random Subspace Method for Constructing "
            "Decision Forests. IEEE PAMI, 20(8), 832-844."
        ]

    def _get_base_class(self) -> Type[Regressor]:
        """Resolve the base regressor specification to a class.

        Returns
        -------
        base_class : Type[Regressor]
            The resolved regressor class.
        """
        if isinstance(self.base_regressor, str):
            return registry.get(self.base_regressor)
        elif isinstance(self.base_regressor, type):
            return self.base_regressor
        elif isinstance(self.base_regressor, Regressor):
            return type(self.base_regressor)
        else:
            raise ValueError(f"Invalid base_regressor: {self.base_regressor}")

    def _get_n_features_subset(self) -> int:
        """Compute the number of features to select for each subspace.

        Returns
        -------
        n_features_subset : int
            Number of features for each random subspace.
        """
        if isinstance(self.subspace_size, float) and 0 < self.subspace_size <= 1:
            return max(1, int(self.n_features_ * self.subspace_size))
        elif isinstance(self.subspace_size, int) and self.subspace_size > 0:
            return min(self.subspace_size, self.n_features_)
        else:
            return max(1, self.n_features_ // 2)

    def _fit_estimator(self, args) -> tuple:
        """Fit a single estimator on a random feature subspace.

        Parameters
        ----------
        args : tuple of (np.ndarray, np.ndarray, int)
            Tuple containing (X, y, seed).

        Returns
        -------
        result : tuple of (Regressor, np.ndarray)
            Tuple of (fitted estimator, selected feature indices).
        """
        X, y, seed = args
        rng = np.random.RandomState(seed)

        n_subset = self._get_n_features_subset()
        feature_indices = rng.choice(self.n_features_, size=n_subset, replace=False)
        feature_indices = np.sort(feature_indices)

        estimator = self._base_class()
        estimator.fit(X[:, feature_indices], y)

        return (estimator, feature_indices)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomSubspaceRegressor":
        """Fit the RandomSubspaceRegressor.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data.
        y : np.ndarray of shape (n_samples,)
            Target values.

        Returns
        -------
        self : RandomSubspaceRegressor
            Returns the fitted instance.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self.n_features_ = X.shape[1]
        self._base_class = self._get_base_class()

        base_seed = self.random_state if self.random_state is not None else 0
        args = [(X, y, base_seed + i) for i in range(self.n_estimators)]

        if self.n_jobs == 1:
            self.estimators_ = [self._fit_estimator(arg) for arg in args]
        else:
            n_jobs = self.n_jobs if self.n_jobs > 0 else None
            with ThreadPoolExecutor(max_workers=n_jobs) as executor:
                self.estimators_ = list(executor.map(self._fit_estimator, args))

        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict target values by averaging subspace regressor predictions.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Test data.

        Returns
        -------
        y_pred : np.ndarray of shape (n_samples,)
            Predicted target values.
        """
        self._check_is_fitted()
        X = np.asarray(X, dtype=float)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        all_predictions = []
        for estimator, feature_indices in self.estimators_:
            preds = estimator.predict(X[:, feature_indices])
            all_predictions.append(preds)

        return np.mean(all_predictions, axis=0)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute R-squared score.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Test features.
        y : np.ndarray of shape (n_samples,)
            True target values.

        Returns
        -------
        score : float
            R-squared score.
        """
        self._check_is_fitted()
        y_pred = self.predict(X)
        y = np.asarray(y)

        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)

        if ss_tot == 0:
            return 0.0

        return 1 - (ss_res / ss_tot)

    def __repr__(self) -> str:
        """Return string representation."""
        if self._is_fitted:
            return (f"RandomSubspaceRegressor(base={self.base_regressor}, "
                    f"n_estimators={self.n_estimators}, subspace={self.subspace_size})")
        return f"RandomSubspaceRegressor(base_regressor={self.base_regressor})"
