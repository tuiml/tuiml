"""Random Committee ensemble implementations for classification and regression."""

import numpy as np
from typing import Dict, List, Any, Optional, Type
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from tuiml.base.algorithms import Classifier, classifier, Regressor, regressor
from tuiml.hub import registry

@classifier(tags=["ensemble", "committee", "randomizable"], version="1.0.0")
class RandomCommitteeClassifier(Classifier):
    """RandomCommitteeClassifier ensemble classifier.

    Builds an ensemble of randomizable base classifiers, each with a 
    different random number seed. Diversity is achieved through the 
    internal randomness of the base learners, as each member is trained 
    on the full dataset.

    Parameters
    ----------
    base_classifier : str or class, default='RandomTreeClassifier'
        The base classifier to use. Should be randomizable.
    n_estimators : int, default=10
        The number of committee members (base earners).
    random_state : int, optional
        Base random seed used to generate seeds for committee members.
    n_jobs : int, default=1
        The number of jobs to run in parallel. -1 means use all processors.

    Attributes
    ----------
    estimators_ : list
        The collection of fitted base classifiers.
    classes_ : np.ndarray
        The unique class labels.

    Examples
    --------
    >>> from tuiml.algorithms.ensemble import RandomCommitteeClassifier
    >>> clf = RandomCommitteeClassifier(base_classifier='RandomTreeClassifier', n_estimators=50)
    >>> clf.fit(X_train, y_train)
    RandomCommitteeClassifier(...)
    >>> predictions = clf.predict(X_test)

    References
    ----------
    .. [1] Breiman, L. (1996). Bagging Predictors. Machine Learning, 24(2), 123-140.
    """

    def __init__(self, base_classifier: Any = 'RandomTreeClassifier',
                 n_estimators: int = 10,
                 random_state: Optional[int] = None,
                 n_jobs: int = 1):
        super().__init__()
        self.base_classifier = base_classifier
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.estimators_ = None
        self.classes_ = None
        self._base_class = None

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "base_classifier": {
                "type": "string", "default": "RandomTree",
                "description": "Base classifier name (should be randomizable)"
            },
            "n_estimators": {
                "type": "integer", "default": 10, "minimum": 1,
                "description": "Number of committee members"
            },
            "random_state": {
                "type": "integer", "default": None,
                "description": "Base random seed"
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
        return "O(n_estimators * base_complexity)"

    @classmethod
    def get_references(cls) -> List[str]:
        return ["Ensemble of randomizable base classifiers with different seeds."]

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

    def _fit_estimator(self, args) -> Classifier:
        """Fit a single estimator."""
        X, y, seed = args
        estimator = self._base_class()

        # Try to set random_state if the classifier supports it
        if hasattr(estimator, 'random_state'):
            estimator.random_state = seed
        elif hasattr(estimator, 'seed'):
            estimator.seed = seed

        estimator.fit(X, y)
        return estimator

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomCommitteeClassifier":
        """Fit the RandomCommitteeClassifier classifier.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data.
        y : np.ndarray of shape (n_samples,)
            Target labels.

        Returns
        -------
        self : RandomCommitteeClassifier
            Returns the fitted instance.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

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

        all_predictions = np.array([est.predict(X) for est in self.estimators_])

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
        for est in self.estimators_:
            try:
                proba = est.predict_proba(X)
                all_proba.append(proba)
            except NotImplementedError:
                # Fall back to hard predictions
                preds = est.predict(X)
                proba = np.zeros((len(X), len(self.classes_)))
                for i, pred in enumerate(preds):
                    idx = np.where(self.classes_ == pred)[0]
                    if len(idx) > 0:
                        proba[i, idx[0]] = 1.0
                all_proba.append(proba)

        return np.mean(all_proba, axis=0)

    def __repr__(self) -> str:
        if self._is_fitted:
            return f"RandomCommitteeClassifier(base={self.base_classifier}, n_estimators={self.n_estimators})"
        return f"RandomCommitteeClassifier(base_classifier={self.base_classifier})"


@regressor(tags=["ensemble", "committee", "regression"], version="1.0.0")
class RandomCommitteeRegressor(Regressor):
    """RandomCommitteeRegressor ensemble regressor.

    Builds an ensemble of randomizable base regressors, each with a
    different random number seed. Diversity is achieved through the
    internal randomness of the base learners, as each member is trained
    on the **full dataset**.

    Overview
    --------
    The algorithm proceeds as follows:

    1. For each of :math:`T` committee members:
       a. Create a base regressor with a unique random seed
       b. Train the regressor on the full training set
    2. To predict, average predictions from all :math:`T` regressors

    Theory
    ------
    Given :math:`T` base regressors :math:`h_1, \\ldots, h_T` each trained with
    a different random seed on the same data:

    .. math::
        H(x) = \\frac{1}{T} \\sum_{t=1}^{T} h_t(x)

    Diversity arises solely from the stochastic nature of each base learner
    (e.g., random feature selection in tree construction), not from data
    perturbation.

    Parameters
    ----------
    base_regressor : str or class, default='AdditiveRegression'
        The base regressor to use. Should be randomizable (i.e., its
        behaviour depends on a random seed).

    n_estimators : int, default=10
        The number of committee members (base regressors).

    random_state : int or None, default=None
        Base random seed used to generate seeds for committee members.

    n_jobs : int, default=1
        The number of jobs to run in parallel. ``-1`` means use all processors.

    Attributes
    ----------
    estimators_ : list
        The collection of fitted base regressors.

    Notes
    -----
    **Complexity:**

    - Training: :math:`O(T \\cdot C_{\\text{base}})` where :math:`T` = n_estimators
    - Prediction: :math:`O(T \\cdot C_{\\text{predict}})` per sample

    **When to use RandomCommitteeRegressor:**

    - When the base learner has internal randomness (e.g., random trees)
    - When you want ensemble diversity without bootstrap sampling
    - When all training data should be used for every committee member

    References
    ----------
    .. [Breiman2001] Breiman, L. (2001).
           **Random Forests.**
           *Machine Learning*, 45(1), 5-32.
           DOI: `10.1023/A:1010933404324 <https://doi.org/10.1023/A:1010933404324>`_

    See Also
    --------
    :class:`~tuiml.algorithms.ensemble.RandomCommitteeClassifier` : Committee ensemble for classification.
    :class:`~tuiml.algorithms.ensemble.BaggingRegressor` : Bootstrap aggregating for regression.
    :class:`~tuiml.algorithms.ensemble.RandomSubspaceRegressor` : Feature subsampling for regression.

    Examples
    --------
    Basic usage for regression with a random committee:

    >>> from tuiml.algorithms.ensemble import RandomCommitteeRegressor
    >>> import numpy as np
    >>>
    >>> # Create sample training data
    >>> X_train = np.array([[1, 2], [2, 3], [3, 1], [4, 3], [5, 2]])
    >>> y_train = np.array([1.5, 2.3, 3.1, 4.2, 5.0])
    >>>
    >>> # Fit the Random Committee regressor
    >>> reg = RandomCommitteeRegressor(n_estimators=10, random_state=42)
    >>> reg.fit(X_train, y_train)
    RandomCommitteeRegressor(...)
    >>> predictions = reg.predict(X_train)
    """

    def __init__(self, base_regressor: Any = 'AdditiveRegression',
                 n_estimators: int = 10,
                 random_state: Optional[int] = None,
                 n_jobs: int = 1):
        """Initialize RandomCommitteeRegressor.

        Parameters
        ----------
        base_regressor : str or class, default='AdditiveRegression'
            The base regressor to use. Should be randomizable.
        n_estimators : int, default=10
            Number of committee members.
        random_state : int or None, default=None
            Base random seed.
        n_jobs : int, default=1
            Number of parallel jobs. ``-1`` uses all processors.
        """
        super().__init__()
        self.base_regressor = base_regressor
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.estimators_ = None
        self._base_class = None

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        """Return JSON Schema for constructor parameters."""
        return {
            "base_regressor": {
                "type": "string", "default": "AdditiveRegression",
                "description": "Base regressor name (should be randomizable)"
            },
            "n_estimators": {
                "type": "integer", "default": 10, "minimum": 1,
                "description": "Number of committee members"
            },
            "random_state": {
                "type": "integer", "default": None,
                "description": "Base random seed"
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
        return "O(n_estimators * base_complexity)"

    @classmethod
    def get_references(cls) -> List[str]:
        """Return academic references."""
        return ["Ensemble of randomizable base regressors with different seeds."]

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

    def _fit_estimator(self, args) -> Regressor:
        """Fit a single estimator with a unique random seed.

        Parameters
        ----------
        args : tuple of (np.ndarray, np.ndarray, int)
            Tuple containing (X, y, seed).

        Returns
        -------
        estimator : Regressor
            The fitted base regressor.
        """
        X, y, seed = args
        estimator = self._base_class()

        if hasattr(estimator, 'random_state'):
            estimator.random_state = seed
        elif hasattr(estimator, 'seed'):
            estimator.seed = seed

        estimator.fit(X, y)
        return estimator

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomCommitteeRegressor":
        """Fit the RandomCommitteeRegressor.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data.
        y : np.ndarray of shape (n_samples,)
            Target values.

        Returns
        -------
        self : RandomCommitteeRegressor
            Returns the fitted instance.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

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
        """Predict target values by averaging committee member predictions.

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

        all_predictions = np.array([est.predict(X) for est in self.estimators_])
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
            return f"RandomCommitteeRegressor(base={self.base_regressor}, n_estimators={self.n_estimators})"
        return f"RandomCommitteeRegressor(base_regressor={self.base_regressor})"
