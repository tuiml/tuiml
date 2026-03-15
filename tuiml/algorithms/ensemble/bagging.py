"""Bagging (Bootstrap Aggregating) ensemble implementations for classification and regression."""

import numpy as np
from typing import Dict, List, Any, Optional, Type
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from tuiml.base.algorithms import Classifier, classifier, Regressor, regressor
from tuiml.hub import registry

@classifier(tags=["ensembles", "bagging", "meta"], version="1.0.0")
class BaggingClassifier(Classifier):
    """BaggingClassifier for **bootstrap aggregating** ensemble classification.

    BaggingClassifier improves stability and accuracy by training multiple base
    classifiers on different **bootstrap samples** of the training data and
    combining their predictions through **majority voting**.

    Overview
    --------
    The algorithm proceeds as follows:

    1. For each of :math:`T` ensemble members:
       a. Draw a bootstrap sample of size :math:`n'` (with replacement) from the training set
       b. Train an independent base classifier on the bootstrap sample
    2. To predict, aggregate predictions from all :math:`T` classifiers via majority vote

    Theory
    ------
    Each bootstrap sample :math:`S_t` is drawn with replacement from the original
    dataset :math:`D` of size :math:`n`:

    .. math::
        S_t = \\{(x_{i_1}, y_{i_1}), \\ldots, (x_{i_{n'}}, y_{i_{n'}})\\},
        \\quad i_j \\sim \\text{Uniform}(1, n)

    where :math:`n' = n \\cdot \\text{bag\\_size\\_percent} / 100`.

    The final ensemble prediction uses majority voting:

    .. math::
        H(x) = \\arg\\max_{k} \\sum_{t=1}^{T} \\mathbb{1}[h_t(x) = k]

    The variance reduction from bagging is:

    .. math::
        \\text{Var}(H) = \\rho \\cdot \\sigma^2 + \\frac{1 - \\rho}{T} \\cdot \\sigma^2

    where :math:`\\rho` is the pairwise correlation between base learners and
    :math:`\\sigma^2` is the variance of a single base learner.

    Parameters
    ----------
    base_classifier : str or class, default='C45TreeClassifier'
        The base classifier to use. Unstable learners (e.g., decision trees)
        benefit most from bagging.

    n_estimators : int, default=10
        The number of base classifiers in the ensemble. More estimators
        generally improve accuracy at the cost of computation time.

    bag_size_percent : int, default=100
        Size of each bootstrap sample as a percentage of the training set.
        100 means each bag is the same size as the original dataset.

    random_state : int or None, default=None
        Random seed for reproducibility.

    n_jobs : int, default=1
        The number of jobs to run in parallel for fitting base classifiers.
        ``-1`` means use all available processors.

    Attributes
    ----------
    estimators_ : list
        The collection of fitted base classifiers.

    classes_ : np.ndarray
        The unique class labels discovered during ``fit()``.

    Notes
    -----
    **Complexity:**

    - Training: :math:`O(T \\cdot n' \\cdot C_{\\text{base}})` where :math:`T` = n_estimators,
      :math:`n'` = bootstrap sample size, :math:`C_{\\text{base}}` = base classifier complexity
    - Prediction: :math:`O(T \\cdot C_{\\text{predict}})` per sample

    **When to use BaggingClassifier:**

    - When the base learner is unstable (high variance), such as decision trees
    - When you want to reduce overfitting without increasing bias
    - When parallel training is desirable (each estimator is independent)
    - As a building block for more complex ensemble methods (e.g., Random Forest)

    References
    ----------
    .. [Breiman1996] Breiman, L. (1996).
           **Bagging Predictors.**
           *Machine Learning*, 24(2), 123-140.
           DOI: `10.1007/BF00058655 <https://doi.org/10.1007/BF00058655>`_

    .. [Breiman1996b] Breiman, L. (1996).
           **Heuristics of Instability and Stabilization in Model Selection.**
           *The Annals of Statistics*, 24(6), 2350-2383.

    See Also
    --------
    :class:`~tuiml.algorithms.ensemble.AdaBoostClassifier` : Adaptive boosting ensemble method.
    :class:`~tuiml.algorithms.ensemble.RandomSubspaceClassifier` : Feature subsampling ensemble method.
    :class:`~tuiml.algorithms.ensemble.RandomCommitteeClassifier` : Ensemble using randomizable base classifiers.

    Examples
    --------
    Basic usage for classification with bootstrap aggregating:

    >>> from tuiml.algorithms.ensemble import BaggingClassifier
    >>> import numpy as np
    >>>
    >>> # Create sample training data
    >>> X_train = np.array([[1, 2], [2, 3], [3, 1], [4, 3], [5, 2]])
    >>> y_train = np.array([0, 0, 1, 1, 1])
    >>>
    >>> # Fit the Bagging classifier
    >>> clf = BaggingClassifier(base_classifier='C45TreeClassifier', n_estimators=10)
    >>> clf.fit(X_train, y_train)
    BaggingClassifier(...)
    >>> predictions = clf.predict(X_train)
    """

    def __init__(self, base_classifier: Any = 'C45TreeClassifier',
                 n_estimators: int = 10,
                 bag_size_percent: int = 100,
                 random_state: Optional[int] = None,
                 n_jobs: int = 1):
        """Initialize BaggingClassifier.

        Parameters
        ----------
        base_classifier : str or class, default='C45TreeClassifier'
            The base classifier to use.
        n_estimators : int, default=10
            Number of base classifiers in the ensemble.
        bag_size_percent : int, default=100
            Bootstrap sample size as percentage of training set.
        random_state : int or None, default=None
            Random seed for reproducibility.
        n_jobs : int, default=1
            Number of parallel jobs. ``-1`` uses all processors.
        """
        super().__init__()
        self.base_classifier = base_classifier
        self.n_estimators = n_estimators
        self.bag_size_percent = bag_size_percent
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.estimators_ = None
        self.classes_ = None
        self._base_class = None

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "base_classifier": {"type": "string", "default": "C45TreeClassifier",
                               "description": "Base classifier name or class"},
            "n_estimators": {"type": "integer", "default": 10, "minimum": 1,
                            "description": "Number of base classifiers"},
            "bag_size_percent": {"type": "integer", "default": 100,
                                "minimum": 1, "maximum": 100,
                                "description": "Bag size as percentage"},
            "random_state": {"type": "integer", "default": None,
                            "description": "Random seed"},
            "n_jobs": {"type": "integer", "default": 1,
                      "description": "Parallel jobs (-1 for all CPUs)"}
        }

    @classmethod
    def get_capabilities(cls) -> List[str]:
        return ["numeric", "nominal", "missing_values", "binary_class", "multiclass"]

    @classmethod
    def get_complexity(cls) -> str:
        return "O(n * base_complexity * n_estimators)"

    @classmethod
    def get_references(cls) -> List[str]:
        return ["Breiman, L. (1996). BaggingClassifier Predictors. Machine Learning, 24(2), 123-140."]

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

    def _fit_estimator(self, args) -> Classifier:
        """Fit a single base estimator on a bootstrap sample.

        Parameters
        ----------
        args : tuple of (np.ndarray, np.ndarray, int)
            Tuple containing (X, y, seed) where X is the training data,
            y is the target labels, and seed is the random seed.

        Returns
        -------
        estimator : Classifier
            The fitted base classifier.
        """
        X, y, seed = args
        rng = np.random.RandomState(seed)

        # Bootstrap sample
        n_samples = len(y)
        bag_size = int(n_samples * self.bag_size_percent / 100)
        indices = rng.choice(n_samples, size=bag_size, replace=True)

        X_bag = X[indices]
        y_bag = y[indices]

        estimator = self._base_class()
        estimator.fit(X_bag, y_bag)
        return estimator

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaggingClassifier":
        """Fit the BaggingClassifier classifier.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data.
        y : np.ndarray of shape (n_samples,)
            Target labels.

        Returns
        -------
        self : BaggingClassifier
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

        # Average probabilities from all estimators
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
            return f"BaggingClassifier(base={self.base_classifier}, n_estimators={self.n_estimators})"
        return f"BaggingClassifier(base_classifier={self.base_classifier})"


@regressor(tags=["ensemble", "bagging", "regression"], version="1.0.0")
class BaggingRegressor(Regressor):
    """BaggingRegressor for **bootstrap aggregating** ensemble regression.

    BaggingRegressor improves stability and accuracy by training multiple base
    regressors on different **bootstrap samples** of the training data and
    combining their predictions through **averaging**.

    Overview
    --------
    The algorithm proceeds as follows:

    1. For each of :math:`T` ensemble members:
       a. Draw a bootstrap sample of size :math:`n'` (with replacement) from the training set
       b. Train an independent base regressor on the bootstrap sample
    2. To predict, average predictions from all :math:`T` regressors

    Theory
    ------
    Each bootstrap sample :math:`S_t` is drawn with replacement from the original
    dataset :math:`D` of size :math:`n`:

    .. math::
        S_t = \\{(x_{i_1}, y_{i_1}), \\ldots, (x_{i_{n'}}, y_{i_{n'}})\\},
        \\quad i_j \\sim \\text{Uniform}(1, n)

    where :math:`n' = n \\cdot \\text{bag\\_size\\_percent} / 100`.

    The final ensemble prediction averages across all base regressors:

    .. math::
        H(x) = \\frac{1}{T} \\sum_{t=1}^{T} h_t(x)

    The variance reduction from bagging is:

    .. math::
        \\text{Var}(H) = \\rho \\cdot \\sigma^2 + \\frac{1 - \\rho}{T} \\cdot \\sigma^2

    where :math:`\\rho` is the pairwise correlation between base learners and
    :math:`\\sigma^2` is the variance of a single base learner.

    Parameters
    ----------
    base_regressor : str or class, default='AdditiveRegression'
        The base regressor to use. Unstable learners (e.g., decision trees)
        benefit most from bagging.

    n_estimators : int, default=10
        The number of base regressors in the ensemble. More estimators
        generally improve accuracy at the cost of computation time.

    bag_size_percent : int, default=100
        Size of each bootstrap sample as a percentage of the training set.
        100 means each bag is the same size as the original dataset.

    random_state : int or None, default=None
        Random seed for reproducibility.

    n_jobs : int, default=1
        The number of jobs to run in parallel for fitting base regressors.
        ``-1`` means use all available processors.

    Attributes
    ----------
    estimators_ : list
        The collection of fitted base regressors.

    Notes
    -----
    **Complexity:**

    - Training: :math:`O(T \\cdot n' \\cdot C_{\\text{base}})` where :math:`T` = n_estimators,
      :math:`n'` = bootstrap sample size, :math:`C_{\\text{base}}` = base regressor complexity
    - Prediction: :math:`O(T \\cdot C_{\\text{predict}})` per sample

    **When to use BaggingRegressor:**

    - When the base learner is unstable (high variance), such as decision trees
    - When you want to reduce overfitting without increasing bias
    - When parallel training is desirable (each estimator is independent)
    - When prediction averaging can smooth out individual model noise

    References
    ----------
    .. [Breiman1996] Breiman, L. (1996).
           **Bagging Predictors.**
           *Machine Learning*, 24(2), 123-140.
           DOI: `10.1007/BF00058655 <https://doi.org/10.1007/BF00058655>`_

    See Also
    --------
    :class:`~tuiml.algorithms.ensemble.BaggingClassifier` : Bootstrap aggregating for classification.
    :class:`~tuiml.algorithms.ensemble.AdditiveRegression` : Gradient boosting for regression.
    :class:`~tuiml.algorithms.ensemble.RandomSubspaceRegressor` : Feature subsampling ensemble for regression.

    Examples
    --------
    Basic usage for regression with bootstrap aggregating:

    >>> from tuiml.algorithms.ensemble import BaggingRegressor
    >>> import numpy as np
    >>>
    >>> # Create sample training data
    >>> X_train = np.array([[1, 2], [2, 3], [3, 1], [4, 3], [5, 2]])
    >>> y_train = np.array([1.5, 2.3, 3.1, 4.2, 5.0])
    >>>
    >>> # Fit the Bagging regressor
    >>> reg = BaggingRegressor(n_estimators=10, random_state=42)
    >>> reg.fit(X_train, y_train)
    BaggingRegressor(...)
    >>> predictions = reg.predict(X_train)
    """

    def __init__(self, base_regressor: Any = 'AdditiveRegression',
                 n_estimators: int = 10,
                 bag_size_percent: int = 100,
                 random_state: Optional[int] = None,
                 n_jobs: int = 1):
        """Initialize BaggingRegressor.

        Parameters
        ----------
        base_regressor : str or class, default='AdditiveRegression'
            The base regressor to use.
        n_estimators : int, default=10
            Number of base regressors in the ensemble.
        bag_size_percent : int, default=100
            Bootstrap sample size as percentage of training set.
        random_state : int or None, default=None
            Random seed for reproducibility.
        n_jobs : int, default=1
            Number of parallel jobs. ``-1`` uses all processors.
        """
        super().__init__()
        self.base_regressor = base_regressor
        self.n_estimators = n_estimators
        self.bag_size_percent = bag_size_percent
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.estimators_ = None
        self._base_class = None

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        """Return JSON Schema for constructor parameters."""
        return {
            "base_regressor": {"type": "string", "default": "AdditiveRegression",
                               "description": "Base regressor name or class"},
            "n_estimators": {"type": "integer", "default": 10, "minimum": 1,
                            "description": "Number of base regressors"},
            "bag_size_percent": {"type": "integer", "default": 100,
                                "minimum": 1, "maximum": 100,
                                "description": "Bag size as percentage"},
            "random_state": {"type": "integer", "default": None,
                            "description": "Random seed"},
            "n_jobs": {"type": "integer", "default": 1,
                      "description": "Parallel jobs (-1 for all CPUs)"}
        }

    @classmethod
    def get_capabilities(cls) -> List[str]:
        """Return algorithm capabilities."""
        return ["numeric", "numeric_class"]

    @classmethod
    def get_complexity(cls) -> str:
        """Return time/space complexity."""
        return "O(n * base_complexity * n_estimators)"

    @classmethod
    def get_references(cls) -> List[str]:
        """Return academic references."""
        return ["Breiman, L. (1996). Bagging Predictors. Machine Learning, 24(2), 123-140."]

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
        """Fit a single base estimator on a bootstrap sample.

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
        rng = np.random.RandomState(seed)

        n_samples = len(y)
        bag_size = int(n_samples * self.bag_size_percent / 100)
        indices = rng.choice(n_samples, size=bag_size, replace=True)

        estimator = self._base_class()
        estimator.fit(X[indices], y[indices])
        return estimator

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaggingRegressor":
        """Fit the BaggingRegressor.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data.
        y : np.ndarray of shape (n_samples,)
            Target values.

        Returns
        -------
        self : BaggingRegressor
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
        """Predict target values by averaging base regressor predictions.

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
            return f"BaggingRegressor(base={self.base_regressor}, n_estimators={self.n_estimators})"
        return f"BaggingRegressor(base_regressor={self.base_regressor})"
