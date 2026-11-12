"""EasyEnsemble classifier for imbalanced classification."""

import numpy as np
from typing import Any, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor

from tuiml.base.algorithms import Classifier, classifier
from tuiml.algorithms.ensemble.adaboost import AdaBoostClassifier


def _fit_easy_ensemble_member(args):
    """Fit one AdaBoost classifier on a balanced subset."""

    X, y, seed, n_estimators, base_classifier, weight_threshold = args

    rng = np.random.RandomState(seed)

    classes, counts = np.unique(y, return_counts=True)

    if len(classes) < 2:
        raise ValueError(
            "EasyEnsembleClassifier requires at least two classes."
        )

    min_count = np.min(counts)

    balanced_indices = []

    for target_class in classes:
        class_indices = np.flatnonzero(y == target_class)

        # Keep all minority samples by sampling to the minority-class size.
        # For majority classes this performs random undersampling.
        selected = rng.choice(
            class_indices,
            size=min_count,
            replace=False,
        )

        balanced_indices.append(selected)

    indices = np.concatenate(balanced_indices)
    rng.shuffle(indices)

    estimator = AdaBoostClassifier(
        base_classifier=base_classifier,
        n_estimators=n_estimators,
        weight_threshold=weight_threshold,
        random_state=seed,
    )

    estimator.fit(
        X[indices],
        y[indices],
    )

    return estimator, indices


@classifier(
    tags=["ensembles", "boosting", "imbalanced", "meta"],
    version="1.0.0",
)
class EasyEnsembleClassifier(Classifier):
    """EasyEnsemble classifier for imbalanced classification.

    EasyEnsemble creates several balanced training subsets by randomly
    undersampling the majority class. A native TuiML AdaBoost classifier is
    trained on each subset and the resulting ensembles are combined through
    weighted voting.
    Overview
    --------
    Easy Ensemble handles class imbalance by training multiple AdaBoost
    classifiers on different balanced subsets of the training data.

    Theory
    ------
    Each ensemble member is trained on a different balanced subset created
    through random undersampling of the majority classes. The probability
    estimates from the individual AdaBoost models are averaged to produce
    the final prediction.

    Notes
    -----
    The implementation uses TuiML's native AdaBoostClassifier.

    References
    ----------
    Liu, X.-Y., Wu, J., & Zhou, Z.-H. (2009). Exploratory Undersampling for
    Class-Imbalance Learning. IEEE Transactions on Systems, Man, and Cybernetics,
    Part B, 39(2), 539-550.
    DOI: 10.1109/TSMCB.2008.2007853

    See Also
    --------
    BalancedRandomForestClassifier
    AdaBoostClassifier
    Parameters
    ----------
    n_estimators : int, default=10
        Number of AdaBoost ensembles.

    base_classifier : str or class, default='DecisionStumpClassifier'
        Base classifier used by each AdaBoost ensemble.

    adaboost_estimators : int, default=50
        Number of boosting iterations in each AdaBoost ensemble.

    weight_threshold : float, default=100
        Weight threshold passed to each AdaBoost ensemble.

    random_state : int or None, default=None
        Random seed for reproducibility.

    n_jobs : int, default=1
        Number of parallel jobs. ``-1`` uses all available processors.

    Attributes
    ----------
    estimators_ : list
        Fitted native AdaBoost classifiers.

    bootstrap_indices_ : list
        Training indices used by each balanced subset.

    classes_ : ndarray
        Unique class labels.

    Examples
    --------
    >>> from tuiml.datasets import load_iris
    >>> from tuiml.algorithms.ensemble import EasyEnsembleClassifier
    >>> data = load_iris()
    >>> X = data.data
    >>> y = data.target
    >>> clf = EasyEnsembleClassifier(n_estimators=3, random_state=42)
    >>> clf.fit(X, y)
    EasyEnsembleClassifier(...)
    >>> clf.predict(X[:3])
    array([0, 0, 0])
    """

    def __init__(
        self,
        n_estimators: int = 10,
        base_classifier: Any = "DecisionStumpClassifier",
        adaboost_estimators: int = 50,
        weight_threshold: float = 100,
        random_state: Optional[int] = None,
        n_jobs: int = 1,
    ):
        super().__init__()

        self.n_estimators = n_estimators
        self.base_classifier = base_classifier
        self.adaboost_estimators = adaboost_estimators
        self.weight_threshold = weight_threshold
        self.random_state = random_state
        self.n_jobs = n_jobs

        self.estimators_ = None
        self.bootstrap_indices_ = None
        self.classes_ = None

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        """Return parameter schema."""

        return {
            "n_estimators": {
                "type": "integer",
                "default": 10,
                "minimum": 1,
                "description": "Number of AdaBoost ensembles",
            },
            "base_classifier": {
                "type": "string",
                "default": "DecisionStumpClassifier",
                "description": "Base classifier used by AdaBoost",
            },
            "adaboost_estimators": {
                "type": "integer",
                "default": 50,
                "minimum": 1,
                "description": "Number of boosting iterations",
            },
            "weight_threshold": {
                "type": "number",
                "default": 100,
                "minimum": 1,
                "description": "AdaBoost weight threshold",
            },
            "random_state": {
                "type": ["integer", "null"],
                "default": None,
                "description": "Random seed for reproducibility",
            },
            "n_jobs": {
                "type": "integer",
                "default": 1,
                "description": "Number of parallel jobs",
            },
        }

    @classmethod
    def get_capabilities(cls) -> List[str]:
        """Return classifier capabilities."""

        return [
            "numeric",
            "nominal",
            "binary_class",
            "multiclass",
        ]

    @classmethod
    def get_complexity(cls) -> str:
        """Return time/space complexity."""

        return "O(E * B * C) training, O(E * B) prediction"

    @classmethod
    def get_references(cls) -> List[str]:
        """Return academic references."""

        return [
            (
                "Liu, X.-Y., Wu, J., & Zhou, Z.-H. (2009). "
                "Exploratory Undersampling for Class-Imbalance Learning. "
                "IEEE Transactions on Systems, Man, and Cybernetics, "
                "Part B, 39(2), 539-550."
            )
        ]

    def _fit_member(self, args):
        """Fit one balanced AdaBoost ensemble."""

        return _fit_easy_ensemble_member(args)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> "EasyEnsembleClassifier":
        """Fit the ensemble.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training samples.

        y : ndarray of shape (n_samples,)
            Target values.

        Returns
        -------
        self
            Fitted classifier.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if len(X) != len(y):
            raise ValueError(
                "X and y must contain the same number of samples."
            )

        self.classes_ = np.unique(y)

        if len(self.classes_) < 2:
            raise ValueError(
                "EasyEnsembleClassifier requires at least two classes."
            )

        master_rng = np.random.RandomState(self.random_state)

        seeds = master_rng.randint(
            0,
            2**31,
            size=self.n_estimators,
        )

        args = [
            (
                X,
                y,
                int(seed),
                self.adaboost_estimators,
                self.base_classifier,
                self.weight_threshold,
            )
            for seed in seeds
        ]

        if self.n_jobs == 1:
            results = [
                self._fit_member(arg)
                for arg in args
            ]
        else:
            workers = None if self.n_jobs < 0 else self.n_jobs

            with ThreadPoolExecutor(
                max_workers=workers
            ) as executor:
                results = list(
                    executor.map(
                        self._fit_member,
                        args,
                    )
                )

        self.estimators_ = [
            result[0]
            for result in results
        ]

        self.bootstrap_indices_ = [
            result[1]
            for result in results
        ]

        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for samples.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        ndarray of shape (n_samples,)
            Predicted class labels.
        """
        self._check_is_fitted()

        probabilities = self.predict_proba(X)

        return self.classes_[np.argmax(probabilities, axis=1)]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        ndarray of shape (n_samples, n_classes)
            Predicted class probabilities.
        """
        self._check_is_fitted()

        X = np.asarray(X, dtype=float)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        probabilities = []

        for estimator in self.estimators_:
            probabilities.append(estimator.predict_proba(X))

        return np.mean(probabilities, axis=0)

    def __repr__(self) -> str:
        return (
            f"EasyEnsembleClassifier("
            f"n_estimators={self.n_estimators}, "
            f"adaboost_estimators={self.adaboost_estimators}"
            f")"
        )
    