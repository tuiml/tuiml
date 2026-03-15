"""Voting ensemble implementations for classification and regression."""

import numpy as np
from typing import Dict, List, Any, Optional, Type, Union
from collections import Counter

from tuiml.base.algorithms import Classifier, classifier, Regressor, regressor
from tuiml.hub import registry

@classifier(tags=["ensembles", "voting", "meta"], version="1.0.0")
class VotingClassifier(Classifier):
    """VotingClassifier for **combining heterogeneous classifiers** via voting rules.

    VotingClassifier (also known as Vote) combines predictions from multiple
    **diverse classifiers** using various **combination rules** such as average
    probabilities, product rule, or majority voting.

    Overview
    --------
    The algorithm proceeds as follows:

    1. Train each of the :math:`L` base classifiers independently on the full training set
    2. To predict, collect outputs (predictions or probabilities) from all classifiers
    3. Apply the selected combination rule to aggregate the outputs
    4. Return the class with the highest aggregated score

    Theory
    ------
    Let :math:`p_l(k|x)` denote the posterior probability estimate from classifier
    :math:`h_l` for class :math:`k` given input :math:`x`. The combination rules are:

    **Average rule:**

    .. math::
        P(k|x) = \\frac{1}{L} \\sum_{l=1}^{L} p_l(k|x)

    **Product rule:**

    .. math::
        P(k|x) = \\frac{\\prod_{l=1}^{L} p_l(k|x)}{\\sum_{k'} \\prod_{l=1}^{L} p_l(k'|x)}

    **Majority voting rule:**

    .. math::
        H(x) = \\arg\\max_{k} \\sum_{l=1}^{L} \\mathbb{1}[h_l(x) = k]

    **Median rule:**

    .. math::
        P(k|x) = \\text{median}_{l=1}^{L}\\, p_l(k|x)

    Parameters
    ----------
    classifiers : list, default=['NaiveBayesClassifier', 'C45TreeClassifier']
        The collection of classifiers to be combined. Can be classifier
        names (strings), classes, or instances.

    combination_rule : {'average', 'product', 'majority', 'median', 'max', 'min'}, default='average'
        The rule used to combine the predictions of the base classifiers.

    Attributes
    ----------
    estimators_ : list
        The collection of fitted base classifiers.

    classes_ : np.ndarray
        The unique class labels discovered during ``fit()``.

    Notes
    -----
    **Complexity:**

    - Training: :math:`O(\\sum_{l=1}^{L} C_l)` where :math:`C_l` is the training
      complexity of each base classifier
    - Prediction: :math:`O(\\sum_{l=1}^{L} C_l^{\\text{pred}})` per sample

    **When to use VotingClassifier:**

    - When you have multiple diverse classifiers with comparable performance
    - When you want a simple combination without learning combination weights
    - When base classifiers make independent errors (low correlation)
    - As a baseline before trying more complex methods like stacking

    References
    ----------
    .. [Kittler1998] Kittler, J., Hatef, M., Duin, R.P.W. and Matas, J. (1998).
           **On Combining Classifiers.**
           *IEEE Transactions on Pattern Analysis and Machine Intelligence*,
           20(3), 226-239.
           DOI: `10.1109/34.667881 <https://doi.org/10.1109/34.667881>`_

    .. [Polikar2006] Polikar, R. (2006).
           **Ensemble Based Systems in Decision Making.**
           *IEEE Circuits and Systems Magazine*, 6(3), 21-45.
           DOI: `10.1109/MCAS.2006.1688199 <https://doi.org/10.1109/MCAS.2006.1688199>`_

    See Also
    --------
    :class:`~tuiml.algorithms.ensemble.StackingClassifier` : Learns combination weights via a meta-classifier.
    :class:`~tuiml.algorithms.ensemble.AdaBoostClassifier` : Weighted voting with adaptive boosting.
    :class:`~tuiml.algorithms.ensemble.BaggingClassifier` : Majority voting with bootstrap samples.

    Examples
    --------
    Basic usage for combining classifiers with voting:

    >>> from tuiml.algorithms.ensemble import VotingClassifier
    >>> import numpy as np
    >>>
    >>> # Create sample training data
    >>> X_train = np.array([[1, 2], [2, 3], [3, 1], [4, 3], [5, 2]])
    >>> y_train = np.array([0, 0, 1, 1, 1])
    >>>
    >>> # Fit the Voting classifier with average rule
    >>> clf = VotingClassifier(
    ...     classifiers=['NaiveBayesClassifier', 'C45TreeClassifier'],
    ...     combination_rule='average'
    ... )
    >>> clf.fit(X_train, y_train)
    VotingClassifier(...)
    >>> predictions = clf.predict(X_train)
    """

    def __init__(self, classifiers: List[Any] = None,
                 combination_rule: str = 'average'):
        """Initialize VotingClassifier.

        Parameters
        ----------
        classifiers : list or None, default=None
            Base classifiers to combine. Defaults to
            ``['NaiveBayesClassifier', 'C45TreeClassifier']``.
        combination_rule : str, default='average'
            Rule for combining predictions. One of ``'average'``,
            ``'product'``, ``'majority'``, ``'median'``, ``'max'``, ``'min'``.
        """
        super().__init__()
        self.classifiers = classifiers or ['NaiveBayesClassifier', 'C45TreeClassifier']
        self.combination_rule = combination_rule
        self.estimators_ = None
        self.classes_ = None

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "classifiers": {"type": "array", "default": ["NaiveBayesClassifier", "C45TreeClassifier"],
                           "description": "List of classifier names"},
            "combination_rule": {"type": "string", "default": "average",
                                "enum": ["average", "product", "majority",
                                        "median", "max", "min"],
                                "description": "Combination rule"}
        }

    @classmethod
    def get_capabilities(cls) -> List[str]:
        return ["numeric", "nominal", "binary_class", "multiclass"]

    @classmethod
    def get_complexity(cls) -> str:
        return "O(sum of base classifier complexities)"

    @classmethod
    def get_references(cls) -> List[str]:
        return ["Kittler, J., et al. (1998). On Combining Classifiers. "
                "IEEE Transactions on Pattern Analysis and Machine Intelligence."]

    def _get_classifier(self, clf_spec: Any) -> Classifier:
        """Instantiate a classifier from a specification.

        Parameters
        ----------
        clf_spec : str, type, or Classifier
            Classifier name, class, or instance to resolve.

        Returns
        -------
        classifier : Classifier
            A classifier instance ready for fitting.
        """
        if isinstance(clf_spec, str):
            clf_class = registry.get(clf_spec)
            return clf_class()
        elif isinstance(clf_spec, type):
            return clf_spec()
        elif isinstance(clf_spec, Classifier):
            return clf_spec
        else:
            raise ValueError(f"Invalid classifier specification: {clf_spec}")

    def fit(self, X: np.ndarray, y: np.ndarray) -> "VotingClassifier":
        """Fit all classifiers.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data.
        y : np.ndarray of shape (n_samples,)
            Target labels.

        Returns
        -------
        self : VotingClassifier
            Returns the fitted instance.
        """
        """Fit all classifiers."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self.classes_ = np.unique(y)

        self.estimators_ = []
        for clf_spec in self.classifiers:
            clf = self._get_classifier(clf_spec)
            clf.fit(X, y)
            self.estimators_.append(clf)

        self._is_fitted = True
        return self

    def _get_all_probas(self, X: np.ndarray) -> np.ndarray:
        """Collect probability estimates from all base classifiers.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        all_probas : np.ndarray of shape (n_classifiers, n_samples, n_classes)
            Probability estimates from each classifier.
        """
        all_probas = []
        for est in self.estimators_:
            try:
                proba = est.predict_proba(X)
                all_probas.append(proba)
            except NotImplementedError:
                # Fall back to hard predictions
                preds = est.predict(X)
                proba = np.zeros((len(X), len(self.classes_)))
                for i, pred in enumerate(preds):
                    idx = np.where(self.classes_ == pred)[0]
                    if len(idx) > 0:
                        proba[i, idx[0]] = 1.0
                all_probas.append(proba)
        return np.array(all_probas)

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
        """Predict class probabilities using combination rule."""
        self._check_is_fitted()
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        all_probas = self._get_all_probas(X)

        if self.combination_rule == 'average':
            combined = np.mean(all_probas, axis=0)
        elif self.combination_rule == 'product':
            combined = np.prod(all_probas + 1e-10, axis=0)
            combined /= combined.sum(axis=1, keepdims=True)
        elif self.combination_rule == 'median':
            combined = np.median(all_probas, axis=0)
        elif self.combination_rule == 'max':
            combined = np.max(all_probas, axis=0)
        elif self.combination_rule == 'min':
            combined = np.min(all_probas, axis=0)
        elif self.combination_rule == 'majority':
            # Use voting counts as pseudo-probabilities
            predictions = np.array([est.predict(X) for est in self.estimators_])
            n_samples = X.shape[0]
            combined = np.zeros((n_samples, len(self.classes_)))
            for i in range(n_samples):
                votes = predictions[:, i]
                for v in votes:
                    idx = np.where(self.classes_ == v)[0]
                    if len(idx) > 0:
                        combined[i, idx[0]] += 1
            combined /= len(self.estimators_)
        else:
            raise ValueError(f"Unknown combination rule: {self.combination_rule}")

        # Normalize
        row_sums = combined.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        combined /= row_sums

        return combined

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
        """Predict class labels."""
        self._check_is_fitted()
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if self.combination_rule == 'majority':
            # Direct majority voting
            predictions = np.array([est.predict(X) for est in self.estimators_])
            result = np.empty(X.shape[0], dtype=object)
            for i in range(X.shape[0]):
                votes = predictions[:, i]
                result[i] = Counter(votes).most_common(1)[0][0]
            return result
        else:
            proba = self.predict_proba(X)
            return self.classes_[np.argmax(proba, axis=1)]

    def __repr__(self) -> str:
        if self._is_fitted:
            return f"VotingClassifier(n_classifiers={len(self.estimators_)}, rule='{self.combination_rule}')"
        return f"VotingClassifier(classifiers={self.classifiers})"


@regressor(tags=["ensemble", "voting", "regression"], version="1.0.0")
class VotingRegressor(Regressor):
    """VotingRegressor for **combining heterogeneous regressors** via aggregation rules.

    VotingRegressor combines predictions from multiple **diverse regressors**
    using aggregation rules such as **averaging**, **median**, or **weighted average**.

    Overview
    --------
    The algorithm proceeds as follows:

    1. Train each of the :math:`L` base regressors independently on the full training set
    2. To predict, collect outputs (predictions) from all regressors
    3. Apply the selected combination rule to aggregate the outputs

    Theory
    ------
    Let :math:`h_l(x)` denote the prediction from regressor :math:`h_l`
    for input :math:`x`. The combination rules are:

    **Average rule:**

    .. math::
        H(x) = \\frac{1}{L} \\sum_{l=1}^{L} h_l(x)

    **Median rule:**

    .. math::
        H(x) = \\text{median}_{l=1}^{L}\\, h_l(x)

    **Max / Min rules:**

    .. math::
        H(x) = \\max_{l} h_l(x) \\quad \\text{or} \\quad H(x) = \\min_{l} h_l(x)

    Parameters
    ----------
    regressors : list, default=['AdditiveRegression']
        The collection of regressors to be combined. Can be regressor
        names (strings), classes, or instances.

    combination_rule : {'average', 'median', 'max', 'min'}, default='average'
        The rule used to combine the predictions of the base regressors.

    Attributes
    ----------
    estimators_ : list
        The collection of fitted base regressors.

    Notes
    -----
    **Complexity:**

    - Training: :math:`O(\\sum_{l=1}^{L} C_l)` where :math:`C_l` is the training
      complexity of each base regressor
    - Prediction: :math:`O(\\sum_{l=1}^{L} C_l^{\\text{pred}})` per sample

    **When to use VotingRegressor:**

    - When you have multiple diverse regressors with comparable performance
    - When you want a simple combination without learning combination weights
    - When base regressors make independent errors (low correlation)
    - As a baseline before trying more complex methods like stacking

    References
    ----------
    .. [Perrone1993] Perrone, M.P. and Cooper, L.N. (1993).
           **When Networks Disagree: Ensemble Methods for Hybrid Neural Networks.**
           *Neural Networks for Speech and Image Processing*, Chapman & Hall.

    See Also
    --------
    :class:`~tuiml.algorithms.ensemble.VotingClassifier` : Voting combination for classification.
    :class:`~tuiml.algorithms.ensemble.StackingRegressor` : Learns combination weights via a meta-regressor.
    :class:`~tuiml.algorithms.ensemble.BaggingRegressor` : Averaging with bootstrap samples.

    Examples
    --------
    Basic usage for combining regressors with voting:

    >>> from tuiml.algorithms.ensemble import VotingRegressor
    >>> import numpy as np
    >>>
    >>> # Create sample training data
    >>> X_train = np.array([[1, 2], [2, 3], [3, 1], [4, 3], [5, 2]])
    >>> y_train = np.array([1.5, 2.3, 3.1, 4.2, 5.0])
    >>>
    >>> # Fit the Voting regressor with average rule
    >>> reg = VotingRegressor(
    ...     regressors=['AdditiveRegression'],
    ...     combination_rule='average'
    ... )
    >>> reg.fit(X_train, y_train)
    VotingRegressor(...)
    >>> predictions = reg.predict(X_train)
    """

    def __init__(self, regressors: List[Any] = None,
                 combination_rule: str = 'average'):
        """Initialize VotingRegressor.

        Parameters
        ----------
        regressors : list or None, default=None
            Base regressors to combine. Defaults to
            ``['AdditiveRegression']``.
        combination_rule : str, default='average'
            Rule for combining predictions. One of ``'average'``,
            ``'median'``, ``'max'``, ``'min'``.
        """
        super().__init__()
        self.regressors = regressors or ['AdditiveRegression']
        self.combination_rule = combination_rule
        self.estimators_ = None

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        """Return JSON Schema for constructor parameters."""
        return {
            "regressors": {"type": "array", "default": ["AdditiveRegression"],
                           "description": "List of regressor names"},
            "combination_rule": {"type": "string", "default": "average",
                                "enum": ["average", "median", "max", "min"],
                                "description": "Combination rule"}
        }

    @classmethod
    def get_capabilities(cls) -> List[str]:
        """Return algorithm capabilities."""
        return ["numeric", "numeric_class"]

    @classmethod
    def get_complexity(cls) -> str:
        """Return time/space complexity."""
        return "O(sum of base regressor complexities)"

    @classmethod
    def get_references(cls) -> List[str]:
        """Return academic references."""
        return ["Perrone, M.P. and Cooper, L.N. (1993). When Networks Disagree: "
                "Ensemble Methods for Hybrid Neural Networks."]

    def _get_regressor(self, reg_spec: Any) -> Regressor:
        """Instantiate a regressor from a specification.

        Parameters
        ----------
        reg_spec : str, type, or Regressor
            Regressor name, class, or instance to resolve.

        Returns
        -------
        regressor : Regressor
            A regressor instance ready for fitting.
        """
        if isinstance(reg_spec, str):
            reg_class = registry.get(reg_spec)
            return reg_class()
        elif isinstance(reg_spec, type):
            return reg_spec()
        elif isinstance(reg_spec, Regressor):
            return reg_spec
        else:
            raise ValueError(f"Invalid regressor specification: {reg_spec}")

    def fit(self, X: np.ndarray, y: np.ndarray) -> "VotingRegressor":
        """Fit all regressors.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data.
        y : np.ndarray of shape (n_samples,)
            Target values.

        Returns
        -------
        self : VotingRegressor
            Returns the fitted instance.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self.estimators_ = []
        for reg_spec in self.regressors:
            reg = self._get_regressor(reg_spec)
            reg.fit(X, y)
            self.estimators_.append(reg)

        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict target values using combination rule.

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

        if self.combination_rule == 'average':
            return np.mean(all_predictions, axis=0)
        elif self.combination_rule == 'median':
            return np.median(all_predictions, axis=0)
        elif self.combination_rule == 'max':
            return np.max(all_predictions, axis=0)
        elif self.combination_rule == 'min':
            return np.min(all_predictions, axis=0)
        else:
            raise ValueError(f"Unknown combination rule: {self.combination_rule}")

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
            return f"VotingRegressor(n_regressors={len(self.estimators_)}, rule='{self.combination_rule}')"
        return f"VotingRegressor(regressors={self.regressors})"
