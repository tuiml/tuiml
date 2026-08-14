"""HIVE-COTE - a meta-ensemble over distinct time-series representations."""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from tuiml.algorithms.timeseries.classification._base import TimeSeriesClassifier
from tuiml.base.algorithms import classifier


@classifier(
    tags=["timeseries", "classification", "ensemble", "meta"],
    version="1.0.0",
)
class HIVECOTEClassifier(TimeSeriesClassifier):
    """Combine **different representations** of a series, weighted by competence.

    Every other classifier in this package looks at a series one way: elastic
    alignment, convolutional response, local subsequences, symbolic word
    counts, interval statistics. Each is strong somewhere and blind
    elsewhere. HIVE-COTE's insight is that those blind spots barely overlap,
    so combining the views beats tuning any one of them.

    Crucially it is **not** a majority vote. Each component's probabilities are
    weighted by its own cross-validated accuracy raised to a power, so a
    component that is right on this dataset dominates one that is not — and
    the ensemble degrades gracefully when a member is simply unsuited.

    Overview
    --------
    1. Cross-validate every component on the training data.
    2. Raise each accuracy to ``alpha`` to get its weight, so the gap between
       a good and a mediocre component is amplified.
    3. Fit every component on the full training set.
    4. At prediction time, take the weighted sum of the components'
       probabilities.

    Theory
    ------
    With component :math:`c` achieving cross-validated accuracy
    :math:`a_c` and predicting :math:`P_c(y \\mid x)`, the ensemble predicts

    .. math::
        P(y \\mid x) \\ \\propto \\ \\sum_c a_c^{\\alpha} \\ P_c(y \\mid x)

    The exponent :math:`\\alpha` controls how sharply competence is rewarded.
    At :math:`\\alpha = 0` every component counts equally; as
    :math:`\\alpha \\to \\infty` only the best survives. The published value of
    4 sits deliberately between: a component 10% more accurate than another
    receives roughly 1.46 times the weight, enough to matter without letting
    one noisy cross-validation estimate take over.

    The weights are estimated by **cross-validation on the training set**, not
    on the training predictions themselves — a component that memorises its
    training data would otherwise earn a weight of 1 and swamp the rest.

    Parameters
    ----------
    components : list of tuple, optional
        ``(name, classifier)`` pairs. Defaults to one member per
        representation: MINIROCKET (convolutional), BOSS (dictionary),
        TimeSeriesForest (interval) and DTW-1NN (elastic distance).
    alpha : float, default=4.0
        Exponent applied to each component's cross-validated accuracy.
    cv : int, default=3
        Folds used to estimate the weights. Raising it steadies the weights
        and multiplies the fitting cost.
    random_state : int, optional
        Seed passed to the default components and the fold split.

    Attributes
    ----------
    components_ : list of tuple
        ``(name, fitted classifier)`` pairs.
    weights_ : np.ndarray of shape (n_components,)
        Normalised ensemble weights.
    component_accuracy_ : dict
        Cross-validated accuracy per component name — the diagnostic worth
        reading, since it says which view the data actually rewards.
    classes_ : np.ndarray of shape (n_classes,)
        Class labels seen during :meth:`fit`.

    Notes
    -----
    **Complexity.** Fitting costs ``cv + 1`` fits of every component, so it is
    the most expensive classifier here by a wide margin. Prediction costs one
    pass per component. This buys robustness, not speed: if a single model is
    wanted, use
    :class:`~tuiml.algorithms.timeseries.classification.MiniRocketClassifier`.

    **When to use — and be sceptical.** The case for HIVE-COTE is that you
    cannot say in advance whether the signal is a motif, a frequency, a trend
    in one stretch, or a global shape. Measured across three synthetic
    problems built to favour different views, 120 train / 120 test:

    ==============  ========  ==========  ========  ========  ==========
    problem         rocket    dictionary  interval  distance  HIVE-COTE
    ==============  ========  ==========  ========  ========  ==========
    localised       1.000     0.825       1.000     1.000     1.000
    trend
    local motif,    1.000     0.783       0.958     1.000     1.000
    random position
    frequency       1.000     0.983       1.000     0.983     0.992
    under noise
    **worst case**  **1.000** 0.783       0.958     0.983     0.992
    ==============  ========  ==========  ========  ========  ==========

    The weighting works as designed — the dictionary component was
    down-weighted to 0.10-0.13 where it was weak and to 0.25 where all four
    were equal — and the ensemble tracks the best member without being told
    which it is. But **MINIROCKET alone matched or beat it on every row**, at
    roughly a quarter of the cost. That is the usual outcome when one
    component is already near-perfect: an ensemble insures against picking
    wrong, and insurance is a loss when you would have picked right.

    So: fit the components individually first and read
    ``component_accuracy_``. If one dominates, use it and keep the compute.
    Reach for the ensemble when the components disagree, when several are
    close, or when the deployment will see data unlike the sample you tuned
    on — the case the table above cannot show.

    The published HIVE-COTE 2.0 uses four specific components with tuned
    internals. This class keeps the **structure** — cross-validated
    competence weighting over diverse representations — while letting the
    components be chosen, because the structure is what carries the benefit
    and a fixed component list would rot.

    References
    ----------
    .. [Lines2018] Lines, J., Taylor, S., & Bagnall, A. (2018). Time Series
       Classification with HIVE-COTE: The Hierarchical Vote Collective of
       Transformation-Based Ensembles. *ACM Transactions on Knowledge
       Discovery from Data*, 12(5), 1-35. :doi:`10.1145/3182382`
    .. [Middlehurst2021] Middlehurst, M., Large, J., Flynn, M., Lines, J.,
       Bostrom, A., & Bagnall, A. (2021). HIVE-COTE 2.0: A New Meta Ensemble
       for Time Series Classification. *Machine Learning*, 110(11),
       3211-3243. :doi:`10.1007/s10994-021-06057-9`

    See Also
    --------
    :class:`~tuiml.algorithms.timeseries.classification.MiniRocketClassifier` : Usually the strongest single component, and far cheaper.
    :class:`~tuiml.algorithms.ensemble.VotingClassifier` : Unweighted combination, for feature-matrix models.

    Examples
    --------
    >>> import numpy as np
    >>> from tuiml.algorithms.timeseries.classification import (
    ...     HIVECOTEClassifier, MiniRocketClassifier, TimeSeriesForestClassifier)
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(0, 1.0, (60, 80))
    >>> y = np.arange(60) % 2
    >>> X[y == 1, 20:50] += np.linspace(0, 4, 30)
    >>> model = HIVECOTEClassifier(
    ...     components=[("rocket", MiniRocketClassifier(n_features=840, random_state=0)),
    ...                 ("interval", TimeSeriesForestClassifier(n_estimators=50, random_state=0))],
    ...     cv=2, random_state=0).fit(X, y)
    >>> float((model.predict(X) == y).mean())
    1.0
    >>> sorted(model.component_accuracy_)
    ['interval', 'rocket']
    """

    def __init__(
        self,
        components: Optional[List[Tuple[str, Any]]] = None,
        alpha: float = 4.0,
        cv: int = 3,
        random_state: Optional[int] = None,
    ):
        """Initialize the HIVE-COTE meta-ensemble.

        Parameters
        ----------
        components : list of tuple, optional
            ``(name, classifier)`` pairs.
        alpha : float, default=4.0
            Exponent applied to cross-validated accuracy.
        cv : int, default=3
            Folds used to estimate the weights.
        random_state : int, optional
            Seed for the default components and the fold split.
        """
        super().__init__()
        if alpha < 0.0:
            raise ValueError(f"alpha must be non-negative, got {alpha}")
        if cv < 2:
            raise ValueError(f"cv must be at least 2, got {cv}")
        if components is not None and len(components) < 2:
            raise ValueError(
                f"HIVE-COTE needs at least 2 components, got {len(components)}"
            )
        self.components = components
        self.alpha = alpha
        self.cv = cv
        self.random_state = random_state

        # Fitted attributes
        self.components_ = None
        self.weights_ = None
        self.component_accuracy_ = None

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        """Return JSON Schema for algorithm parameters."""
        return {
            "alpha": {
                "type": "number",
                "default": 4.0,
                "minimum": 0,
                "description": "Exponent applied to cross-validated accuracy"
            },
            "cv": {
                "type": "integer",
                "default": 3,
                "minimum": 2,
                "description": "Folds used to estimate component weights"
            },
            "random_state": {
                "type": ["integer", "null"],
                "default": None,
                "description": "Random seed"
            }
        }

    @classmethod
    def get_capabilities(cls) -> List[str]:
        """Return supported capabilities."""
        return [
            "numeric",
            "multi_class",
            "timeseries",
            "multivariate_timeseries",
            "ensemble",
        ]

    @classmethod
    def get_complexity(cls) -> str:
        """Return complexity analysis."""
        return "Fit: (cv + 1) fits of every component; Predict: one pass per component"

    @classmethod
    def get_references(cls) -> List[str]:
        """Return academic citations."""
        return [
            "Lines, J., Taylor, S. and Bagnall, A., 2018. Time series "
            "classification with HIVE-COTE. ACM TKDD.",
            "Middlehurst, M., Large, J., Flynn, M., Lines, J., Bostrom, A. and "
            "Bagnall, A., 2021. HIVE-COTE 2.0: a new meta ensemble for time "
            "series classification. Machine Learning."
        ]

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HIVECOTEClassifier":
        """Weight the components by cross-validated accuracy, then fit them.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_timepoints) or (n_samples, n_channels, n_timepoints)
            Training series.
        y : np.ndarray of shape (n_samples,)
            Training labels.

        Returns
        -------
        self : HIVECOTEClassifier
            The fitted ensemble.
        """
        panel, y = self._validate_fit(X, y)
        specification = self._resolve_components()

        self.component_accuracy_ = {}
        accuracies = []
        for name, component in specification:
            accuracy = self._cross_validated_accuracy(component, panel, y)
            self.component_accuracy_[name] = accuracy
            accuracies.append(accuracy)

        weights = np.asarray(accuracies, dtype=np.float64) ** self.alpha
        total = weights.sum()
        # Every component failing to beat zero accuracy leaves nothing to
        # weight by; fall back to an equal vote rather than dividing by zero.
        self.weights_ = (
            weights / total if total > 0 else np.full(len(weights), 1.0 / len(weights))
        )

        self.components_ = [
            (name, copy.deepcopy(component).fit(panel, y))
            for name, component in specification
        ]
        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Classify each series by the weighted component vote.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_timepoints) or (n_samples, n_channels, n_timepoints)
            Series to classify.

        Returns
        -------
        y_pred : np.ndarray of shape (n_samples,)
            Predicted labels.
        """
        return self.classes_[self.predict_proba(X).argmax(axis=1)]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return the weighted average of the components' probabilities.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_timepoints) or (n_samples, n_channels, n_timepoints)
            Series to classify.

        Returns
        -------
        proba : np.ndarray of shape (n_samples, n_classes)
            Ensemble probabilities, rows summing to one.
        """
        panel = self._validate_predict(X)
        proba = np.zeros((len(panel), len(self.classes_)))

        for weight, (_, component) in zip(self.weights_, self.components_):
            proba += weight * self._aligned_proba(component, panel)

        total = proba.sum(axis=1, keepdims=True)
        return np.divide(proba, total, out=proba, where=total > 0)

    def _aligned_proba(self, component: Any, panel: np.ndarray) -> np.ndarray:
        """Get a component's probabilities in the ensemble's class order.

        A component fitted on a fold may have seen fewer classes than the
        ensemble, so its columns cannot be assumed to line up.

        Parameters
        ----------
        component : Classifier
            A fitted component.
        panel : np.ndarray of shape (n_samples, n_channels, n_timepoints)
            Series to classify.

        Returns
        -------
        proba : np.ndarray of shape (n_samples, n_classes)
            Probabilities widened to the ensemble's classes.
        """
        raw = np.asarray(component.predict_proba(panel), dtype=np.float64)
        component_classes = np.asarray(
            getattr(component, "classes_", self.classes_)
        )

        if len(component_classes) == len(self.classes_) and np.array_equal(
            component_classes, self.classes_
        ):
            return raw

        aligned = np.zeros((len(panel), len(self.classes_)))
        for column, label in enumerate(component_classes):
            position = np.searchsorted(self.classes_, label)
            if position < len(self.classes_) and self.classes_[position] == label:
                aligned[:, position] = raw[:, column]
        return aligned

    def _cross_validated_accuracy(
        self, component: Any, panel: np.ndarray, y: np.ndarray
    ) -> float:
        """Estimate a component's accuracy on held-out folds.

        Parameters
        ----------
        component : Classifier
            An unfitted component; it is deep-copied per fold.
        panel : np.ndarray of shape (n_samples, n_channels, n_timepoints)
            Training panel.
        y : np.ndarray of shape (n_samples,)
            Training labels.

        Returns
        -------
        accuracy : float
            Mean accuracy over the folds. Zero if every fold failed.
        """
        rng = np.random.default_rng(self.random_state)
        order = rng.permutation(len(panel))
        folds = np.array_split(order, min(self.cv, len(panel)))

        correct = 0
        counted = 0
        for test_index in folds:
            train_index = np.setdiff1d(order, test_index)
            # A fold that leaves a single class trains nothing meaningful.
            if len(train_index) == 0 or len(np.unique(y[train_index])) < 2:
                continue
            model = copy.deepcopy(component).fit(panel[train_index], y[train_index])
            correct += int((model.predict(panel[test_index]) == y[test_index]).sum())
            counted += len(test_index)

        return correct / counted if counted else 0.0

    def _resolve_components(self) -> List[Tuple[str, Any]]:
        """Return the components to ensemble.

        Returns
        -------
        components : list of tuple
            The caller's components, or one member per representation.
        """
        if self.components is not None:
            return list(self.components)

        from tuiml.algorithms.timeseries.classification.dictionary import (
            BOSSClassifier,
        )
        from tuiml.algorithms.timeseries.classification.interval import (
            TimeSeriesForestClassifier,
        )
        from tuiml.algorithms.timeseries.classification.knn import (
            DTWNeighborsClassifier,
        )
        from tuiml.algorithms.timeseries.classification.rocket import (
            MiniRocketClassifier,
        )

        # One member per representation: convolutional, dictionary, interval
        # and elastic distance. Diversity is the point — adding a second
        # convolutional member would cost compute and add nothing.
        return [
            ("rocket", MiniRocketClassifier(random_state=self.random_state)),
            ("dictionary", BOSSClassifier()),
            (
                "interval",
                TimeSeriesForestClassifier(random_state=self.random_state),
            ),
            ("distance", DTWNeighborsClassifier(window=0.1)),
        ]

    def __repr__(self) -> str:
        """Return a readable representation of the ensemble."""
        names = (
            [name for name, _ in self.components_]
            if self.components_ is not None
            else "unfitted"
        )
        return f"HIVECOTEClassifier(components={names}, alpha={self.alpha})"
