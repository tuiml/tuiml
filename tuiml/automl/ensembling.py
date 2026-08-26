"""Greedy ensemble selection over an AutoML trial pool.

A search leaves behind dozens of fitted models and keeps one. Caruana et al.
(2004) showed that a **weighted average of the pool**, assembled greedily, is
reliably better than its single best member and costs nothing more to build:
the models are already fitted, and selection works purely on their stored
validation predictions.

This module implements that procedure and the small combiner that carries the
result. It is not a replacement for
:class:`~tuiml.algorithms.ensemble.VotingClassifier`, which builds an ensemble
by *fitting* a declared list of learners with fixed weights; here the members
are already fitted, the pool is whatever the search happened to produce, and
the weights are what selection discovers.

References
----------
.. [Caruana2004] Caruana, R., Niculescu-Mizil, A., Crew, G., & Ksikes, A.
   (2004). Ensemble selection from libraries of models. *Proceedings of the
   21st International Conference on Machine Learning (ICML)*, 18.
   :doi:`10.1145/1015330.1015432`
"""

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


def greedy_selection(
    predictions: Sequence[np.ndarray],
    y_true: np.ndarray,
    scorer: Callable[[np.ndarray, np.ndarray], float],
    *,
    n_rounds: int = 25,
    decode: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> Tuple[np.ndarray, float, List[int]]:
    """Select an ensemble greedily, with replacement, from a pool of models.

    Overview
    --------
    1. Start from the empty ensemble.
    2. In each round, try appending **every** pool member to the current
       ensemble, score the resulting average, and keep the best append.
    3. Repeat for ``n_rounds``. Selection is *with replacement*: a strong
       model can be added several times, which is exactly how the procedure
       expresses a non-uniform weight without ever solving for one.
    4. Return the prefix of the selection sequence with the highest score,
       so that rounds which only made things worse are discarded.

    Theory
    ------
    After :math:`k` rounds the ensemble prediction is the running mean of the
    selected members,

    .. math::
        \\bar{p}_k = \\frac{1}{k} \\sum_{i=1}^{k} p_{s_i},

    so the number of times model :math:`j` was selected, divided by :math:`k`,
    is its weight. Because each round maximises the score of
    :math:`\\bar{p}_k` directly, selection optimises the target metric itself
    rather than a differentiable surrogate, and it cannot do worse than the
    single best model, which is always available as the first pick.

    Parameters
    ----------
    predictions : sequence of np.ndarray
        One validation prediction array per pool member, all the same shape:
        ``(n_samples, n_classes)`` probabilities for classification, or
        ``(n_samples,)`` values for regression.
    y_true : np.ndarray of shape (n_samples,)
        Validation targets.
    scorer : callable
        Function ``(y_true, y_pred) -> float``, higher is better.
    n_rounds : int, default=25
        Number of greedy rounds, i.e. the ensemble's size counted with
        multiplicity.
    decode : callable, optional
        Maps an averaged prediction array to the form ``scorer`` expects,
        e.g. probabilities to hard class labels. Identity when omitted.

    Returns
    -------
    weights : np.ndarray of shape (n_models,)
        Selection frequency per pool member, summing to 1.
    score : float
        The score of the returned ensemble on the validation predictions.
    order : list of int
        The indices selected, in the order they were picked.

    Raises
    ------
    ValueError
        If ``predictions`` is empty.

    Examples
    --------
    >>> import numpy as np
    >>> from tuiml.automl.ensembling import greedy_selection
    >>> y = np.array([0, 0, 1, 1])
    >>> good = np.array([[.9, .1], [.8, .2], [.2, .8], [.1, .9]])
    >>> bad = np.array([[.4, .6], [.3, .7], [.6, .4], [.7, .3]])
    >>> accuracy = lambda t, p: float(np.mean(t == p))
    >>> weights, score, order = greedy_selection(
    ...     [good, bad], y, accuracy, n_rounds=4,
    ...     decode=lambda p: p.argmax(axis=1))
    >>> weights.tolist()
    [1.0, 0.0]
    >>> score
    1.0
    """
    pool = [np.asarray(p, dtype=float) for p in predictions]
    if not pool:
        raise ValueError("greedy_selection needs at least one set of predictions.")
    if decode is None:
        def decode(values):  # noqa: E306 - trivial identity default
            return values

    y_true = np.asarray(y_true)
    n_models = len(pool)
    counts = np.zeros(n_models, dtype=float)
    running = np.zeros_like(pool[0])

    order: List[int] = []
    scores: List[float] = []

    for round_index in range(1, int(n_rounds) + 1):
        best_index, best_score = -1, -np.inf
        for index, candidate in enumerate(pool):
            averaged = (running + candidate) / round_index
            try:
                score = float(scorer(y_true, decode(averaged)))
            except Exception:
                # A member whose predictions the metric cannot digest (for
                # example a degenerate constant column) simply never wins.
                continue
            if score > best_score:
                best_index, best_score = index, score
        if best_index < 0:
            break
        running = running + pool[best_index]
        counts[best_index] += 1
        order.append(best_index)
        scores.append(best_score)

    if not order:
        raise ValueError("No pool member could be scored; ensemble selection failed.")

    # Truncate to the best prefix: later rounds are kept only if they helped.
    best_round = int(np.argmax(scores))
    kept = order[: best_round + 1]
    counts = np.zeros(n_models, dtype=float)
    for index in kept:
        counts[index] += 1

    return counts / counts.sum(), float(scores[best_round]), kept


class GreedyEnsemble:
    """A weighted average of already-fitted models, chosen by selection.

    Built by :func:`greedy_selection` from an AutoML trial pool and used as
    the final predictor when the ensemble scores better than the best single
    trial. It holds fitted models, so it has no ``fit``: everything it needs
    was learned during the search.

    Parameters
    ----------
    models : list of Algorithm
        The fitted pool members, in the same order as the predictions passed
        to :func:`greedy_selection`.
    weights : np.ndarray of shape (n_models,)
        Selection frequencies. Members with weight zero are ignored.
    task : {"classification", "regression"}
        Whether to average class probabilities or predicted values.
    classes : np.ndarray, optional
        Class labels, in the column order of the averaged probability matrix.
        Required for classification.

    Attributes
    ----------
    members_ : list of tuple
        The ``(weight, model)`` pairs that actually contribute, heaviest
        first.

    See Also
    --------
    :func:`~tuiml.automl.ensembling.greedy_selection` : Builds the weights.
    :class:`~tuiml.algorithms.ensemble.VotingClassifier` : Fits and combines a
        declared list of learners instead of a discovered pool.

    Examples
    --------
    >>> import numpy as np
    >>> from tuiml.algorithms.trees import DecisionTreeClassifier
    >>> from tuiml.automl.ensembling import GreedyEnsemble
    >>> from tuiml.datasets import load_iris
    >>> data = load_iris()
    >>> trees = [DecisionTreeClassifier(max_depth=d).fit(data.X, data.y)
    ...          for d in (1, 3)]
    >>> ensemble = GreedyEnsemble(trees, np.array([0.25, 0.75]),
    ...                           task="classification",
    ...                           classes=np.unique(data.y))
    >>> ensemble.predict(data.X[:3]).tolist()
    [0, 0, 0]
    """

    def __init__(
        self,
        models: Sequence[Any],
        weights: np.ndarray,
        task: str = "classification",
        classes: Optional[np.ndarray] = None,
    ):
        """Store the fitted members and their weights."""
        self.models = list(models)
        self.weights = np.asarray(weights, dtype=float)
        self.task = task
        self.classes_ = None if classes is None else np.asarray(classes)
        self.members_ = sorted(
            ((float(w), m) for w, m in zip(self.weights, self.models) if w > 0),
            key=lambda pair: -pair[0],
        )

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict weighted-average class probabilities.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        proba : np.ndarray of shape (n_samples, n_classes)
            Averaged class probabilities, columns ordered as ``classes_``.

        Raises
        ------
        RuntimeError
            If the ensemble was built for regression.
        """
        if self.task != "classification":
            raise RuntimeError("predict_proba is only defined for classification.")
        total = None
        for weight, model in self.members_:
            proba = weight * align_proba(model, X, self.classes_)
            total = proba if total is None else total + proba
        return total

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels (classification) or values (regression).

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        y_pred : np.ndarray of shape (n_samples,)
            Ensemble predictions.
        """
        if self.task == "classification":
            return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
        total = None
        for weight, model in self.members_:
            values = weight * np.asarray(model.predict(X), dtype=float)
            total = values if total is None else total + values
        return total

    def describe(self) -> List[Dict[str, Any]]:
        """Return the contributing members as plain dicts.

        Returns
        -------
        rows : list of dict
            One ``{"model", "weight"}`` dict per contributing member,
            heaviest first.
        """
        return [
            {"model": type(model).__name__, "weight": weight}
            for weight, model in self.members_
        ]

    def __repr__(self) -> str:
        """Return a short summary naming the members and their weights."""
        parts = ", ".join(
            f"{type(model).__name__}:{weight:.2f}" for weight, model in self.members_
        )
        return f"GreedyEnsemble({parts})"


def align_proba(model: Any, X: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """Return a model's class probabilities in a fixed column order.

    Pool members may know a different subset of classes, or the same classes
    in a different order, than the ensemble as a whole. Averaging their raw
    matrices would silently add unrelated columns together, so every matrix is
    projected onto the ensemble's ``classes`` first.

    Parameters
    ----------
    model : Algorithm
        A fitted classifier.
    X : np.ndarray of shape (n_samples, n_features)
        Input samples.
    classes : np.ndarray of shape (n_classes,)
        The target column order.

    Returns
    -------
    proba : np.ndarray of shape (n_samples, n_classes)
        Probabilities, one column per entry of ``classes``.
    """
    import warnings

    with warnings.catch_warnings():
        # Classifiers without native probabilities fall back to one-hot and
        # warn; inside an ensemble that is a deliberate, harmless choice.
        warnings.simplefilter("ignore")
        raw = np.asarray(model.predict_proba(X), dtype=float)

    model_classes = getattr(model, "classes_", None)
    if model_classes is None or len(model_classes) != raw.shape[1]:
        if raw.shape[1] == len(classes):
            return raw
        model_classes = classes[: raw.shape[1]]

    aligned = np.zeros((raw.shape[0], len(classes)), dtype=float)
    lookup = {label: column for column, label in enumerate(classes)}
    for column, label in enumerate(np.asarray(model_classes)):
        target = lookup.get(label)
        if target is not None:
            aligned[:, target] = raw[:, column]
    return aligned


__all__ = ["greedy_selection", "GreedyEnsemble", "align_proba"]
