"""TabICL — in-context learning on tabular data, with no gradient step.

Wraps the upstream ``tabicl`` package from Inria's Soda team. TuiML ships no
weights; ``tabicl`` fetches its own checkpoint on first use, and both its code
and its weights are BSD-3-Clause, the same license as TuiML.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from tuiml.base.algorithms import Classifier, Regressor
from tuiml.algorithms.tabular_foundation._foundation import (
    fd_classifier,
    fd_regressor,
    require_package,
)

#: Constructor parameters shared by the classifier and the regressor, as
#: ``{name: (json_type, default, description)}``. Kept in one place so the two
#: schemas cannot drift apart.
_SHARED_SCHEMA = {
    "n_estimators": (
        "integer", 8,
        "Number of ensemble members. Each sees a different feature/class "
        "permutation and normalisation; their logits are averaged. More "
        "members cost proportionally more forward passes.",
    ),
    "batch_size": (
        "integer", 8,
        "Ensemble members evaluated per forward pass. Lower this before "
        "lowering n_estimators when memory is tight — it trades speed for "
        "peak memory without changing the prediction.",
    ),
    "outlier_threshold": (
        "number", 4.0,
        "Standard deviations beyond which a value is clipped during "
        "preprocessing.",
    ),
    "device": (
        "string", None,
        "Torch device: 'cpu', 'cuda', 'mps', or None to let tabicl choose. "
        "A GPU is worth an order of magnitude here.",
    ),
    "allow_auto_download": (
        "boolean", True,
        "Download the checkpoint from the Hugging Face Hub on first use "
        "(~150 MB, cached in ~/.cache/huggingface). Set False in air-gapped "
        "environments and supply model_path instead.",
    ),
    "model_path": (
        "string", None,
        "Path to a local checkpoint, bypassing the download entirely.",
    ),
    "random_state": (
        "integer", 42,
        "Seed for the ensemble's permutations. Fixed by default, so repeated "
        "runs agree.",
    ),
    "n_jobs": (
        "integer", None,
        "Threads for preprocessing. None lets tabicl decide.",
    ),
    "verbose": (
        "boolean", False,
        "Print progress, including checkpoint download.",
    ),
}


def _schema_from(spec: Dict[str, tuple]) -> Dict[str, Dict[str, Any]]:
    """Turn the compact ``{name: (type, default, doc)}`` table into JSON Schema.

    Parameters
    ----------
    spec : dict
        Mapping of parameter name to ``(json_type, default, description)``.

    Returns
    -------
    schema : dict
        JSON-Schema-shaped dict, one entry per parameter.
    """
    return {
        name: {"type": json_type, "default": default, "description": doc}
        for name, (json_type, default, doc) in spec.items()
    }


class _TabICLBase:
    """Delegation shared by the TabICL classifier and regressor.

    Holds the constructor bookkeeping, the upstream-estimator construction and
    ``fit``/``predict``. The two public classes differ only in their base class,
    their extra parameters and their docstrings.
    """

    #: Name of the upstream class in the ``tabicl`` package.
    _upstream: str = ""

    def _upstream_kwargs(self) -> Dict[str, Any]:
        """Return the keyword arguments to construct the upstream estimator."""
        return dict(self._params)

    def _build_estimator(self) -> Any:
        """Import ``tabicl`` and return a fresh, configured upstream estimator.

        Returns
        -------
        estimator : object
            An unfitted upstream TabICL estimator.
        """
        module = require_package("tabicl", type(self).__name__)
        cls = getattr(module, self._upstream)
        return cls(**self._upstream_kwargs())

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_TabICLBase":
        """Store the training set as in-context examples.

        No gradient step happens here and no parameter is updated: TabICL is
        frozen. The cost of "fitting" is only preprocessing and memorisation,
        and essentially all the compute lands in :meth:`predict`, which is the
        reverse of every other algorithm in TuiML.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : object
            The fitted wrapper.
        """
        self.model_ = self._build_estimator()
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.model_.fit(X, y)
        self.n_features_in_ = X.shape[1]
        if hasattr(self.model_, "classes_"):
            self.classes_ = self.model_.classes_
        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict by a single forward pass conditioned on the training set.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.

        Returns
        -------
        y_pred : np.ndarray of shape (n_samples,)
            Predicted values.
        """
        self._check_is_fitted()
        return self.model_.predict(np.asarray(X, dtype=float))


@fd_classifier(tags=["foundation-model", "pretrained", "in-context-learning",
                     "transformer", "ensemble"], version="1.0.0")
class TabICLClassifier(_TabICLBase, Classifier):
    r"""Tabular classification by **in-context learning**, with no training.

    TabICL is a **pretrained foundation model**: a transformer trained once, on
    millions of synthetic tabular tasks, to *be* a learning algorithm. Your
    training rows are fed to the frozen network as context alongside the rows
    you want predicted, and the answer falls out of a single forward pass.
    Nothing is optimised on your data — there is no gradient step, no
    hyperparameter search, and no fitted coefficient anywhere.

    Overview
    --------
    1. Each column is normalised and outliers are clipped.
    2. A column-then-row attention stack embeds every cell in the context of
       its column's distribution and its row's other values.
    3. Training rows and their labels form the *context*; test rows are
       appended without labels.
    4. One forward pass returns a distribution over classes for each test row.
    5. Steps 1-4 repeat for ``n_estimators`` permutations of the features and
       classes, and the logits are averaged.

    Theory
    ------
    A conventional classifier searches for parameters :math:`\theta` minimising
    a loss on your data. TabICL does not. It approximates the **posterior
    predictive distribution** directly, having been pretrained to map a context
    set onto predictions:

    .. math::
        p(y \\mid x, D_{\\text{train}})
            = \\int p(y \\mid x, \\theta) \\, p(\\theta \\mid D_{\\text{train}})
              \\, d\\theta

    The integral is what a Bayesian method would have to approximate by
    sampling. Pretraining on millions of synthetic datasets amortises it into
    the network's weights, so at your dataset the whole thing is one forward
    pass. This is why the method needs no tuning: the "learning algorithm"
    was itself learned, once, in advance.

    Parameters
    ----------
    n_estimators : int, default=8
        Number of ensemble members, each with a different feature and class
        permutation. Logits are averaged.
    softmax_temperature : float, default=0.9
        Temperature applied to the logits. Below 1 sharpens the predicted
        distribution; raise it if probabilities look overconfident.
    average_logits : bool, default=True
        Average ensemble members in logit space rather than probability space.
    outlier_threshold : float, default=4.0
        Standard deviations beyond which a value is clipped in preprocessing.
    batch_size : int, default=8
        Ensemble members evaluated per forward pass. Lower it before lowering
        ``n_estimators`` when memory is tight.
    device : str or None, default=None
        Torch device. ``None`` lets tabicl choose.
    allow_auto_download : bool, default=True
        Fetch the checkpoint (~150 MB) from the Hugging Face Hub on first use.
    model_path : str or None, default=None
        Local checkpoint path, bypassing the download.
    random_state : int, default=42
        Seed for the ensemble permutations.
    n_jobs : int or None, default=None
        Preprocessing threads.
    verbose : bool, default=False
        Print progress, including the download.

    Attributes
    ----------
    model_ : object
        The underlying ``tabicl.TabICLClassifier``.
    classes_ : np.ndarray of shape (n_classes,)
        Class labels seen in ``y``.
    n_features_in_ : int
        Number of features seen during ``fit``.

    Notes
    -----
    **Requires** ``pip install 'tuiml[foundation]'``, which pulls in ``tabicl``
    and PyTorch. Importing TuiML and constructing this class work without
    either; only :meth:`fit` needs them.

    **Weights are not shipped by TuiML.** On first use ``tabicl`` downloads its
    checkpoint from the Hugging Face Hub into ``~/.cache/huggingface``. Code
    and weights are both BSD-3-Clause, matching TuiML's own license, so there
    is no commercial-use restriction to accept. That is *not* true of every
    tabular foundation model — several publish weights under non-commercial
    terms — which is why TabICL is the only one integrated here.

    **Complexity:** fitting is :math:`O(1)` in the sense that matters — no
    optimisation runs. Prediction is a transformer forward pass over
    :math:`n_{\\text{train}} + n_{\\text{test}}` rows per ensemble member, so
    cost grows with the *training* set at predict time. A GPU is worth roughly
    an order of magnitude.

    **When to use TabICL:**

    - Small to medium tabular data (up to roughly 100k rows) where you want a
      strong result without tuning anything.
    - As a baseline that costs one line and no search budget.
    - When labelled data is scarce — amortised pretraining is at its most
      valuable where a from-scratch model would overfit.

    Prefer a gradient-boosted ensemble
    (:class:`~tuiml.algorithms.gradient_boosting.XGBoostClassifier`) on large
    data, where TabICL's per-prediction cost stops being worth it.

    References
    ----------
    .. [Qu2025] Qu, J., Holzmüller, D., Varoquaux, G. and Le Morvan, M. (2025).
           **TabICL: A Tabular Foundation Model for In-Context Learning on
           Large Data.** *Proceedings of the 42nd International Conference on
           Machine Learning (ICML)*.
           arXiv: `2502.05564 <https://arxiv.org/abs/2502.05564>`_

    See Also
    --------
    :class:`~tuiml.algorithms.tabular_foundation.TabICLRegressor` : Regression counterpart.
    :class:`~tuiml.algorithms.tabular_foundation.FTTransformerClassifier` : Native transformer trained on your data.
    :class:`~tuiml.algorithms.gradient_boosting.XGBoostClassifier` : The usual strong tabular baseline.

    Examples
    --------
    Constructing and inspecting the model needs no optional dependency:

    >>> from tuiml.algorithms.tabular_foundation import TabICLClassifier
    >>> clf = TabICLClassifier(n_estimators=2)
    >>> clf.n_estimators
    2
    >>> "softmax_temperature" in clf.get_parameter_schema()
    True

    Fitting requires ``pip install 'tuiml[foundation]'`` and downloads the
    checkpoint on first use:

    >>> import numpy as np  # doctest: +SKIP
    >>> rng = np.random.default_rng(0)  # doctest: +SKIP
    >>> X = rng.normal(size=(60, 4))  # doctest: +SKIP
    >>> y = (X[:, 0] > 0).astype(int)  # doctest: +SKIP
    >>> clf.fit(X, y).predict(X[:5])  # doctest: +SKIP
    array([1, 0, 0, 1, 1])
    """

    _upstream = "TabICLClassifier"

    def __init__(
        self,
        n_estimators: int = 8,
        softmax_temperature: float = 0.9,
        average_logits: bool = True,
        outlier_threshold: float = 4.0,
        batch_size: int = 8,
        device: Optional[str] = None,
        allow_auto_download: bool = True,
        model_path: Optional[str] = None,
        random_state: int = 42,
        n_jobs: Optional[int] = None,
        verbose: bool = False,
    ) -> None:
        """Record hyperparameters. Imports nothing — see the class Notes."""
        super().__init__()
        self.n_estimators = n_estimators
        self.softmax_temperature = softmax_temperature
        self.average_logits = average_logits
        self.outlier_threshold = outlier_threshold
        self.batch_size = batch_size
        self.device = device
        self.allow_auto_download = allow_auto_download
        self.model_path = model_path
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose
        self._params = {
            "n_estimators": n_estimators,
            "softmax_temperature": softmax_temperature,
            "average_logits": average_logits,
            "outlier_threshold": outlier_threshold,
            "batch_size": batch_size,
            "device": device,
            "allow_auto_download": allow_auto_download,
            "model_path": model_path,
            "random_state": random_state,
            "n_jobs": n_jobs,
            "verbose": verbose,
        }
        self.model_ = None
        self.classes_ = None
        self.n_features_in_ = None

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probabilities from the frozen network.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to score.

        Returns
        -------
        proba : np.ndarray of shape (n_samples, n_classes)
            Per-class probabilities, rows summing to 1.
        """
        self._check_is_fitted()
        return self.model_.predict_proba(np.asarray(X, dtype=float))

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        """Return JSON Schema for constructor parameters."""
        schema = _schema_from(_SHARED_SCHEMA)
        schema.update(_schema_from({
            "softmax_temperature": (
                "number", 0.9,
                "Temperature on the logits. Below 1 sharpens the predicted "
                "distribution.",
            ),
            "average_logits": (
                "boolean", True,
                "Average ensemble members in logit rather than probability "
                "space.",
            ),
        }))
        return schema

    @classmethod
    def get_capabilities(cls) -> List[str]:
        """Return the data and task properties this model supports."""
        return [
            "numeric", "categorical", "missing_values",
            "binary_class", "multiclass", "probabilistic",
            "ensemble", "non_linear",
        ]


@fd_regressor(tags=["foundation-model", "pretrained", "in-context-learning",
                    "transformer", "ensemble"], version="1.0.0")
class TabICLRegressor(_TabICLBase, Regressor):
    r"""Tabular regression by **in-context learning**, with no training.

    The regression counterpart of
    :class:`~tuiml.algorithms.tabular_foundation.TabICLClassifier`, using the
    same frozen pretrained transformer and the same one-forward-pass
    prediction. See that class for the full description of the method; this
    docstring covers only what differs.

    Overview
    --------
    1. Columns are normalised and outliers clipped.
    2. Column-then-row attention embeds each cell.
    3. Training rows and their targets form the context; test rows are
       appended without targets.
    4. One forward pass returns a predicted value per test row.
    5. Steps 1-4 repeat over ``n_estimators`` feature permutations and the
       predictions are averaged.

    Theory
    ------
    As in the classifier, the network approximates the posterior predictive
    distribution rather than fitting parameters to your data:

    .. math::
        p(y \\mid x, D_{\\text{train}})
            = \\int p(y \\mid x, \\theta) \\, p(\\theta \\mid D_{\\text{train}})
              \\, d\\theta

    with a continuous rather than categorical target. There is no class
    permutation to average over, so the regressor has neither
    ``class_shuffle_method`` nor ``softmax_temperature``.

    Parameters
    ----------
    n_estimators : int, default=8
        Ensemble members, each with a different feature permutation.
    outlier_threshold : float, default=4.0
        Standard deviations beyond which a value is clipped in preprocessing.
    batch_size : int, default=8
        Ensemble members per forward pass.
    device : str or None, default=None
        Torch device. ``None`` lets tabicl choose.
    allow_auto_download : bool, default=True
        Fetch the checkpoint (~150 MB) on first use.
    model_path : str or None, default=None
        Local checkpoint path, bypassing the download.
    random_state : int, default=42
        Seed for the ensemble permutations.
    n_jobs : int or None, default=None
        Preprocessing threads.
    verbose : bool, default=False
        Print progress, including the download.

    Attributes
    ----------
    model_ : object
        The underlying ``tabicl.TabICLRegressor``.
    n_features_in_ : int
        Number of features seen during ``fit``.

    Notes
    -----
    **Requires** ``pip install 'tuiml[foundation]'``. Importing and
    constructing work without it; only :meth:`fit` needs it. TuiML ships no
    weights — ``tabicl`` downloads its own checkpoint, BSD-3-Clause like TuiML
    itself.

    **Complexity:** no optimisation runs at fit time; prediction is a forward
    pass over training plus test rows, per ensemble member.

    **When to use TabICLRegressor:** small to medium tabular regression where
    you want a strong untuned baseline. Prefer
    :class:`~tuiml.algorithms.gradient_boosting.XGBoostRegressor` on large
    data, and :class:`~tuiml.algorithms.gradient_boosting.NGBoostRegressor`
    when you need a calibrated predictive *distribution* rather than a point.

    References
    ----------
    .. [Qu2025] Qu, J., Holzmüller, D., Varoquaux, G. and Le Morvan, M. (2025).
           **TabICL: A Tabular Foundation Model for In-Context Learning on
           Large Data.** *Proceedings of the 42nd International Conference on
           Machine Learning (ICML)*.
           arXiv: `2502.05564 <https://arxiv.org/abs/2502.05564>`_

    See Also
    --------
    :class:`~tuiml.algorithms.tabular_foundation.TabICLClassifier` : Classification counterpart.
    :class:`~tuiml.algorithms.gradient_boosting.NGBoostRegressor` : Native probabilistic boosting.
    :class:`~tuiml.algorithms.tabular_foundation.FTTransformerRegressor` : Native transformer trained on your data.

    Examples
    --------
    >>> from tuiml.algorithms.tabular_foundation import TabICLRegressor
    >>> reg = TabICLRegressor(n_estimators=2)
    >>> reg.n_estimators
    2
    >>> "softmax_temperature" in reg.get_parameter_schema()
    False
    """

    _upstream = "TabICLRegressor"

    def __init__(
        self,
        n_estimators: int = 8,
        outlier_threshold: float = 4.0,
        batch_size: int = 8,
        device: Optional[str] = None,
        allow_auto_download: bool = True,
        model_path: Optional[str] = None,
        random_state: int = 42,
        n_jobs: Optional[int] = None,
        verbose: bool = False,
    ) -> None:
        """Record hyperparameters. Imports nothing — see the class Notes."""
        super().__init__()
        self.n_estimators = n_estimators
        self.outlier_threshold = outlier_threshold
        self.batch_size = batch_size
        self.device = device
        self.allow_auto_download = allow_auto_download
        self.model_path = model_path
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose
        self._params = {
            "n_estimators": n_estimators,
            "outlier_threshold": outlier_threshold,
            "batch_size": batch_size,
            "device": device,
            "allow_auto_download": allow_auto_download,
            "model_path": model_path,
            "random_state": random_state,
            "n_jobs": n_jobs,
            "verbose": verbose,
        }
        self.model_ = None
        self.n_features_in_ = None

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        """Return JSON Schema for constructor parameters."""
        return _schema_from(_SHARED_SCHEMA)

    @classmethod
    def get_capabilities(cls) -> List[str]:
        """Return the data and task properties this model supports."""
        return [
            "numeric", "categorical", "missing_values",
            "regression", "numeric_class",
            "ensemble", "non_linear",
        ]


__all__ = ["TabICLClassifier", "TabICLRegressor"]
