"""Weka Bayesian classifier wrappers.

Probabilistic classifiers from ``weka.classifiers.bayes``, registered under
``weka.<ClassName>`` hub keys.

Notes
-----
Requires the optional Weka extra: ``pip install 'tuiml[weka]'`` plus a Java
runtime (11+) on ``PATH``.
"""

from typing import Any, Dict, List, Optional, Sequence

from tuiml.base.algorithms import Classifier
from tuiml.weka._base import _WekaSupervisedMixin, weka_classifier

__all__ = ["NaiveBayes", "NaiveBayesMultinomial", "BayesNet"]


@weka_classifier(tags=["bayes", "probabilistic"])
class NaiveBayes(_WekaSupervisedMixin, Classifier):
    """**NaiveBayes** — Weka's naive Bayes classifier (hub key ``weka.NaiveBayes``).

    Wraps ``weka.classifiers.bayes.NaiveBayes``. Numeric attributes are modelled
    with a single Gaussian per class by default; the alternatives below trade
    that assumption for a kernel or a discretization.

    Parameters
    ----------
    use_kernel_estimator : bool, default=False
        Model numeric attributes with a kernel density estimate rather than a
        single normal distribution (Weka ``-K``). Better when an attribute is
        clearly not Gaussian.
    use_supervised_discretization : bool, default=False
        Discretize numeric attributes with supervised (MDL) binning
        (Weka ``-D``). Mutually exclusive with ``use_kernel_estimator``.
    nominal_features : sequence of int or None, default=None
        0-based indices of columns to treat as categorical.
    options : sequence of str or None, default=None
        Extra raw Weka options.

    Raises
    ------
    ValueError
        If both ``use_kernel_estimator`` and ``use_supervised_discretization``
        are set: Weka accepts only one numeric-attribute strategy at a time.

    See Also
    --------
    :class:`~tuiml.algorithms.bayesian.NaiveBayesClassifier` : TuiML's native naive Bayes.

    Examples
    --------
    >>> from tuiml.weka import NaiveBayes
    >>> from tuiml.datasets import load_iris
    >>> data = load_iris()
    >>> clf = NaiveBayes().fit(data.X, data.y)
    >>> clf.predict(data.X[:5]).shape
    (5,)
    """

    _weka_classname = "weka.classifiers.bayes.NaiveBayes"
    _is_classification = True

    def __init__(
        self,
        use_kernel_estimator: bool = False,
        use_supervised_discretization: bool = False,
        nominal_features: Optional[Sequence[int]] = None,
        options: Optional[Sequence[str]] = None,
    ):
        super().__init__()
        self.use_kernel_estimator = use_kernel_estimator
        self.use_supervised_discretization = use_supervised_discretization
        self.nominal_features = nominal_features
        self.options = options

    def _options(self) -> List[str]:
        """Return the Weka option tokens for this configuration."""
        if self.use_kernel_estimator and self.use_supervised_discretization:
            raise ValueError(
                "use_kernel_estimator and use_supervised_discretization are "
                "mutually exclusive; Weka accepts only one of -K and -D."
            )
        opts: List[str] = []
        if self.use_kernel_estimator:
            opts.append("-K")
        if self.use_supervised_discretization:
            opts.append("-D")
        return opts

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Any]:
        """Return JSON Schema for constructor parameters."""
        return {
            "use_kernel_estimator": {"type": "boolean", "default": False},
            "use_supervised_discretization": {"type": "boolean", "default": False},
        }

    @classmethod
    def get_capabilities(cls) -> List[str]:
        """Return the capability names this wrapper supports."""
        return ["numeric", "nominal", "missing_values", "binary_class", "multiclass"]


@weka_classifier(tags=["bayes", "probabilistic", "text"])
class NaiveBayesMultinomial(_WekaSupervisedMixin, Classifier):
    """**NaiveBayesMultinomial** — multinomial naive Bayes (hub key ``weka.NaiveBayesMultinomial``).

    Wraps ``weka.classifiers.bayes.NaiveBayesMultinomial``. Models each
    attribute as a count, which is the right assumption for bag-of-words text
    features.

    Parameters
    ----------
    nominal_features : sequence of int or None, default=None
        0-based indices of columns to treat as categorical.
    options : sequence of str or None, default=None
        Extra raw Weka options.

    Notes
    -----
    Expects **non-negative counts**, not arbitrary real-valued features.

    See Also
    --------
    :class:`~tuiml.algorithms.bayesian.NaiveBayesMultinomialClassifier` : TuiML's native version.

    Examples
    --------
    >>> from tuiml.weka import NaiveBayesMultinomial
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> X = rng.integers(0, 5, size=(60, 4)).astype(float)
    >>> y = (X[:, 0] > 2).astype(int)
    >>> clf = NaiveBayesMultinomial().fit(X, y)
    >>> clf.predict(X[:5]).shape
    (5,)
    """

    _weka_classname = "weka.classifiers.bayes.NaiveBayesMultinomial"
    _is_classification = True

    def __init__(
        self,
        nominal_features: Optional[Sequence[int]] = None,
        options: Optional[Sequence[str]] = None,
    ):
        super().__init__()
        self.nominal_features = nominal_features
        self.options = options

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Any]:
        """Return JSON Schema for constructor parameters."""
        return {}

    @classmethod
    def get_capabilities(cls) -> List[str]:
        """Return the capability names this wrapper supports."""
        return ["numeric", "binary_class", "multiclass"]


@weka_classifier(tags=["bayes", "network", "probabilistic"])
class BayesNet(_WekaSupervisedMixin, Classifier):
    """**BayesNet** — Bayesian network classifier (hub key ``weka.BayesNet``).

    Wraps ``weka.classifiers.bayes.BayesNet``. Learns a network structure over
    the attributes and estimates its conditional probability tables, so unlike
    naive Bayes it can represent dependencies between attributes.

    Parameters
    ----------
    search_algorithm : str, default="weka.classifiers.bayes.net.search.local.K2"
        Fully-qualified Weka class implementing the structure search
        (Weka ``-Q``).
    estimator : str, default="weka.classifiers.bayes.net.estimate.SimpleEstimator"
        Fully-qualified Weka class estimating the conditional probability
        tables (Weka ``-E``).
    nominal_features : sequence of int or None, default=None
        0-based indices of columns to treat as categorical.
    options : sequence of str or None, default=None
        Extra raw Weka options.

    Notes
    -----
    ``search_algorithm`` and ``estimator`` take a **bare class name**: Weka's
    ``-Q`` and ``-E`` do not accept options embedded in the value, so
    per-search or per-estimator settings (such as the estimator's ``-A``
    smoothing prior) cannot be passed here.

    BayesNet requires **nominal** attributes. Numeric columns must be
    discretized first — either by passing their indices in ``nominal_features``
    when they already hold integer codes, or by binning them beforehand with
    :mod:`tuiml.preprocessing.discretization`.

    See Also
    --------
    :class:`~tuiml.algorithms.bayesian.NaiveBayesClassifier` : TuiML's native Bayesian classifier.

    Examples
    --------
    >>> from tuiml.weka import BayesNet
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> X = rng.integers(0, 3, size=(60, 4)).astype(float)
    >>> y = (X[:, 0] > 1).astype(int)
    >>> clf = BayesNet(nominal_features=range(4)).fit(X, y)
    >>> clf.predict(X[:5]).shape
    (5,)
    """

    _weka_classname = "weka.classifiers.bayes.BayesNet"
    _is_classification = True

    def __init__(
        self,
        search_algorithm: str = "weka.classifiers.bayes.net.search.local.K2",
        estimator: str = "weka.classifiers.bayes.net.estimate.SimpleEstimator",
        nominal_features: Optional[Sequence[int]] = None,
        options: Optional[Sequence[str]] = None,
    ):
        super().__init__()
        self.search_algorithm = search_algorithm
        self.estimator = estimator
        self.nominal_features = nominal_features
        self.options = options

    def _options(self) -> List[str]:
        """Return the Weka option tokens for this configuration."""
        return ["-Q", self.search_algorithm, "-E", self.estimator]

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Any]:
        """Return JSON Schema for constructor parameters."""
        return {
            "search_algorithm": {
                "type": "string",
                "default": "weka.classifiers.bayes.net.search.local.K2"},
            "estimator": {
                "type": "string",
                "default": "weka.classifiers.bayes.net.estimate.SimpleEstimator"},
        }

    @classmethod
    def get_capabilities(cls) -> List[str]:
        """Return the capability names this wrapper supports."""
        return ["nominal", "missing_values", "binary_class", "multiclass"]
