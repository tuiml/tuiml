"""Weka meta-learner wrappers.

Ensemble and wrapper schemes from ``weka.classifiers.meta``, registered under
``weka.<ClassName>`` hub keys. Every learner here takes a **base learner**,
named by its fully-qualified Weka class.

Notes
-----
Requires the optional Weka extra: ``pip install 'tuiml[weka]'`` plus a Java
runtime (11+) on ``PATH``.

Weka's command line nests a base learner's own options after a ``--``
separator::

    -W weka.classifiers.trees.J48 -- -C 0.25 -M 2

:func:`_base_spec` builds that tail, so a wrapper only needs ``base_classifier``
and ``base_options``.
"""

from typing import Any, Dict, List, Optional, Sequence

from tuiml.base.algorithms import Classifier, Regressor
from tuiml.weka._base import (
    _WekaSupervisedMixin,
    fmt_num,
    weka_classifier,
    weka_regressor,
)

__all__ = [
    "AdaBoostM1",
    "Bagging",
    "LogitBoost",
    "RandomCommittee",
    "RandomSubSpace",
    "MultiClassClassifier",
    "FilteredClassifier",
    "AdditiveRegression",
    "RegressionByDiscretization",
    "Vote",
    "Stacking",
]


def _base_spec(
    base_classifier: str, base_options: Optional[Sequence[str]] = None
) -> List[str]:
    """Build the ``-W <class> [-- <options>]`` tail for a meta-learner.

    Parameters
    ----------
    base_classifier : str
        Fully-qualified Weka class of the base learner.
    base_options : sequence of str or None, default=None
        Options for the base learner. They are placed after ``--`` so Weka
        routes them to the base learner rather than the meta-learner.

    Returns
    -------
    tokens : list of str
        The option tokens.

    Examples
    --------
    >>> from tuiml.weka.meta import _base_spec
    >>> _base_spec("weka.classifiers.trees.J48", ["-C", "0.1"])
    ['-W', 'weka.classifiers.trees.J48', '--', '-C', '0.1']
    """
    tokens = ["-W", base_classifier]
    if base_options:
        tokens.append("--")
        tokens.extend(base_options)
    return tokens


class _MetaBase(_WekaSupervisedMixin):
    """Shared constructor and option handling for single-base-learner metas.

    Attributes
    ----------
    base_classifier : str
        Fully-qualified Weka class used as the base learner.
    base_options : sequence of str or None
        Options routed to the base learner.
    """

    #: Default base learner when the caller does not name one.
    _default_base = "weka.classifiers.trees.DecisionStump"

    def __init__(
        self,
        base_classifier: Optional[str] = None,
        base_options: Optional[Sequence[str]] = None,
        nominal_features: Optional[Sequence[int]] = None,
        options: Optional[Sequence[str]] = None,
    ):
        super().__init__()
        self.base_classifier = base_classifier or self._default_base
        self.base_options = base_options
        self.nominal_features = nominal_features
        self.options = options

    def _meta_options(self) -> List[str]:
        """Return the meta-learner's own options, excluding the base spec."""
        return []

    def _options(self) -> List[str]:
        """Return the meta options followed by the base-learner spec."""
        return self._meta_options() + _base_spec(self.base_classifier, self.base_options)


@weka_classifier(tags=["meta", "ensemble", "boosting"])
class AdaBoostM1(_MetaBase, Classifier):
    """**AdaBoostM1** — adaptive boosting (hub key ``weka.AdaBoostM1``).

    Wraps ``weka.classifiers.meta.AdaBoostM1``.

    Parameters
    ----------
    num_iterations : int, default=10
        Number of boosting rounds (Weka ``-I``).
    weight_threshold : int, default=100
        Weight pruning threshold as a percentage (Weka ``-P``).
    seed : int, default=1
        Random seed (Weka ``-S``).
    use_resampling : bool, default=False
        Resample instead of reweighting (Weka ``-Q``), for base learners that
        cannot use instance weights.
    base_classifier : str or None, default=None
        Base learner; defaults to ``weka.classifiers.trees.DecisionStump``.
    base_options : sequence of str or None, default=None
        Options for the base learner.
    nominal_features : sequence of int or None, default=None
        0-based indices of columns to treat as categorical.
    options : sequence of str or None, default=None
        Extra raw Weka options.

    See Also
    --------
    :class:`~tuiml.algorithms.ensemble.AdaBoostClassifier` : TuiML's native AdaBoost.

    Examples
    --------
    >>> from tuiml.weka import AdaBoostM1
    >>> from tuiml.datasets import load_iris
    >>> data = load_iris()
    >>> clf = AdaBoostM1(num_iterations=10).fit(data.X, data.y)
    >>> clf.predict(data.X[:5]).shape
    (5,)
    """

    _weka_classname = "weka.classifiers.meta.AdaBoostM1"
    _is_classification = True
    _default_base = "weka.classifiers.trees.DecisionStump"

    def __init__(
        self,
        num_iterations: int = 10,
        weight_threshold: int = 100,
        seed: int = 1,
        use_resampling: bool = False,
        base_classifier: Optional[str] = None,
        base_options: Optional[Sequence[str]] = None,
        nominal_features: Optional[Sequence[int]] = None,
        options: Optional[Sequence[str]] = None,
    ):
        super().__init__(base_classifier, base_options, nominal_features, options)
        self.num_iterations = num_iterations
        self.weight_threshold = weight_threshold
        self.seed = seed
        self.use_resampling = use_resampling

    def _meta_options(self) -> List[str]:
        """Return the boosting options."""
        opts = ["-I", fmt_num(self.num_iterations),
                "-P", fmt_num(self.weight_threshold),
                "-S", fmt_num(self.seed)]
        if self.use_resampling:
            opts.append("-Q")
        return opts

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Any]:
        """Return JSON Schema for constructor parameters."""
        return {
            "num_iterations": {"type": "integer", "default": 10, "minimum": 1},
            "weight_threshold": {"type": "integer", "default": 100},
            "seed": {"type": "integer", "default": 1},
            "use_resampling": {"type": "boolean", "default": False},
            "base_classifier": {"type": "string",
                                "default": "weka.classifiers.trees.DecisionStump"},
        }

    @classmethod
    def get_capabilities(cls) -> List[str]:
        """Return the capability names this wrapper supports."""
        return ["numeric", "nominal", "missing_values", "binary_class", "multiclass"]


@weka_classifier(tags=["meta", "ensemble", "bagging"])
class Bagging(_MetaBase, Classifier):
    """**Bagging** — bootstrap aggregating (hub key ``weka.Bagging``).

    Wraps ``weka.classifiers.meta.Bagging``.

    Parameters
    ----------
    num_iterations : int, default=10
        Number of bags (Weka ``-I``).
    bag_size_percent : int, default=100
        Size of each bag as a percentage of the training set (Weka ``-P``).
    seed : int, default=1
        Random seed (Weka ``-S``).
    calc_out_of_bag : bool, default=False
        Compute the out-of-bag error (Weka ``-O``).
    base_classifier : str or None, default=None
        Base learner; defaults to ``weka.classifiers.trees.REPTree``.
    base_options : sequence of str or None, default=None
        Options for the base learner.
    nominal_features : sequence of int or None, default=None
        0-based indices of columns to treat as categorical.
    options : sequence of str or None, default=None
        Extra raw Weka options.

    See Also
    --------
    :class:`~tuiml.algorithms.ensemble.BaggingClassifier` : TuiML's native bagging.

    Examples
    --------
    >>> from tuiml.weka import Bagging
    >>> from tuiml.datasets import load_iris
    >>> data = load_iris()
    >>> clf = Bagging(num_iterations=10).fit(data.X, data.y)
    >>> clf.predict(data.X[:5]).shape
    (5,)
    """

    _weka_classname = "weka.classifiers.meta.Bagging"
    _is_classification = True
    _default_base = "weka.classifiers.trees.REPTree"

    def __init__(
        self,
        num_iterations: int = 10,
        bag_size_percent: int = 100,
        seed: int = 1,
        calc_out_of_bag: bool = False,
        base_classifier: Optional[str] = None,
        base_options: Optional[Sequence[str]] = None,
        nominal_features: Optional[Sequence[int]] = None,
        options: Optional[Sequence[str]] = None,
    ):
        super().__init__(base_classifier, base_options, nominal_features, options)
        self.num_iterations = num_iterations
        self.bag_size_percent = bag_size_percent
        self.seed = seed
        self.calc_out_of_bag = calc_out_of_bag

    def _meta_options(self) -> List[str]:
        """Return the bagging options."""
        opts = ["-I", fmt_num(self.num_iterations),
                "-P", fmt_num(self.bag_size_percent),
                "-S", fmt_num(self.seed)]
        if self.calc_out_of_bag:
            opts.append("-O")
        return opts

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Any]:
        """Return JSON Schema for constructor parameters."""
        return {
            "num_iterations": {"type": "integer", "default": 10, "minimum": 1},
            "bag_size_percent": {"type": "integer", "default": 100, "minimum": 1},
            "seed": {"type": "integer", "default": 1},
            "calc_out_of_bag": {"type": "boolean", "default": False},
            "base_classifier": {"type": "string",
                                "default": "weka.classifiers.trees.REPTree"},
        }

    @classmethod
    def get_capabilities(cls) -> List[str]:
        """Return the capability names this wrapper supports."""
        return ["numeric", "nominal", "missing_values", "binary_class", "multiclass"]


@weka_classifier(tags=["meta", "ensemble", "boosting"])
class LogitBoost(_MetaBase, Classifier):
    """**LogitBoost** — additive logistic regression (hub key ``weka.LogitBoost``).

    Wraps ``weka.classifiers.meta.LogitBoost``. Performs additive logistic
    regression by fitting a regression base learner to the working response at
    each round.

    Parameters
    ----------
    num_iterations : int, default=10
        Number of boosting rounds (Weka ``-I``).
    shrinkage : float, default=1.0
        Shrinkage applied to each round (Weka ``-H``).
    seed : int, default=1
        Random seed (Weka ``-S``).
    base_classifier : str or None, default=None
        Base learner, which must be a **regressor**; defaults to
        ``weka.classifiers.trees.DecisionStump``.
    base_options : sequence of str or None, default=None
        Options for the base learner.
    nominal_features : sequence of int or None, default=None
        0-based indices of columns to treat as categorical.
    options : sequence of str or None, default=None
        Extra raw Weka options.

    See Also
    --------
    :class:`~tuiml.algorithms.ensemble.AdaBoostClassifier` : TuiML's native boosting meta-classifier.

    Examples
    --------
    >>> from tuiml.weka import LogitBoost
    >>> from tuiml.datasets import load_iris
    >>> data = load_iris()
    >>> clf = LogitBoost(num_iterations=10).fit(data.X, data.y)
    >>> clf.predict(data.X[:5]).shape
    (5,)
    """

    _weka_classname = "weka.classifiers.meta.LogitBoost"
    _is_classification = True
    _default_base = "weka.classifiers.trees.DecisionStump"

    def __init__(
        self,
        num_iterations: int = 10,
        shrinkage: float = 1.0,
        seed: int = 1,
        base_classifier: Optional[str] = None,
        base_options: Optional[Sequence[str]] = None,
        nominal_features: Optional[Sequence[int]] = None,
        options: Optional[Sequence[str]] = None,
    ):
        super().__init__(base_classifier, base_options, nominal_features, options)
        self.num_iterations = num_iterations
        self.shrinkage = shrinkage
        self.seed = seed

    def _meta_options(self) -> List[str]:
        """Return the boosting options."""
        return ["-I", fmt_num(self.num_iterations), "-H", str(self.shrinkage),
                "-S", fmt_num(self.seed)]

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Any]:
        """Return JSON Schema for constructor parameters."""
        return {
            "num_iterations": {"type": "integer", "default": 10, "minimum": 1},
            "shrinkage": {"type": "number", "default": 1.0, "minimum": 0.0},
            "seed": {"type": "integer", "default": 1},
        }

    @classmethod
    def get_capabilities(cls) -> List[str]:
        """Return the capability names this wrapper supports."""
        return ["numeric", "nominal", "missing_values", "binary_class", "multiclass"]


@weka_classifier(tags=["meta", "ensemble", "random"])
class RandomCommittee(_MetaBase, Classifier):
    """**RandomCommittee** — average over randomized base learners (hub key ``weka.RandomCommittee``).

    Wraps ``weka.classifiers.meta.RandomCommittee``. Builds several copies of a
    randomizable base learner, each with a different seed, and averages their
    predictions. Unlike bagging, every model sees the **full** training set;
    the only source of diversity is the base learner's own randomness.

    Parameters
    ----------
    num_iterations : int, default=10
        Number of committee members (Weka ``-I``).
    seed : int, default=1
        Random seed (Weka ``-S``).
    base_classifier : str or None, default=None
        Base learner, which must be randomizable; defaults to
        ``weka.classifiers.trees.RandomTree``.
    base_options : sequence of str or None, default=None
        Options for the base learner.
    nominal_features : sequence of int or None, default=None
        0-based indices of columns to treat as categorical.
    options : sequence of str or None, default=None
        Extra raw Weka options.

    Examples
    --------
    >>> from tuiml.weka import RandomCommittee
    >>> from tuiml.datasets import load_iris
    >>> data = load_iris()
    >>> clf = RandomCommittee(num_iterations=10).fit(data.X, data.y)
    >>> clf.predict(data.X[:5]).shape
    (5,)
    """

    _weka_classname = "weka.classifiers.meta.RandomCommittee"
    _is_classification = True
    _default_base = "weka.classifiers.trees.RandomTree"

    def __init__(
        self,
        num_iterations: int = 10,
        seed: int = 1,
        base_classifier: Optional[str] = None,
        base_options: Optional[Sequence[str]] = None,
        nominal_features: Optional[Sequence[int]] = None,
        options: Optional[Sequence[str]] = None,
    ):
        super().__init__(base_classifier, base_options, nominal_features, options)
        self.num_iterations = num_iterations
        self.seed = seed

    def _meta_options(self) -> List[str]:
        """Return the committee options."""
        return ["-I", fmt_num(self.num_iterations), "-S", fmt_num(self.seed)]

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Any]:
        """Return JSON Schema for constructor parameters."""
        return {
            "num_iterations": {"type": "integer", "default": 10, "minimum": 1},
            "seed": {"type": "integer", "default": 1},
            "base_classifier": {"type": "string",
                                "default": "weka.classifiers.trees.RandomTree"},
        }

    @classmethod
    def get_capabilities(cls) -> List[str]:
        """Return the capability names this wrapper supports."""
        return ["numeric", "nominal", "missing_values", "binary_class", "multiclass"]


@weka_classifier(tags=["meta", "ensemble", "subspace"])
class RandomSubSpace(_MetaBase, Classifier):
    """**RandomSubSpace** — random feature subspace ensemble (hub key ``weka.RandomSubSpace``).

    Wraps ``weka.classifiers.meta.RandomSubSpace``. Each member is trained on a
    random subset of the attributes.

    Parameters
    ----------
    num_iterations : int, default=10
        Number of ensemble members (Weka ``-I``).
    subspace_size : float, default=0.5
        Fraction of attributes per member (Weka ``-P``); values above 1 are
        read as an absolute count.
    seed : int, default=1
        Random seed (Weka ``-S``).
    base_classifier : str or None, default=None
        Base learner; defaults to ``weka.classifiers.trees.REPTree``.
    base_options : sequence of str or None, default=None
        Options for the base learner.
    nominal_features : sequence of int or None, default=None
        0-based indices of columns to treat as categorical.
    options : sequence of str or None, default=None
        Extra raw Weka options.

    See Also
    --------
    :class:`~tuiml.algorithms.ensemble.BaggingClassifier` : TuiML's native bagging meta-classifier.

    Examples
    --------
    >>> from tuiml.weka import RandomSubSpace
    >>> from tuiml.datasets import load_iris
    >>> data = load_iris()
    >>> clf = RandomSubSpace(num_iterations=10).fit(data.X, data.y)
    >>> clf.predict(data.X[:5]).shape
    (5,)
    """

    _weka_classname = "weka.classifiers.meta.RandomSubSpace"
    _is_classification = True
    _default_base = "weka.classifiers.trees.REPTree"

    def __init__(
        self,
        num_iterations: int = 10,
        subspace_size: float = 0.5,
        seed: int = 1,
        base_classifier: Optional[str] = None,
        base_options: Optional[Sequence[str]] = None,
        nominal_features: Optional[Sequence[int]] = None,
        options: Optional[Sequence[str]] = None,
    ):
        super().__init__(base_classifier, base_options, nominal_features, options)
        self.num_iterations = num_iterations
        self.subspace_size = subspace_size
        self.seed = seed

    def _meta_options(self) -> List[str]:
        """Return the subspace options."""
        return ["-I", fmt_num(self.num_iterations), "-P", str(self.subspace_size),
                "-S", fmt_num(self.seed)]

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Any]:
        """Return JSON Schema for constructor parameters."""
        return {
            "num_iterations": {"type": "integer", "default": 10, "minimum": 1},
            "subspace_size": {"type": "number", "default": 0.5, "minimum": 0.0},
            "seed": {"type": "integer", "default": 1},
        }

    @classmethod
    def get_capabilities(cls) -> List[str]:
        """Return the capability names this wrapper supports."""
        return ["numeric", "nominal", "missing_values", "binary_class", "multiclass"]


@weka_classifier(tags=["meta", "multiclass", "decomposition"])
class MultiClassClassifier(_MetaBase, Classifier):
    """**MultiClassClassifier** — binary decomposition of a multiclass problem (hub key ``weka.MultiClassClassifier``).

    Wraps ``weka.classifiers.meta.MultiClassClassifier``. Turns a binary base
    learner into a multiclass one by one-vs-rest, one-vs-one, or
    error-correcting output codes.

    Parameters
    ----------
    method : {'ova', 'random_codes', 'exhaustive_codes', 'ovo'}, default='ova'
        Decomposition method (Weka ``-M`` as ``0``-``3``).
    random_width : float, default=2.0
        Code width multiplier for random error-correcting codes (Weka ``-R``).
    seed : int, default=1
        Random seed (Weka ``-S``).
    base_classifier : str or None, default=None
        Base learner; defaults to ``weka.classifiers.functions.Logistic``.
    base_options : sequence of str or None, default=None
        Options for the base learner.
    nominal_features : sequence of int or None, default=None
        0-based indices of columns to treat as categorical.
    options : sequence of str or None, default=None
        Extra raw Weka options.

    See Also
    --------
    :class:`~tuiml.algorithms.ensemble.OneVsRestClassifier` : TuiML's native version.

    Examples
    --------
    >>> from tuiml.weka import MultiClassClassifier
    >>> from tuiml.datasets import load_iris
    >>> data = load_iris()
    >>> clf = MultiClassClassifier(method='ovo').fit(data.X, data.y)
    >>> clf.predict(data.X[:5]).shape
    (5,)
    """

    _weka_classname = "weka.classifiers.meta.MultiClassClassifier"
    _is_classification = True
    _default_base = "weka.classifiers.functions.Logistic"

    #: Weka's ``-M`` takes an integer code per decomposition method.
    _METHODS = {"ova": "0", "random_codes": "1", "exhaustive_codes": "2", "ovo": "3"}

    def __init__(
        self,
        method: str = "ova",
        random_width: float = 2.0,
        seed: int = 1,
        base_classifier: Optional[str] = None,
        base_options: Optional[Sequence[str]] = None,
        nominal_features: Optional[Sequence[int]] = None,
        options: Optional[Sequence[str]] = None,
    ):
        super().__init__(base_classifier, base_options, nominal_features, options)
        self.method = method
        self.random_width = random_width
        self.seed = seed

    def _meta_options(self) -> List[str]:
        """Return the decomposition options."""
        try:
            code = self._METHODS[self.method]
        except KeyError:
            raise ValueError(
                f"method must be one of {sorted(self._METHODS)}, got {self.method!r}"
            ) from None
        return ["-M", code, "-R", str(self.random_width), "-S", fmt_num(self.seed)]

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Any]:
        """Return JSON Schema for constructor parameters."""
        return {
            "method": {"type": "string", "default": "ova",
                       "enum": ["ova", "random_codes", "exhaustive_codes", "ovo"]},
            "random_width": {"type": "number", "default": 2.0, "minimum": 0.0},
            "seed": {"type": "integer", "default": 1},
        }

    @classmethod
    def get_capabilities(cls) -> List[str]:
        """Return the capability names this wrapper supports."""
        return ["numeric", "nominal", "missing_values", "binary_class", "multiclass"]


@weka_classifier(tags=["meta", "filter", "pipeline"])
class FilteredClassifier(_MetaBase, Classifier):
    """**FilteredClassifier** — filter then classify (hub key ``weka.FilteredClassifier``).

    Wraps ``weka.classifiers.meta.FilteredClassifier``. Runs an arbitrary Weka
    filter over the data and trains the base learner on the result, with the
    filter fitted on training data only.

    Parameters
    ----------
    filter_name : str, default="weka.filters.unsupervised.attribute.Standardize"
        Fully-qualified Weka filter class (Weka ``-F``).
    filter_options : sequence of str or None, default=None
        Options for the filter, appended to its class name.
    base_classifier : str or None, default=None
        Base learner; defaults to ``weka.classifiers.trees.J48``.
    base_options : sequence of str or None, default=None
        Options for the base learner.
    nominal_features : sequence of int or None, default=None
        0-based indices of columns to treat as categorical.
    options : sequence of str or None, default=None
        Extra raw Weka options.

    Notes
    -----
    TuiML's own :class:`~tuiml.workflow.Workflow` covers the same ground for
    native components; this wrapper exists so a **Weka filter** can be used in
    the same position.

    Examples
    --------
    >>> from tuiml.weka import FilteredClassifier
    >>> from tuiml.datasets import load_iris
    >>> data = load_iris()
    >>> clf = FilteredClassifier().fit(data.X, data.y)
    >>> clf.predict(data.X[:5]).shape
    (5,)
    """

    _weka_classname = "weka.classifiers.meta.FilteredClassifier"
    _is_classification = True
    _default_base = "weka.classifiers.trees.J48"

    def __init__(
        self,
        filter_name: str = "weka.filters.unsupervised.attribute.Standardize",
        filter_options: Optional[Sequence[str]] = None,
        base_classifier: Optional[str] = None,
        base_options: Optional[Sequence[str]] = None,
        nominal_features: Optional[Sequence[int]] = None,
        options: Optional[Sequence[str]] = None,
    ):
        super().__init__(base_classifier, base_options, nominal_features, options)
        self.filter_name = filter_name
        self.filter_options = filter_options

    def _meta_options(self) -> List[str]:
        """Return the filter specification."""
        spec = self.filter_name
        if self.filter_options:
            spec = f"{self.filter_name} {' '.join(self.filter_options)}"
        return ["-F", spec]

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Any]:
        """Return JSON Schema for constructor parameters."""
        return {
            "filter_name": {
                "type": "string",
                "default": "weka.filters.unsupervised.attribute.Standardize"},
            "base_classifier": {"type": "string",
                                "default": "weka.classifiers.trees.J48"},
        }

    @classmethod
    def get_capabilities(cls) -> List[str]:
        """Return the capability names this wrapper supports."""
        return ["numeric", "nominal", "missing_values", "binary_class", "multiclass"]


@weka_regressor(tags=["meta", "ensemble", "boosting", "regression"])
class AdditiveRegression(_MetaBase, Regressor):
    """**AdditiveRegression** — gradient boosting for regression (hub key ``weka.AdditiveRegression``).

    Wraps ``weka.classifiers.meta.AdditiveRegression``. Fits the base regressor
    to the residuals of the current ensemble, round after round.

    Parameters
    ----------
    num_iterations : int, default=10
        Number of boosting rounds (Weka ``-I``).
    shrinkage : float, default=1.0
        Shrinkage applied to each round's contribution (Weka ``-S``). Values
        below 1 trade more rounds for better generalization.
    base_classifier : str or None, default=None
        Base regressor; defaults to ``weka.classifiers.trees.DecisionStump``.
    base_options : sequence of str or None, default=None
        Options for the base learner.
    nominal_features : sequence of int or None, default=None
        0-based indices of columns to treat as categorical.
    options : sequence of str or None, default=None
        Extra raw Weka options.

    See Also
    --------
    :class:`~tuiml.algorithms.ensemble.GradientBoostingRegressor` : TuiML's native version.

    Examples
    --------
    >>> from tuiml.weka import AdditiveRegression
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(60, 3))
    >>> y = 2.0 * X[:, 0] + rng.normal(scale=0.1, size=60)
    >>> reg = AdditiveRegression(num_iterations=20).fit(X, y)
    >>> reg.predict(X[:5]).shape
    (5,)
    """

    _weka_classname = "weka.classifiers.meta.AdditiveRegression"
    _is_classification = False
    _default_base = "weka.classifiers.trees.DecisionStump"

    def __init__(
        self,
        num_iterations: int = 10,
        shrinkage: float = 1.0,
        base_classifier: Optional[str] = None,
        base_options: Optional[Sequence[str]] = None,
        nominal_features: Optional[Sequence[int]] = None,
        options: Optional[Sequence[str]] = None,
    ):
        super().__init__(base_classifier, base_options, nominal_features, options)
        self.num_iterations = num_iterations
        self.shrinkage = shrinkage

    def _meta_options(self) -> List[str]:
        """Return the boosting options."""
        return ["-I", fmt_num(self.num_iterations), "-S", str(self.shrinkage)]

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Any]:
        """Return JSON Schema for constructor parameters."""
        return {
            "num_iterations": {"type": "integer", "default": 10, "minimum": 1},
            "shrinkage": {"type": "number", "default": 1.0, "minimum": 0.0},
        }

    @classmethod
    def get_capabilities(cls) -> List[str]:
        """Return the capability names this wrapper supports."""
        return ["numeric", "nominal", "missing_values", "regression"]


@weka_regressor(tags=["meta", "regression", "discretization"])
class RegressionByDiscretization(_MetaBase, Regressor):
    """**RegressionByDiscretization** — regression via a classifier (hub key ``weka.RegressionByDiscretization``).

    Wraps ``weka.classifiers.meta.RegressionByDiscretization``. Bins the numeric
    target, trains a **classifier** on the bins, and predicts the weighted mean
    of the bin centres — which lets any classifier tackle a regression problem.

    Parameters
    ----------
    num_bins : int, default=10
        Number of bins the target is split into (Weka ``-B``).
    delete_empty_bins : bool, default=False
        Drop bins that end up empty (Weka ``-E``).
    base_classifier : str or None, default=None
        Base **classifier**; defaults to ``weka.classifiers.trees.J48``.
    base_options : sequence of str or None, default=None
        Options for the base learner.
    nominal_features : sequence of int or None, default=None
        0-based indices of columns to treat as categorical.
    options : sequence of str or None, default=None
        Extra raw Weka options.

    Examples
    --------
    >>> from tuiml.weka import RegressionByDiscretization
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(60, 3))
    >>> y = 2.0 * X[:, 0] + rng.normal(scale=0.1, size=60)
    >>> reg = RegressionByDiscretization(num_bins=8).fit(X, y)
    >>> reg.predict(X[:5]).shape
    (5,)
    """

    _weka_classname = "weka.classifiers.meta.RegressionByDiscretization"
    _is_classification = False
    _default_base = "weka.classifiers.trees.J48"

    def __init__(
        self,
        num_bins: int = 10,
        delete_empty_bins: bool = False,
        base_classifier: Optional[str] = None,
        base_options: Optional[Sequence[str]] = None,
        nominal_features: Optional[Sequence[int]] = None,
        options: Optional[Sequence[str]] = None,
    ):
        super().__init__(base_classifier, base_options, nominal_features, options)
        self.num_bins = num_bins
        self.delete_empty_bins = delete_empty_bins

    def _meta_options(self) -> List[str]:
        """Return the discretization options."""
        opts = ["-B", fmt_num(self.num_bins)]
        if self.delete_empty_bins:
            opts.append("-E")
        return opts

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Any]:
        """Return JSON Schema for constructor parameters."""
        return {
            "num_bins": {"type": "integer", "default": 10, "minimum": 2},
            "delete_empty_bins": {"type": "boolean", "default": False},
        }

    @classmethod
    def get_capabilities(cls) -> List[str]:
        """Return the capability names this wrapper supports."""
        return ["numeric", "nominal", "missing_values", "regression"]


@weka_classifier(tags=["meta", "ensemble", "voting"])
class Vote(_WekaSupervisedMixin, Classifier):
    """**Vote** — combine several classifiers by voting (hub key ``weka.Vote``).

    Wraps ``weka.classifiers.meta.Vote``. Unlike the other metas this takes a
    **list** of base learners, each supplied with its own ``-B`` option.

    Parameters
    ----------
    classifiers : sequence of str or None, default=None
        Fully-qualified Weka classes to combine. Defaults to J48, NaiveBayes
        and IBk. Each may carry its own options in the same string, e.g.
        ``"weka.classifiers.trees.J48 -C 0.1"``.
    combination_rule : {'avg', 'product', 'majority', 'min', 'max', 'median'}, default='avg'
        How member predictions are combined (Weka ``-R``).
    seed : int, default=1
        Random seed (Weka ``-S``).
    nominal_features : sequence of int or None, default=None
        0-based indices of columns to treat as categorical.
    options : sequence of str or None, default=None
        Extra raw Weka options.

    See Also
    --------
    :class:`~tuiml.algorithms.ensemble.VotingClassifier` : TuiML's native version.

    Examples
    --------
    >>> from tuiml.weka import Vote
    >>> from tuiml.datasets import load_iris
    >>> data = load_iris()
    >>> clf = Vote(combination_rule='majority').fit(data.X, data.y)
    >>> clf.predict(data.X[:5]).shape
    (5,)
    """

    _weka_classname = "weka.classifiers.meta.Vote"
    _is_classification = True

    #: Weka's ``-R`` takes a lowercase keyword per combination rule.
    _RULES = {"avg": "AVG", "product": "PROD", "majority": "MAJ",
              "min": "MIN", "max": "MAX", "median": "MED"}

    _DEFAULT_MEMBERS = (
        "weka.classifiers.trees.J48",
        "weka.classifiers.bayes.NaiveBayes",
        "weka.classifiers.lazy.IBk",
    )

    def __init__(
        self,
        classifiers: Optional[Sequence[str]] = None,
        combination_rule: str = "avg",
        seed: int = 1,
        nominal_features: Optional[Sequence[int]] = None,
        options: Optional[Sequence[str]] = None,
    ):
        super().__init__()
        self.classifiers = list(classifiers) if classifiers else list(self._DEFAULT_MEMBERS)
        self.combination_rule = combination_rule
        self.seed = seed
        self.nominal_features = nominal_features
        self.options = options

    def _options(self) -> List[str]:
        """Return one ``-B`` per member plus the combination rule."""
        try:
            rule = self._RULES[self.combination_rule]
        except KeyError:
            raise ValueError(
                f"combination_rule must be one of {sorted(self._RULES)}, "
                f"got {self.combination_rule!r}"
            ) from None
        opts: List[str] = ["-S", fmt_num(self.seed), "-R", rule]
        for member in self.classifiers:
            opts += ["-B", member]
        return opts

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Any]:
        """Return JSON Schema for constructor parameters."""
        return {
            "classifiers": {"type": "array", "items": {"type": "string"},
                            "default": list(cls._DEFAULT_MEMBERS)},
            "combination_rule": {"type": "string", "default": "avg",
                                 "enum": ["avg", "product", "majority",
                                          "min", "max", "median"]},
            "seed": {"type": "integer", "default": 1},
        }

    @classmethod
    def get_capabilities(cls) -> List[str]:
        """Return the capability names this wrapper supports."""
        return ["numeric", "nominal", "missing_values", "binary_class", "multiclass"]


@weka_classifier(tags=["meta", "ensemble", "stacking"])
class Stacking(_WekaSupervisedMixin, Classifier):
    """**Stacking** — combine classifiers with a meta-learner (hub key ``weka.Stacking``).

    Wraps ``weka.classifiers.meta.Stacking``. Trains several base classifiers,
    then trains a meta-classifier on their cross-validated predictions.

    Parameters
    ----------
    classifiers : sequence of str or None, default=None
        Fully-qualified Weka classes used as level-0 learners (Weka ``-B``).
        Defaults to J48, NaiveBayes and IBk.
    meta_classifier : str, default="weka.classifiers.functions.Logistic"
        Level-1 learner combining the base predictions (Weka ``-M``).
    num_folds : int, default=10
        Folds used to generate the level-1 training set (Weka ``-X``).
    seed : int, default=1
        Random seed (Weka ``-S``).
    nominal_features : sequence of int or None, default=None
        0-based indices of columns to treat as categorical.
    options : sequence of str or None, default=None
        Extra raw Weka options.

    Notes
    -----
    ``num_folds`` matters more than it looks: the meta-learner only ever sees
    the level-1 table built from these folds, so too few folds starve it. On
    iris, dropping from the default 10 to 3 takes accuracy from 0.98 to 0.67.
    Lower it only when training cost forces you to.

    See Also
    --------
    :class:`~tuiml.algorithms.ensemble.StackingClassifier` : TuiML's native version.

    Examples
    --------
    >>> from tuiml.weka import Stacking
    >>> from tuiml.datasets import load_iris
    >>> data = load_iris()
    >>> clf = Stacking().fit(data.X, data.y)
    >>> clf.predict(data.X[:5]).shape
    (5,)
    """

    _weka_classname = "weka.classifiers.meta.Stacking"
    _is_classification = True

    _DEFAULT_MEMBERS = (
        "weka.classifiers.trees.J48",
        "weka.classifiers.bayes.NaiveBayes",
        "weka.classifiers.lazy.IBk",
    )

    def __init__(
        self,
        classifiers: Optional[Sequence[str]] = None,
        meta_classifier: str = "weka.classifiers.functions.Logistic",
        num_folds: int = 10,
        seed: int = 1,
        nominal_features: Optional[Sequence[int]] = None,
        options: Optional[Sequence[str]] = None,
    ):
        super().__init__()
        self.classifiers = list(classifiers) if classifiers else list(self._DEFAULT_MEMBERS)
        self.meta_classifier = meta_classifier
        self.num_folds = num_folds
        self.seed = seed
        self.nominal_features = nominal_features
        self.options = options

    def _options(self) -> List[str]:
        """Return the level-0 members, the meta-learner and the fold count."""
        opts: List[str] = ["-X", fmt_num(self.num_folds), "-S", fmt_num(self.seed)]
        for member in self.classifiers:
            opts += ["-B", member]
        opts += ["-M", self.meta_classifier]
        return opts

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Any]:
        """Return JSON Schema for constructor parameters."""
        return {
            "classifiers": {"type": "array", "items": {"type": "string"},
                            "default": list(cls._DEFAULT_MEMBERS)},
            "meta_classifier": {"type": "string",
                                "default": "weka.classifiers.functions.Logistic"},
            "num_folds": {"type": "integer", "default": 10, "minimum": 2},
            "seed": {"type": "integer", "default": 1},
        }

    @classmethod
    def get_capabilities(cls) -> List[str]:
        """Return the capability names this wrapper supports."""
        return ["numeric", "nominal", "missing_values", "binary_class", "multiclass"]
