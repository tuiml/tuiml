"""Weka tree wrappers.

Decision and model trees from ``weka.classifiers.trees``, registered under
``weka.<ClassName>`` hub keys so they sit alongside the native TuiML ``trees``
family without colliding with it.

Notes
-----
Requires the optional Weka extra: ``pip install 'tuiml[weka]'`` plus a Java
runtime (11+) on ``PATH``.
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
    "J48",
    "REPTree",
    "RandomTree",
    "RandomForest",
    "DecisionStump",
    "LMT",
    "M5P",
]


@weka_classifier(tags=["tree", "c45"])
class J48(_WekaSupervisedMixin, Classifier):
    """**J48** — Weka's C4.5 decision tree (hub key ``weka.J48``).

    Wraps ``weka.classifiers.trees.J48``, Weka's implementation of Quinlan's
    C4.5. Splits on the attribute with the highest gain ratio and prunes with
    subtree raising by default.

    Parameters
    ----------
    confidence_factor : float, default=0.25
        Confidence threshold for pruning (Weka ``-C``). Lower values prune more
        aggressively.
    min_num_obj : int, default=2
        Minimum number of instances per leaf (Weka ``-M``).
    unpruned : bool, default=False
        Build an unpruned tree (Weka ``-U``).
    binary_splits : bool, default=False
        Use binary splits on nominal attributes (Weka ``-B``).
    nominal_features : sequence of int or None, default=None
        0-based indices of columns to treat as categorical rather than numeric.
    options : sequence of str or None, default=None
        Extra raw Weka options appended to the generated ones, for anything not
        exposed as a named parameter.

    Attributes
    ----------
    model_ : weka.classifiers.Classifier
        The fitted backing Weka classifier.
    classes_ : np.ndarray
        Class labels in the caller's original label space.

    See Also
    --------
    :class:`~tuiml.algorithms.trees.DecisionTreeClassifier` : TuiML's native decision tree (CART-style).
    :class:`~tuiml.weka.trees.REPTree` : Reduced-error pruning tree.

    Examples
    --------
    >>> from tuiml.weka import J48
    >>> from tuiml.datasets import load_iris
    >>> data = load_iris()
    >>> clf = J48(confidence_factor=0.25).fit(data.X, data.y)
    >>> clf.predict(data.X[:5]).shape
    (5,)
    """

    _weka_classname = "weka.classifiers.trees.J48"
    _is_classification = True

    def __init__(
        self,
        confidence_factor: float = 0.25,
        min_num_obj: int = 2,
        unpruned: bool = False,
        binary_splits: bool = False,
        nominal_features: Optional[Sequence[int]] = None,
        options: Optional[Sequence[str]] = None,
    ):
        super().__init__()
        self.confidence_factor = confidence_factor
        self.min_num_obj = min_num_obj
        self.unpruned = unpruned
        self.binary_splits = binary_splits
        self.nominal_features = nominal_features
        self.options = options

    def _options(self) -> List[str]:
        """Return the Weka option tokens for this configuration."""
        opts: List[str] = []
        if self.unpruned:
            opts.append("-U")
        else:
            opts += ["-C", fmt_num(self.confidence_factor)]
        opts += ["-M", fmt_num(self.min_num_obj)]
        if self.binary_splits:
            opts.append("-B")
        return opts

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Any]:
        """Return JSON Schema for constructor parameters."""
        return {
            "confidence_factor": {"type": "number", "default": 0.25,
                                  "minimum": 0.0, "maximum": 1.0},
            "min_num_obj": {"type": "integer", "default": 2, "minimum": 1},
            "unpruned": {"type": "boolean", "default": False},
            "binary_splits": {"type": "boolean", "default": False},
        }

    @classmethod
    def get_capabilities(cls) -> List[str]:
        """Return the capability names this wrapper supports."""
        return ["numeric", "nominal", "missing_values", "binary_class", "multiclass"]


@weka_classifier(tags=["tree", "pruning"])
class REPTree(_WekaSupervisedMixin, Classifier):
    """**REPTree** — fast tree with reduced-error pruning (hub key ``weka.REPTree``).

    Wraps ``weka.classifiers.trees.REPTree``. Builds a decision/regression tree
    using information gain (variance for numeric targets) and prunes it with
    reduced-error pruning and backfitting.

    Parameters
    ----------
    max_depth : int, default=-1
        Maximum tree depth (Weka ``-L``); ``-1`` means unlimited.
    min_num : float, default=2.0
        Minimum total instance weight in a leaf (Weka ``-M``).
    num_folds : int, default=3
        Folds used for pruning; one is held out (Weka ``-N``).
    no_pruning : bool, default=False
        Disable pruning entirely (Weka ``-P``).
    seed : int, default=1
        Random seed (Weka ``-S``).
    nominal_features : sequence of int or None, default=None
        0-based indices of columns to treat as categorical.
    options : sequence of str or None, default=None
        Extra raw Weka options.

    Attributes
    ----------
    model_ : weka.classifiers.Classifier
        The fitted backing Weka classifier.

    Examples
    --------
    >>> from tuiml.weka import REPTree
    >>> from tuiml.datasets import load_iris
    >>> data = load_iris()
    >>> clf = REPTree(max_depth=5).fit(data.X, data.y)
    >>> clf.predict(data.X[:5]).shape
    (5,)
    """

    _weka_classname = "weka.classifiers.trees.REPTree"
    _is_classification = True

    def __init__(
        self,
        max_depth: int = -1,
        min_num: float = 2.0,
        num_folds: int = 3,
        no_pruning: bool = False,
        seed: int = 1,
        nominal_features: Optional[Sequence[int]] = None,
        options: Optional[Sequence[str]] = None,
    ):
        super().__init__()
        self.max_depth = max_depth
        self.min_num = min_num
        self.num_folds = num_folds
        self.no_pruning = no_pruning
        self.seed = seed
        self.nominal_features = nominal_features
        self.options = options

    def _options(self) -> List[str]:
        """Return the Weka option tokens for this configuration."""
        opts = ["-M", fmt_num(self.min_num), "-N", fmt_num(self.num_folds),
                "-S", fmt_num(self.seed), "-L", fmt_num(self.max_depth)]
        if self.no_pruning:
            opts.append("-P")
        return opts

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Any]:
        """Return JSON Schema for constructor parameters."""
        return {
            "max_depth": {"type": "integer", "default": -1},
            "min_num": {"type": "number", "default": 2.0, "minimum": 0.0},
            "num_folds": {"type": "integer", "default": 3, "minimum": 2},
            "no_pruning": {"type": "boolean", "default": False},
            "seed": {"type": "integer", "default": 1},
        }

    @classmethod
    def get_capabilities(cls) -> List[str]:
        """Return the capability names this wrapper supports."""
        return ["numeric", "nominal", "missing_values", "binary_class", "multiclass"]


@weka_classifier(tags=["tree", "random"])
class RandomTree(_WekaSupervisedMixin, Classifier):
    """**RandomTree** — tree over a random attribute subset (hub key ``weka.RandomTree``).

    Wraps ``weka.classifiers.trees.RandomTree``. Considers ``k`` randomly chosen
    attributes at each node and performs no pruning. Mainly used as the base
    learner inside :class:`~tuiml.weka.trees.RandomForest`.

    Parameters
    ----------
    k_value : int, default=0
        Attributes sampled per split (Weka ``-K``); ``0`` means
        :math:`\\lfloor \\log_2(p) \\rfloor + 1`.
    max_depth : int, default=0
        Maximum depth (Weka ``-depth``); ``0`` means unlimited.
    min_num : float, default=1.0
        Minimum total instance weight in a leaf (Weka ``-M``).
    seed : int, default=1
        Random seed (Weka ``-S``).
    nominal_features : sequence of int or None, default=None
        0-based indices of columns to treat as categorical.
    options : sequence of str or None, default=None
        Extra raw Weka options.

    Examples
    --------
    >>> from tuiml.weka import RandomTree
    >>> from tuiml.datasets import load_iris
    >>> data = load_iris()
    >>> clf = RandomTree(seed=42).fit(data.X, data.y)
    >>> clf.predict(data.X[:5]).shape
    (5,)
    """

    _weka_classname = "weka.classifiers.trees.RandomTree"
    _is_classification = True

    def __init__(
        self,
        k_value: int = 0,
        max_depth: int = 0,
        min_num: float = 1.0,
        seed: int = 1,
        nominal_features: Optional[Sequence[int]] = None,
        options: Optional[Sequence[str]] = None,
    ):
        super().__init__()
        self.k_value = k_value
        self.max_depth = max_depth
        self.min_num = min_num
        self.seed = seed
        self.nominal_features = nominal_features
        self.options = options

    def _options(self) -> List[str]:
        """Return the Weka option tokens for this configuration."""
        return ["-K", fmt_num(self.k_value), "-M", fmt_num(self.min_num),
                "-S", fmt_num(self.seed), "-depth", fmt_num(self.max_depth)]

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Any]:
        """Return JSON Schema for constructor parameters."""
        return {
            "k_value": {"type": "integer", "default": 0, "minimum": 0},
            "max_depth": {"type": "integer", "default": 0, "minimum": 0},
            "min_num": {"type": "number", "default": 1.0, "minimum": 0.0},
            "seed": {"type": "integer", "default": 1},
        }

    @classmethod
    def get_capabilities(cls) -> List[str]:
        """Return the capability names this wrapper supports."""
        return ["numeric", "nominal", "missing_values", "binary_class", "multiclass"]


@weka_classifier(tags=["tree", "ensemble", "forest"])
class RandomForest(_WekaSupervisedMixin, Classifier):
    """**RandomForest** — Weka's random forest (hub key ``weka.RandomForest``).

    Wraps ``weka.classifiers.trees.RandomForest``, a bagged ensemble of
    :class:`~tuiml.weka.trees.RandomTree`.

    Parameters
    ----------
    num_iterations : int, default=100
        Number of trees (Weka ``-I``).
    num_features : int, default=0
        Attributes sampled per split (Weka ``-K``); ``0`` means
        :math:`\\lfloor \\log_2(p) \\rfloor + 1`, which is Weka's default and
        differs from scikit-learn's :math:`\\sqrt{p}`.
    max_depth : int, default=0
        Maximum depth (Weka ``-depth``); ``0`` means unlimited.
    seed : int, default=1
        Random seed (Weka ``-S``).
    nominal_features : sequence of int or None, default=None
        0-based indices of columns to treat as categorical.
    options : sequence of str or None, default=None
        Extra raw Weka options.

    See Also
    --------
    :class:`~tuiml.algorithms.trees.RandomForestClassifier` : TuiML's native forest.

    Examples
    --------
    >>> from tuiml.weka import RandomForest
    >>> from tuiml.datasets import load_iris
    >>> data = load_iris()
    >>> clf = RandomForest(num_iterations=20, seed=42).fit(data.X, data.y)
    >>> clf.predict(data.X[:5]).shape
    (5,)
    """

    _weka_classname = "weka.classifiers.trees.RandomForest"
    _is_classification = True

    def __init__(
        self,
        num_iterations: int = 100,
        num_features: int = 0,
        max_depth: int = 0,
        seed: int = 1,
        nominal_features: Optional[Sequence[int]] = None,
        options: Optional[Sequence[str]] = None,
    ):
        super().__init__()
        self.num_iterations = num_iterations
        self.num_features = num_features
        self.max_depth = max_depth
        self.seed = seed
        self.nominal_features = nominal_features
        self.options = options

    def _options(self) -> List[str]:
        """Return the Weka option tokens for this configuration."""
        return ["-I", fmt_num(self.num_iterations), "-K", fmt_num(self.num_features),
                "-S", fmt_num(self.seed), "-depth", fmt_num(self.max_depth)]

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Any]:
        """Return JSON Schema for constructor parameters."""
        return {
            "num_iterations": {"type": "integer", "default": 100, "minimum": 1},
            "num_features": {"type": "integer", "default": 0, "minimum": 0},
            "max_depth": {"type": "integer", "default": 0, "minimum": 0},
            "seed": {"type": "integer", "default": 1},
        }

    @classmethod
    def get_capabilities(cls) -> List[str]:
        """Return the capability names this wrapper supports."""
        return ["numeric", "nominal", "missing_values", "binary_class", "multiclass"]


@weka_classifier(tags=["tree", "stump", "baseline"])
class DecisionStump(_WekaSupervisedMixin, Classifier):
    """**DecisionStump** — one-level decision tree (hub key ``weka.DecisionStump``).

    Wraps ``weka.classifiers.trees.DecisionStump``. Splits on a single
    attribute; normally used as a weak learner inside a boosting ensemble.

    Parameters
    ----------
    nominal_features : sequence of int or None, default=None
        0-based indices of columns to treat as categorical.
    options : sequence of str or None, default=None
        Extra raw Weka options.

    Examples
    --------
    >>> from tuiml.weka import DecisionStump
    >>> from tuiml.datasets import load_iris
    >>> data = load_iris()
    >>> clf = DecisionStump().fit(data.X, data.y)
    >>> clf.predict(data.X[:5]).shape
    (5,)
    """

    _weka_classname = "weka.classifiers.trees.DecisionStump"
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
        return ["numeric", "nominal", "missing_values", "binary_class", "multiclass"]


@weka_classifier(tags=["tree", "logistic", "model-tree"])
class LMT(_WekaSupervisedMixin, Classifier):
    """**LMT** — Logistic Model Tree (hub key ``weka.LMT``).

    Wraps ``weka.classifiers.trees.LMT``. A decision tree carrying a logistic
    regression model at each leaf, fitted with LogitBoost.

    Parameters
    ----------
    min_num_instances : int, default=15
        Minimum instances at which a node is considered for splitting
        (Weka ``-M``).
    fast_regression : bool, default=True
        Use the heuristic that speeds up LogitBoost fitting. Passing False
        emits Weka's ``-R`` flag to disable it.
    nominal_features : sequence of int or None, default=None
        0-based indices of columns to treat as categorical.
    options : sequence of str or None, default=None
        Extra raw Weka options.

    Notes
    -----
    LMT is considerably slower to train than :class:`~tuiml.weka.trees.J48`,
    because every candidate node fits a boosted logistic model.

    Examples
    --------
    >>> from tuiml.weka import LMT
    >>> from tuiml.datasets import load_iris
    >>> data = load_iris()
    >>> clf = LMT().fit(data.X, data.y)
    >>> clf.predict(data.X[:5]).shape
    (5,)
    """

    _weka_classname = "weka.classifiers.trees.LMT"
    _is_classification = True

    def __init__(
        self,
        min_num_instances: int = 15,
        fast_regression: bool = True,
        nominal_features: Optional[Sequence[int]] = None,
        options: Optional[Sequence[str]] = None,
    ):
        super().__init__()
        self.min_num_instances = min_num_instances
        self.fast_regression = fast_regression
        self.nominal_features = nominal_features
        self.options = options

    def _options(self) -> List[str]:
        """Return the Weka option tokens for this configuration."""
        opts = ["-M", fmt_num(self.min_num_instances)]
        if not self.fast_regression:
            opts.append("-R")
        return opts

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Any]:
        """Return JSON Schema for constructor parameters."""
        return {
            "min_num_instances": {"type": "integer", "default": 15, "minimum": 1},
            "fast_regression": {"type": "boolean", "default": True},
        }

    @classmethod
    def get_capabilities(cls) -> List[str]:
        """Return the capability names this wrapper supports."""
        return ["numeric", "nominal", "missing_values", "binary_class", "multiclass"]


@weka_regressor(tags=["tree", "model-tree", "regression"])
class M5P(_WekaSupervisedMixin, Regressor):
    """**M5P** — M5 model tree for regression (hub key ``weka.M5P``).

    Wraps ``weka.classifiers.trees.M5P``. Builds a regression tree whose leaves
    hold linear regression models rather than constants.

    Parameters
    ----------
    min_num_instances : float, default=4.0
        Minimum instances allowed at a leaf (Weka ``-M``).
    unpruned : bool, default=False
        Build an unpruned tree (Weka ``-N``).
    use_unsmoothed : bool, default=False
        Skip the smoothing step applied to leaf predictions (Weka ``-U``).
    build_regression_tree : bool, default=False
        Produce a plain regression tree — constants at the leaves instead of
        linear models (Weka ``-R``).
    nominal_features : sequence of int or None, default=None
        0-based indices of columns to treat as categorical.
    options : sequence of str or None, default=None
        Extra raw Weka options.

    See Also
    --------
    :class:`~tuiml.algorithms.trees.DecisionTreeRegressor` : TuiML's native regression tree (constant leaves, not linear models).

    Examples
    --------
    >>> from tuiml.weka import M5P
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(80, 3))
    >>> y = 2.0 * X[:, 0] + rng.normal(scale=0.1, size=80)
    >>> reg = M5P().fit(X, y)
    >>> reg.predict(X[:5]).shape
    (5,)
    """

    _weka_classname = "weka.classifiers.trees.M5P"
    _is_classification = False

    def __init__(
        self,
        min_num_instances: float = 4.0,
        unpruned: bool = False,
        use_unsmoothed: bool = False,
        build_regression_tree: bool = False,
        nominal_features: Optional[Sequence[int]] = None,
        options: Optional[Sequence[str]] = None,
    ):
        super().__init__()
        self.min_num_instances = min_num_instances
        self.unpruned = unpruned
        self.use_unsmoothed = use_unsmoothed
        self.build_regression_tree = build_regression_tree
        self.nominal_features = nominal_features
        self.options = options

    def _options(self) -> List[str]:
        """Return the Weka option tokens for this configuration."""
        opts = ["-M", fmt_num(self.min_num_instances)]
        if self.unpruned:
            opts.append("-N")
        if self.use_unsmoothed:
            opts.append("-U")
        if self.build_regression_tree:
            opts.append("-R")
        return opts

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Any]:
        """Return JSON Schema for constructor parameters."""
        return {
            "min_num_instances": {"type": "number", "default": 4.0, "minimum": 1.0},
            "unpruned": {"type": "boolean", "default": False},
            "use_unsmoothed": {"type": "boolean", "default": False},
            "build_regression_tree": {"type": "boolean", "default": False},
        }

    @classmethod
    def get_capabilities(cls) -> List[str]:
        """Return the capability names this wrapper supports."""
        return ["numeric", "nominal", "missing_values", "regression"]
