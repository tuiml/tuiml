"""Weka rule-learner wrappers.

Rule and decision-table learners from ``weka.classifiers.rules``, registered
under ``weka.<ClassName>`` hub keys. These are the models to reach for when the
result has to be readable by a person: every one of them can be printed with
``to_weka_string()``.

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

__all__ = ["ZeroR", "OneR", "JRip", "PART", "DecisionTable", "M5Rules"]


@weka_classifier(tags=["rules", "baseline"])
class ZeroR(_WekaSupervisedMixin, Classifier):
    """**ZeroR** — majority-class baseline (hub key ``weka.ZeroR``).

    Wraps ``weka.classifiers.rules.ZeroR``. Predicts the majority class (or the
    mean, for a numeric target) and ignores every attribute. Its only real use
    is as the floor any serious model must beat.

    Parameters
    ----------
    nominal_features : sequence of int or None, default=None
        0-based indices of columns to treat as categorical.
    options : sequence of str or None, default=None
        Extra raw Weka options.

    See Also
    --------
    :class:`~tuiml.algorithms.rules.ZeroRuleClassifier` : TuiML's native ZeroR.

    Examples
    --------
    >>> from tuiml.weka import ZeroR
    >>> from tuiml.datasets import load_iris
    >>> data = load_iris()
    >>> clf = ZeroR().fit(data.X, data.y)
    >>> clf.predict(data.X[:5]).shape
    (5,)
    """

    _weka_classname = "weka.classifiers.rules.ZeroR"
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


@weka_classifier(tags=["rules", "baseline", "interpretable"])
class OneR(_WekaSupervisedMixin, Classifier):
    """**OneR** — one-attribute rule classifier (hub key ``weka.OneR``).

    Wraps ``weka.classifiers.rules.OneR``. Builds a single rule on the one
    attribute that yields the lowest training error — Holte's demonstration
    that very simple rules often perform surprisingly well.

    Parameters
    ----------
    min_bucket_size : int, default=6
        Minimum bucket size used when discretizing numeric attributes
        (Weka ``-B``). Larger values give simpler, more general rules.
    nominal_features : sequence of int or None, default=None
        0-based indices of columns to treat as categorical.
    options : sequence of str or None, default=None
        Extra raw Weka options.

    See Also
    --------
    :class:`~tuiml.algorithms.trees.DecisionStumpClassifier` : TuiML's native single-attribute rule learner.

    Examples
    --------
    >>> from tuiml.weka import OneR
    >>> from tuiml.datasets import load_iris
    >>> data = load_iris()
    >>> clf = OneR(min_bucket_size=6).fit(data.X, data.y)
    >>> clf.predict(data.X[:5]).shape
    (5,)
    """

    _weka_classname = "weka.classifiers.rules.OneR"
    _is_classification = True

    def __init__(
        self,
        min_bucket_size: int = 6,
        nominal_features: Optional[Sequence[int]] = None,
        options: Optional[Sequence[str]] = None,
    ):
        super().__init__()
        self.min_bucket_size = min_bucket_size
        self.nominal_features = nominal_features
        self.options = options

    def _options(self) -> List[str]:
        """Return the Weka option tokens for this configuration."""
        return ["-B", fmt_num(self.min_bucket_size)]

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Any]:
        """Return JSON Schema for constructor parameters."""
        return {"min_bucket_size": {"type": "integer", "default": 6, "minimum": 1}}

    @classmethod
    def get_capabilities(cls) -> List[str]:
        """Return the capability names this wrapper supports."""
        return ["numeric", "nominal", "missing_values", "binary_class", "multiclass"]


@weka_classifier(tags=["rules", "ripper", "interpretable"])
class JRip(_WekaSupervisedMixin, Classifier):
    """**JRip** — Weka's RIPPER rule learner (hub key ``weka.JRip``).

    Wraps ``weka.classifiers.rules.JRip``, Weka's implementation of Cohen's
    RIPPER: grow a rule set, prune it, then optimize it over several passes.

    Parameters
    ----------
    folds : int, default=3
        Folds used for reduced-error pruning; one is held out (Weka ``-F``).
    min_no : float, default=2.0
        Minimum total instance weight covered by a rule (Weka ``-N``).
    optimizations : int, default=2
        Number of optimization passes over the rule set (Weka ``-O``).
    seed : int, default=1
        Random seed (Weka ``-S``).
    prune : bool, default=True
        Run pruning. Passing False emits Weka's ``-P`` flag to disable it.
    nominal_features : sequence of int or None, default=None
        0-based indices of columns to treat as categorical.
    options : sequence of str or None, default=None
        Extra raw Weka options.

    See Also
    --------
    :class:`~tuiml.algorithms.trees.DecisionTreeClassifier` : TuiML's nearest native model — axis-aligned rules as a tree.

    Examples
    --------
    >>> from tuiml.weka import JRip
    >>> from tuiml.datasets import load_iris
    >>> data = load_iris()
    >>> clf = JRip().fit(data.X, data.y)
    >>> clf.predict(data.X[:5]).shape
    (5,)
    """

    _weka_classname = "weka.classifiers.rules.JRip"
    _is_classification = True

    def __init__(
        self,
        folds: int = 3,
        min_no: float = 2.0,
        optimizations: int = 2,
        seed: int = 1,
        prune: bool = True,
        nominal_features: Optional[Sequence[int]] = None,
        options: Optional[Sequence[str]] = None,
    ):
        super().__init__()
        self.folds = folds
        self.min_no = min_no
        self.optimizations = optimizations
        self.seed = seed
        self.prune = prune
        self.nominal_features = nominal_features
        self.options = options

    def _options(self) -> List[str]:
        """Return the Weka option tokens for this configuration."""
        opts = ["-F", fmt_num(self.folds), "-N", fmt_num(self.min_no),
                "-O", fmt_num(self.optimizations), "-S", fmt_num(self.seed)]
        if not self.prune:
            opts.append("-P")
        return opts

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Any]:
        """Return JSON Schema for constructor parameters."""
        return {
            "folds": {"type": "integer", "default": 3, "minimum": 2},
            "min_no": {"type": "number", "default": 2.0, "minimum": 0.0},
            "optimizations": {"type": "integer", "default": 2, "minimum": 0},
            "seed": {"type": "integer", "default": 1},
            "prune": {"type": "boolean", "default": True},
        }

    @classmethod
    def get_capabilities(cls) -> List[str]:
        """Return the capability names this wrapper supports."""
        return ["numeric", "nominal", "missing_values", "binary_class", "multiclass"]


@weka_classifier(tags=["rules", "interpretable"])
class PART(_WekaSupervisedMixin, Classifier):
    """**PART** — partial decision tree rule learner (hub key ``weka.PART``).

    Wraps ``weka.classifiers.rules.PART``. Repeatedly builds a partial C4.5 tree
    and turns its best leaf into a rule, which avoids RIPPER's global
    optimization step while still producing a compact rule list.

    Parameters
    ----------
    confidence_factor : float, default=0.25
        Confidence threshold for pruning (Weka ``-C``).
    min_num_obj : int, default=2
        Minimum instances per rule (Weka ``-M``).
    unpruned : bool, default=False
        Build unpruned rules (Weka ``-U``).
    binary_splits : bool, default=False
        Use binary splits on nominal attributes (Weka ``-B``).
    nominal_features : sequence of int or None, default=None
        0-based indices of columns to treat as categorical.
    options : sequence of str or None, default=None
        Extra raw Weka options.

    See Also
    --------
    :class:`~tuiml.algorithms.trees.DecisionTreeClassifier` : TuiML's nearest native model — PART's rules come from partial trees.

    Examples
    --------
    >>> from tuiml.weka import PART
    >>> from tuiml.datasets import load_iris
    >>> data = load_iris()
    >>> clf = PART().fit(data.X, data.y)
    >>> clf.predict(data.X[:5]).shape
    (5,)
    """

    _weka_classname = "weka.classifiers.rules.PART"
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


@weka_classifier(tags=["rules", "table", "interpretable"])
class DecisionTable(_WekaSupervisedMixin, Classifier):
    """**DecisionTable** — simple decision table majority classifier (hub key ``weka.DecisionTable``).

    Wraps ``weka.classifiers.rules.DecisionTable``. Selects a feature subset by
    search, then predicts using a lookup table of majority classes over that
    subset, falling back to the global majority for unseen combinations.

    Parameters
    ----------
    cross_val : int, default=1
        Cross-validation setting used to evaluate subsets (Weka ``-X``);
        ``1`` means leave-one-out.
    use_ibk : bool, default=False
        Fall back to a nearest-neighbour prediction rather than the global
        majority for combinations missing from the table (Weka ``-I``).
    nominal_features : sequence of int or None, default=None
        0-based indices of columns to treat as categorical.
    options : sequence of str or None, default=None
        Extra raw Weka options.

    Examples
    --------
    >>> from tuiml.weka import DecisionTable
    >>> from tuiml.datasets import load_iris
    >>> data = load_iris()
    >>> clf = DecisionTable().fit(data.X, data.y)
    >>> clf.predict(data.X[:5]).shape
    (5,)
    """

    _weka_classname = "weka.classifiers.rules.DecisionTable"
    _is_classification = True

    def __init__(
        self,
        cross_val: int = 1,
        use_ibk: bool = False,
        nominal_features: Optional[Sequence[int]] = None,
        options: Optional[Sequence[str]] = None,
    ):
        super().__init__()
        self.cross_val = cross_val
        self.use_ibk = use_ibk
        self.nominal_features = nominal_features
        self.options = options

    def _options(self) -> List[str]:
        """Return the Weka option tokens for this configuration."""
        opts = ["-X", fmt_num(self.cross_val)]
        if self.use_ibk:
            opts.append("-I")
        return opts

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Any]:
        """Return JSON Schema for constructor parameters."""
        return {
            "cross_val": {"type": "integer", "default": 1, "minimum": 1},
            "use_ibk": {"type": "boolean", "default": False},
        }

    @classmethod
    def get_capabilities(cls) -> List[str]:
        """Return the capability names this wrapper supports."""
        return ["numeric", "nominal", "missing_values", "binary_class", "multiclass"]


@weka_regressor(tags=["rules", "regression", "model-tree"])
class M5Rules(_WekaSupervisedMixin, Regressor):
    """**M5Rules** — regression rules from M5 model trees (hub key ``weka.M5Rules``).

    Wraps ``weka.classifiers.rules.M5Rules``. Repeatedly builds an M5 model tree
    and converts its best branch into a rule, giving a rule list whose
    consequents are linear models.

    Parameters
    ----------
    min_num_instances : float, default=4.0
        Minimum instances allowed at a leaf (Weka ``-M``).
    unpruned : bool, default=False
        Build unpruned rules (Weka ``-N``).
    use_unsmoothed : bool, default=False
        Skip the smoothing step (Weka ``-U``).
    nominal_features : sequence of int or None, default=None
        0-based indices of columns to treat as categorical.
    options : sequence of str or None, default=None
        Extra raw Weka options.

    Examples
    --------
    >>> from tuiml.weka import M5Rules
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(80, 3))
    >>> y = 2.0 * X[:, 0] + rng.normal(scale=0.1, size=80)
    >>> reg = M5Rules().fit(X, y)
    >>> reg.predict(X[:5]).shape
    (5,)
    """

    _weka_classname = "weka.classifiers.rules.M5Rules"
    _is_classification = False

    def __init__(
        self,
        min_num_instances: float = 4.0,
        unpruned: bool = False,
        use_unsmoothed: bool = False,
        nominal_features: Optional[Sequence[int]] = None,
        options: Optional[Sequence[str]] = None,
    ):
        super().__init__()
        self.min_num_instances = min_num_instances
        self.unpruned = unpruned
        self.use_unsmoothed = use_unsmoothed
        self.nominal_features = nominal_features
        self.options = options

    def _options(self) -> List[str]:
        """Return the Weka option tokens for this configuration."""
        opts = ["-M", fmt_num(self.min_num_instances)]
        if self.unpruned:
            opts.append("-N")
        if self.use_unsmoothed:
            opts.append("-U")
        return opts

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Any]:
        """Return JSON Schema for constructor parameters."""
        return {
            "min_num_instances": {"type": "number", "default": 4.0, "minimum": 1.0},
            "unpruned": {"type": "boolean", "default": False},
            "use_unsmoothed": {"type": "boolean", "default": False},
        }

    @classmethod
    def get_capabilities(cls) -> List[str]:
        """Return the capability names this wrapper supports."""
        return ["numeric", "nominal", "missing_values", "regression"]
