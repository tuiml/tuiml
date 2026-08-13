"""Weka instance-based (lazy) learner wrappers.

Learners from ``weka.classifiers.lazy``, which store the training data and defer
all work to prediction time. Registered under ``weka.<ClassName>`` hub keys.

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

__all__ = ["IBk", "KStar", "LWL"]


@weka_classifier(tags=["lazy", "instance-based", "knn"])
class IBk(_WekaSupervisedMixin, Classifier):
    """**IBk** — k-nearest-neighbour classifier (hub key ``weka.IBk``).

    Wraps ``weka.classifiers.lazy.IBk``, Weka's implementation of Aha's
    instance-based learner.

    Parameters
    ----------
    k : int, default=1
        Number of neighbours (Weka ``-K``).
    distance_weighting : {'none', 'inverse', 'similarity'}, default='none'
        How neighbour votes are weighted: uniformly, by ``1/distance``
        (Weka ``-I``), or by ``1 - distance`` (Weka ``-F``).
    cross_validate : bool, default=False
        Select the best ``k`` up to the given value by leave-one-out
        cross-validation (Weka ``-X``).
    nominal_features : sequence of int or None, default=None
        0-based indices of columns to treat as categorical.
    options : sequence of str or None, default=None
        Extra raw Weka options.

    See Also
    --------
    :class:`~tuiml.algorithms.neighbors.KNearestNeighborsClassifier` : TuiML's native kNN.

    Examples
    --------
    >>> from tuiml.weka import IBk
    >>> from tuiml.datasets import load_iris
    >>> data = load_iris()
    >>> clf = IBk(k=3).fit(data.X, data.y)
    >>> clf.predict(data.X[:5]).shape
    (5,)
    """

    _weka_classname = "weka.classifiers.lazy.IBk"
    _is_classification = True

    def __init__(
        self,
        k: int = 1,
        distance_weighting: str = "none",
        cross_validate: bool = False,
        nominal_features: Optional[Sequence[int]] = None,
        options: Optional[Sequence[str]] = None,
    ):
        super().__init__()
        self.k = k
        self.distance_weighting = distance_weighting
        self.cross_validate = cross_validate
        self.nominal_features = nominal_features
        self.options = options

    def _options(self) -> List[str]:
        """Return the Weka option tokens for this configuration."""
        opts = ["-K", fmt_num(self.k)]
        if self.distance_weighting == "inverse":
            opts.append("-I")
        elif self.distance_weighting == "similarity":
            opts.append("-F")
        elif self.distance_weighting != "none":
            raise ValueError(
                "distance_weighting must be 'none', 'inverse' or 'similarity', "
                f"got {self.distance_weighting!r}"
            )
        if self.cross_validate:
            opts.append("-X")
        return opts

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Any]:
        """Return JSON Schema for constructor parameters."""
        return {
            "k": {"type": "integer", "default": 1, "minimum": 1},
            "distance_weighting": {"type": "string", "default": "none",
                                   "enum": ["none", "inverse", "similarity"]},
            "cross_validate": {"type": "boolean", "default": False},
        }

    @classmethod
    def get_capabilities(cls) -> List[str]:
        """Return the capability names this wrapper supports."""
        return ["numeric", "nominal", "missing_values", "binary_class", "multiclass"]


@weka_classifier(tags=["lazy", "instance-based", "entropy"])
class KStar(_WekaSupervisedMixin, Classifier):
    """**K\\*** — entropy-based instance classifier (hub key ``weka.KStar``).

    Wraps ``weka.classifiers.lazy.KStar``. Instead of a geometric distance it
    uses an entropic measure — the complexity of transforming one instance into
    another — which handles mixed numeric and symbolic attributes coherently.

    Parameters
    ----------
    global_blend : int, default=20
        Blending parameter as a percentage (Weka ``-B``), between 0 and 100.
        0 behaves like nearest-neighbour, 100 weights all instances equally.
    missing_mode : {'average', 'ignore', 'max', 'normal'}, default='average'
        How missing values are treated (Weka ``-M`` with ``a``/``d``/``m``/``n``).
    entropic_auto_blend : bool, default=False
        Set the blend automatically from the data (Weka ``-E``).
    nominal_features : sequence of int or None, default=None
        0-based indices of columns to treat as categorical.
    options : sequence of str or None, default=None
        Extra raw Weka options.

    References
    ----------
    .. [Cleary1995] Cleary, J.G. and Trigg, L.E. (1995).
           **K*: An Instance-based Learner Using an Entropic Distance Measure.**
           *Proceedings of the 12th International Conference on Machine
           Learning*, 108-114.

    Examples
    --------
    >>> from tuiml.weka import KStar
    >>> from tuiml.datasets import load_iris
    >>> data = load_iris()
    >>> clf = KStar(global_blend=20).fit(data.X, data.y)
    >>> clf.predict(data.X[:5]).shape
    (5,)
    """

    _weka_classname = "weka.classifiers.lazy.KStar"
    _is_classification = True

    #: Weka's ``-M`` takes a single letter per missing-value mode.
    _MISSING_MODES = {"average": "a", "ignore": "d", "max": "m", "normal": "n"}

    def __init__(
        self,
        global_blend: int = 20,
        missing_mode: str = "average",
        entropic_auto_blend: bool = False,
        nominal_features: Optional[Sequence[int]] = None,
        options: Optional[Sequence[str]] = None,
    ):
        super().__init__()
        self.global_blend = global_blend
        self.missing_mode = missing_mode
        self.entropic_auto_blend = entropic_auto_blend
        self.nominal_features = nominal_features
        self.options = options

    def _options(self) -> List[str]:
        """Return the Weka option tokens for this configuration."""
        try:
            mode = self._MISSING_MODES[self.missing_mode]
        except KeyError:
            raise ValueError(
                f"missing_mode must be one of {sorted(self._MISSING_MODES)}, "
                f"got {self.missing_mode!r}"
            ) from None
        opts = ["-B", fmt_num(self.global_blend), "-M", mode]
        if self.entropic_auto_blend:
            opts.append("-E")
        return opts

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Any]:
        """Return JSON Schema for constructor parameters."""
        return {
            "global_blend": {"type": "integer", "default": 20,
                             "minimum": 0, "maximum": 100},
            "missing_mode": {"type": "string", "default": "average",
                             "enum": ["average", "ignore", "max", "normal"]},
            "entropic_auto_blend": {"type": "boolean", "default": False},
        }

    @classmethod
    def get_capabilities(cls) -> List[str]:
        """Return the capability names this wrapper supports."""
        return ["numeric", "nominal", "missing_values", "binary_class", "multiclass"]


@weka_regressor(tags=["lazy", "instance-based", "local"])
class LWL(_WekaSupervisedMixin, Regressor):
    """**LWL** — locally weighted learning (hub key ``weka.LWL``).

    Wraps ``weka.classifiers.lazy.LWL``. For each query it weights the training
    instances by their distance to that query and fits the base learner on the
    weighted set, which turns a global model into a local one.

    Parameters
    ----------
    k_nn : int, default=-1
        Neighbours used for setting the kernel bandwidth (Weka ``-K``);
        ``-1`` uses all training instances.
    weighting_kernel : int, default=0
        Weighting kernel (Weka ``-U``): 0 linear, 1 epanechnikov, 2 tricube,
        3 inverse, 4 gaussian, 5 constant.
    base_classifier : str, default="weka.classifiers.trees.DecisionStump"
        Fully-qualified Weka class used as the local model (Weka ``-W``).
    nominal_features : sequence of int or None, default=None
        0-based indices of columns to treat as categorical.
    options : sequence of str or None, default=None
        Extra raw Weka options.

    Notes
    -----
    All the work happens at prediction time: every query refits the base
    learner, so prediction is far more expensive than training.

    Examples
    --------
    >>> from tuiml.weka import LWL
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(60, 2))
    >>> y = 2.0 * X[:, 0] + rng.normal(scale=0.1, size=60)
    >>> reg = LWL(k_nn=20).fit(X, y)
    >>> reg.predict(X[:5]).shape
    (5,)
    """

    _weka_classname = "weka.classifiers.lazy.LWL"
    _is_classification = False

    def __init__(
        self,
        k_nn: int = -1,
        weighting_kernel: int = 0,
        base_classifier: str = "weka.classifiers.trees.DecisionStump",
        nominal_features: Optional[Sequence[int]] = None,
        options: Optional[Sequence[str]] = None,
    ):
        super().__init__()
        self.k_nn = k_nn
        self.weighting_kernel = weighting_kernel
        self.base_classifier = base_classifier
        self.nominal_features = nominal_features
        self.options = options

    def _options(self) -> List[str]:
        """Return the Weka option tokens for this configuration."""
        return ["-K", fmt_num(self.k_nn), "-U", fmt_num(self.weighting_kernel),
                "-W", self.base_classifier]

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Any]:
        """Return JSON Schema for constructor parameters."""
        return {
            "k_nn": {"type": "integer", "default": -1},
            "weighting_kernel": {"type": "integer", "default": 0,
                                 "minimum": 0, "maximum": 5},
            "base_classifier": {"type": "string",
                                "default": "weka.classifiers.trees.DecisionStump"},
        }

    @classmethod
    def get_capabilities(cls) -> List[str]:
        """Return the capability names this wrapper supports."""
        return ["numeric", "nominal", "missing_values", "regression"]
