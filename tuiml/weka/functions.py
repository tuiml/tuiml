"""Weka function-based learner wrappers.

Learners from ``weka.classifiers.functions`` — support vector machines, logistic
and linear regression, neural networks — registered under ``weka.<ClassName>``
hub keys.

Notes
-----
Requires the optional Weka extra: ``pip install 'tuiml[weka]'`` plus a Java
runtime (11+) on ``PATH``.

These learners apply their own internal preprocessing: ``SMO`` and
``MultilayerPerceptron`` normalize numeric attributes and binarize nominal ones
by default, so results differ from the same nominal encoding fed to a tree.
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
    "SMO",
    "SMOreg",
    "Logistic",
    "SimpleLogistic",
    "LinearRegression",
    "SimpleLinearRegression",
    "MultilayerPerceptron",
    "GaussianProcesses",
]


@weka_classifier(tags=["svm", "kernel", "function"])
class SMO(_WekaSupervisedMixin, Classifier):
    """**SMO** — support vector classifier (hub key ``weka.SMO``).

    Wraps ``weka.classifiers.functions.SMO``, Weka's sequential minimal
    optimization trainer for support vector machines.

    Parameters
    ----------
    c : float, default=1.0
        Complexity / regularization constant (Weka ``-C``).
    kernel : str, default="weka.classifiers.functions.supportVector.PolyKernel"
        Fully-qualified Weka kernel class (Weka ``-K``). The common
        alternatives are ``...supportVector.RBFKernel`` and
        ``...supportVector.Puk``.
    kernel_options : sequence of str or None, default=None
        Options passed to the kernel, e.g. ``["-E", "2"]`` for a quadratic
        polynomial or ``["-G", "0.01"]`` for an RBF gamma.
    nominal_features : sequence of int or None, default=None
        0-based indices of columns to treat as categorical.
    options : sequence of str or None, default=None
        Extra raw Weka options.

    Notes
    -----
    SMO is a **binary** learner; Weka handles multiclass targets internally by
    pairwise coupling, so no extra wrapping is needed here.

    By default SMO normalizes the training data itself and fits a linear
    ``PolyKernel``, which is why its numbers differ from an unnormalized
    scikit-learn ``SVC`` with the same ``C``.

    See Also
    --------
    :class:`~tuiml.algorithms.svm.SVC` : TuiML's native support vector classifier.

    Examples
    --------
    >>> from tuiml.weka import SMO
    >>> from tuiml.datasets import load_iris
    >>> data = load_iris()
    >>> clf = SMO(c=1.0).fit(data.X, data.y)
    >>> clf.predict(data.X[:5]).shape
    (5,)
    """

    _weka_classname = "weka.classifiers.functions.SMO"
    _is_classification = True

    def __init__(
        self,
        c: float = 1.0,
        kernel: str = "weka.classifiers.functions.supportVector.PolyKernel",
        kernel_options: Optional[Sequence[str]] = None,
        nominal_features: Optional[Sequence[int]] = None,
        options: Optional[Sequence[str]] = None,
    ):
        super().__init__()
        self.c = c
        self.kernel = kernel
        self.kernel_options = kernel_options
        self.nominal_features = nominal_features
        self.options = options

    def _options(self) -> List[str]:
        """Return the Weka option tokens for this configuration."""
        kernel_spec = self.kernel
        if self.kernel_options:
            kernel_spec = f"{self.kernel} {' '.join(self.kernel_options)}"
        return ["-C", fmt_num(self.c), "-K", kernel_spec]

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Any]:
        """Return JSON Schema for constructor parameters."""
        return {
            "c": {"type": "number", "default": 1.0, "minimum": 0.0},
            "kernel": {
                "type": "string",
                "default": "weka.classifiers.functions.supportVector.PolyKernel"},
        }

    @classmethod
    def get_capabilities(cls) -> List[str]:
        """Return the capability names this wrapper supports."""
        return ["numeric", "nominal", "missing_values", "binary_class", "multiclass"]


@weka_regressor(tags=["svm", "kernel", "function", "regression"])
class SMOreg(_WekaSupervisedMixin, Regressor):
    """**SMOreg** — support vector regressor (hub key ``weka.SMOreg``).

    Wraps ``weka.classifiers.functions.SMOreg``, the regression counterpart of
    :class:`~tuiml.weka.functions.SMO`.

    Parameters
    ----------
    c : float, default=1.0
        Complexity / regularization constant (Weka ``-C``).
    kernel : str, default="weka.classifiers.functions.supportVector.PolyKernel"
        Fully-qualified Weka kernel class (Weka ``-K``).
    kernel_options : sequence of str or None, default=None
        Options passed to the kernel.
    nominal_features : sequence of int or None, default=None
        0-based indices of columns to treat as categorical.
    options : sequence of str or None, default=None
        Extra raw Weka options.

    See Also
    --------
    :class:`~tuiml.algorithms.svm.SVR` : TuiML's native support vector regressor.

    Examples
    --------
    >>> from tuiml.weka import SMOreg
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(60, 3))
    >>> y = 2.0 * X[:, 0] + rng.normal(scale=0.1, size=60)
    >>> reg = SMOreg().fit(X, y)
    >>> reg.predict(X[:5]).shape
    (5,)
    """

    _weka_classname = "weka.classifiers.functions.SMOreg"
    _is_classification = False

    def __init__(
        self,
        c: float = 1.0,
        kernel: str = "weka.classifiers.functions.supportVector.PolyKernel",
        kernel_options: Optional[Sequence[str]] = None,
        nominal_features: Optional[Sequence[int]] = None,
        options: Optional[Sequence[str]] = None,
    ):
        super().__init__()
        self.c = c
        self.kernel = kernel
        self.kernel_options = kernel_options
        self.nominal_features = nominal_features
        self.options = options

    def _options(self) -> List[str]:
        """Return the Weka option tokens for this configuration."""
        kernel_spec = self.kernel
        if self.kernel_options:
            kernel_spec = f"{self.kernel} {' '.join(self.kernel_options)}"
        return ["-C", fmt_num(self.c), "-K", kernel_spec]

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Any]:
        """Return JSON Schema for constructor parameters."""
        return {
            "c": {"type": "number", "default": 1.0, "minimum": 0.0},
            "kernel": {
                "type": "string",
                "default": "weka.classifiers.functions.supportVector.PolyKernel"},
        }

    @classmethod
    def get_capabilities(cls) -> List[str]:
        """Return the capability names this wrapper supports."""
        return ["numeric", "nominal", "missing_values", "regression"]


@weka_classifier(tags=["linear", "logistic", "function"])
class Logistic(_WekaSupervisedMixin, Classifier):
    """**Logistic** — multinomial logistic regression (hub key ``weka.Logistic``).

    Wraps ``weka.classifiers.functions.Logistic``, ridge-penalized multinomial
    logistic regression fitted by quasi-Newton optimization.

    Parameters
    ----------
    ridge : float, default=1e-8
        Ridge penalty (Weka ``-R``).
    max_its : int, default=-1
        Maximum optimizer iterations (Weka ``-M``); ``-1`` runs until
        convergence.
    nominal_features : sequence of int or None, default=None
        0-based indices of columns to treat as categorical.
    options : sequence of str or None, default=None
        Extra raw Weka options.

    See Also
    --------
    :class:`~tuiml.algorithms.linear.LogisticRegression` : TuiML's native version.

    Examples
    --------
    >>> from tuiml.weka import Logistic
    >>> from tuiml.datasets import load_iris
    >>> data = load_iris()
    >>> clf = Logistic().fit(data.X, data.y)
    >>> clf.predict(data.X[:5]).shape
    (5,)
    """

    _weka_classname = "weka.classifiers.functions.Logistic"
    _is_classification = True

    def __init__(
        self,
        ridge: float = 1e-8,
        max_its: int = -1,
        nominal_features: Optional[Sequence[int]] = None,
        options: Optional[Sequence[str]] = None,
    ):
        super().__init__()
        self.ridge = ridge
        self.max_its = max_its
        self.nominal_features = nominal_features
        self.options = options

    def _options(self) -> List[str]:
        """Return the Weka option tokens for this configuration."""
        return ["-R", str(self.ridge), "-M", fmt_num(self.max_its)]

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Any]:
        """Return JSON Schema for constructor parameters."""
        return {
            "ridge": {"type": "number", "default": 1e-8, "minimum": 0.0},
            "max_its": {"type": "integer", "default": -1},
        }

    @classmethod
    def get_capabilities(cls) -> List[str]:
        """Return the capability names this wrapper supports."""
        return ["numeric", "nominal", "missing_values", "binary_class", "multiclass"]


@weka_classifier(tags=["linear", "logistic", "boosting", "function"])
class SimpleLogistic(_WekaSupervisedMixin, Classifier):
    """**SimpleLogistic** — logistic regression fitted by LogitBoost (hub key ``weka.SimpleLogistic``).

    Wraps ``weka.classifiers.functions.SimpleLogistic``. Fits linear logistic
    models with LogitBoost over simple regression functions, using
    cross-validation to pick the number of boosting iterations, which performs
    built-in attribute selection.

    Parameters
    ----------
    num_boosting_iterations : int, default=0
        Fixed number of boosting iterations (Weka ``-I``); ``0`` selects the
        count by cross-validation.
    use_cross_validation : bool, default=True
        Choose the iteration count by cross-validation. Passing False emits
        Weka's ``-S`` flag to use the training error instead.
    heuristic_stop : int, default=50
        Stop early if the minimum has not moved for this many iterations
        (Weka ``-H``). Set to 0 to disable the heuristic.
    nominal_features : sequence of int or None, default=None
        0-based indices of columns to treat as categorical.
    options : sequence of str or None, default=None
        Extra raw Weka options.

    See Also
    --------
    :class:`~tuiml.algorithms.linear.LogisticRegression` : TuiML's native logistic regression.

    Examples
    --------
    >>> from tuiml.weka import SimpleLogistic
    >>> from tuiml.datasets import load_iris
    >>> data = load_iris()
    >>> clf = SimpleLogistic().fit(data.X, data.y)
    >>> clf.predict(data.X[:5]).shape
    (5,)
    """

    _weka_classname = "weka.classifiers.functions.SimpleLogistic"
    _is_classification = True

    def __init__(
        self,
        num_boosting_iterations: int = 0,
        use_cross_validation: bool = True,
        heuristic_stop: int = 50,
        nominal_features: Optional[Sequence[int]] = None,
        options: Optional[Sequence[str]] = None,
    ):
        super().__init__()
        self.num_boosting_iterations = num_boosting_iterations
        self.use_cross_validation = use_cross_validation
        self.heuristic_stop = heuristic_stop
        self.nominal_features = nominal_features
        self.options = options

    def _options(self) -> List[str]:
        """Return the Weka option tokens for this configuration."""
        opts = ["-I", fmt_num(self.num_boosting_iterations),
                "-H", fmt_num(self.heuristic_stop)]
        if not self.use_cross_validation:
            opts.append("-S")
        return opts

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Any]:
        """Return JSON Schema for constructor parameters."""
        return {
            "num_boosting_iterations": {"type": "integer", "default": 0, "minimum": 0},
            "use_cross_validation": {"type": "boolean", "default": True},
            "heuristic_stop": {"type": "integer", "default": 50, "minimum": 0},
        }

    @classmethod
    def get_capabilities(cls) -> List[str]:
        """Return the capability names this wrapper supports."""
        return ["numeric", "nominal", "missing_values", "binary_class", "multiclass"]


@weka_regressor(tags=["linear", "regression", "function"])
class LinearRegression(_WekaSupervisedMixin, Regressor):
    """**LinearRegression** — least squares with attribute selection (hub key ``weka.LinearRegression``).

    Wraps ``weka.classifiers.functions.LinearRegression``.

    Parameters
    ----------
    attribute_selection : {'m5', 'none', 'greedy'}, default='m5'
        Attribute selection method (Weka ``-S`` as ``0``/``1``/``2``).
    ridge : float, default=1e-8
        Ridge penalty (Weka ``-R``).
    eliminate_colinear : bool, default=True
        Remove collinear attributes. Passing False emits Weka's ``-C`` flag to
        keep them.
    nominal_features : sequence of int or None, default=None
        0-based indices of columns to treat as categorical.
    options : sequence of str or None, default=None
        Extra raw Weka options.

    Notes
    -----
    Weka's default runs **M5 attribute selection and collinearity removal**, so
    it does not reproduce a plain ordinary-least-squares fit. Pass
    ``attribute_selection='none'`` and ``eliminate_colinear=False`` for that.

    See Also
    --------
    :class:`~tuiml.algorithms.linear.LinearRegression` : TuiML's native version.

    Examples
    --------
    >>> from tuiml.weka import LinearRegression
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(60, 3))
    >>> y = 2.0 * X[:, 0] + rng.normal(scale=0.1, size=60)
    >>> reg = LinearRegression().fit(X, y)
    >>> reg.predict(X[:5]).shape
    (5,)
    """

    _weka_classname = "weka.classifiers.functions.LinearRegression"
    _is_classification = False

    #: Weka's ``-S`` takes an integer code per selection method.
    _SELECTION = {"m5": "0", "none": "1", "greedy": "2"}

    def __init__(
        self,
        attribute_selection: str = "m5",
        ridge: float = 1e-8,
        eliminate_colinear: bool = True,
        nominal_features: Optional[Sequence[int]] = None,
        options: Optional[Sequence[str]] = None,
    ):
        super().__init__()
        self.attribute_selection = attribute_selection
        self.ridge = ridge
        self.eliminate_colinear = eliminate_colinear
        self.nominal_features = nominal_features
        self.options = options

    def _options(self) -> List[str]:
        """Return the Weka option tokens for this configuration."""
        try:
            sel = self._SELECTION[self.attribute_selection]
        except KeyError:
            raise ValueError(
                f"attribute_selection must be one of {sorted(self._SELECTION)}, "
                f"got {self.attribute_selection!r}"
            ) from None
        opts = ["-S", sel, "-R", str(self.ridge)]
        if not self.eliminate_colinear:
            opts.append("-C")
        return opts

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Any]:
        """Return JSON Schema for constructor parameters."""
        return {
            "attribute_selection": {"type": "string", "default": "m5",
                                    "enum": ["m5", "none", "greedy"]},
            "ridge": {"type": "number", "default": 1e-8, "minimum": 0.0},
            "eliminate_colinear": {"type": "boolean", "default": True},
        }

    @classmethod
    def get_capabilities(cls) -> List[str]:
        """Return the capability names this wrapper supports."""
        return ["numeric", "nominal", "missing_values", "regression"]


@weka_regressor(tags=["linear", "regression", "baseline"])
class SimpleLinearRegression(_WekaSupervisedMixin, Regressor):
    """**SimpleLinearRegression** — least squares on one attribute (hub key ``weka.SimpleLinearRegression``).

    Wraps ``weka.classifiers.functions.SimpleLinearRegression``. Fits a straight
    line on whichever single attribute yields the lowest squared error — the
    regression counterpart of :class:`~tuiml.weka.rules.OneR`.

    Parameters
    ----------
    nominal_features : sequence of int or None, default=None
        0-based indices of columns to treat as categorical.
    options : sequence of str or None, default=None
        Extra raw Weka options.

    Examples
    --------
    >>> from tuiml.weka import SimpleLinearRegression
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(60, 3))
    >>> y = 2.0 * X[:, 0] + rng.normal(scale=0.1, size=60)
    >>> reg = SimpleLinearRegression().fit(X, y)
    >>> reg.predict(X[:5]).shape
    (5,)
    """

    _weka_classname = "weka.classifiers.functions.SimpleLinearRegression"
    _is_classification = False

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
        return ["numeric", "missing_values", "regression"]


@weka_classifier(tags=["neural", "mlp", "function"])
class MultilayerPerceptron(_WekaSupervisedMixin, Classifier):
    """**MultilayerPerceptron** — backpropagation neural network (hub key ``weka.MultilayerPerceptron``).

    Wraps ``weka.classifiers.functions.MultilayerPerceptron``.

    Parameters
    ----------
    hidden_layers : str, default="a"
        Hidden layer specification (Weka ``-H``). A comma-separated list of
        node counts, or one of Weka's shorthands: ``"a"`` for
        ``(attribs + classes) / 2``, ``"i"``, ``"o"``, ``"t"``.
    learning_rate : float, default=0.3
        Learning rate (Weka ``-L``).
    momentum : float, default=0.2
        Momentum (Weka ``-M``).
    training_time : int, default=500
        Number of epochs (Weka ``-N``).
    seed : int, default=0
        Random seed (Weka ``-S``).
    nominal_features : sequence of int or None, default=None
        0-based indices of columns to treat as categorical.
    options : sequence of str or None, default=None
        Extra raw Weka options.

    Notes
    -----
    Weka normalizes attributes and the numeric class internally by default, and
    training runs for the full ``training_time`` epochs, which makes this the
    slowest learner in the package on large data.

    See Also
    --------
    :class:`~tuiml.algorithms.neural.MultilayerPerceptronClassifier` : TuiML's native MLP.

    Examples
    --------
    >>> from tuiml.weka import MultilayerPerceptron
    >>> from tuiml.datasets import load_iris
    >>> data = load_iris()
    >>> clf = MultilayerPerceptron(training_time=50).fit(data.X, data.y)
    >>> clf.predict(data.X[:5]).shape
    (5,)
    """

    _weka_classname = "weka.classifiers.functions.MultilayerPerceptron"
    _is_classification = True

    def __init__(
        self,
        hidden_layers: str = "a",
        learning_rate: float = 0.3,
        momentum: float = 0.2,
        training_time: int = 500,
        seed: int = 0,
        nominal_features: Optional[Sequence[int]] = None,
        options: Optional[Sequence[str]] = None,
    ):
        super().__init__()
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.training_time = training_time
        self.seed = seed
        self.nominal_features = nominal_features
        self.options = options

    def _options(self) -> List[str]:
        """Return the Weka option tokens for this configuration."""
        return ["-H", str(self.hidden_layers),
                "-L", fmt_num(self.learning_rate),
                "-M", fmt_num(self.momentum),
                "-N", fmt_num(self.training_time),
                "-S", fmt_num(self.seed)]

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Any]:
        """Return JSON Schema for constructor parameters."""
        return {
            "hidden_layers": {"type": "string", "default": "a"},
            "learning_rate": {"type": "number", "default": 0.3, "minimum": 0.0},
            "momentum": {"type": "number", "default": 0.2, "minimum": 0.0},
            "training_time": {"type": "integer", "default": 500, "minimum": 1},
            "seed": {"type": "integer", "default": 0},
        }

    @classmethod
    def get_capabilities(cls) -> List[str]:
        """Return the capability names this wrapper supports."""
        return ["numeric", "nominal", "missing_values", "binary_class", "multiclass"]


@weka_regressor(tags=["bayesian", "kernel", "regression", "function"])
class GaussianProcesses(_WekaSupervisedMixin, Regressor):
    """**GaussianProcesses** — Gaussian process regression (hub key ``weka.GaussianProcesses``).

    Wraps ``weka.classifiers.functions.GaussianProcesses``.

    Parameters
    ----------
    noise : float, default=1.0
        Noise level, as a fraction of the target standard deviation
        (Weka ``-N``).
    kernel : str, default="weka.classifiers.functions.supportVector.PolyKernel"
        Fully-qualified Weka kernel class (Weka ``-K``).
    kernel_options : sequence of str or None, default=None
        Options passed to the kernel.
    nominal_features : sequence of int or None, default=None
        0-based indices of columns to treat as categorical.
    options : sequence of str or None, default=None
        Extra raw Weka options.

    Notes
    -----
    Fitting inverts an :math:`n \\times n` matrix, so cost grows as
    :math:`O(n^3)` and memory as :math:`O(n^2)` — impractical beyond a few
    thousand training rows.

    See Also
    --------
    :class:`~tuiml.algorithms.bayesian.GaussianProcessesRegressor` : TuiML's native version.

    Examples
    --------
    >>> from tuiml.weka import GaussianProcesses
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(60, 3))
    >>> y = 2.0 * X[:, 0] + rng.normal(scale=0.1, size=60)
    >>> reg = GaussianProcesses().fit(X, y)
    >>> reg.predict(X[:5]).shape
    (5,)
    """

    _weka_classname = "weka.classifiers.functions.GaussianProcesses"
    _is_classification = False

    def __init__(
        self,
        noise: float = 1.0,
        kernel: str = "weka.classifiers.functions.supportVector.PolyKernel",
        kernel_options: Optional[Sequence[str]] = None,
        nominal_features: Optional[Sequence[int]] = None,
        options: Optional[Sequence[str]] = None,
    ):
        super().__init__()
        self.noise = noise
        self.kernel = kernel
        self.kernel_options = kernel_options
        self.nominal_features = nominal_features
        self.options = options

    def _options(self) -> List[str]:
        """Return the Weka option tokens for this configuration."""
        kernel_spec = self.kernel
        if self.kernel_options:
            kernel_spec = f"{self.kernel} {' '.join(self.kernel_options)}"
        return ["-N", fmt_num(self.noise), "-K", kernel_spec]

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Any]:
        """Return JSON Schema for constructor parameters."""
        return {
            "noise": {"type": "number", "default": 1.0, "minimum": 0.0},
            "kernel": {
                "type": "string",
                "default": "weka.classifiers.functions.supportVector.PolyKernel"},
        }

    @classmethod
    def get_capabilities(cls) -> List[str]:
        """Return the capability names this wrapper supports."""
        return ["numeric", "nominal", "missing_values", "regression"]
