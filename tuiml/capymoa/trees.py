"""CapyMOA trees wrappers.

Incremental (Hoeffding-bound based) decision trees backed by CapyMOA/MOA.
Registered under ``capymoa.<ClassName>`` hub keys, mirroring the native
TuiML ``trees`` family.
"""

from typing import Any, Dict, List

from tuiml.base.algorithms import Classifier, Regressor
from tuiml.capymoa._base import (
    _CapyMOAStreamMixin,
    capymoa_classifier,
    capymoa_regressor,
)


@capymoa_classifier(tags=["tree", "hoeffding"])
class HoeffdingTree(_CapyMOAStreamMixin, Classifier):
    """**Hoeffding Tree** (VFDT) incremental classifier (hub key ``capymoa.HoeffdingTree``).

    Wraps :class:`capymoa.classifier.HoeffdingTree`. Grows a decision tree
    one instance at a time, using the Hoeffding bound to decide when enough
    instances have been seen to commit to a split — no need to store or
    revisit past data.

    Parameters
    ----------
    grace_period : int, default=200
        Number of instances a leaf must observe between split attempts.
        Smaller values adapt faster but cost more computation.

    Attributes
    ----------
    schema_ : capymoa.stream.Schema
        Stream schema derived from the training data.
    learner_ : capymoa.classifier.HoeffdingTree
        The fitted backing CapyMOA learner.

    Notes
    -----
    Requires the optional CapyMOA extra: ``pip install 'tuiml[capymoa]'``.
    The classic single-model streaming learner; assumes a mostly stationary
    stream. For drifting streams prefer
    :class:`~tuiml.capymoa.trees.HoeffdingAdaptiveTree` or
    :class:`~tuiml.capymoa.ensemble.AdaptiveRandomForest`.

    See Also
    --------
    :class:`~tuiml.capymoa.trees.HoeffdingAdaptiveTree` : Drift-adaptive variant.
    :class:`~tuiml.capymoa.trees.EFDT` : Extremely fast (split-revising) variant.

    Examples
    --------
    >>> from tuiml.capymoa import HoeffdingTree
    >>> from tuiml.datasets.generators import Agrawal
    >>> data = Agrawal(n_samples=1000, function=2, random_state=42).generate()
    >>> model = HoeffdingTree(grace_period=100).fit(data.X, data.y)
    >>> model.predict(data.X[:5]).shape
    (5,)
    """

    def __init__(self, grace_period: int = 200):
        super().__init__()
        self.grace_period = grace_period

    def _build_learner(self, schema):
        """Construct the backing CapyMOA learner for ``schema``."""
        from capymoa.classifier import HoeffdingTree as _Learner
        return _Learner(schema=schema, grace_period=self.grace_period)

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Any]:
        """Return JSON Schema for constructor parameters."""
        return {"grace_period": {"type": "integer", "default": 200}}

    @classmethod
    def get_capabilities(cls) -> List[str]:
        """Return the capability names this wrapper supports."""
        return ["numeric", "streaming"]


@capymoa_classifier(tags=["tree", "hoeffding", "drift"])
class HoeffdingAdaptiveTree(_CapyMOAStreamMixin, Classifier):
    """**Hoeffding Adaptive Tree** (HAT) drift-aware classifier (hub key ``capymoa.HoeffdingAdaptiveTree``).

    Wraps :class:`capymoa.classifier.HoeffdingAdaptiveTree`. Extends the
    Hoeffding Tree with ADWIN drift detectors at internal nodes: when a
    subtree's accuracy degrades, an alternate subtree is grown in the
    background and swapped in once it performs better.

    Parameters
    ----------
    grace_period : int, default=200
        Number of instances a leaf must observe between split attempts.

    Attributes
    ----------
    schema_ : capymoa.stream.Schema
        Stream schema derived from the training data.
    learner_ : capymoa.classifier.HoeffdingAdaptiveTree
        The fitted backing CapyMOA learner.

    Notes
    -----
    Requires the optional CapyMOA extra: ``pip install 'tuiml[capymoa]'``.
    Prefer this over the plain :class:`~tuiml.capymoa.trees.HoeffdingTree`
    when the data distribution is expected to change over time.

    See Also
    --------
    :class:`~tuiml.capymoa.trees.HoeffdingTree` : Non-adaptive base variant.
    :class:`~tuiml.capymoa.ensemble.AdaptiveRandomForest` : Drift-aware ensemble.

    Examples
    --------
    >>> from tuiml.capymoa import HoeffdingAdaptiveTree
    >>> from tuiml.datasets.generators import Hyperplane
    >>> data = Hyperplane(n_samples=1000, n_drift_features=2,
    ...                   random_state=42).generate()
    >>> model = HoeffdingAdaptiveTree().fit(data.X, data.y)
    >>> model.predict(data.X[:5]).shape
    (5,)
    """

    def __init__(self, grace_period: int = 200):
        super().__init__()
        self.grace_period = grace_period

    def _build_learner(self, schema):
        """Construct the backing CapyMOA learner for ``schema``."""
        from capymoa.classifier import HoeffdingAdaptiveTree as _Learner
        return _Learner(schema=schema, grace_period=self.grace_period)

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Any]:
        """Return JSON Schema for constructor parameters."""
        return {"grace_period": {"type": "integer", "default": 200}}

    @classmethod
    def get_capabilities(cls) -> List[str]:
        """Return the capability names this wrapper supports."""
        return ["numeric", "streaming"]


@capymoa_classifier(tags=["tree", "hoeffding"])
class EFDT(_CapyMOAStreamMixin, Classifier):
    """**Extremely Fast Decision Tree** classifier (hub key ``capymoa.EFDT``).

    Wraps :class:`capymoa.classifier.EFDT`. A Hoeffding Tree variant that
    splits as soon as a split looks *useful* rather than provably *best*,
    then keeps re-evaluating and revising splits as more data arrives —
    converging to the batch tree faster than VFDT.

    Parameters
    ----------
    grace_period : int, default=200
        Number of instances a leaf must observe between split attempts.

    Attributes
    ----------
    schema_ : capymoa.stream.Schema
        Stream schema derived from the training data.
    learner_ : capymoa.classifier.EFDT
        The fitted backing CapyMOA learner.

    Notes
    -----
    Requires the optional CapyMOA extra: ``pip install 'tuiml[capymoa]'``.

    See Also
    --------
    :class:`~tuiml.capymoa.trees.HoeffdingTree` : Classic VFDT.

    Examples
    --------
    >>> from tuiml.capymoa import EFDT
    >>> from tuiml.datasets.generators import Agrawal
    >>> data = Agrawal(n_samples=1000, function=3, random_state=42).generate()
    >>> model = EFDT().fit(data.X, data.y)
    >>> model.predict(data.X[:5]).shape
    (5,)
    """

    def __init__(self, grace_period: int = 200):
        super().__init__()
        self.grace_period = grace_period

    def _build_learner(self, schema):
        """Construct the backing CapyMOA learner for ``schema``."""
        from capymoa.classifier import EFDT as _Learner
        return _Learner(schema=schema, grace_period=self.grace_period)

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Any]:
        """Return JSON Schema for constructor parameters."""
        return {"grace_period": {"type": "integer", "default": 200}}

    @classmethod
    def get_capabilities(cls) -> List[str]:
        """Return the capability names this wrapper supports."""
        return ["numeric", "streaming"]


@capymoa_regressor(tags=["tree", "drift"])
class FIMTDD(_CapyMOAStreamMixin, Regressor):
    """**FIMT-DD** streaming regression tree (hub key ``capymoa.FIMTDD``).

    Wraps :class:`capymoa.regressor.FIMTDD`. Fast Incremental Model Tree
    with Drift Detection: grows a regression tree incrementally, fits
    linear models in the leaves, and replaces subtrees when drift is
    detected.

    Attributes
    ----------
    schema_ : capymoa.stream.Schema
        Stream schema derived from the training data.
    learner_ : capymoa.regressor.FIMTDD
        The fitted backing CapyMOA learner.

    Notes
    -----
    Requires the optional CapyMOA extra: ``pip install 'tuiml[capymoa]'``.

    See Also
    --------
    :class:`~tuiml.capymoa.ensemble.AdaptiveRandomForestRegressor` :
        Ensemble alternative for drifting regression streams.

    Examples
    --------
    >>> from tuiml.capymoa import FIMTDD
    >>> from tuiml.datasets.generators import Friedman
    >>> data = Friedman(n_samples=1000, random_state=42).generate()
    >>> model = FIMTDD().fit(data.X, data.y)
    >>> model.predict(data.X[:5]).shape
    (5,)
    """

    def __init__(self):
        super().__init__()

    def _build_learner(self, schema):
        """Construct the backing CapyMOA learner for ``schema``."""
        from capymoa.regressor import FIMTDD as _Learner
        return _Learner(schema=schema)

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Any]:
        """Return JSON Schema for constructor parameters."""
        return {}

    @classmethod
    def get_capabilities(cls) -> List[str]:
        """Return the capability names this wrapper supports."""
        return ["numeric", "streaming"]


__all__ = ["HoeffdingTree", "HoeffdingAdaptiveTree", "EFDT", "FIMTDD"]
