"""CapyMOA ensemble wrappers.

Incremental, mostly drift-aware ensembles backed by CapyMOA/MOA.
Registered under ``capymoa.<ClassName>`` hub keys, mirroring the native
TuiML ``ensemble`` family.
"""

from typing import Any, Dict, List

from tuiml.base.algorithms import Classifier, Regressor
from tuiml.capymoa._base import (
    _CapyMOAStreamMixin,
    capymoa_classifier,
    capymoa_regressor,
)


@capymoa_classifier(tags=["ensemble", "drift"])
class AdaptiveRandomForest(_CapyMOAStreamMixin, Classifier):
    """**Adaptive Random Forest** drift-aware classifier (hub key ``capymoa.AdaptiveRandomForest``).

    Wraps :class:`capymoa.classifier.AdaptiveRandomForestClassifier`. An
    ensemble of Hoeffding Trees trained with online bagging and per-tree
    random feature subsets; each member carries a drift detector and is
    replaced by a background tree when its accuracy degrades.

    Parameters
    ----------
    ensemble_size : int, default=10
        Number of trees in the forest. More trees improve accuracy at the
        cost of memory and per-instance training time.

    Attributes
    ----------
    schema_ : capymoa.stream.Schema
        Stream schema derived from the training data.
    learner_ : capymoa.classifier.AdaptiveRandomForestClassifier
        The fitted backing CapyMOA learner.

    Notes
    -----
    Requires the optional CapyMOA extra: ``pip install 'tuiml[capymoa]'``.
    The strongest general-purpose choice among the CapyMOA classifiers
    wrapped here, especially under concept drift.

    See Also
    --------
    :class:`~tuiml.capymoa.ensemble.LeveragingBagging` : Bagging with more
        aggressive input randomisation.
    :class:`~tuiml.capymoa.trees.HoeffdingAdaptiveTree` : Single-tree
        drift-aware alternative.

    Examples
    --------
    >>> from tuiml.capymoa import AdaptiveRandomForest
    >>> from tuiml.datasets.generators import Hyperplane
    >>> data = Hyperplane(n_samples=1000, n_drift_features=2,
    ...                   random_state=42).generate()
    >>> model = AdaptiveRandomForest(ensemble_size=5).fit(data.X, data.y)
    >>> model.predict(data.X[:5]).shape
    (5,)
    """

    def __init__(self, ensemble_size: int = 10):
        super().__init__()
        self.ensemble_size = ensemble_size

    def _build_learner(self, schema):
        """Construct the backing CapyMOA learner for ``schema``."""
        from capymoa.classifier import AdaptiveRandomForestClassifier as _Learner
        return _Learner(schema=schema, ensemble_size=self.ensemble_size)

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Any]:
        """Return JSON Schema for constructor parameters."""
        return {"ensemble_size": {"type": "integer", "default": 10}}

    @classmethod
    def get_capabilities(cls) -> List[str]:
        """Return the capability names this wrapper supports."""
        return ["numeric", "streaming"]


@capymoa_classifier(tags=["ensemble", "bagging"])
class OnlineBagging(_CapyMOAStreamMixin, Classifier):
    """**Online Bagging** (Oza & Russell) ensemble classifier (hub key ``capymoa.OnlineBagging``).

    Wraps :class:`capymoa.classifier.OnlineBagging`. Simulates bootstrap
    sampling on a stream by training each ensemble member on every instance
    :math:`k` times, with :math:`k \\sim \\text{Poisson}(1)`.

    Parameters
    ----------
    ensemble_size : int, default=10
        Number of base learners in the ensemble.

    Attributes
    ----------
    schema_ : capymoa.stream.Schema
        Stream schema derived from the training data.
    learner_ : capymoa.classifier.OnlineBagging
        The fitted backing CapyMOA learner.

    Notes
    -----
    Requires the optional CapyMOA extra: ``pip install 'tuiml[capymoa]'``.
    No built-in drift handling; for drifting streams prefer
    :class:`~tuiml.capymoa.ensemble.LeveragingBagging` or
    :class:`~tuiml.capymoa.ensemble.AdaptiveRandomForest`.

    See Also
    --------
    :class:`~tuiml.capymoa.ensemble.LeveragingBagging` : Drift-aware bagging.

    Examples
    --------
    >>> from tuiml.capymoa import OnlineBagging
    >>> from tuiml.datasets.generators import Agrawal
    >>> data = Agrawal(n_samples=1000, function=1, random_state=42).generate()
    >>> model = OnlineBagging(ensemble_size=5).fit(data.X, data.y)
    >>> model.predict(data.X[:5]).shape
    (5,)
    """

    def __init__(self, ensemble_size: int = 10):
        super().__init__()
        self.ensemble_size = ensemble_size

    def _build_learner(self, schema):
        """Construct the backing CapyMOA learner for ``schema``."""
        from capymoa.classifier import OnlineBagging as _Learner
        return _Learner(schema=schema, ensemble_size=self.ensemble_size)

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Any]:
        """Return JSON Schema for constructor parameters."""
        return {"ensemble_size": {"type": "integer", "default": 10}}

    @classmethod
    def get_capabilities(cls) -> List[str]:
        """Return the capability names this wrapper supports."""
        return ["numeric", "streaming"]


@capymoa_classifier(tags=["ensemble", "bagging", "drift"])
class LeveragingBagging(_CapyMOAStreamMixin, Classifier):
    """**Leveraging Bagging** drift-aware ensemble classifier (hub key ``capymoa.LeveragingBagging``).

    Wraps :class:`capymoa.classifier.LeveragingBagging`. Extends online
    bagging with higher resampling weights (:math:`\\text{Poisson}(6)`)
    for more input diversity, plus ADWIN drift detection that resets
    underperforming ensemble members.

    Parameters
    ----------
    ensemble_size : int, default=10
        Number of base learners in the ensemble.

    Attributes
    ----------
    schema_ : capymoa.stream.Schema
        Stream schema derived from the training data.
    learner_ : capymoa.classifier.LeveragingBagging
        The fitted backing CapyMOA learner.

    Notes
    -----
    Requires the optional CapyMOA extra: ``pip install 'tuiml[capymoa]'``.

    See Also
    --------
    :class:`~tuiml.capymoa.ensemble.OnlineBagging` : Simpler, non-adaptive bagging.
    :class:`~tuiml.capymoa.ensemble.AdaptiveRandomForest` : Random-subspace
        drift-aware forest.

    Examples
    --------
    >>> from tuiml.capymoa import LeveragingBagging
    >>> from tuiml.datasets.generators import Hyperplane
    >>> data = Hyperplane(n_samples=1000, n_drift_features=2,
    ...                   random_state=42).generate()
    >>> model = LeveragingBagging(ensemble_size=5).fit(data.X, data.y)
    >>> model.predict(data.X[:5]).shape
    (5,)
    """

    def __init__(self, ensemble_size: int = 10):
        super().__init__()
        self.ensemble_size = ensemble_size

    def _build_learner(self, schema):
        """Construct the backing CapyMOA learner for ``schema``."""
        from capymoa.classifier import LeveragingBagging as _Learner
        return _Learner(schema=schema, ensemble_size=self.ensemble_size)

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Any]:
        """Return JSON Schema for constructor parameters."""
        return {"ensemble_size": {"type": "integer", "default": 10}}

    @classmethod
    def get_capabilities(cls) -> List[str]:
        """Return the capability names this wrapper supports."""
        return ["numeric", "streaming"]


@capymoa_regressor(tags=["ensemble", "drift"])
class AdaptiveRandomForestRegressor(_CapyMOAStreamMixin, Regressor):
    """**Adaptive Random Forest** drift-aware regressor (hub key ``capymoa.AdaptiveRandomForestRegressor``).

    Wraps :class:`capymoa.regressor.AdaptiveRandomForestRegressor`. The
    regression counterpart of the Adaptive Random Forest: an ensemble of
    FIMT-DD trees with online bagging, random feature subsets, and
    per-member drift detection.

    Parameters
    ----------
    ensemble_size : int, default=10
        Number of trees in the forest.

    Attributes
    ----------
    schema_ : capymoa.stream.Schema
        Stream schema derived from the training data.
    learner_ : capymoa.regressor.AdaptiveRandomForestRegressor
        The fitted backing CapyMOA learner.

    Notes
    -----
    Requires the optional CapyMOA extra: ``pip install 'tuiml[capymoa]'``.

    See Also
    --------
    :class:`~tuiml.capymoa.trees.FIMTDD` : Single-tree streaming regressor.
    :class:`~tuiml.capymoa.ensemble.AdaptiveRandomForest` : Classification
        counterpart.

    Examples
    --------
    >>> from tuiml.capymoa import AdaptiveRandomForestRegressor
    >>> from tuiml.datasets.generators import Friedman
    >>> data = Friedman(n_samples=1000, random_state=42).generate()
    >>> model = AdaptiveRandomForestRegressor(ensemble_size=5).fit(data.X, data.y)
    >>> model.predict(data.X[:5]).shape
    (5,)
    """

    def __init__(self, ensemble_size: int = 10):
        super().__init__()
        self.ensemble_size = ensemble_size

    def _build_learner(self, schema):
        """Construct the backing CapyMOA learner for ``schema``."""
        from capymoa.regressor import AdaptiveRandomForestRegressor as _Learner
        return _Learner(schema=schema, ensemble_size=self.ensemble_size)

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Any]:
        """Return JSON Schema for constructor parameters."""
        return {"ensemble_size": {"type": "integer", "default": 10}}

    @classmethod
    def get_capabilities(cls) -> List[str]:
        """Return the capability names this wrapper supports."""
        return ["numeric", "streaming"]


__all__ = [
    "AdaptiveRandomForest",
    "OnlineBagging",
    "LeveragingBagging",
    "AdaptiveRandomForestRegressor",
]
