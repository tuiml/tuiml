"""Curated CapyMOA streaming-learner wrappers.

Registered under ``capymoa.<ClassName>`` hub keys. These wrap genuinely
streaming (incremental) learners backed by CapyMOA/MOA, distinct from TuiML's
own native streaming algorithms in ``tuiml.algorithms.streaming``.
"""

from typing import Any, Dict, List

from tuiml.base.algorithms import Classifier, Regressor
from tuiml.capymoa._base import (
    _CapyMOAStreamMixin,
    capymoa_classifier,
    capymoa_regressor,
)


@capymoa_classifier(tags=["bayesian"])
class NaiveBayes(_CapyMOAStreamMixin, Classifier):
    """CapyMOA incremental Naive Bayes classifier."""

    def __init__(self):
        super().__init__()

    def _build_learner(self, schema):
        from capymoa.classifier import NaiveBayes as _Learner
        return _Learner(schema=schema)

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Any]:
        return {}

    @classmethod
    def get_capabilities(cls) -> List[str]:
        return ["numeric", "streaming"]


@capymoa_classifier(tags=["tree", "hoeffding"])
class HoeffdingTree(_CapyMOAStreamMixin, Classifier):
    """CapyMOA Hoeffding Tree (VFDT) classifier."""

    def __init__(self, grace_period: int = 200):
        super().__init__()
        self.grace_period = grace_period

    def _build_learner(self, schema):
        from capymoa.classifier import HoeffdingTree as _Learner
        return _Learner(schema=schema, grace_period=self.grace_period)

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Any]:
        return {"grace_period": {"type": "integer", "default": 200}}

    @classmethod
    def get_capabilities(cls) -> List[str]:
        return ["numeric", "streaming"]


@capymoa_classifier(tags=["ensemble", "drift"])
class AdaptiveRandomForest(_CapyMOAStreamMixin, Classifier):
    """CapyMOA Adaptive Random Forest classifier (concept-drift aware)."""

    def __init__(self, ensemble_size: int = 10):
        super().__init__()
        self.ensemble_size = ensemble_size

    def _build_learner(self, schema):
        from capymoa.classifier import AdaptiveRandomForestClassifier as _Learner
        return _Learner(schema=schema, ensemble_size=self.ensemble_size)

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Any]:
        return {"ensemble_size": {"type": "integer", "default": 10}}

    @classmethod
    def get_capabilities(cls) -> List[str]:
        return ["numeric", "streaming"]


@capymoa_regressor(tags=["tree", "drift"])
class FIMTDD(_CapyMOAStreamMixin, Regressor):
    """CapyMOA FIMT-DD streaming regression tree."""

    def __init__(self):
        super().__init__()

    def _build_learner(self, schema):
        from capymoa.regressor import FIMTDD as _Learner
        return _Learner(schema=schema)

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Any]:
        return {}

    @classmethod
    def get_capabilities(cls) -> List[str]:
        return ["numeric", "streaming"]


__all__ = ["NaiveBayes", "HoeffdingTree", "AdaptiveRandomForest", "FIMTDD"]
