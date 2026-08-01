"""CapyMOA bayesian wrappers.

Incremental Bayesian learners backed by CapyMOA/MOA. Registered under
``capymoa.<ClassName>`` hub keys, mirroring the native TuiML ``bayesian``
family.
"""

from typing import Any, Dict, List

from tuiml.base.algorithms import Classifier
from tuiml.capymoa._base import _CapyMOAStreamMixin, capymoa_classifier


@capymoa_classifier(tags=["bayesian"])
class NaiveBayes(_CapyMOAStreamMixin, Classifier):
    """**Incremental Naive Bayes** classifier (hub key ``capymoa.NaiveBayes``).

    Wraps :class:`capymoa.classifier.NaiveBayes`. Maintains per-class
    feature statistics that are updated one instance at a time, so the
    model can be trained on data that does not fit in memory and keeps
    learning as new instances arrive.

    Attributes
    ----------
    schema_ : capymoa.stream.Schema
        Stream schema derived from the training data.
    learner_ : capymoa.classifier.NaiveBayes
        The fitted backing CapyMOA learner.

    Notes
    -----
    Requires the optional CapyMOA extra: ``pip install 'tuiml[capymoa]'``.
    A good first baseline for streaming classification: fast, single-pass,
    and surprisingly robust under gradual concept drift.

    See Also
    --------
    :class:`~tuiml.capymoa.trees.HoeffdingTree` : Incremental decision tree.
    :class:`~tuiml.algorithms.bayesian.naive_bayes.NaiveBayes` : Native batch
        TuiML implementation.

    Examples
    --------
    >>> from tuiml.capymoa import NaiveBayes
    >>> from tuiml.datasets.generators import Agrawal
    >>> data = Agrawal(n_samples=1000, function=1, random_state=42).generate()
    >>> model = NaiveBayes().fit(data.X, data.y)
    >>> model.predict(data.X[:5]).shape
    (5,)
    """

    def __init__(self):
        super().__init__()

    def _build_learner(self, schema):
        """Construct the backing CapyMOA learner for ``schema``."""
        from capymoa.classifier import NaiveBayes as _Learner
        return _Learner(schema=schema)

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Any]:
        """Return JSON Schema for constructor parameters."""
        return {}

    @classmethod
    def get_capabilities(cls) -> List[str]:
        """Return the capability names this wrapper supports."""
        return ["numeric", "streaming"]


__all__ = ["NaiveBayes"]
