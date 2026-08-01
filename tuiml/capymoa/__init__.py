"""CapyMOA (streaming / online-learning) bridge for TuiML.

Importing this package registers the wrapped CapyMOA incremental learners into
the TuiML hub under ``capymoa.<ClassName>`` keys, so they never collide with
the native TuiML streaming algorithms in :mod:`tuiml.algorithms.streaming`.

The wrapper modules mirror TuiML's own taxonomy — ``bayesian``, ``trees``,
``ensemble`` — so a wrapped learner sits in the same place as its native
counterpart:

- :mod:`tuiml.capymoa.bayesian` — incremental Naive Bayes.
- :mod:`tuiml.capymoa.trees` — Hoeffding / EFDT trees.
- :mod:`tuiml.capymoa.ensemble` — online bagging, adaptive random forests.

CapyMOA is an optional dependency (it pulls in a JVM-backed MOA runtime). If it
is not installed, importing this package still succeeds (the wrapper classes are
defined), but instantiating a wrapper raises a clear ``ImportError`` pointing
at ``pip install 'tuiml[capymoa]'``.

Examples
--------
>>> from tuiml.capymoa import HoeffdingTree, NaiveBayes
>>> from tuiml.datasets.generators import Agrawal
>>> data = Agrawal(n_samples=500, function=1, random_state=42).generate()
>>> # Incremental training across multiple batches
>>> model = HoeffdingTree(grace_period=100).partial_fit(data.X[:300], data.y[:300])
>>> model.partial_fit(data.X[300:], data.y[300:])
>>> model.predict(data.X[:3]).shape
(3,)

>>> tree = NaiveBayes().partial_fit(data.X[:300], data.y[:300])
>>> tree.partial_fit(data.X[300:], data.y[300:])
>>> tree.predict(data.X[:3]).shape
(3,)
"""

# Importing these modules triggers registration of every wrapper.
from tuiml.capymoa import bayesian, ensemble, trees  # noqa: F401
from tuiml.capymoa.bayesian import *  # noqa: F401,F403
from tuiml.capymoa.ensemble import *  # noqa: F401,F403
from tuiml.capymoa.trees import *  # noqa: F401,F403

__all__ = [
    *bayesian.__all__,
    *ensemble.__all__,
    *trees.__all__,
]
