"""CapyMOA (streaming / online-learning) bridge for TuiML.

Importing this package registers the wrapped CapyMOA incremental learners into
the TuiML hub under ``capymoa.<ClassName>`` keys, so they never collide with
the native TuiML algorithms in :mod:`tuiml.algorithms` — ``"NaiveBayes"`` is a
CapyMOA-only key, but ``"AdaptiveRandomForest"`` would collide without the
namespace.

The wrapper modules mirror TuiML's own taxonomy — ``bayesian``, ``trees``,
``ensemble`` — so a wrapped learner sits in the same place as its native
counterpart:

- :mod:`tuiml.capymoa.bayesian` — incremental Naive Bayes.
- :mod:`tuiml.capymoa.trees` — Hoeffding / EFDT trees.
- :mod:`tuiml.capymoa.ensemble` — online bagging, adaptive random forests.

Installation
------------
CapyMOA is an optional dependency, and it needs a **Java runtime** (11+) on
``PATH`` as well, because MOA runs on the JVM::

    java -version                      # must succeed first
    pip install 'tuiml[capymoa]'       # or: uv pip install 'tuiml[capymoa]'

If CapyMOA is not installed, importing this package still succeeds (the wrapper
classes are defined), but instantiating a wrapper raises a clear
``ImportError`` pointing at ``pip install 'tuiml[capymoa]'``.

Usage
-----
Streaming learners update one instance at a time. Use ``partial_fit`` to feed
successive batches; it returns ``self``, so calls chain. ``fit`` is also
available and simply streams the whole array in one pass:

>>> from tuiml.capymoa import HoeffdingTree, NaiveBayes
>>> from tuiml.datasets.generators import Agrawal
>>> data = Agrawal(n_samples=500, function=1, random_state=42).generate()
>>> # Incremental training across multiple batches
>>> model = HoeffdingTree(grace_period=100).partial_fit(data.X[:300], data.y[:300])
>>> model = model.partial_fit(data.X[300:], data.y[300:])
>>> model.predict(data.X[:3]).shape
(3,)

>>> tree = NaiveBayes().partial_fit(data.X[:300], data.y[:300])
>>> tree = tree.partial_fit(data.X[300:], data.y[300:])
>>> tree.predict(data.X[:3]).shape
(3,)

They are also addressable by hub key, as ``"capymoa.<ClassName>"``, which is
the form :func:`tuiml.train` and an agent over MCP both use:

>>> import tuiml
>>> import tuiml.capymoa                     # registers the capymoa.* keys
>>> model = tuiml.train({
...     "model": {"name": "capymoa.HoeffdingTree"},
...     "data": {"source": "iris", "target": "class"},
...     "evaluation": {"test_size": 0.3},
... })
>>> "accuracy_score" in model.metrics_
True

See Also
--------
:mod:`tuiml.sklearn` : The same bridge pattern for scikit-learn estimators.
:mod:`tuiml.weka` : The same bridge pattern for Weka (JVM) learners.
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
