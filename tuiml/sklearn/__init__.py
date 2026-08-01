"""scikit-learn bridge for TuiML.

Importing this package registers the wrapped scikit-learn estimators into the
TuiML hub under ``sklearn.<ClassName>`` keys, so they never collide with the
native TuiML algorithms of the same name.

The wrapper modules mirror TuiML's own taxonomy, ``linear``, ``svm``,
``trees``, ``ensemble``, ``bayesian``, ``neighbors``, ``neural``,
``clustering``, ``anomaly``, plus ``preprocessing/`` and ``features/``: so a
wrapped estimator sits in the same place as its native counterpart. The
wrapper classes are driven by the spec rows in :mod:`tuiml.sklearn.specs`;
keep a module's classes and its spec rows in sync when editing either.

TuiML is native-first: its own algorithms are the primary implementations, and
estimators are addressed by their namespaced hub key.

Installation
------------
scikit-learn is an optional dependency::

    pip install 'tuiml[sklearn]'      # or: uv pip install 'tuiml[sklearn]'

If it is not installed, importing this package still succeeds (the wrapper
classes are defined), but instantiating a wrapper raises a clear
``ImportError`` pointing at ``pip install tuiml[sklearn]``.

Usage
-----
There are two ways to reach a wrapped estimator, and they are equivalent.

**1. Import the class** and use it like any TuiML algorithm, ``fit`` /
``predict`` / ``predict_proba`` for models, ``fit_transform`` for
transformers. Constructor keywords are passed straight through to
scikit-learn:

>>> from tuiml.datasets import load_iris
>>> from tuiml.sklearn import RandomForestClassifier
>>> data = load_iris()
>>> model = RandomForestClassifier(n_estimators=50, random_state=0)
>>> model.fit(data.X, data.y).predict(data.X[:5]).tolist()
[0, 0, 0, 0, 0]
>>> model.predict_proba(data.X[:1]).shape
(1, 3)

**2. Name it in a spec**, as ``"sklearn.<ClassName>"``. This is the form
:func:`tuiml.train` takes, and the form an agent uses over MCP. Wrappers can
appear as the model or as pipeline steps, mixed freely with native components:

>>> import tuiml
>>> import tuiml.sklearn                       # registers the sklearn.* keys
>>> model = tuiml.train({
...     "model": {"name": "sklearn.LogisticRegression",
...               "params": {"max_iter": 500}},
...     "data": {"source": "iris", "target": "class"},
...     "pipeline": [{"name": "sklearn.StandardScaler"},
...                  {"name": "sklearn.SelectKBest", "params": {"k": 3}}],
...     "evaluation": {"cv": 5, "metrics": ["accuracy_score"]},
... })
>>> round(model.metrics_["cv_accuracy_score_mean"], 2)
0.96

Names collide on purpose: ``sklearn.RandomForestClassifier`` and the native
``RandomForestClassifier`` are different algorithms, and the prefix is what
picks one. Benchmarking the two against each other is the point:

>>> from tuiml import Benchmark
>>> import tuiml.sklearn
>>> bench = Benchmark(
...     models=[{"name": "RandomForestClassifier"},
...             {"name": "sklearn.RandomForestClassifier"}],
...     datasets=[{"source": "iris"}],
...     random_seed=0,
... )
>>> sorted(bench.run().scores_["model"].unique())
['RandomForestClassifier', 'sklearn.RandomForestClassifier']

Discovering what is available
-----------------------------
Every wrapper reports its own parameters, derived from the scikit-learn
actually installed, so the list is never stale:

>>> from tuiml.sklearn import SVC
>>> sorted(SVC.get_parameter_schema())[:4]
['C', 'break_ties', 'cache_size', 'class_weight']

A parameter scikit-learn does not accept is rejected at ``fit`` time with the
valid names listed, rather than being silently ignored.

Notes
-----
Raw scikit-learn estimator *objects* are not accepted by :func:`tuiml.train`
or :class:`tuiml.Benchmark`: to reach an estimator that is not yet wrapped,
add a row to :mod:`tuiml.sklearn.specs` rather than passing an instance.

A handful of wrappers are renamed to avoid clashing with a native TuiML class
of the same name, e.g. scikit-learn's ``PCA`` is exported as ``PCAExtractor``.
``dir(tuiml.sklearn)`` lists the exported names.

See Also
--------
:mod:`tuiml.capymoa` : The same bridge pattern for CapyMOA streaming learners.
:mod:`tuiml.sklearn.specs` : The declarative table driving every wrapper.
"""

# Importing these modules triggers registration of every wrapper.
from tuiml.sklearn import (  # noqa: F401
    anomaly,
    bayesian,
    clustering,
    ensemble,
    features,
    linear,
    neighbors,
    neural,
    preprocessing,
    svm,
    trees,
)
from tuiml.sklearn.anomaly import *  # noqa: F401,F403
from tuiml.sklearn.bayesian import *  # noqa: F401,F403
from tuiml.sklearn.clustering import *  # noqa: F401,F403
from tuiml.sklearn.ensemble import *  # noqa: F401,F403
from tuiml.sklearn.features import *  # noqa: F401,F403
from tuiml.sklearn.linear import *  # noqa: F401,F403
from tuiml.sklearn.neighbors import *  # noqa: F401,F403
from tuiml.sklearn.neural import *  # noqa: F401,F403
from tuiml.sklearn.preprocessing import *  # noqa: F401,F403
from tuiml.sklearn.svm import *  # noqa: F401,F403
from tuiml.sklearn.trees import *  # noqa: F401,F403

__all__ = [
    *anomaly.__all__,
    *bayesian.__all__,
    *clustering.__all__,
    *ensemble.__all__,
    *features.__all__,
    *linear.__all__,
    *neighbors.__all__,
    *neural.__all__,
    *preprocessing.__all__,
    *svm.__all__,
    *trees.__all__,
]
