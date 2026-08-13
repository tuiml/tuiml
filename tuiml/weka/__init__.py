"""Weka bridge for TuiML.

Importing this package registers the wrapped Weka learners into the TuiML hub
under ``weka.<ClassName>`` keys, so they never collide with the native TuiML
algorithms of the same name — ``"J48"`` is not a native TuiML key, but
``"RandomForest"`` would collide without the namespace.

The wrapper modules mirror **Weka's** own taxonomy rather than TuiML's, so a
learner sits where a Weka user expects to find it:

- :mod:`tuiml.weka.trees` — J48, REPTree, RandomTree, RandomForest, LMT, M5P.
- :mod:`tuiml.weka.rules` — ZeroR, OneR, JRip, PART, DecisionTable, M5Rules.
- :mod:`tuiml.weka.lazy` — IBk, KStar, LWL.
- :mod:`tuiml.weka.functions` — SMO, SMOreg, Logistic, LinearRegression, MLP.
- :mod:`tuiml.weka.bayes` — NaiveBayes, NaiveBayesMultinomial, BayesNet.
- :mod:`tuiml.weka.meta` — AdaBoostM1, Bagging, Vote, Stacking, and friends.
- :mod:`tuiml.weka.clusterers` — SimpleKMeans, EM, Canopy, Cobweb.

Installation
------------
Weka is an optional dependency, and it needs a **Java runtime** (11+) on
``PATH`` as well, because Weka runs on the JVM::

    java -version                   # must succeed first
    pip install 'tuiml[weka]'       # or: uv pip install 'tuiml[weka]'

If it is not installed, importing this package still succeeds (the wrapper
classes are defined and registered), but calling ``fit`` raises a clear
``ImportError`` pointing at the extra.

Usage
-----
The wrappers take and return numpy arrays like any other TuiML algorithm:

>>> from tuiml.weka import J48
>>> from tuiml.datasets import load_iris
>>> data = load_iris()
>>> clf = J48(confidence_factor=0.25).fit(data.X, data.y)
>>> clf.predict(data.X[:3]).shape
(3,)

They are also addressable by hub key, as ``"weka.<ClassName>"``, which is the
form :func:`tuiml.train` and an agent over MCP both use:

>>> import tuiml
>>> import tuiml.weka                          # registers the weka.* keys
>>> model = tuiml.train({
...     "model": {"name": "weka.J48"},
...     "data": {"source": "iris", "target": "class"},
...     "evaluation": {"test_size": 0.3},
... })
>>> "accuracy_score" in model.metrics_
True

Categorical attributes
----------------------
Weka distinguishes **nominal** from numeric attributes in its ARFF header, and
its tree and rule learners use genuinely different logic for the two. TuiML
passes plain numeric arrays, so declare which columns hold category codes:

>>> from tuiml.weka import J48
>>> clf = J48(nominal_features=[0, 3])      # columns 0 and 3 are categorical

Left undeclared, a categorical column is treated as a continuous scale, and
Weka will split it with ``<=`` tests that imply an order the codes do not have.

Reading the model
-----------------
Every wrapper exposes Weka's own model dump, which is often the reason to reach
for Weka rather than a native implementation:

>>> print(clf.fit(data.X, data.y).to_weka_string())   # doctest: +SKIP
J48 pruned tree
------------------
x4 <= 0.6: 0.0 (50.0)
...

The JVM
-------
The JVM starts lazily on the first ``fit`` and is **never stopped**, because
JPype cannot restart one inside the same process. To choose a heap size, call
:func:`~tuiml.weka._base.ensure_jvm` before the first fit:

>>> from tuiml.weka import ensure_jvm
>>> ensure_jvm(max_heap_size="4096m")     # doctest: +SKIP

See Also
--------
:mod:`tuiml.sklearn` : The same bridge pattern for scikit-learn estimators.
:mod:`tuiml.capymoa` : The same bridge pattern for CapyMOA streaming learners.
"""

from tuiml.weka._base import ensure_jvm, to_instances  # noqa: F401

# Importing these modules triggers registration of every wrapper.
from tuiml.weka import (  # noqa: F401
    bayes,
    clusterers,
    functions,
    lazy,
    meta,
    rules,
    trees,
)
from tuiml.weka.bayes import *  # noqa: F401,F403
from tuiml.weka.clusterers import *  # noqa: F401,F403
from tuiml.weka.functions import *  # noqa: F401,F403
from tuiml.weka.lazy import *  # noqa: F401,F403
from tuiml.weka.meta import *  # noqa: F401,F403
from tuiml.weka.rules import *  # noqa: F401,F403
from tuiml.weka.trees import *  # noqa: F401,F403

__all__ = [
    "ensure_jvm",
    "to_instances",
    *bayes.__all__,
    *clusterers.__all__,
    *functions.__all__,
    *lazy.__all__,
    *meta.__all__,
    *rules.__all__,
    *trees.__all__,
]
