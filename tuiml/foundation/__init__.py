"""Pretrained foundation model bridge for TuiML.

Importing this package registers the wrapped foundation models into the TuiML
hub under ``foundation.<ClassName>`` keys, so they never collide with the
native TuiML algorithms. Two kinds live here: **tabular** models
(:class:`TabICLClassifier`, :class:`TabICLRegressor`) that learn in-context
from your training rows, and **time series** models
(:class:`ChronosForecaster`, :class:`TimesFMForecaster`) that forecast a
series zero-shot from its own history.

A **pretrained foundation model** is a different animal from every other
model in the library. The native learners -- including the deep ones in
:mod:`tuiml.algorithms.tabular_deep` and
:mod:`tuiml.algorithms.timeseries.deep` -- are TuiML's own code and train
from scratch on your data: ``fit`` runs gradient steps or grows trees, and
the resulting parameters *are* the model. A foundation model arrives already
trained. Its weights are a **pretrained checkpoint**, downloaded from a model
hub and run in a single forward pass with no gradient step at all. Nothing is
fitted here: ``fit`` only memorises the data so it can be handed to the model
as *context* at predict time -- the training rows for a tabular model, the
series' history for a forecaster -- which is why these models are fast to
"train" and why their cost lands almost entirely in :meth:`predict`.

That difference is what the namespace records. A key like
``foundation.TabICLClassifier`` is an honest signal that TuiML is delegating
to someone else's artifact rather than running its own implementation --
exactly the convention :mod:`tuiml.sklearn` uses for ``sklearn.SVC`` and
:mod:`tuiml.weka` for ``weka.J48``. Native algorithms keep their bare names.

Installation
------------
The backing packages are an optional dependency::

    pip install 'tuiml[foundation]'   # or: uv pip install 'tuiml[foundation]'

This also pulls in PyTorch. If it is not installed, importing this package
still succeeds and the wrapper classes remain fully introspectable -- the
algorithm catalog, the parameter schemas and pickling are identical on every
install. The dependency is checked in ``fit``, which raises a clear
``ImportError`` naming the exact install command.

Licensing
---------
TuiML is BSD-3-Clause and **never ships or mirrors model weights**. The
upstream package fetches its own checkpoint, so the download is a direct
transaction between the user and whoever publishes it, under that publisher's
license.

This matters because a permissive wrapper does not relicense what it wraps: a
checkpoint restricted to non-commercial use stays restricted no matter how
TuiML is licensed. Only models whose **weights** carry a license compatible
with TuiML's own are integrated here:

- **TabICL** -- BSD-3-Clause for code *and* weights.
- **Chronos** (Bolt and Chronos-2) -- Apache-2.0 for code and weights.
- **TimesFM 2.5** -- Apache-2.0 for code and weights.

Models whose weights are non-commercial -- TabPFN, Google's TabFM, Salesforce's
Moirai -- are deliberately left out. Adding one would require a consent gate,
which does not exist yet.

Usage
-----
There are two ways to reach a foundation model, and they are equivalent.

**1. Import the class** and use it like any TuiML algorithm. Constructing and
inspecting it needs no optional dependency:

>>> from tuiml.foundation import TabICLClassifier
>>> clf = TabICLClassifier(n_estimators=2)
>>> clf.n_estimators
2
>>> "softmax_temperature" in clf.get_parameter_schema()
True

Fitting downloads the checkpoint on first use (~150 MB, cached under
``~/.cache/huggingface``) and then predicts in a single forward pass:

>>> from tuiml.datasets import load_iris  # doctest: +SKIP
>>> data = load_iris()  # doctest: +SKIP
>>> clf.fit(data.X, data.y).predict(data.X[:5]).tolist()  # doctest: +SKIP
[0, 0, 0, 0, 0]

**2. Name it in a spec**, as ``"foundation.<ClassName>"``. This is the form
:func:`tuiml.train` takes, and the form an agent uses over MCP. The prefix is
part of the name:

>>> import tuiml  # doctest: +SKIP
>>> import tuiml.foundation                    # registers the foundation.* keys
>>> model = tuiml.train({  # doctest: +SKIP
...     "model": {"name": "foundation.TabICLClassifier",
...               "params": {"n_estimators": 4}},
...     "data": {"source": "iris", "target": "class"},
...     "evaluation": {"test_size": 0.3, "metrics": ["accuracy_score"]},
... })
>>> round(model.metrics_["accuracy_score"], 2)  # doctest: +SKIP
0.98

The registration is what makes the key resolvable:

>>> import tuiml.foundation
>>> from tuiml.registry import registry
>>> sorted(k for k in registry.list_names() if k.startswith("foundation."))  # doctest: +NORMALIZE_WHITESPACE
['foundation.ChronosForecaster', 'foundation.TabICLClassifier',
 'foundation.TabICLRegressor', 'foundation.TimesFMForecaster']

The forecasters follow the native forecaster API -- ``fit(y)`` then
``predict(steps)`` -- and add ``predict_quantiles(steps)``:

>>> from tuiml.foundation import ChronosForecaster
>>> ChronosForecaster().checkpoint
'amazon/chronos-bolt-small'

See Also
--------
:mod:`tuiml.algorithms.tabular_deep` : Native deep tabular architectures
    (FT-Transformer, SAINT, NODE) trained from scratch on your data.
:mod:`tuiml.algorithms.timeseries` : Native forecasters, classical and deep
    (Theta, SARIMAX, N-BEATS, PatchTST), trained on your series.
:mod:`tuiml.sklearn` : The same bridge pattern for scikit-learn estimators.
:mod:`tuiml.weka` : The same bridge pattern for Weka (JVM) learners.
"""

# Importing this module triggers registration of every wrapper.
from tuiml.foundation.chronos import ChronosForecaster
from tuiml.foundation.tabicl import TabICLClassifier, TabICLRegressor
from tuiml.foundation.timesfm import TimesFMForecaster

__all__ = [
    "ChronosForecaster",
    "TabICLClassifier",
    "TabICLRegressor",
    "TimesFMForecaster",
]
