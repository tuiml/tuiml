"""scikit-learn bridge for TuiML.

Importing this package registers the wrapped scikit-learn estimators into the
TuiML hub under ``sklearn.<ClassName>`` keys, so they never collide with the
native TuiML algorithms of the same name.

The wrapper modules mirror TuiML's own taxonomy — ``linear``, ``svm``,
``trees``, ``ensemble``, ``bayesian``, ``neighbors``, ``neural``,
``clustering``, ``anomaly``, plus ``preprocessing/`` and ``features/`` — so a
wrapped estimator sits in the same place as its native counterpart. Those
modules are **generated** from :mod:`tuiml.sklearn.specs` by
``scripts/generate_sklearn_wrappers.py``; edit the spec table, not the modules.

TuiML is native-first: its own algorithms are the primary implementations, and
estimators are addressed by their namespaced hub key. Raw scikit-learn estimator
*objects* are not accepted by ``tuiml.train`` / ``tuiml.experiment`` — to reach
an estimator that is not yet wrapped, add a row to
:mod:`tuiml.sklearn.specs` rather than passing an instance.

scikit-learn is an optional dependency. If it is not installed, importing this
package still succeeds (the wrapper classes are defined), but instantiating a
wrapper raises a clear ``ImportError`` pointing at ``pip install tuiml[sklearn]``.

Examples
--------
>>> import tuiml
>>> tuiml.train("sklearn.RandomForestClassifier", {"source": "iris", "target": "class"}, cv=5)
>>> tuiml.train("sklearn.SVC", {"source": "iris", "target": "class"}, C=2.0)

>>> from tuiml.sklearn import Ridge          # sklearn-backed
>>> from tuiml.algorithms import LinearRegression   # native
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
