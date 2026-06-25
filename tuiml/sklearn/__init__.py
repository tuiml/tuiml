"""scikit-learn bridge for TuiML.

Importing this package registers a curated set of scikit-learn estimators into
the TuiML hub under ``sklearn.<ClassName>`` keys (so they never collide with the
native TuiML algorithms), and exposes the generic :class:`SklearnAdapter` for
wrapping *any* scikit-learn-compatible estimator.

scikit-learn is an optional dependency. If it is not installed, importing this
package still succeeds (the wrapper classes are defined), but instantiating a
wrapper raises a clear ``ImportError`` pointing at ``pip install tuiml[sklearn]``.

Examples
--------
>>> from tuiml.sklearn import RandomForestClassifier   # sklearn-backed
>>> from tuiml.algorithms import RandomForestClassifier as NativeRF  # native
>>> import tuiml
>>> tuiml.train("sklearn.RandomForestClassifier", "iris", "class", cv=5)

>>> from sklearn.svm import SVC
>>> tuiml.train(SVC(), "iris", "class")   # any estimator, auto-wrapped
"""

from tuiml.sklearn.adapter import SklearnAdapter, wrap_sklearn, is_sklearn_estimator

# Importing these submodules triggers registration of the curated wrappers.
from tuiml.sklearn import algorithms, preprocessing, features  # noqa: E402,F401
from tuiml.sklearn.algorithms import *  # noqa: F401,F403,E402
from tuiml.sklearn.preprocessing import *  # noqa: F401,F403,E402
from tuiml.sklearn.features import *  # noqa: F401,F403,E402

__all__ = [
    "SklearnAdapter", "wrap_sklearn", "is_sklearn_estimator",
    *algorithms.__all__,
    *preprocessing.__all__,
    *features.__all__,
]
