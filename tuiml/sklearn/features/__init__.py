"""scikit-learn feature wrappers, mirroring TuiML's native layout.

The same split as :mod:`tuiml.features`, so a wrapped transformer sits where
its native counterpart does. Registered under ``sklearn.<ClassName>`` hub
keys, and usable as pipeline steps alongside native components.

Modules
-------
- :mod:`~tuiml.sklearn.features.selection` — ``SelectKBest``,
  ``SelectPercentile``, ``SelectFdr``, ``SelectFpr``, ``SelectFwe``,
  ``GenericUnivariateSelect``, ``VarianceThreshold``.
- :mod:`~tuiml.sklearn.features.extraction` — ``PCAExtractor``,
  ``KernelPCA``, ``TruncatedSVD``, ``FastICA``, ``NMF``, ``Isomap``,
  ``LocallyLinearEmbedding``, and the random-projection and
  kernel-approximation transformers.

Notes
-----
scikit-learn's ``PCA`` is exported as ``PCAExtractor``, to leave the name free
for the native class.

See Also
--------
:mod:`tuiml.sklearn` : Installing the extra, and how to use the wrappers.
:mod:`tuiml.features` : The native equivalents.
"""


from tuiml.sklearn.features import extraction  # noqa: F401
from tuiml.sklearn.features import selection  # noqa: F401
from tuiml.sklearn.features.extraction import *  # noqa: F401,F403
from tuiml.sklearn.features.selection import *  # noqa: F401,F403

__all__ = [
    *extraction.__all__,
    *selection.__all__,
]
