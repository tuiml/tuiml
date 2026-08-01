"""scikit-learn preprocessing wrappers, mirroring TuiML's native layout.

The same split as :mod:`tuiml.preprocessing`, so a wrapped transformer sits
where its native counterpart does. Registered under ``sklearn.<ClassName>``
hub keys, and usable as pipeline steps alongside native components.

Modules
-------
- :mod:`~tuiml.sklearn.preprocessing.scaling` — ``StandardScaler``,
  ``MinMaxScaler``, ``MaxAbsScaler``, ``RobustScaler``,
  ``QuantileTransformer``, ``PowerTransformer``, ``Normalizer``,
  ``KBinsDiscretizer``, ``PolynomialFeatures``, ``SplineTransformer``, and
  the encoders (``OneHotEncoder``, ``OrdinalEncoder``, ``TargetEncoder``).
- :mod:`~tuiml.sklearn.preprocessing.imputation` — ``SimpleImputer``,
  ``KNNImputer``, ``IterativeImputer``, ``MissingIndicator``.
- :mod:`~tuiml.sklearn.preprocessing.text` — ``CountVectorizer``,
  ``TfidfVectorizer``, ``TfidfTransformer``, ``HashingVectorizer``.

See Also
--------
:mod:`tuiml.sklearn` : Installing the extra, and how to use the wrappers.
:mod:`tuiml.preprocessing` : The native equivalents.
"""


from tuiml.sklearn.preprocessing import imputation  # noqa: F401
from tuiml.sklearn.preprocessing import scaling  # noqa: F401
from tuiml.sklearn.preprocessing import text  # noqa: F401
from tuiml.sklearn.preprocessing.imputation import *  # noqa: F401,F403
from tuiml.sklearn.preprocessing.scaling import *  # noqa: F401,F403
from tuiml.sklearn.preprocessing.text import *  # noqa: F401,F403

__all__ = [
    *imputation.__all__,
    *scaling.__all__,
    *text.__all__,
]
