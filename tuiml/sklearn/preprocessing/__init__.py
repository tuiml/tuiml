"""scikit-learn preprocessing wrappers, mirroring TuiML's native layout."""


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
