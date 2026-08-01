"""Custom construction paths for the few estimators a spec cannot express.

A :class:`~tuiml.sklearn._spec.SklearnSpec` describes an estimator declaratively:
resolve a class, pass parameters, done. A small number of estimators need extra
assembly instead. Those name a function here via the spec's ``builder`` field.

Keep this module small, every entry is a documented exception to the
declarative rule, not a place to put ordinary configuration.
"""

from typing import Any, Dict

from tuiml.sklearn._spec import build_estimator


def svc_with_optional_calibration(
    target: str, params: Dict[str, Any], wrapper_name: str
) -> Any:
    """Build an ``SVC``, wrapping it for calibrated probabilities when asked.

    ``SVC(probability=True)`` is deprecated in scikit-learn 1.9 and removed in
    1.11; ``CalibratedClassifierCV`` is the documented replacement. Both perform
    cross-validated Platt scaling, so probabilities stay comparable. The
    ``probability`` parameter is kept on the wrapper because it selects between
    the two assemblies rather than being passed through.

    Parameters
    ----------
    target : str
        Estimator reference as ``"module:ClassName"``.
    params : dict
        Wrapper parameters. ``probability`` is consumed here, not forwarded.
    wrapper_name : str
        Wrapper class name, used in error messages.

    Returns
    -------
    estimator : object
        A bare ``SVC``, or a ``CalibratedClassifierCV`` wrapping one.
    """
    params = dict(params)
    probability = params.pop("probability", True)
    estimator = build_estimator(target, params, wrapper_name)
    if not probability:
        return estimator
    from sklearn.calibration import CalibratedClassifierCV

    return CalibratedClassifierCV(estimator, ensemble=False)


def iterative_imputer(
    target: str, params: Dict[str, Any], wrapper_name: str
) -> Any:
    """Build an ``IterativeImputer``, satisfying its experimental import guard.

    ``sklearn.impute.IterativeImputer`` raises on import until
    ``sklearn.experimental.enable_iterative_imputer`` has been imported, so the
    generic construction path cannot resolve it unaided.

    Parameters
    ----------
    target : str
        Estimator reference as ``"module:ClassName"``.
    params : dict
        Parameters to pass to the estimator constructor.
    wrapper_name : str
        Wrapper class name, used in error messages.

    Returns
    -------
    estimator : object
        A configured ``IterativeImputer``.
    """
    from sklearn.experimental import enable_iterative_imputer  # noqa: F401

    return build_estimator(target, params, wrapper_name)
