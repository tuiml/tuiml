"""Shared helpers for the declarative spec convention.

Every component in a spec uses one shape, ``{"name": ..., "params": {...}}``,
and every data source resolves to ``(source, target, features)``. These
helpers implement that contract for :mod:`tuiml.training` and
:mod:`tuiml.benchmarking`.
"""

from typing import Any

def _reject_foreign_estimator(model: Any) -> None:
    """Raise if ``model`` is a foreign estimator object rather than a TuiML one.

    TuiML is native-first: algorithms are addressed by registry name, and
    external libraries are reached through curated wrappers registered under a
    namespaced key (``sklearn.SVC``, ``capymoa.HoeffdingTree``). Passing a raw
    third-party estimator *instance* is not supported, so this fails early with
    a message naming the wrapper to use instead of failing obscurely later.

    Parameters
    ----------
    model : Any
        The value passed as the ``model`` / algorithm argument.

    Raises
    ------
    TypeError
        If ``model`` looks like a non-TuiML estimator object.
    """
    from tuiml.base.algorithms import Algorithm

    if isinstance(model, (str, dict, type)) or isinstance(model, Algorithm):
        return
    # Duck-typed foreign estimator: has fit(), but is not a TuiML Algorithm.
    if not callable(getattr(model, "fit", None)):
        return

    cls_name = type(model).__name__
    from tuiml.registry import registry

    candidates = [
        name for name in registry.list_names()
        if "." in name and name.split(".", 1)[1] == cls_name
    ]
    if candidates:
        hint = (
            f"Use the registered wrapper instead:  "
            f'tuiml.train("{candidates[0]}", ...)'
        )
    else:
        hint = (
            f"No wrapper is registered for {cls_name}. Use a native TuiML "
            f"algorithm (tuiml.list_algorithms()), or add a wrapper under "
            f"tuiml/sklearn/, tuiml/capymoa/ or tuiml/weka/."
        )
    raise TypeError(
        f"{cls_name} is not a TuiML algorithm. TuiML addresses algorithms by "
        f"registry name; raw third-party estimator objects are not accepted. "
        f"{hint}"
    )


def _resolve_data_spec(data, target, features):
    """Normalize the ``data`` argument into ``(source, target, features)``.

    The ``data`` argument may be a **data spec** dict or a bare source. A spec
    dict groups everything about the data in one place:

    - ``{"source": "sales.csv", "target": "label", "features": [...]}``: a file
      path or builtin name plus its target column (and optional feature subset).
    - ``{"X": X_array, "y": y_array}``: in-memory arrays, already split.

    Anything that is not such a dict (a path string, builtin name, ``DataFrame``,
    or ``ndarray``) is returned unchanged, paired with the separately supplied
    ``target``/``features`` arguments.

    Parameters
    ----------
    data : str, DataFrame, ndarray, or dict
        The raw ``data`` argument passed to :func:`train`.
    target : str, ndarray, or None
        The separately supplied target (used only for bare-``data`` forms).
    features : list of str or None
        The separately supplied feature subset.

    Returns
    -------
    source : object
        The data source to hand to :class:`~tuiml.workflow.Workflow`.
    target : object
        The resolved target column name / array (or ``None``).
    features : list of str or None
        The resolved feature subset.
    """
    if not isinstance(data, dict):
        return data, target, features

    spec = dict(data)
    features = spec.get("features", features)

    # In-memory arrays, already split.
    if "X" in spec:
        return spec["X"], spec.get("y", target), features

    # File path / builtin name / DataFrame reference.
    source = spec.get("source")
    if source is None and ("path" in spec or "data" in spec):
        raise ValueError(
            'Data specs use the key "source" for the file path or builtin '
            'name, e.g. {"source": "sales.csv", "target": "label"}.'
        )
    target = spec.get("target", target)
    return source, target, features
