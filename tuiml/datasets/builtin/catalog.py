"""Catalog of the datasets bundled with TuiML.

Holds :data:`DATASET_REGISTRY`, the metadata table describing every built-in
dataset, plus the functions that read it (:func:`get_dataset_info`,
:func:`get_datasets_by_task`) and the name-based loaders that work for any
dataset in the catalog (:func:`list_datasets`, :func:`load_dataset`).

Each registry entry records the task type, shape, and a one-line description,
which makes the catalog directly usable as context for an LLM deciding which
dataset to reach for. The per-dataset convenience loaders (``load_iris`` and
friends) live in the sibling ``classification``, ``regression``, and ``other``
modules.

Examples
--------
>>> from tuiml.datasets import list_datasets, load_dataset, get_datasets_by_task
>>> list_datasets("regression")
['airline', 'cpu', 'cpu_with_vendor']
>>> data = load_dataset("iris")
>>> sorted(get_datasets_by_task("association"))
['supermarket']
"""

from typing import List, Optional

import re

from tuiml.datasets.builtin._paths import _CATEGORY_DIRS
from tuiml.datasets.loaders import Dataset, load_arff

# =============================================================================
# Dataset Registry - LLM-friendly metadata for all built-in datasets
# =============================================================================

DATASET_REGISTRY = {
    # Classification datasets
    "iris": {
        "task": "classification",
        "samples": 150,
        "features": 4,
        "classes": 3,
        "description": "Classic flower classification: setosa, versicolor, virginica",
        "loader": "load_iris"
    },
    "diabetes": {
        "task": "classification",
        "samples": 768,
        "features": 8,
        "classes": 2,
        "description": "Pima Indians diabetes prediction",
        "loader": "load_diabetes"
    },
    "breast_cancer": {
        "task": "classification",
        "samples": 286,
        "features": 9,
        "classes": 2,
        "description": "Breast cancer recurrence prediction",
        "loader": "load_breast_cancer"
    },
    "glass": {
        "task": "classification",
        "samples": 214,
        "features": 9,
        "classes": 7,
        "description": "Glass type identification from chemical properties",
        "loader": "load_glass"
    },
    "ionosphere": {
        "task": "classification",
        "samples": 351,
        "features": 34,
        "classes": 2,
        "description": "Radar signal classification (good/bad)",
        "loader": "load_ionosphere"
    },
    "vote": {
        "task": "classification",
        "samples": 435,
        "features": 16,
        "classes": 2,
        "description": "Congressional voting records (democrat/republican)",
        "loader": "load_vote"
    },
    "credit": {
        "task": "classification",
        "samples": 1000,
        "features": 20,
        "classes": 2,
        "description": "German credit risk assessment",
        "loader": "load_credit"
    },
    "soybean": {
        "task": "classification",
        "samples": 683,
        "features": 35,
        "classes": 19,
        "description": "Soybean disease classification",
        "loader": "load_soybean"
    },
    "labor": {
        "task": "classification",
        "samples": 57,
        "features": 16,
        "classes": 2,
        "description": "Labor relations negotiation outcomes",
        "loader": "load_labor"
    },
    "contact_lenses": {
        "task": "classification",
        "samples": 24,
        "features": 4,
        "classes": 3,
        "description": "Contact lens prescription recommendation",
        "loader": "load_contact_lenses"
    },
    "hypothyroid": {
        "task": "classification",
        "samples": 3772,
        "features": 29,
        "classes": 4,
        "description": "Hypothyroid disease diagnosis",
        "loader": "load_hypothyroid"
    },
    # Regression datasets
    "cpu": {
        "task": "regression",
        "samples": 209,
        "features": 6,
        "classes": None,
        "description": "CPU performance prediction",
        "loader": "load_cpu"
    },
    # Other datasets
}


#: Short names people commonly type for a bundled dataset. Values are the
#: ARFF file stem that :func:`load_dataset` resolves.
DATASET_ALIASES = {
    "credit": "credit-g",
    "breast": "breast-cancer",
    "lenses": "contact-lenses",
}


def _canonical(name: str) -> str:
    """Lower-case a dataset name and unify ``_``/``-``/space separators to ``-``.

    Parameters
    ----------
    name : str
        User-supplied dataset name.

    Returns
    -------
    str
        Normalised key, e.g. ``'Breast_Cancer'`` -> ``'breast-cancer'``.
    """
    key = name.strip().lower()
    if key.endswith(".arff"):
        key = key[:-5]
    return re.sub(r"[\s_]+", "-", key)


def resolve_dataset_name(name: str) -> str:
    """Map a loosely written dataset name to the bundled ARFF file stem.

    Accepts case, separator (``breast_cancer`` / ``breast-cancer`` /
    ``Breast Cancer``) and alias (``credit`` -> ``credit-g``) variants, so
    the same name works whether it came from :data:`DATASET_REGISTRY`
    (underscore keys), a file stem (hyphens) or an agent's guess.

    Parameters
    ----------
    name : str
        Dataset name in any of the accepted spellings.

    Returns
    -------
    str
        The file stem to load, e.g. ``'breast-cancer'``.

    Raises
    ------
    ValueError
        If nothing matches; the message lists close matches and every
        available name.

    Examples
    --------
    >>> from tuiml.datasets.builtin.catalog import resolve_dataset_name
    >>> resolve_dataset_name("breast_cancer")
    'breast-cancer'
    >>> resolve_dataset_name("credit")
    'credit-g'
    """
    available = list_datasets()
    by_key = {_canonical(stem): stem for stem in available}
    key = _canonical(name)
    if key in by_key:
        return by_key[key]
    if key in DATASET_ALIASES:
        return DATASET_ALIASES[key]
    import difflib
    close = difflib.get_close_matches(key, list(by_key), n=3, cutoff=0.6)
    close += [stem for k, stem in by_key.items() if key in k and stem not in close]
    hint = f" Did you mean: {', '.join(dict.fromkeys(close))}?" if close else ""
    raise ValueError(f"Dataset '{name}' not found.{hint} Available: {available}")


def _registry_key(name: str) -> str:
    """Return the :data:`DATASET_REGISTRY` key (underscore form) for ``name``.

    Parameters
    ----------
    name : str
        Dataset name in any accepted spelling.

    Returns
    -------
    str
        The registry key, or ``name`` unchanged when it has no registry entry.
    """
    try:
        stem = resolve_dataset_name(name)
    except ValueError:
        return name
    for key in DATASET_REGISTRY:
        if _canonical(key) == stem or DATASET_ALIASES.get(_canonical(key)) == stem:
            return key
    return name


def get_dataset_info(name: str = None) -> dict:
    """Get metadata about built-in datasets in an LLM-friendly format.

    Parameters
    ----------
    name : str or None, default=None
        The name of a specific dataset (e.g., "iris"). If None, metadata
        for all registered datasets will be returned.

    Returns
    -------
    dict
        A dictionary containing metadata such as task type, sample count,
        feature count, class count, and description.

    Raises
    ------
    ValueError
        If ``name`` is not a registered dataset.

    Examples
    --------
    >>> from tuiml.datasets import get_dataset_info
    >>> info = get_dataset_info("diabetes")
    >>> print(info["samples"])
    768
    """
    if name:
        key = _registry_key(name)
        if key not in DATASET_REGISTRY:
            raise ValueError(f"Unknown dataset: {name}. Available: {list(DATASET_REGISTRY.keys())}")
        return DATASET_REGISTRY[key]
    return DATASET_REGISTRY


def get_datasets_by_task(task: str) -> dict:
    """Get datasets filtered by task type (LLM-friendly).

    Parameters
    ----------
    task : str
        Task type: "classification", "regression", "association", "text_classification"

    Returns
    -------
    dict
        Filtered dataset registry.

    Examples
    --------
    >>> from tuiml.datasets import get_datasets_by_task
    >>> get_datasets_by_task("classification")
    >>> get_datasets_by_task("regression")
    """
    return {
        name: info for name, info in DATASET_REGISTRY.items()
        if info["task"] == task
    }


def list_datasets(category: Optional[str] = None) -> List[str]:
    """List names of all available built-in datasets.

    Parameters
    ----------
    category : str or None, default=None
        Optional filter to restrict results to a specific category:

        - ``"classification"``
        - ``"regression"``
        - ``"other"`` (Association, Text, etc.)

    Returns
    -------
    List[str]
        Alphabetical list of dataset names.

    Raises
    ------
    ValueError
        If ``category`` is not one of the three known categories.

    Examples
    --------
    >>> from tuiml.datasets import list_datasets
    >>> available = list_datasets("regression")
    >>> print(available)
    ['airline', 'cpu', 'cpu_with_vendor']
    """
    if category:
        if category not in _CATEGORY_DIRS:
            raise ValueError(f"Unknown category: {category}. Use: {list(_CATEGORY_DIRS.keys())}")
        return sorted([f.stem for f in _CATEGORY_DIRS[category].glob("*.arff")])

    # Return all
    all_datasets = []
    for d in _CATEGORY_DIRS.values():
        all_datasets.extend([f.stem for f in d.glob("*.arff")])
    return sorted(set(all_datasets))


def load_dataset(name: str) -> Dataset:
    """Load a built-in dataset by its registry name.

    Automatically identifies the correct file path and uses the ARFF loader
    to return a standardized Dataset object.

    Parameters
    ----------
    name : str
        The name of the dataset to load (e.g., ``'iris'``, ``'diabetes'``,
        ``'cpu'``).

    Returns
    -------
    Dataset
        Standardized dataset object containing the data and metadata.

    Raises
    ------
    ValueError
        If no bundled dataset matches ``name``.

    Examples
    --------
    >>> from tuiml.datasets import load_dataset
    >>> iris = load_dataset('iris')
    >>> X, y = iris
    """
    # Exact file name first (``iris.arff``), then the normalised stem so
    # ``breast_cancer``, ``Breast-Cancer`` and ``credit`` all resolve.
    for category_dir in _CATEGORY_DIRS.values():
        path = category_dir / name
        if path.exists():
            return load_arff(path)

    stem = resolve_dataset_name(name)
    for category_dir in _CATEGORY_DIRS.values():
        path = category_dir / f"{stem}.arff"
        if path.exists():
            return load_arff(path)

    raise ValueError(f"Dataset '{name}' not found. Available: {list_datasets()}")


__all__ = [
    "DATASET_REGISTRY",
    "get_dataset_info",
    "get_datasets_by_task",
    "list_datasets",
    "load_dataset",
    "resolve_dataset_name",
    "DATASET_ALIASES",
]
