"""Classic datasets, shipped with the library.

Real data available offline and by name, so an example, a benchmark or an
agent's first request needs no download and no file path. This is what makes
``{"source": "iris"}`` work anywhere TuiML takes a data spec.

Datasets
--------
- **Classification:** ``iris``, ``breast_cancer``, ``glass``, ``diabetes``
  and others.
- **Regression:** ``cpu``, ``airline``.
- **Other:** ``supermarket`` (association rule mining), ``reuters`` (text).

Layout
------
The ARFF files ship under ``builtin/data/``; ``catalog`` holds the metadata
table used to look a dataset up by name or by task.

Notes
-----
Loaders return a :class:`~tuiml.datasets.loaders.Dataset`, which also unpacks
as ``X, y`` — so both styles below are the same call.

Examples
--------
>>> from tuiml.datasets import load_iris, list_datasets
>>> data = load_iris()
>>> data.X.shape
(150, 4)
>>> X, y = load_iris()          # unpacks directly
>>> len(set(y.tolist()))
3
>>> "iris" in list_datasets("classification")
True
"""

from tuiml.datasets.builtin.catalog import (
    DATASET_REGISTRY,
    get_dataset_info,
    get_datasets_by_task,
    list_datasets,
    load_dataset,
)
from tuiml.datasets.builtin.classification import (
    load_breast_cancer,
    load_contact_lenses,
    load_credit,
    load_diabetes,
    load_glass,
    load_hypothyroid,
    load_ionosphere,
    load_iris,
    load_iris_2d,
    load_labor,
    load_segment,
    load_segment_test,
    load_soybean,
    load_unbalanced,
    load_vote,
    load_weather,
    load_weather_nominal,
)
from tuiml.datasets.builtin.regression import (
    load_airline,
    load_cpu,
    load_cpu_with_vendor,
)
from tuiml.datasets.builtin.other import (
    load_reuters_corn,
    load_reuters_grain,
    load_supermarket,
)

__all__ = [
    # LLM-friendly metadata
    "DATASET_REGISTRY",
    "get_dataset_info",
    "get_datasets_by_task",
    # Utilities
    "list_datasets",
    "load_dataset",
    # Classification
    "load_iris",
    "load_iris_2d",
    "load_diabetes",
    "load_breast_cancer",
    "load_glass",
    "load_ionosphere",
    "load_vote",
    "load_credit",
    "load_weather",
    "load_weather_nominal",
    "load_soybean",
    "load_labor",
    "load_contact_lenses",
    "load_hypothyroid",
    "load_segment",
    "load_segment_test",
    "load_unbalanced",
    # Regression
    "load_cpu",
    "load_cpu_with_vendor",
    "load_airline",
    # Other
    "load_supermarket",
    "load_reuters_corn",
    "load_reuters_grain",
]
