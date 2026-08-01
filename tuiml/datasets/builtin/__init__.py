"""
Built-in datasets for TuiML.

Provides easy access to classic ML datasets organized by task type:

- ``classification``: iris, diabetes, breast_cancer, glass, etc.
- ``regression``: cpu, airline
- ``other``: supermarket (association), reuters (text)

The ARFF files themselves ship under ``builtin/data/``, and ``catalog``
carries the metadata table used to look datasets up by name or by task.

Usage:
    >>> from tuiml.datasets import load_iris
    >>> data = load_iris()
    >>> X, y = load_iris()  # Can unpack

    >>> from tuiml.datasets import list_datasets
    >>> list_datasets()  # All datasets
    >>> list_datasets("classification")  # Only classification
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
