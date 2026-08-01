"""Convenience loaders for the bundled regression datasets.

Each function reads one ARFF file that ships with TuiML and returns a
:class:`~tuiml.datasets.loaders.arff.Dataset` whose target is continuous, so
they pair with any :class:`~tuiml.base.algorithms.Regressor`.

Every loader here is also reachable by name through
:func:`~tuiml.datasets.builtin.catalog.load_dataset`.

Examples
--------
>>> from tuiml.datasets import load_cpu
>>> data = load_cpu()
>>> data.X.shape
(209, 6)
"""

from tuiml.datasets.builtin._paths import _get_path
from tuiml.datasets.loaders import Dataset, load_arff


def load_cpu() -> Dataset:
    """Load the Computer Hardware (CPU) Performance dataset.

    209 instances with 6 continuous features for regression tasks.

    Returns
    -------
    Dataset
        Standardized dataset object containing the data and metadata.

    Examples
    --------
    >>> from tuiml.datasets import load_cpu
    >>> data = load_cpu()
    >>> print(data.X.shape)
    (209, 6)
    """
    return load_arff(_get_path("regression", "cpu.arff"))


def load_cpu_with_vendor() -> Dataset:
    """Load the CPU Performance dataset with vendor information.

    209 samples, 7 features (regression task).

    Returns
    -------
    dataset : Dataset
        CPU performance regression dataset including vendor feature.
    """
    return load_arff(_get_path("regression", "cpu.with.vendor.arff"))


def load_airline() -> Dataset:
    """Load the Airline dataset.

    Small dataset for time series / scheduling examples.

    Returns
    -------
    dataset : Dataset
        Airline scheduling regression dataset.
    """
    return load_arff(_get_path("regression", "airline.arff"))


__all__ = [
    "load_cpu",
    "load_cpu_with_vendor",
    "load_airline",
]
