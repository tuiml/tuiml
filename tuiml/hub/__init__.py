"""TuiML Hub - Local registry and remote access for algorithms and datasets.

This module provides three main interfaces:

1. **registry** - For managing local components
2. **remote** - For accessing remote algorithms from the hub
3. **datasets** - For browsing and loading datasets from the hub

Examples
--------
Local component registration::

    from tuiml.hub import registry

    @registry.register("classifier")
    class MyClassifier:
        pass

    model = registry.create("MyClassifier", param=value)

Remote algorithm usage::

    from tuiml.hub import remote

    # Browse and search
    algorithms = remote.browse()
    results = remote.search("random forest")

    # Install and use
    remote.install("my_algorithm")
    model = remote.use("my_algorithm", n_estimators=100)

Remote dataset usage::

    from tuiml.hub import datasets

    # Browse datasets
    results = datasets.browse(task_type="classification")

    # Load a dataset as DataFrame
    df = datasets.load("wine-quality")
"""

from tuiml.hub.registry import registry
from tuiml.hub.remote import remote
from tuiml.hub.datasets_remote import datasets, DatasetInfo
from tuiml.hub.types import ComponentType, AlgorithmInfo, Registrable

__all__ = [
    "registry",
    "remote",
    "datasets",
    "ComponentType",
    "AlgorithmInfo",
    "DatasetInfo",
    "Registrable",
]
