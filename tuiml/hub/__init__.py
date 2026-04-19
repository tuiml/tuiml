"""TuiML Hub - Local component registry.

Provides the local algorithm/feature/preprocessor registry used by
``@classifier``, ``@regressor``, ``@transformer`` decorators across the
library.

Examples
--------
Local component registration::

    from tuiml.hub import registry

    @registry.register("classifier")
    class MyClassifier:
        pass

    model = registry.create("MyClassifier", param=value)
"""

from tuiml.hub.registry import registry
from tuiml.hub.types import ComponentType, Registrable

__all__ = [
    "registry",
    "ComponentType",
    "Registrable",
]
