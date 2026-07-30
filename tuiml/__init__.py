"""
TuiML - Modern machine learning toolkit.

A Python-based ML framework with a plugin architecture for extensibility.

Three levels of API:
    1. High-Level (Functional): tuiml.train() - one call, spec-dict driven
    2. Mid-Level (Workflow): tuiml.Workflow([...]) - a pipeline of steps
    3. Low-Level (OOP): Direct class imports - fit/predict/score/save/load
"""

from tuiml.hub import registry, ComponentType

# High-level API
from tuiml.api import (
    train,
    experiment,
    list_algorithms,
    describe_algorithm,
    search_algorithms,
    serve,
    stop_server,
    server_status,
    PRESETS,
)

# Mid-level API
from tuiml.workflow import Workflow, On

# Agent entry points (tools for every major framework + one-liner agent)
from tuiml.agent import agent

# Optional third-party bridges. Each registers its wrappers into the hub under a
# namespaced key (``sklearn.*`` / ``capymoa.*``) when its backing library is
# installed. They are best-effort: a missing backing library must never break
# ``import tuiml`` or the native algorithms.
try:
    import tuiml.sklearn  # noqa: F401
except Exception:  # pragma: no cover - defensive; native must still import
    pass
try:
    import tuiml.capymoa  # noqa: F401
except Exception:  # pragma: no cover
    pass

__version__ = "0.1.6"

__all__ = [
    # Core registry
    "registry",
    "ComponentType",

    # High-level API (one-liner functions)
    "train",
    "experiment",
    "list_algorithms",
    "describe_algorithm",
    "search_algorithms",
    "PRESETS",
    "serve",
    "stop_server",
    "server_status",

    # Mid-level API (pipeline objects)
    "Workflow",
    "On",

    # Agent / framework integration
    "agent",
]
