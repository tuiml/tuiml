"""
TuiML - Modern machine learning toolkit.

A Python-based ML framework with a plugin architecture for extensibility.

Three levels of API:
    1. High-Level (Functional): tuiml.train() - one call, spec-dict driven
    2. Mid-Level (Workflow): tuiml.Workflow([...]) - a pipeline of steps
    3. Low-Level (OOP): Direct class imports - fit/predict/score/save/load
"""

import importlib
import importlib.util

# Every public name is resolved on first use rather than imported here.
# Importing them eagerly cost ~2.3s and pulled in the plotting stack, which
# `import tuiml` paid even to read `__version__` -- and the console script
# behind `tuiml --version` cannot avoid importing this package.
#
# Components still register themselves by being imported; that is now driven
# by Registry._ensure_populated() on first read, so the catalogue is complete
# whenever anything actually looks at it. See :mod:`tuiml.registry`.
_LAZY_ATTRS = {
    # Core registry
    "registry": "tuiml.registry:registry",
    "ComponentType": "tuiml.registry:ComponentType",

    # High-level API, one root module per concern
    "train": "tuiml.training:train",
    "PRESETS": "tuiml.training:PRESETS",
    "Benchmark": "tuiml.benchmarking:Benchmark",
    "list_algorithms": "tuiml.discovery:list_algorithms",
    "describe_algorithm": "tuiml.discovery:describe_algorithm",
    "search_algorithms": "tuiml.discovery:search_algorithms",
    "serve": "tuiml.serving:serve",
    "stop_server": "tuiml.serving:stop_server",
    "server_status": "tuiml.serving:server_status",

    # Mid-level API
    "Workflow": "tuiml.workflow:Workflow",
    "On": "tuiml.workflow:On",
}

__version__ = "0.1.9"


def __getattr__(name: str):
    """Resolve a public name, or a submodule, on first access (PEP 562).

    Submodules are handled too because ``__init__`` used to import several of
    them (``tuiml.algorithms``, ``tuiml.agent``), which left them reachable as
    attributes; ``import tuiml; tuiml.agent.agent()`` must keep working.
    """
    target = _LAZY_ATTRS.get(name)
    if target is not None:
        module_name, _, attr = target.partition(":")
        value = getattr(importlib.import_module(module_name), attr)
        globals()[name] = value          # cache: __getattr__ runs once per name
        return value

    # `tuiml.<submodule>` — resolve only real submodules, so a typo still
    # raises AttributeError rather than ImportError from somewhere deeper.
    if not name.startswith("_"):
        try:
            found = importlib.util.find_spec(f"tuiml.{name}") is not None
        except (ImportError, ValueError):
            found = False
        if found:
            module = importlib.import_module(f"tuiml.{name}")
            globals()[name] = module
            return module

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    """List the lazily-resolved names alongside whatever is already bound."""
    return sorted(set(globals()) | set(_LAZY_ATTRS))

__all__ = [
    # Core registry
    "registry",
    "ComponentType",

    # High-level API (one-liner functions)
    "train",
    "Benchmark",
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
]
