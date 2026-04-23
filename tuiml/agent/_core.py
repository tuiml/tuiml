"""Shared helpers for the agent framework adapters.

Every adapter in ``tuiml.agent.*`` (langchain, openai, anthropic, pydantic_ai,
crewai, …) derives its tool list from the same source of truth — the
``WORKFLOW_TOOLS`` dict in ``tuiml.agent.tools`` — and routes invocations
through ``execute_tool``. This module holds the bits that don't belong to any
one framework: the canonical tool iterator, the JSON-Schema → Pydantic model
converter, the framework-agnostic callable adapter, and the bundled system
prompt (``SKILL.md``) loader.
"""

from __future__ import annotations

from functools import lru_cache
from importlib import resources
from typing import Any, Callable, Dict, Iterator, List, Tuple, Type


# ---------------------------------------------------------------------------
# Canonical tool iteration
# ---------------------------------------------------------------------------

def iter_tools() -> Iterator[Tuple[str, str, Dict[str, Any]]]:
    """Yield ``(name, description, input_schema)`` for every workflow tool.

    A single place every adapter should consume. If we ever add new tools they
    automatically show up in every framework integration.
    """
    from tuiml.agent.tools import WORKFLOW_TOOLS
    for name, spec in WORKFLOW_TOOLS.items():
        yield name, spec["description"], spec.get("inputSchema", {"type": "object", "properties": {}})


# ---------------------------------------------------------------------------
# Invocation
# ---------------------------------------------------------------------------

def invoke(name: str, **kwargs: Any) -> Dict[str, Any]:
    """Run a TuiML tool and return its structured result.

    Thin alias for ``tuiml.agent.execute_tool`` — kept here so every adapter
    imports from one consistent path.
    """
    from tuiml.agent.tools import execute_tool
    return execute_tool(name, **kwargs)


def callables() -> Dict[str, Callable[..., Dict[str, Any]]]:
    """Return a ``{tool_name: python_callable}`` dict (framework-agnostic).

    Useful when a framework only takes plain callables (e.g. smolagents,
    custom loops). Each callable accepts keyword arguments matching the
    tool's JSON Schema and returns the usual ``{status: ..., ...}`` dict.
    """
    return {
        name: _bind(name)
        for name, _desc, _schema in iter_tools()
    }


def _bind(tool_name: str) -> Callable[..., Dict[str, Any]]:
    """Bind a tool name to a callable — kept as a separate function so each
    closure captures a distinct ``tool_name`` via its default-argument trick."""
    def _call(__tool_name: str = tool_name, **kwargs: Any) -> Dict[str, Any]:
        return invoke(__tool_name, **kwargs)
    _call.__name__ = tool_name
    return _call


# ---------------------------------------------------------------------------
# JSON Schema → Pydantic model (for LangChain / CrewAI args_schema)
# ---------------------------------------------------------------------------

_TYPE_MAP: Dict[str, type] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "array": list,
    "object": dict,
    "null": type(None),
}


def _py_type_from_prop(prop: Dict[str, Any]) -> type:
    """Map a single JSON Schema property to a Python type.

    Best-effort: array/object get ``list``/``dict`` (no item type); ``anyOf``
    collapses to ``Any``; missing ``type`` also falls back to ``Any``.
    """
    from typing import Any as _Any

    if "anyOf" in prop or "oneOf" in prop:
        return _Any  # type: ignore[return-value]
    t = prop.get("type")
    if isinstance(t, list):
        # Multiple types: pick the first non-null as a pragmatic default.
        t = next((x for x in t if x != "null"), "string")
    return _TYPE_MAP.get(t, _Any)  # type: ignore[return-value]


def build_args_model(tool_name: str, schema: Dict[str, Any]) -> Type:
    """Build a Pydantic v2 ``BaseModel`` subclass from a JSON Schema.

    The resulting model becomes ``StructuredTool(args_schema=…)`` for
    LangChain, or any framework that needs a concrete type.
    """
    from pydantic import create_model, Field  # type: ignore

    props: Dict[str, Dict[str, Any]] = schema.get("properties", {}) or {}
    required = set(schema.get("required", []) or [])

    fields: Dict[str, Any] = {}
    for pname, pschema in props.items():
        py_type = _py_type_from_prop(pschema)
        description = pschema.get("description", "")
        default = pschema.get("default", ... if pname in required else None)
        fields[pname] = (py_type, Field(default=default, description=description))

    if not fields:
        # Pydantic requires at least one field for a useful model; add a
        # no-op marker so `...Tool(args_schema=Model)` works for no-arg tools.
        fields["_"] = (str, Field(default="", description="no arguments"))

    return create_model(f"{tool_name}Args", **fields)


# ---------------------------------------------------------------------------
# Skill / system prompt loader
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def load_skill() -> str:
    """Return the bundled ``SKILL.md`` contents (canonical system prompt)."""
    return resources.files("tuiml.agent").joinpath("SKILL.md").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Lazy-import helpers
# ---------------------------------------------------------------------------

def require(module_name: str, extra: str) -> None:
    """Raise a friendly ImportError if an optional framework is missing.

    Every adapter calls ``require("langchain_core", "langchain")`` etc. so the
    error message points the user to ``pip install tuiml[langchain]``.
    """
    import importlib
    try:
        importlib.import_module(module_name)
    except ImportError as e:
        raise ImportError(
            f"{module_name} is not installed. Add it with:\n"
            f"    pip install tuiml[{extra}]\n"
            f"or directly:\n"
            f"    pip install {module_name.split('.')[0]}"
        ) from e


__all__ = [
    "iter_tools",
    "invoke",
    "callables",
    "build_args_model",
    "load_skill",
    "require",
]
