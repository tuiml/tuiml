"""Task-oriented tools that LLMs use to run complete ML workflows.

Each tool is one module under this package, declaring a :class:`ToolSpec`
alongside its executor. Everything the rest of the codebase consumes -- the
schema dicts, the dispatch table, the MCP annotations, the notebook-export
skip list -- is derived from the specs collected in :data:`SPECS` below.

To add a tool: create the module with its ``ToolSpec``, then import it here
and append its spec to :data:`SPECS`. That list is the registration point and
is deliberately explicit -- it fixes tool ordering and makes a missing
registration a visible omission rather than a silent one.

Layout
------
``workflow/``   train, predict, evaluate, benchmark, tune, plot, save_model
``data/``       upload, profile, generate, read, preprocess, select_features
``analysis/``   statistics
``discovery/``  list, describe
``authoring/``  agent-authored algorithms (skeleton, create, read, edit, ...)
``system/``     install info, self-update, restart, REST serving
``notebook/``   session export to .ipynb
"""

from typing import Any, Dict, List

from ._session import record_session_call
from ._state import get_session_seed, set_session_seed
from ._shared import _load_data, _load_model_from_disk, _save_model_to_disk
from ._spec import ToolSpec
from .analysis import statistics as _statistics
from .authoring import (
    create as _create,
    delete as _delete,
    edit as _edit,
    files as _files,
    read as _read_algo,
    search as _search_source,
    skeleton as _skeleton,
)
from .data import (
    generate as _generate,
    preprocess as _preprocess,
    profile as _profile,
    read as _read_data,
    select_features as _select_features,
    upload as _upload,
)
from .discovery import describe as _describe, list_components as _list
from .notebook import export as _export
from .system import info as _info, self_update as _self_update, restart as _restart, serving as _serving
from .workflow import (
    benchmark as _benchmark,
    evaluate as _evaluate,
    plot as _plot,
    predict as _predict,
    save_model as _save_model,
    train as _train,
    tune as _tune,
)

# ---------------------------------------------------------------------------
# The single source of truth
# ---------------------------------------------------------------------------

SPECS: List[ToolSpec] = [
    _train.SPEC,
    _predict.SPEC,
    _evaluate.SPEC,
    _benchmark.SPEC,
    _upload.SPEC,
    _save_model.SPEC,
    _serving.SERVE_SPEC,
    _serving.STOP_SPEC,
    _serving.STATUS_SPEC,
    _plot.SPEC,
    _profile.SPEC,
    _generate.SPEC,
    _preprocess.SPEC,
    _select_features.SPEC,
    _statistics.SPEC,
    _tune.SPEC,
    _read_data.SPEC,
    _info.SPEC,
    _skeleton.SPEC,
    _create.SPEC,
    _delete.SPEC,
    _self_update.SPEC,
    _restart.SPEC,
    _export.SPEC,
    _list.SPEC,
    _describe.SPEC,
    _read_algo.SPEC,
    _files.SPEC,
    _search_source.SPEC,
    _edit.SPEC,
]

_BY_NAME: Dict[str, ToolSpec] = {}
for _spec in SPECS:
    if _spec.name in _BY_NAME:
        raise RuntimeError(f"duplicate tool name: {_spec.name}")
    _BY_NAME[_spec.name] = _spec

# ---------------------------------------------------------------------------
# Derived views (previously eight hand-maintained dicts)
# ---------------------------------------------------------------------------

WORKFLOW_TOOLS: Dict[str, Dict] = {
    s.name: s.as_mcp_tool() for s in SPECS if s.group == "workflow"
}
DISCOVERY_TOOLS: Dict[str, Dict] = {
    s.name: s.as_mcp_tool() for s in SPECS if s.group == "discovery"
}
CODE_TOOLS: Dict[str, Dict] = {
    s.name: s.as_mcp_tool() for s in SPECS if s.group == "code"
}

TOOL_EXECUTORS = {s.name: s.execute for s in SPECS}

OUTPUT_SCHEMAS: Dict[str, Dict] = {
    s.name: s.output_schema for s in SPECS if s.output_schema is not None
}

# Component tool output schema (generic for all component tools)
COMPONENT_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "status": {"type": "string", "enum": ["success", "error"]},
        "result": {"type": "string", "description": "String representation of the component"},
        "type": {"type": "string", "description": "Component class name"},
        "error": {"type": "string"}
    },
    "required": ["status"]
}

_SEEDED_TOOLS = frozenset(s.name for s in SPECS if s.seeded)


def get_tool_spec(tool_name: str):
    """Look up a tool's spec.

    Parameters
    ----------
    tool_name : str
        Name of the MCP tool.

    Returns
    -------
    spec : ToolSpec or None
        The spec, or None for component tools and unknown names.
    """
    return _BY_NAME.get(tool_name)


def get_tool_output_schema(tool_name: str) -> Dict[str, Any]:
    """Get the JSON output schema for a tool.

    Parameters
    ----------
    tool_name : str
        Name of the MCP tool (e.g. ``'tuiml_train'``).

    Returns
    -------
    schema : dict
        The tool's output JSON Schema, or the generic component output
        schema when the tool has no dedicated one.
    """
    spec = _BY_NAME.get(tool_name)
    if spec is not None and spec.output_schema is not None:
        return spec.output_schema
    return COMPONENT_OUTPUT_SCHEMA


def get_tool_annotations(tool_name: str) -> Dict[str, bool]:
    """Get MCP behavior annotations for a tool.

    Parameters
    ----------
    tool_name : str
        Name of the MCP tool.

    Returns
    -------
    annotations : dict
        Annotation flags (``readOnlyHint``, ``destructiveHint``,
        ``idempotentHint``, ``openWorldHint``); component tools fall back
        to read-only defaults.
    """
    spec = _BY_NAME.get(tool_name)
    if spec is not None:
        return spec.as_annotations()
    # Component tools instantiate a class and return its repr, nothing more.
    return {
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    }


def is_reproducible(tool_name: str) -> bool:
    """Whether a successful call should become a notebook cell.

    Parameters
    ----------
    tool_name : str
        Name of the MCP tool.

    Returns
    -------
    reproducible : bool
        False for discovery and admin tools, which produce no Python worth
        exporting. Unknown (component) tools are treated as reproducible.
    """
    spec = _BY_NAME.get(tool_name)
    return True if spec is None else spec.reproducible


def get_workflow_tools() -> Dict[str, Dict]:
    """Get all workflow, discovery and code tool schemas.

    Returns
    -------
    tools : dict
        Mapping of tool name to its MCP input schema definition.
    """
    return {**WORKFLOW_TOOLS, **DISCOVERY_TOOLS, **CODE_TOOLS}


def execute_tool(tool_name: str, **kwargs) -> Dict[str, Any]:
    """Execute a tool by name, resolving the random seed first.

    Sets a process-wide seed (explicit ``random_seed`` kwarg, else the
    session seed) before dispatching to the workflow executor or, failing
    that, a registered component tool.

    Parameters
    ----------
    tool_name : str
        Name of the tool to execute.
    random_seed : int, default=None
        Random seed for the call. When omitted the call runs under the
        session seed from :func:`~tuiml.agent.tools._state.get_session_seed`,
        so repeating a call within one session reproduces its numbers
        (arrives via ``**kwargs``, like the tool arguments).
    **kwargs
        Remaining arguments are forwarded to the tool executor.

    Returns
    -------
    result : dict
        The executor's result dict; successful workflow results also get
        the effective ``random_seed`` added. Component tools return
        ``status``, ``result`` (stringified) and ``type``. Unknown tools
        return ``status`` (``'error'``) and ``error``.
    """
    random_seed = kwargs.pop('random_seed', None)

    # Falling back to the session seed rather than a fresh draw is what makes a
    # conversation reproducible: two identical calls return identical numbers,
    # and comparing two runs measures the change rather than the seed.
    if random_seed is None:
        random_seed = get_session_seed()

    from tuiml.utils.seed import set_global_seed
    set_global_seed(random_seed)

    if tool_name in _SEEDED_TOOLS:
        kwargs['random_seed'] = random_seed

    # Check workflow tools first
    spec = _BY_NAME.get(tool_name)
    if spec is not None:
        result = spec.execute(**kwargs)
        if isinstance(result, dict) and result.get('status') == 'success':
            result['random_seed'] = random_seed
        return result

    # For any component tool, ensure full registry is loaded — including the
    # user's own algorithms, which a tool call may well be naming.
    from tuiml.agent.user_algorithms import ensure_loaded
    ensure_loaded()

    from ._components import get_tool
    tool = get_tool(tool_name)
    if tool:
        try:
            result = tool.executor(kwargs)
            return {
                'status': 'success',
                'result': str(result),
                'type': result.__class__.__name__
            }
        except Exception as e:
            return {'status': 'error', 'error': str(e)}

    return {'status': 'error', 'error': f"Unknown tool: {tool_name}"}


# Agent-authored algorithms are re-registered from disk by
# ``user_algorithms.ensure_loaded()``, which the MCP server, the CLI and
# ``execute_tool`` each call before touching the registry. It deliberately does
# not run here: importing this module is not a reason to execute user code, and
# every CLI command imports it transitively — which is how ``tuiml --version``
# used to load user algorithms and print about it.


# Re-exported for callers that reach past the tool layer (the `tuiml update`
# and `tuiml info` CLI front ends call these executors directly).
execute_self_update = _self_update.execute_self_update
execute_system_info = _info.execute_system_info

__all__ = [
    "ToolSpec",
    "SPECS",
    "WORKFLOW_TOOLS",
    "DISCOVERY_TOOLS",
    "CODE_TOOLS",
    "TOOL_EXECUTORS",
    "OUTPUT_SCHEMAS",
    "COMPONENT_OUTPUT_SCHEMA",
    "execute_tool",
    "execute_self_update",
    "execute_system_info",
    "get_workflow_tools",
    "get_tool_spec",
    "get_tool_output_schema",
    "get_tool_annotations",
    "is_reproducible",
    "record_session_call",
    "_load_data",
    "_load_model_from_disk",
    "_save_model_to_disk",
]
