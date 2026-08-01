"""Single declaration of what a TuiML agent tool is.

Every tool is declared exactly once, as a :class:`ToolSpec` living next to its
executor. The package ``__init__`` derives everything else from those specs:
the ``WORKFLOW_TOOLS`` / ``DISCOVERY_TOOLS`` / ``CODE_TOOLS`` schema dicts, the
executor dispatch table, the MCP output schemas and behaviour annotations, the
notebook-export skip list, and the set of tools that take an injected random
seed. Before this, those were eight hand-maintained dicts scattered across a
6700-line module, and they had drifted: a tool with schemas but no executor, and
executors with no annotations. Deriving them from one source makes that class of
drift unrepresentable.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional


@dataclass(frozen=True)
class ToolSpec:
    """Everything the agent layer needs to know about one tool.

    Parameters
    ----------
    name : str
        MCP tool name, e.g. ``'tuiml_train'``.
    description : str
        Human/LLM-readable description advertised to clients.
    input_schema : dict
        JSON Schema for the tool's arguments.
    execute : callable
        Executor invoked as ``execute(**kwargs)``, returning a result dict
        that always carries a ``status`` key.
    output_schema : dict, default=None
        JSON Schema for the result. None means the generic component output
        schema is advertised instead.
    group : str, default='workflow'
        Which schema dict the tool belongs to: ``'workflow'``,
        ``'discovery'`` or ``'code'``.
    read_only : bool, default=False
        MCP ``readOnlyHint``: the tool does not modify its environment.
    destructive : bool, default=False
        MCP ``destructiveHint``: the tool may delete or overwrite state.
    idempotent : bool, default=False
        MCP ``idempotentHint``: repeating the call has no extra effect.
    open_world : bool, default=False
        MCP ``openWorldHint``: the tool touches systems outside TuiML.
    seeded : bool, default=False
        Whether ``execute_tool`` forwards the resolved ``random_seed``.
    reproducible : bool, default=True
        Whether a successful call becomes a cell in the exported notebook.
        Discovery and admin tools produce no reproducible Python.
    """

    name: str
    description: str
    input_schema: Dict[str, Any]
    execute: Callable[..., Dict[str, Any]]
    output_schema: Optional[Dict[str, Any]] = None
    group: str = "workflow"
    read_only: bool = False
    destructive: bool = False
    idempotent: bool = False
    open_world: bool = False
    seeded: bool = False
    reproducible: bool = True

    def as_mcp_tool(self) -> Dict[str, Any]:
        """Render the MCP tool definition clients see.

        Returns
        -------
        definition : dict
            Mapping with ``name``, ``description`` and ``inputSchema``.
        """
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.input_schema,
        }

    def as_annotations(self) -> Dict[str, bool]:
        """Render the MCP behaviour annotations.

        Returns
        -------
        annotations : dict
            The four ``*Hint`` flags.
        """
        return {
            "readOnlyHint": self.read_only,
            "destructiveHint": self.destructive,
            "idempotentHint": self.idempotent,
            "openWorldHint": self.open_world,
        }
