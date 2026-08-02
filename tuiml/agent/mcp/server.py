"""MCP (Model Context Protocol) server for TuiML.

Exposes workflow and discovery tools that give LLMs access to all TuiML
components (algorithms, preprocessors, datasets, features). New algorithms
added to the component registry are automatically discoverable via
``tuiml_list`` / ``tuiml_describe`` and usable via
``tuiml_train`` / ``tuiml_benchmark``.

Usage
-----
Run the server (installed with ``pip install tuiml``)::

    tuiml-mcp

Or run it as a Python module::

    python -m tuiml.agent.mcp.server

Server options::

    tuiml-mcp --help   # Show help
    tuiml-mcp --info   # Show server info

Configure in Claude Desktop (``claude_desktop_config.json``)::

    {
        "mcpServers": {
            "tuiml": {
                "command": "tuiml-mcp"
            }
        }
    }
"""

import asyncio
import datetime as _dt
import json
import os
import queue
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional


# ─── MCP call tracing ────────────────────────────────────────────────
# Every call_tool invocation is appended as a JSONL record so users can
# audit / replay what their AI clients did. Disable with
# TUIML_MCP_TRACE=0 (default: enabled).
_TRACE_ENABLED = os.environ.get("TUIML_MCP_TRACE", "1") != "0"
_TRACE_PATH = Path(os.environ.get(
    "TUIML_MCP_TRACE_FILE",
    str(Path.home() / ".tuiml" / "logs" / "mcp.jsonl"),
))
_TRACE_LOCK = threading.Lock()
_TRACE_PID = os.getpid()


def _trace_write(record: dict) -> None:
    """Append one JSONL record to the MCP trace file.

    Parameters
    ----------
    record : dict
        JSON-serializable trace record to append.

    Returns
    -------
    None
        Failures are swallowed; tracing must never break the server.
    """
    if not _TRACE_ENABLED:
        return
    try:
        _TRACE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _TRACE_LOCK, _TRACE_PATH.open("a") as f:
            f.write(json.dumps(record, default=str) + "\n")
    except Exception:
        # Tracing must never break the server
        pass


def _trace_call_start(name: str, args: dict) -> None:
    """Write a trace record for the start of a tool call.

    Parameters
    ----------
    name : str
        Tool name being invoked.
    args : dict
        Tool arguments; large values are truncated before logging.

    Returns
    -------
    None
    """
    # Truncate large arg values so the log file doesn't explode.
    safe_args = {}
    for k, v in (args or {}).items():
        if k == "_progress_callback":
            continue
        s = str(v)
        safe_args[k] = s if len(s) <= 500 else (s[:500] + f"…[truncated, {len(s)} chars]")
    _trace_write({
        "ts": _dt.datetime.now().isoformat(timespec="milliseconds"),
        "pid": _TRACE_PID,
        "phase": "call",
        "tool": name,
        "args": safe_args,
    })


def _trace_call_end(name: str, result: Optional[dict], duration_ms: int,
                    error: Optional[str]) -> None:
    """Write a trace record for the completion of a tool call.

    Parameters
    ----------
    name : str
        Tool name that was invoked.
    result : dict or None
        Tool result; only its status and top-level keys are logged.
    duration_ms : int
        Wall-clock duration of the call in milliseconds.
    error : str or None
        Error message if the call failed, else None.

    Returns
    -------
    None
    """
    summary: Dict[str, Any] = {}
    if result is not None:
        # Don't write the full result (may include base64 images, large
        # arrays). Keep just status + top-level keys for traceability.
        summary["status"] = result.get("status", "unknown") if isinstance(result, dict) else "non-dict"
        if isinstance(result, dict):
            summary["keys"] = sorted(k for k in result.keys() if not k.startswith("_"))
    _trace_write({
        "ts": _dt.datetime.now().isoformat(timespec="milliseconds"),
        "pid": _TRACE_PID,
        "phase": "return",
        "tool": name,
        "duration_ms": duration_ms,
        "summary": summary,
        "error": error,
    })

# MCP imports - optional dependency.
#
# This module targets the 2.x SDK. The handlers below are registered through
# the ``Server(...)`` constructor (``on_list_tools=`` and friends) and return
# whole result models; 1.x registered them with decorators and returned bare
# lists. The two shapes are mutually exclusive, so an SDK that is too old is
# reported as unavailable rather than half-wired.
_MCP_UNAVAILABLE_REASON = None

try:
    import jsonschema
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import (
        Tool,
        Resource,
        ResourceTemplate,
        TextContent,
        ImageContent,
        TextResourceContents,
        CallToolResult,
        ListToolsResult,
        ListResourcesResult,
        ListResourceTemplatesResult,
        ReadResourceResult,
    )
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    Server = None
    _MCP_UNAVAILABLE_REASON = (
        "The MCP package is not installed. Install it with: pip install 'mcp>=2'"
    )

# 1.x also exports ``Server`` from ``mcp.server``, so the import above
# succeeds against it and the failure would otherwise surface much later as a
# bare TypeError about unexpected keyword arguments — which reaches the user
# as an opaque "connection closed" in their MCP client. The decorator method
# is the cheapest 1.x fingerprint: 2.x has no ``Server.list_tools``.
if MCP_AVAILABLE and hasattr(Server, "list_tools"):
    MCP_AVAILABLE = False
    try:
        from importlib.metadata import version as _pkg_version
        _found = _pkg_version("mcp")
    except Exception:
        _found = "1.x"
    _MCP_UNAVAILABLE_REASON = (
        f"Installed MCP SDK {_found} is too old. TuiML's MCP server targets "
        f"the 2.x SDK. Upgrade with: pip install 'mcp>=2'"
    )

def _strip_none(obj):
    """Recursively remove None values from dicts so absent optional fields pass schema validation.

    Parameters
    ----------
    obj : Any
        Value to clean; dicts and lists are walked recursively.

    Returns
    -------
    Any
        Copy of ``obj`` with all None-valued dict entries removed.
    """
    if isinstance(obj, dict):
        return {k: _strip_none(v) for k, v in obj.items() if v is not None}
    if isinstance(obj, list):
        return [_strip_none(item) for item in obj]
    return obj

def _format_progress(info: Dict[str, Any]) -> str:
    """Format a progress callback dict into a human-readable log message.

    Parameters
    ----------
    info : dict
        Progress payload from a workflow tool. The ``"type"`` key selects
        the format: ``"tune_progress"``, ``"benchmark_progress"``, or
        ``"experiment_progress"``; anything else is JSON-dumped as-is.

    Returns
    -------
    str
        One-line progress message suitable for MCP log notifications.
    """
    ptype = info.get('type', '')
    if ptype == 'tune_progress':
        iteration = info.get('iteration', '?')
        total = info.get('total', '?')
        mean = info.get('mean_score', 0)
        best = info.get('best_score', 0)
        params = info.get('params', {})
        return (
            f"[Tuning {iteration}/{total}] "
            f"score={mean:.4f}, best={best:.4f}, params={params}"
        )
    elif ptype == 'benchmark_progress':
        return (
            f"[Benchmark {info.get('dataset', '?')}] "
            f"{info.get('model', '?')}: fold "
            f"{info.get('fold', '?')}/{info.get('folds', '?')}"
        )
    elif ptype == 'experiment_progress':
        ds = info.get('dataset', '?')
        model = info.get('model', '?')
        di = info.get('dataset_index', '?')
        dt = info.get('total_datasets', '?')
        mi = info.get('model_index', '?')
        mt = info.get('total_models', '?')
        scores = info.get('mean_scores', {})
        scores_str = ', '.join(f"{k}={v:.4f}" for k, v in scores.items()) if scores else 'computing...'
        return (
            f"[Experiment dataset {di}/{dt} model {mi}/{mt}] "
            f"{model} on {ds}: {scores_str}"
        )
    else:
        return json.dumps(info, default=str)


def create_server() -> "Server":
    """
    Create and configure the TuiML MCP server.

    Only workflow and discovery tools are exposed as MCP tools (30 total).
    The internal registry still tracks all 200+ components so that
    tuiml_list, tuiml_describe, and tuiml_train
    can dynamically access any algorithm - including new ones added later.

    Returns
    -------
    Server
        Configured MCP server exposing TuiML workflow tools, with every
        handler registered on the constructor as the 2.x SDK expects.

    Raises
    ------
    ImportError
        If the 2.x ``mcp`` package is not installed.
    """
    if not MCP_AVAILABLE:
        raise ImportError(_MCP_UNAVAILABLE_REASON)

    # Tools that return image content blocks cannot use outputSchema
    # (MCP validates structured output against the schema, but image
    # responses use [TextContent, ImageContent] which is unstructured)
    IMAGE_TOOLS = {"tuiml_plot"}

    # Tools that benefit from real-time progress notifications
    _PROGRESS_TOOLS = {"tuiml_tune", "tuiml_benchmark"}

    def _build_tools() -> Dict[str, Tool]:
        """Build the exposed tool set, keyed by name.

        Returns
        -------
        dict of str to mcp.types.Tool
            One Tool per workflow/discovery tool, each carrying name,
            description, inputSchema, annotations, and (except for
            image-returning tools) outputSchema.
        """
        from tuiml.agent.tools import (
            get_workflow_tools,
            get_tool_output_schema,
            get_tool_annotations,
        )

        tools: Dict[str, Tool] = {}
        for name, schema in get_workflow_tools().items():
            tool_kwargs = dict(
                name=name,
                description=schema["description"],
                input_schema=schema["inputSchema"],
                annotations=get_tool_annotations(name),
            )
            if name not in IMAGE_TOOLS:
                tool_kwargs["output_schema"] = get_tool_output_schema(name)
            tools[name] = Tool(**tool_kwargs)
        return tools

    def _error_result(message: str) -> "CallToolResult":
        """Wrap a message as a failed tool call.

        Parameters
        ----------
        message : str
            Human-readable failure description.

        Returns
        -------
        mcp.types.CallToolResult
            Result carrying the message as text with ``isError`` set.
        """
        return CallToolResult(
            content=[TextContent(type="text", text=message)],
            is_error=True,
        )

    # =========================================================================
    # List Tools Handler - only workflow + discovery tools
    # =========================================================================
    async def on_list_tools(ctx, params) -> "ListToolsResult":
        """Serve ``tools/list`` with the workflow and discovery tools.

        The 200+ registry components are deliberately not exposed as tools;
        they stay reachable through tuiml_list / tuiml_describe / tuiml_train.

        Parameters
        ----------
        ctx : mcp.server.context.ServerRequestContext
            Per-request context (unused here).
        params : mcp.types.PaginatedRequestParams or None
            Pagination params; the tool set is small enough to send whole.

        Returns
        -------
        mcp.types.ListToolsResult
            Every exposed tool, unpaginated.
        """
        return ListToolsResult(tools=list(_build_tools().values()))

    # =========================================================================
    # Call Tool Handler - runs CPU-bound work off the event loop
    # =========================================================================
    async def on_call_tool(ctx, params) -> "CallToolResult":
        """Serve ``tools/call`` for any TuiML tool.

        Parameters
        ----------
        ctx : mcp.server.context.ServerRequestContext
            Per-request context, used for progress notifications.
        params : mcp.types.CallToolRequestParams
            Carries the tool ``name`` and its ``arguments``.

        Returns
        -------
        mcp.types.CallToolResult
            Structured result on success, or a result with ``"status":
            "error"`` when the tool itself raised. Tools that produce images
            return ``[TextContent, ImageContent]`` as unstructured content.
        """
        from tuiml.agent.tools import execute_tool, record_session_call

        name = params.name
        arguments: Dict[str, Any] = dict(params.arguments or {})

        # 1.x validated arguments against inputSchema inside the decorator.
        # 2.x hands the handler the raw params, so do it here or a bad
        # argument reaches execute_tool as a confusing TypeError.
        tool = _build_tools().get(name)
        if tool is None:
            return _error_result(f"Unknown tool: {name}")
        try:
            jsonschema.validate(instance=arguments, schema=tool.input_schema)
        except jsonschema.ValidationError as e:
            return _error_result(f"Input validation error: {e.message}")

        _trace_call_start(name, arguments)
        _t0 = time.perf_counter()

        try:
            # For long-running tools, stream real-time progress. This is
            # opt-in per the spec: notifications/progress is only legal when
            # the client attached a progressToken to the request, and 2.x
            # gates notifications/message behind a separate per-request
            # opt-in, so there is nowhere to send otherwise.
            progress_token = (ctx.meta or {}).get("progress_token")
            if name in _PROGRESS_TOOLS and progress_token is not None:
                progress_queue: queue.Queue = queue.Queue()

                def _sync_progress_callback(info):
                    """Sync callback invoked from worker thread, posts to queue."""
                    progress_queue.put(info)

                async def _drain_progress():
                    """Drain the queue and forward each item as notifications/progress."""
                    # `progress` must increase monotonically across the
                    # request. Benchmark fold counters restart per model, so
                    # count notifications sent rather than trusting the
                    # payload; `total` is only meaningful when a tool knows
                    # its own iteration count up front.
                    sent = 0
                    while True:
                        try:
                            info = progress_queue.get_nowait()
                        except queue.Empty:
                            await asyncio.sleep(0.1)
                            continue
                        sent += 1
                        total = None
                        if info.get("type") == "tune_progress":
                            raw_total = info.get("total")
                            if isinstance(raw_total, (int, float)):
                                total = float(raw_total)
                        try:
                            await ctx.session.send_progress_notification(
                                progress_token=progress_token,
                                progress=float(sent),
                                total=total,
                                message=_format_progress(info),
                                related_request_id=ctx.request_id,
                            )
                        except Exception:
                            pass  # Don't break execution if notification fails

                drain_task = asyncio.create_task(_drain_progress())
                arguments['_progress_callback'] = _sync_progress_callback

                try:
                    result = await asyncio.to_thread(execute_tool, name, **arguments)
                finally:
                    # Give the drain task a moment to flush remaining messages
                    await asyncio.sleep(0.2)
                    drain_task.cancel()
                    try:
                        await drain_task
                    except asyncio.CancelledError:
                        pass
            else:
                result = await asyncio.to_thread(execute_tool, name, **arguments)

            # Round-trip through JSON to ensure all values are serializable
            # (handles numpy types, datetimes, etc.)
            result = json.loads(json.dumps(result, default=str))
            # Strip None values, outputSchema allows absent optional fields
            # but not null when typed as "string"/"integer"/etc.
            result = _strip_none(result)

            duration_ms = int((time.perf_counter() - _t0) * 1000)
            _trace_call_end(name, result, duration_ms, error=None)

            # Record this call in the in-session log for tuiml_export_notebook.
            # Pass the original client arguments (before _progress_callback injection).
            record_session_call(name, {k: v for k, v in arguments.items()
                                       if not k.startswith('_')}, result)

            # If the result contains image data, return mixed content. These
            # tools publish no outputSchema, so there is nothing to validate
            # and the payload rides as unstructured content only.
            if '_image_base64' in result:
                image_data = result.pop('_image_base64')
                mime_type = result.pop('_image_mime', 'image/png')
                return CallToolResult(content=[
                    TextContent(type="text", text=json.dumps(result)),
                    ImageContent(type="image", data=image_data, mime_type=mime_type),
                ])

            # 1.x built this pair from a bare dict return: the structured
            # payload plus a JSON rendering for clients that only read text.
            if tool.output_schema is not None:
                try:
                    jsonschema.validate(instance=result, schema=tool.output_schema)
                except jsonschema.ValidationError as e:
                    return _error_result(f"Output validation error: {e.message}")

            return CallToolResult(
                content=[TextContent(type="text", text=json.dumps(result, indent=2))],
                structured_content=result,
            )

        except Exception as e:
            duration_ms = int((time.perf_counter() - _t0) * 1000)
            _trace_call_end(name, None, duration_ms, error=str(e))
            error = {
                "status": "error",
                "error": str(e),
                "tool": name,
            }
            return CallToolResult(
                content=[TextContent(type="text", text=json.dumps(error, indent=2))],
                structured_content=error,
                is_error=True,
            )

    # =========================================================================
    # List Resources Handler (Datasets)
    # =========================================================================
    async def on_list_resources(ctx, params) -> "ListResourcesResult":
        """Serve ``resources/list`` with the built-in datasets.

        Parameters
        ----------
        ctx : mcp.server.context.ServerRequestContext
            Per-request context (unused here).
        params : mcp.types.PaginatedRequestParams or None
            Pagination params; the dataset list is sent whole.

        Returns
        -------
        mcp.types.ListResourcesResult
            One Resource per built-in dataset, with a
            ``tuiml://dataset/{name}`` URI and JSON mime type.
        """
        resources = []

        try:
            from tuiml.datasets.builtin import DATASET_REGISTRY

            for name, info in DATASET_REGISTRY.items():
                resources.append(Resource(
                    uri=f"tuiml://dataset/{name}",
                    name=name,
                    description=info.get("description", f"{name} dataset"),
                    mime_type="application/json"
                ))
        except ImportError:
            pass

        return ListResourcesResult(resources=resources)

    # =========================================================================
    # Read Resource Handler
    # =========================================================================
    async def on_read_resource(ctx, params) -> "ReadResourceResult":
        """Serve ``resources/read`` for a dataset URI.

        Parameters
        ----------
        ctx : mcp.server.context.ServerRequestContext
            Per-request context (unused here).
        params : mcp.types.ReadResourceRequestParams
            Carries the ``tuiml://dataset/{name}`` URI to read.

        Returns
        -------
        mcp.types.ReadResourceResult
            One JSON text block with dataset name, info, shape, feature
            names, and a 5-row preview; or a JSON error object for unknown
            URIs.
        """
        uri = str(params.uri)
        payload = None

        if uri.startswith("tuiml://dataset/"):
            dataset_name = uri.replace("tuiml://dataset/", "")

            try:
                from tuiml.datasets import load_dataset
                from tuiml.datasets.builtin import get_dataset_info

                info = get_dataset_info(dataset_name)
                dataset = load_dataset(dataset_name)

                payload = json.dumps({
                    "name": dataset_name,
                    "info": info,
                    "shape": list(dataset.X.shape) if hasattr(dataset, 'X') else None,
                    "feature_names": dataset.feature_names if hasattr(dataset, 'feature_names') else None,
                    "preview": dataset.X[:5].tolist() if hasattr(dataset, 'X') else None
                }, indent=2, default=str)
            except Exception as e:
                payload = json.dumps({"error": str(e)})
        else:
            payload = json.dumps({"error": f"Unknown resource: {uri}"})

        return ReadResourceResult(contents=[
            TextResourceContents(uri=uri, mime_type="application/json", text=payload)
        ])

    # =========================================================================
    # Resource Templates Handler
    # =========================================================================
    async def on_list_resource_templates(ctx, params) -> "ListResourceTemplatesResult":
        """Serve ``resources/templates/list``.

        Parameters
        ----------
        ctx : mcp.server.context.ServerRequestContext
            Per-request context (unused here).
        params : mcp.types.PaginatedRequestParams or None
            Pagination params; a single template is sent.

        Returns
        -------
        mcp.types.ListResourceTemplatesResult
            A single template describing the ``tuiml://dataset/{name}``
            URI scheme for built-in datasets.
        """
        return ListResourceTemplatesResult(resource_templates=[
            ResourceTemplate(
                uri_template="tuiml://dataset/{name}",
                name="TuiML Dataset",
                description="Load a built-in TuiML dataset",
                mime_type="application/json"
            )
        ])

    # 2.x registers handlers on the constructor instead of by decorator, and
    # derives advertised capabilities from which ones are passed.
    from tuiml import __version__

    return Server(
        "tuiml",
        version=__version__,
        on_list_tools=on_list_tools,
        on_call_tool=on_call_tool,
        on_list_resources=on_list_resources,
        on_read_resource=on_read_resource,
        on_list_resource_templates=on_list_resource_templates,
    )

async def run_server():
    """Run the MCP server using stdio transport.

    Returns
    -------
    None
        Blocks until the client disconnects; exits the process if the
        ``mcp`` package is not installed.
    """
    if not MCP_AVAILABLE:
        print(f"Error: {_MCP_UNAVAILABLE_REASON}", file=sys.stderr)
        sys.exit(1)

    # Pre-load the component registry in the background so the MCP
    # handshake (initialize / tools/list) responds immediately. Clients
    # like Claude Desktop index tools right after connecting; blocking
    # here for seconds makes tools miss the client's first index pass.
    def _preload_components():
        try:
            from tuiml.agent.tools import get_workflow_tools
            from tuiml.agent.tools._components import get_all_tools
            exposed = len(get_workflow_tools())
            discoverable = len(get_all_tools())
            print(f"✓ {exposed} MCP tools exposed, {discoverable} components discoverable", file=sys.stderr)
        except Exception as e:
            print(f"[tuiml] component preload failed: {e}", file=sys.stderr)

    threading.Thread(target=_preload_components, name="tuiml-preload", daemon=True).start()

    print("✓ TuiML MCP Server started (stdio transport, local only)", file=sys.stderr)
    print("  Loading TuiML components in the background...", file=sys.stderr)
    print("  Setup and docs: https://tuiml.ai/getting_started.html", file=sys.stderr)
    print("  Waiting for client...", file=sys.stderr)

    server = create_server()

    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )

def main():
    """Main entry point for the MCP server (the ``tuiml-mcp`` command).

    Handles the ``--info`` and ``--help`` flags, otherwise runs the
    stdio server.

    Returns
    -------
    None
    """
    import sys

    # Check for --info flag
    if len(sys.argv) > 1 and sys.argv[1] in ['--info', '-i', 'info']:
        info = get_server_info()
        print("TuiML MCP Server")
        print("=" * 40)
        print(f"MCP Available: {info['mcp_available']}")
        print(f"Exposed Tools: {info['tools']['exposed_tools']}")
        print(f"Discoverable Components: {info['tools']['discoverable_components']}")
        print()
        print("Exposed tools (workflow + discovery):")
        from tuiml.agent.tools import get_workflow_tools
        for name in get_workflow_tools():
            print(f"  - {name}")
        print()
        print("All components are accessible via tuiml_train, tuiml_list,")
        print("and tuiml_describe.")
        print()
        print("To run the server:")
        print("  tuiml-mcp")
        print()
        print("Configure in Claude Desktop:")
        print('  {"mcpServers": {"tuiml": {"command": "tuiml-mcp"}}}')
        return

    # Check for --help flag
    if len(sys.argv) > 1 and sys.argv[1] in ['--help', '-h', 'help']:
        print("TuiML MCP Server")
        print()
        print("Usage:")
        print("  tuiml-mcp          Run MCP server (stdio)")
        print("  tuiml-mcp --info   Show server info")
        print("  tuiml-mcp --help   Show this help")
        return

    # Run MCP server
    print("Starting TuiML MCP Server...", file=sys.stderr)
    asyncio.run(run_server())

# =============================================================================
# Server Info
# =============================================================================

def get_server_info() -> Dict[str, Any]:
    """Get information about the MCP server.

    Returns
    -------
    dict
        Server metadata with keys ``"name"``, ``"version"``,
        ``"description"``, ``"mcp_available"``, and ``"tools"`` (a dict
        with ``"exposed_tools"``, ``"discoverable_components"``, and
        ``"components_by_category"`` counts).
    """
    from tuiml.agent.tools._components import get_tool_count
    from tuiml.agent.tools import get_workflow_tools

    workflow_count = len(get_workflow_tools())
    component_counts = get_tool_count()

    # The package version, not a version for the server itself: it is what
    # create_server() advertises in the initialize handshake, so `--info` and
    # the client's view of the server have to agree.
    from tuiml import __version__

    return {
        "name": "tuiml",
        "version": __version__,
        "description": "TuiML Machine Learning MCP Server",
        "mcp_available": MCP_AVAILABLE,
        "tools": {
            "exposed_tools": workflow_count,
            "discoverable_components": sum(component_counts.values()),
            "components_by_category": component_counts,
        }
    }

if __name__ == "__main__":
    main()
