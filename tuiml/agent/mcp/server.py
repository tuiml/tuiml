"""
MCP (Model Context Protocol) Server for TuiML.

Exposes workflow and discovery tools that give LLMs access to ALL
TuiML components (algorithms, preprocessors, datasets, features).
New algorithms added to the hub or community are automatically
discoverable via tuiml_list / tuiml_search / tuiml_describe
and usable via tuiml_train / tuiml_experiment.

Usage:
    # Simplest way (after pip install)
    tuiml-mcp

    # Or run as Python module
    python -m tuiml.agent.mcp.server

    # Server options
    tuiml-mcp --help   # Show help
    tuiml-mcp --info   # Show server info

    # Configure in Claude Desktop (claude_desktop_config.json):
    {
        "mcpServers": {
            "tuiml": {
                "command": "tuiml-mcp"
            }
        }
    }
"""

import asyncio
import json
import queue
import sys
import threading
from typing import Any, Dict, List, Optional

# MCP imports - optional dependency
try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import (
        Tool,
        Resource,
        ResourceTemplate,
        TextContent,
        ImageContent,
    )
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    Server = None

def _strip_none(obj):
    """Recursively remove None values from dicts so absent optional fields pass schema validation."""
    if isinstance(obj, dict):
        return {k: _strip_none(v) for k, v in obj.items() if v is not None}
    if isinstance(obj, list):
        return [_strip_none(item) for item in obj]
    return obj

def _format_progress(info: Dict[str, Any]) -> str:
    """Format a progress callback dict into a human-readable log message."""
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

    Only workflow and discovery tools are exposed as MCP tools (11 total).
    The internal registry still tracks all 200+ components so that
    tuiml_list, tuiml_search, tuiml_describe, and tuiml_train
    can dynamically access any algorithm - including new ones added later.

    Returns
    -------
    Server
        Configured MCP server exposing TuiML workflow tools.

    Raises
    ------
    ImportError
        If MCP package is not installed.
    """
    if not MCP_AVAILABLE:
        raise ImportError(
            "MCP package not installed. Install with: pip install mcp"
        )

    server = Server("tuiml")

    # =========================================================================
    # List Tools Handler - only workflow + discovery tools
    # =========================================================================
    @server.list_tools()
    async def list_tools() -> List[Tool]:
        """List workflow and discovery tools (not 200+ component tools)."""
        tools = []

        from tuiml.agent.tools import get_workflow_tools, get_tool_output_schema, get_tool_annotations

        # Tools that return image content blocks cannot use outputSchema
        # (MCP validates structured output against the schema, but image
        # responses use [TextContent, ImageContent] which is unstructured)
        IMAGE_TOOLS = {"tuiml_plot"}

        for name, schema in get_workflow_tools().items():
            tool_kwargs = dict(
                name=name,
                description=schema["description"],
                inputSchema=schema["inputSchema"],
                annotations=get_tool_annotations(name),
            )
            if name not in IMAGE_TOOLS:
                tool_kwargs["outputSchema"] = get_tool_output_schema(name)
            tools.append(Tool(**tool_kwargs))

        return tools

    # =========================================================================
    # Call Tool Handler - runs CPU-bound work off the event loop
    # =========================================================================
    # Tools that benefit from real-time progress notifications
    _PROGRESS_TOOLS = {"tuiml_tune", "tuiml_experiment"}

    @server.call_tool()
    async def call_tool(name: str, arguments: Dict[str, Any]):
        """Execute any TuiML tool."""
        from tuiml.agent.tools import execute_tool

        try:
            # For long-running tools, set up real-time progress notifications
            if name in _PROGRESS_TOOLS:
                progress_queue: queue.Queue = queue.Queue()
                loop = asyncio.get_running_loop()

                def _sync_progress_callback(info):
                    """Sync callback invoked from worker thread — posts to queue."""
                    progress_queue.put(info)

                async def _drain_progress():
                    """Async task that drains the queue and sends MCP log notifications."""
                    while True:
                        try:
                            info = progress_queue.get_nowait()
                        except queue.Empty:
                            await asyncio.sleep(0.1)
                            continue
                        # Format a human-readable progress message
                        msg = _format_progress(info)
                        try:
                            await server.request_context.session.send_log_message(
                                level="info",
                                data=msg,
                                logger="tuiml.progress"
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
            # Strip None values — outputSchema allows absent optional fields
            # but not null when typed as "string"/"integer"/etc.
            result = _strip_none(result)

            # If the result contains image data, return mixed content
            if '_image_base64' in result:
                image_data = result.pop('_image_base64')
                mime_type = result.pop('_image_mime', 'image/png')
                return [
                    TextContent(type="text", text=json.dumps(result)),
                    ImageContent(type="image", data=image_data, mimeType=mime_type),
                ]

            return result

        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "tool": name
            }

    # =========================================================================
    # List Resources Handler (Datasets)
    # =========================================================================
    @server.list_resources()
    async def list_resources() -> List[Resource]:
        """List available datasets as MCP resources."""
        resources = []

        try:
            from tuiml.datasets.builtin import DATASET_REGISTRY

            for name, info in DATASET_REGISTRY.items():
                resources.append(Resource(
                    uri=f"tuiml://dataset/{name}",
                    name=name,
                    description=info.get("description", f"{name} dataset"),
                    mimeType="application/json"
                ))
        except ImportError:
            pass

        return resources

    # =========================================================================
    # Read Resource Handler
    # =========================================================================
    @server.read_resource()
    async def read_resource(uri: str) -> str:
        """Read a dataset resource."""
        if uri.startswith("tuiml://dataset/"):
            dataset_name = uri.replace("tuiml://dataset/", "")

            try:
                from tuiml.datasets.builtin import DATASET_REGISTRY, get_dataset_info
                from tuiml.datasets import load_dataset

                info = get_dataset_info(dataset_name)
                dataset = load_dataset(dataset_name)

                result = {
                    "name": dataset_name,
                    "info": info,
                    "shape": list(dataset.X.shape) if hasattr(dataset, 'X') else None,
                    "feature_names": dataset.feature_names if hasattr(dataset, 'feature_names') else None,
                    "preview": dataset.X[:5].tolist() if hasattr(dataset, 'X') else None
                }

                return json.dumps(result, indent=2, default=str)
            except Exception as e:
                return json.dumps({"error": str(e)})

        return json.dumps({"error": f"Unknown resource: {uri}"})

    # =========================================================================
    # Resource Templates Handler
    # =========================================================================
    @server.list_resource_templates()
    async def list_resource_templates() -> List[ResourceTemplate]:
        """List resource templates."""
        return [
            ResourceTemplate(
                uriTemplate="tuiml://dataset/{name}",
                name="TuiML Dataset",
                description="Load a built-in TuiML dataset",
                mimeType="application/json"
            )
        ]

    return server

async def run_server():
    """Run the MCP server using stdio transport."""
    if not MCP_AVAILABLE:
        print(
            "Error: MCP package not installed.\n"
            "Install with: pip install mcp\n"
            "Or: uv add mcp",
            file=sys.stderr
        )
        sys.exit(1)

    # Pre-load component registry so discovery tools are fast
    print("Loading TuiML components...", file=sys.stderr)
    from tuiml.agent.registry import get_all_tools
    get_all_tools()  # Trigger registry loading

    # Report counts
    info = get_server_info()
    exposed = info['tools']['exposed_tools']
    discoverable = info['tools']['discoverable_components']
    print(f"✓ {exposed} MCP tools exposed, {discoverable} components discoverable", file=sys.stderr)
    print("✓ TuiML MCP Server started (waiting for client)", file=sys.stderr)

    server = create_server()

    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )

def main():
    """Main entry point for the MCP server."""
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
        print("tuiml_describe, and tuiml_search.")
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
    """Get information about the MCP server."""
    from tuiml.agent.registry import get_tool_count
    from tuiml.agent.tools import get_workflow_tools

    workflow_count = len(get_workflow_tools())
    component_counts = get_tool_count()

    return {
        "name": "tuiml",
        "version": "1.0.0",
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
