#!/usr/bin/env python3
"""
Utility to convert chat logs from various LLM interfaces (Antigravity, LM Studio, OpenAI, Claude Code)
into reproducible Jupyter Notebooks (.ipynb). It extracts user prompts into Markdown cells,
assistant responses into Markdown cells, shell commands into Code cells (prefixed with !), 
and Python code blocks into standard Python Code cells.
"""

import os
import re
import json
import argparse
import sys
from typing import List, Dict, Any

def create_notebook(cells: List[Dict[str, Any]], output_path: str):
    """Save cells into a valid Jupyter Notebook JSON structure."""
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=2)
    print(f"[OK] Notebook successfully saved to: {output_path}")

def make_markdown_cell(text: str, source_label: str = "") -> Dict[str, Any]:
    """Helper to format a Markdown cell."""
    header = f"### {source_label}\n" if source_label else ""
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [header + text]
    }

def make_code_cell(code: str, is_shell: bool = False) -> Dict[str, Any]:
    """Helper to format a Code cell (adds ! prefix for shell commands)."""
    lines = code.splitlines(keepends=True)
    if is_shell:
        # Prefix lines with ! for shell execution in Jupyter
        processed_lines = []
        for line in lines:
            if line.strip() and not line.strip().startswith('!'):
                processed_lines.append("!" + line)
            else:
                processed_lines.append(line)
        lines = processed_lines

    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines
    }

def extract_code_blocks(text: str) -> List[Dict[str, Any]]:
    """
    Parses Markdown text and extracts code blocks.
    Returns a sequence of Markdown and Code cells representing the document.
    """
    # Pattern to find code blocks with language specifiers
    pattern = re.compile(r'```(\w*)\s*\n(.*?)```', re.DOTALL)
    cells = []
    
    last_end = 0
    for match in pattern.finditer(text):
        lang = match.group(1).lower()
        code = match.group(2)
        start, end = match.span()
        
        # Add markdown before the code block
        md_text = text[last_end:start].strip()
        if md_text:
            cells.append(make_markdown_cell(md_text))
            
        # Add code cell based on language
        if lang in ('python', 'py'):
            cells.append(make_code_cell(code, is_shell=False))
        elif lang in ('bash', 'sh', 'shell', 'powershell', 'cmd'):
            cells.append(make_code_cell(code, is_shell=True))
        else:
            # Re-wrap other code blocks (like JSON, diff, XML) in markdown
            cells.append(make_markdown_cell(f"```{lang}\n{code}```"))
            
        last_end = end
        
    # Add any remaining markdown
    remaining = text[last_end:].strip()
    if remaining:
        cells.append(make_markdown_cell(remaining))
        
    return cells

def mcp_to_cli(tool_name: str, arguments: Dict[str, Any]) -> str:
    """Maps an MCP tool name and arguments to its equivalent tuiml CLI command."""
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return f"# Failed to parse arguments for {tool_name}"

    cmd_parts = ["tuiml"]

    if tool_name == "execute_train":
        cmd_parts.append("train")
        # Extract main options
        for opt in ['stage', 'algorithm', 'data', 'target', 'save_path', 'model_id', 'model_path']:
            val = arguments.get(opt)
            if val is not None:
                cmd_parts.append(f"--{opt.replace('_', '-')} {val}")
        
        # Check cv, preset, test_size, etc.
        for opt in ['cv', 'preset', 'test_size', 'output']:
            val = arguments.get(opt)
            if val is not None:
                cmd_parts.append(f"--{opt.replace('_', '-')} {val}")
                
        # Check lists like preprocessing
        preproc = arguments.get('preprocessing')
        if preproc:
            if isinstance(preproc, list):
                for p in preproc:
                    cmd_parts.append(f"--preprocessing {p}")
            else:
                cmd_parts.append(f"--preprocessing {preproc}")
                
        # Check parameters
        params = arguments.get('algorithm_params') or arguments.get('params')
        if params:
            cmd_parts.append(f"--params '{json.dumps(params)}'")
            
        # stage_kwargs extra options
        stage_kwargs = arguments.get('stage_kwargs') or {}
        for k, v in stage_kwargs.items():
            if v is True:
                cmd_parts.append(f"--{k.replace('_', '-')}")
            elif v is not False and v is not None:
                cmd_parts.append(f"--{k.replace('_', '-')} {v}")

    elif tool_name == "execute_predict":
        cmd_parts.append("predict")
        for opt in ['stage', 'model_path', 'model_id', 'data', 'output']:
            val = arguments.get(opt)
            if val is not None:
                cmd_parts.append(f"--{opt.replace('_', '-')} {val}")
        
        stage_kwargs = arguments.get('stage_kwargs') or {}
        for k, v in stage_kwargs.items():
            if v is True:
                cmd_parts.append(f"--{k.replace('_', '-')}")
            elif v is not False and v is not None:
                cmd_parts.append(f"--{k.replace('_', '-')} {v}")

    elif tool_name == "execute_evaluate":
        cmd_parts.append("evaluate")
        for opt in ['stage', 'model_path', 'model_id', 'data', 'target', 'output']:
            val = arguments.get(opt)
            if val is not None:
                cmd_parts.append(f"--{opt.replace('_', '-')} {val}")

        stage_kwargs = arguments.get('stage_kwargs') or {}
        for k, v in stage_kwargs.items():
            if v is True:
                cmd_parts.append(f"--{k.replace('_', '-')}")
            elif v is not False and v is not None:
                cmd_parts.append(f"--{k.replace('_', '-')} {v}")

    elif tool_name == "execute_preprocess":
        cmd_parts.append("preprocess")
        for opt in ['stage', 'data', 'target', 'output']:
            val = arguments.get(opt)
            if val is not None:
                cmd_parts.append(f"--{opt.replace('_', '-')} {val}")

        stage_kwargs = arguments.get('stage_kwargs') or {}
        for k, v in stage_kwargs.items():
            if v is True:
                cmd_parts.append(f"--{k.replace('_', '-')}")
            elif v is not False and v is not None:
                cmd_parts.append(f"--{k.replace('_', '-')} {v}")

    elif tool_name == "execute_tune":
        cmd_parts.append("tune")
        for opt in ['stage', 'algorithm', 'data', 'target', 'method', 'output', 'save_path', 'model_id', 'model_path']:
            val = arguments.get(opt)
            if val is not None:
                cmd_parts.append(f"--{opt.replace('_', '-')} {val}")
                
        # param_grid
        pg = arguments.get('param_grid')
        if pg:
            cmd_parts.append(f"--param-grid '{json.dumps(pg)}'")

        stage_kwargs = arguments.get('stage_kwargs') or {}
        for k, v in stage_kwargs.items():
            if v is True:
                cmd_parts.append(f"--{k.replace('_', '-')}")
            elif v is not False and v is not None:
                cmd_parts.append(f"--{k.replace('_', '-')} {v}")

    elif tool_name == "execute_benchmark":
        cmd_parts.append("experiment")
        for opt in ['stage', 'name', 'output']:
            val = arguments.get(opt)
            if val is not None:
                cmd_parts.append(f"--{opt.replace('_', '-')} {val}")
        
        # Check algorithms
        algos = arguments.get('algorithms')
        if algos:
            for a in algos:
                cmd_parts.append(f"--algorithm {a}")
                
        # Check datasets
        datasets = arguments.get('datasets')
        if datasets:
            for d in datasets:
                cmd_parts.append(f"--dataset {d}")

        stage_kwargs = arguments.get('stage_kwargs') or {}
        for k, v in stage_kwargs.items():
            if v is True:
                cmd_parts.append(f"--{k.replace('_', '-')}")
            elif v is not False and v is not None:
                cmd_parts.append(f"--{k.replace('_', '-')} {v}")

    else:
        cmd_name = tool_name.replace("execute_", "").replace("_", "-")
        cmd_parts.append(cmd_name)
        for k, v in arguments.items():
            if v is True:
                cmd_parts.append(f"--{k.replace('_', '-')}")
            elif v is not False and v is not None:
                cmd_parts.append(f"--{k.replace('_', '-')} {v}")

    return " ".join(cmd_parts)

def parse_antigravity(file_path: str) -> List[Dict[str, Any]]:
    """Parse Antigravity JSONL transcripts."""
    cells = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            step = json.loads(line)
            
            # User turn
            if step.get("type") == "USER_INPUT":
                cells.append(make_markdown_cell(step.get("content", ""), "👤 User"))
                
            # Assistant turn
            elif step.get("type") == "PLANNER_RESPONSE":
                content = step.get("content", "")
                if content:
                    # Append assistant's thoughts and any parsed code blocks
                    cells.append(make_markdown_cell("Assistant Thoughts...", "🤖 Assistant"))
                    cells.extend(extract_code_blocks(content))
                
                # Check for direct tool calls (e.g. command runs or MCP executions)
                tool_calls = step.get("tool_calls", [])
                for call in tool_calls:
                    name = call.get("name")
                    if name == "run_command":
                        args = call.get("arguments", {})
                        if isinstance(args, str):
                            args = json.loads(args)
                        cmd = args.get("CommandLine")
                        if cmd:
                            cells.append(make_markdown_cell("**Executed Command via tool call:**"))
                            cells.append(make_code_cell(cmd, is_shell=True))
                    elif name and name.startswith("execute_"):
                        args = call.get("arguments", {})
                        cli_cmd = mcp_to_cli(name, args)
                        cells.append(make_markdown_cell(f"**MCP Tool Call: {name} (Translated to CLI)**"))
                        cells.append(make_code_cell(cli_cmd, is_shell=True))
    return cells

def parse_lm_studio(file_path: str) -> List[Dict[str, Any]]:
    """Parse LM Studio JSON exports."""
    cells = []
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    messages = data.get("messages", [])
    if not messages and isinstance(data, list):
        messages = data
        
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content", "")
        
        if role == "user":
            cells.append(make_markdown_cell(content, "👤 User"))
        elif role == "assistant":
            cells.append(make_markdown_cell("Assistant Response...", "🤖 Assistant"))
            cells.extend(extract_code_blocks(content))
            
    return cells

def parse_openai(file_path: str) -> List[Dict[str, Any]]:
    """Parse OpenAI (ChatGPT) conversations.json mapping tree structure."""
    cells = []
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    # Handles single conversation or list of conversations
    conversations = data if isinstance(data, list) else [data]
    
    for conv in conversations:
        mapping = conv.get("mapping", {})
        if not mapping:
            continue
            
        # Find leaf node to trace conversation back to root
        leaf_node_id = None
        for node_id, node in mapping.items():
            if not node.get("children"):
                leaf_node_id = node_id
                break
                
        # Reconstruct linear list of messages
        messages = []
        curr_id = leaf_node_id
        while curr_id:
            node = mapping[curr_id]
            msg = node.get("message")
            if msg and msg.get("author"):
                messages.append(msg)
            curr_id = node.get("parent")
            
        messages.reverse()
        
        for msg in messages:
            role = msg.get("author", {}).get("role")
            
            # Extract text parts
            content_obj = msg.get("content", {})
            parts = content_obj.get("parts", [])
            content_text = "".join(parts) if parts else ""
            
            if role == "user":
                cells.append(make_markdown_cell(content_text, "👤 User"))
            elif role == "assistant":
                cells.append(make_markdown_cell("Assistant Response...", "🤖 Assistant"))
                cells.extend(extract_code_blocks(content_text))
                
    return cells

def parse_claude_code(file_path: str) -> List[Dict[str, Any]]:
    """Parse Claude Code JSONL transcripts from ~/.claude/projects/."""
    cells = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            turn = json.loads(line)
            
            # Claude Code JSONL usually has messages / turns structured in roles
            role = turn.get("role")
            content = turn.get("content", "")
            
            # Content can be a string or structured list of blocks
            text_content = ""
            if isinstance(content, list):
                for block in content:
                    if block.get("type") == "text":
                        text_content += block.get("text", "")
            else:
                text_content = str(content)
                
            if role == "user":
                cells.append(make_markdown_cell(text_content, "👤 User"))
            elif role == "assistant":
                cells.append(make_markdown_cell("Assistant Response...", "🤖 Assistant"))
                cells.extend(extract_code_blocks(text_content))
                
                # Check for tool use/command execution
                if isinstance(content, list):
                    for block in content:
                        if block.get("type") == "tool_use" and block.get("name") == "bash":
                            cmd = block.get("input", {}).get("command")
                            if cmd:
                                cells.append(make_markdown_cell("**Executed Command:**"))
                                cells.append(make_code_cell(cmd, is_shell=True))
    return cells

def main():
    parser = argparse.ArgumentParser(description="Convert LLM chat exports into Jupyter Notebooks (.ipynb)")
    parser.add_argument("-f", "--format", required=True, 
                        choices=["antigravity", "lm_studio", "openai", "claude_code"],
                        help="The source chat log format type")
    parser.add_argument("-i", "--input", required=True, help="Path to input log/export file")
    parser.add_argument("-o", "--output", required=True, help="Path to output .ipynb file")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"[Error] Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)
        
    print(f"Parsing {args.format} chat log from {args.input}...")
    try:
        if args.format == "antigravity":
            cells = parse_antigravity(args.input)
        elif args.format == "lm_studio":
            cells = parse_lm_studio(args.input)
        elif args.format == "openai":
            cells = parse_openai(args.input)
        elif args.format == "claude_code":
            cells = parse_claude_code(args.input)
        else:
            cells = []
            
        if not cells:
            print("[Warning] No messages or commands were extracted.")
            
        create_notebook(cells, args.output)
    except Exception as e:
        print(f"[Error] Failed to convert chat log: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
