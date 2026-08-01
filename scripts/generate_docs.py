#!/usr/bin/env python3
"""
Documentation Generator for TuiML

Extracts docstrings from Python files and generates HTML documentation
while preserving the original folder structure.

Usage:
    python generate_docs.py [source_dir] [output_dir]

Example:
    python generate_docs.py tuiml/tuiml docs_html
"""

import ast
import os
import re
import sys
import html
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Set
from datetime import datetime

# ============================================================================
# Global config, single source of truth for generated HTML docs
# ============================================================================
DOC_CONFIG = {
    "project_name": "TuiML",
    "github_url": "https://github.com/tuiml/tuiml",
    "version": "0.1.0",
    "status": "Alpha",
    "copyright_year": datetime.now().year,
}
DOC_CONFIG["version_label"] = f"v{DOC_CONFIG['version']} {DOC_CONFIG['status']}"

#: Branch the "view source" links point at. ``main`` tracks the code the docs
#: were generated from, since the site rebuilds on every push to it. Pin this
#: to a tag if you ever publish docs for a release separately from main.
GITHUB_BRANCH = "main"

#: Path prefix from the repository root down to the package the docs cover.
#: The generator's source root is ``tuiml/tuiml``, so a module recorded as
#: ``evaluation/metrics.py`` lives at ``tuiml/evaluation/metrics.py`` in the
#: repo.
GITHUB_SOURCE_PREFIX = "tuiml"


@dataclass
class DocItem:
    """Represents a documented item (module, class, function, method)."""
    name: str
    docstring: Optional[str]
    item_type: str  # 'module', 'class', 'function', 'method'
    lineno: int = 0
    signature: str = ""
    decorators: List[str] = field(default_factory=list)
    bases: List[str] = field(default_factory=list)  # For classes
    cli_usage: str = ""  # For click commands, e.g. "tuiml benchmark [OPTIONS]"
    cli_options: List[Dict[str, Any]] = field(default_factory=list)  # click options
    methods: List["DocItem"] = field(default_factory=list)  # For classes
    children: List["DocItem"] = field(default_factory=list)  # Nested classes/functions
    module: str = ""  # Module path for display


@dataclass
class ModuleDoc:
    """Represents documentation for a Python module."""
    filepath: Path
    relative_path: Path
    module_name: str
    docstring: Optional[str]
    classes: List[DocItem] = field(default_factory=list)
    functions: List[DocItem] = field(default_factory=list)
    constants: List[str] = field(default_factory=list)


class DocstringParser:
    """Parses NumPy-style docstrings into structured sections."""

    SECTION_HEADERS = [
        'Parameters', 'Returns', 'Yields', 'Raises', 'Warns',
        'Attributes', 'Methods', 'Notes', 'Examples', 'See Also',
        'References', 'Other Parameters', 'Warnings',
        # Extended headers
        'Theory', 'Algorithm', 'Description', 'Overview', 'Usage',
        'Implementation', 'Technical Details', 'Background'
    ]

    def __init__(self, docstring: str, class_map: dict = None):
        self.raw = docstring
        self.sections: Dict[str, str] = {}
        self.summary = ""
        self.extended_summary = ""
        self._class_map = class_map or {}
        self._parse()

    def _parse(self):
        """Parse the docstring into sections."""
        if not self.raw:
            return

        lines = self.raw.strip().split('\n')

        # Find summary (first non-empty line or lines until blank line)
        summary_lines = []
        i = 0
        while i < len(lines):
            stripped = lines[i].strip()
            if not stripped:
                i += 1  # Move past the blank line
                break
            summary_lines.append(stripped)
            i += 1
            # If we only have summary lines (single paragraph), stop after them
            if i >= len(lines):
                break
        
        self.summary = ' '.join(summary_lines)

        # Skip blank lines after summary
        while i < len(lines) and not lines[i].strip():
            i += 1

        # Find extended summary (until first section header)
        extended_lines = []
        while i < len(lines):
            line = lines[i]
            # Check if this is a section header
            if self._is_section_header(lines, i):
                break
            extended_lines.append(line)
            i += 1

        self.extended_summary = '\n'.join(extended_lines).strip()

        # Parse sections
        current_section = None
        section_content = []

        while i < len(lines):
            line = lines[i]

            if self._is_section_header(lines, i):
                # Save previous section
                if current_section:
                    self.sections[current_section] = '\n'.join(section_content).strip()

                current_section = line.strip()
                section_content = []
                i += 2  # Skip header and underline
            else:
                section_content.append(line)
                i += 1

        # Save last section
        if current_section:
            self.sections[current_section] = '\n'.join(section_content).strip()

    def _is_section_header(self, lines: List[str], index: int) -> bool:
        """Check if line at index is a section header."""
        if index >= len(lines):
            return False

        line = lines[index].strip()

        # A header is a short title line at the left margin. Besides the known
        # names, accept any custom title (NumPy style allows e.g. "Example" or
        # "Example, one-liner") as long as it carries a dash underline.
        if not line or lines[index][:1] in (' ', '\t'):
            return False
        if line not in self.SECTION_HEADERS and len(line) > 60:
            return False

        # Check for underline
        if index + 1 < len(lines):
            underline = lines[index + 1].strip()
            if len(underline) >= 3 and all(c == '-' for c in underline):
                return True

        return False

    def to_html(self) -> str:
        """Convert parsed docstring to HTML."""
        parts = []

        if self.summary:
            parts.append(
                f'<p class="doc-summary text-lg font-semibold text-gray-900 mb-3">'
                f'{self._format_inline_code(html.escape(self.summary))}</p>'
            )

        if self.extended_summary:
            parts.append(f'<div class="doc-extended">{self._format_text(self.extended_summary)}</div>')

        for section_name, content in self.sections.items():
            parts.append(self._format_section(section_name, content))

        return '\n'.join(parts)

    def _format_section(self, name: str, content: str) -> str:
        """Format a docstring section as HTML."""
        html_content = ""

        if name in ('Parameters', 'Returns', 'Yields', 'Raises', 'Attributes', 'Other Parameters'):
            html_content = self._format_param_section(content)
        elif name.startswith('Example') or name.startswith('Usage'):
            # The dark code block is self-explanatory; skip the section box
            # chrome and render the example content directly.
            return f'<div class="mb-8 last:mb-0">{self._format_examples(content)}</div>'
        elif name == 'See Also':
            html_content = self._format_see_also(content)
        elif name == 'References':
            html_content = self._format_references(content)
        else:
            html_content = self._format_text(content)

        # Choose icon based on section type
        icon = ''
        if name == 'Parameters':
            icon = '<i class="fa-solid fa-sliders text-indigo-500 mr-2"></i>'
        elif name == 'Returns':
            icon = '<i class="fa-solid fa-arrow-right-from-bracket text-green-500 mr-2"></i>'
        elif name == 'Attributes':
            icon = '<i class="fa-solid fa-list-check text-blue-500 mr-2"></i>'
        elif name.startswith('Example') or name.startswith('Usage'):
            icon = '<i class="fa-solid fa-code text-purple-500 mr-2"></i>'
        elif name == 'Raises':
            icon = '<i class="fa-solid fa-triangle-exclamation text-red-500 mr-2"></i>'
        elif name == 'See Also':
            icon = '<i class="fa-solid fa-link text-gray-500 mr-2"></i>'
        elif name == 'References':
            icon = '<i class="fa-solid fa-book text-amber-500 mr-2"></i>'
        elif name == 'Notes':
            icon = '<i class="fa-solid fa-sticky-note text-yellow-500 mr-2"></i>'
        else:
            icon = '<i class="fa-solid fa-info-circle text-gray-400 mr-2"></i>'
            
        return f'''
        <div class="mb-8 last:mb-0 bg-white rounded-lg border border-gray-200 overflow-hidden">
            <div class="bg-gray-50 px-4 py-3 border-b border-gray-200">
                <h4 class="font-bold text-gray-800 text-sm uppercase tracking-wide flex items-center">{icon}{html.escape(name)}</h4>
            </div>
            <div class="p-4">{html_content}</div>
        </div>
        '''

    def _format_param_section(self, content: str) -> str:
        """Format Parameters/Returns/Attributes sections with table-like layout."""
        lines = content.split('\n')
        items = []
        current_param = None
        current_desc = []

        for line in lines:
            # Check if this is a parameter definition line
            if line and not line.startswith(' ') and not line.startswith('\t'):
                # Save previous parameter
                if current_param:
                    items.append((current_param, current_desc))

                # Parse new parameter
                if ':' in line:
                    parts = line.split(':', 1)
                    param_name = parts[0].strip()
                    param_type = parts[1].strip() if len(parts) > 1 else ''
                    current_param = (param_name, param_type)
                else:
                    current_param = (line.strip(), '')
                current_desc = []
            else:
                current_desc.append(line.strip())

        # Save last parameter
        if current_param:
            items.append((current_param, current_desc))

        if not items:
            return f'<p class="text-gray-500 text-sm italic">No parameters documented.</p>'

        # Build HTML as a responsive table/grid
        html_items = ['<div class="divide-y divide-gray-100">']
        for (name, type_info), desc_lines in items:
            # Parse type info for default values
            default_match = re.search(r'default[=:\s]+([\'"]?[^\'",\)]+[\'"]?)', type_info, re.IGNORECASE)
            default_value = default_match.group(1).strip() if default_match else None
            
            # Clean up type info (remove 'optional' and 'default' parts for the badge)
            clean_type = re.sub(r',?\s*optional', '', type_info)
            clean_type = re.sub(r',?\s*default[=:\s]+[^,\)]+', '', clean_type).strip()
            
            # Type badge color based on type
            type_color = 'bg-gray-100 text-gray-700'
            if clean_type:
                if 'int' in clean_type.lower():
                    type_color = 'bg-blue-100 text-blue-700'
                elif 'float' in clean_type.lower():
                    type_color = 'bg-green-100 text-green-700'
                elif 'str' in clean_type.lower():
                    type_color = 'bg-yellow-100 text-yellow-800'
                elif 'bool' in clean_type.lower():
                    type_color = 'bg-purple-100 text-purple-700'
                elif 'list' in clean_type.lower() or 'array' in clean_type.lower():
                    type_color = 'bg-orange-100 text-orange-700'
                elif 'dict' in clean_type.lower():
                    type_color = 'bg-pink-100 text-pink-700'
            
            type_badge = f'<span class="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium {type_color}">{html.escape(clean_type)}</span>' if clean_type else ''
            default_badge = f'<span class="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-blue-50 text-blue-600 ml-2">= {html.escape(default_value)}</span>' if default_value else ''
            
            # Format description with bullet point support
            desc_html = self._format_param_description(desc_lines)
            
            html_items.append(f'''
            <div class="py-4 grid grid-cols-1 md:grid-cols-12 gap-2 md:gap-4 hover:bg-gray-50 transition-colors px-2 rounded">
                <div class="md:col-span-3">
                    <code class="font-bold text-indigo-600 text-sm">{html.escape(name)}</code>
                </div>
                <div class="md:col-span-3 flex flex-wrap items-center gap-1">
                    {type_badge}
                    {default_badge}
                </div>
                <div class="md:col-span-6 text-sm text-gray-600">
                    {desc_html}
                </div>
            </div>
            ''')
        
        html_items.append('</div>')
        return '\n'.join(html_items)

    def _format_param_description(self, desc_lines: list) -> str:
        """Format parameter description, handling bullet points and inline code."""
        if not desc_lines:
            return ''
        
        result_parts = []
        in_list = False
        list_items = []
        text_buffer = []
        
        for line in desc_lines:
            # Check if line is a bullet point (starts with - or *)
            bullet_match = re.match(r'^[\-\*]\s+(.+)$', line)
            
            if bullet_match:
                # Save any accumulated text before the list
                if text_buffer and not in_list:
                    text = ' '.join(text_buffer)
                    result_parts.append(f'<p class="mb-2">{self._format_inline_code(html.escape(text))}</p>')
                    text_buffer = []
                
                in_list = True
                list_items.append(bullet_match.group(1))
            else:
                # If we were in a list, close it
                if in_list and list_items:
                    result_parts.append(self._render_option_list(list_items))
                    list_items = []
                    in_list = False
                
                # Skip empty lines
                if line.strip():
                    text_buffer.append(line)
        
        # Handle remaining content
        if in_list and list_items:
            result_parts.append(self._render_option_list(list_items))
        elif text_buffer:
            text = ' '.join(text_buffer)
            result_parts.append(f'{self._format_inline_code(html.escape(text))}')
        
        return ''.join(result_parts)

    def _render_option_list(self, items: list) -> str:
        """Render a list of options as styled HTML."""
        html_items = ['<ul class="mt-2 space-y-1">']
        for item in items:
            # Parse option format: ``"value"``: Description
            option_match = re.match(r'``([^`]+)``\s*[-\-]\s*(.+)', item)
            if option_match:
                option_value = option_match.group(1)
                option_desc = option_match.group(2)
                html_items.append(f'''
                    <li class="flex items-start gap-2">
                        <code class="bg-slate-100 text-slate-700 px-1.5 py-0.5 rounded text-xs font-mono flex-shrink-0">{html.escape(option_value)}</code>
                        <span class="text-gray-600">{html.escape(option_desc)}</span>
                    </li>
                ''')
            else:
                # Simple bullet item
                html_items.append(f'<li class="flex items-start gap-2"><span class="text-gray-400">•</span><span>{self._format_inline_code(html.escape(item))}</span></li>')
        html_items.append('</ul>')
        return '\n'.join(html_items)

    def _format_inline_code(self, text: str) -> str:
        """Format inline code markers `` `` to HTML code tags."""
        # RST cross-reference roles: show the target name as inline code.
        text = re.sub(
            r':(?:func|class|meth|attr|data|mod|obj|exc):`~?([^`]+)`',
            lambda m: '<code class="bg-slate-100 text-slate-700 px-1 py-0.5 rounded text-xs font-mono">'
                      + m.group(1).rsplit('.', 1)[-1] + '</code>',
            text,
        )
        # Replace ``code`` with styled code spans
        text = re.sub(
            r'``([^`]+)``',
            r'<code class="bg-slate-100 text-slate-700 px-1 py-0.5 rounded text-xs font-mono">\1</code>',
            text
        )
        # Summaries are written with **bold** key concepts (see the docstring
        # convention); without this they render as literal asterisks.
        return self._format_emphasis(text)

    @staticmethod
    def _format_emphasis(text: str) -> str:
        """Convert ``**bold**`` and ``*italic*`` markers to HTML.

        Parameters
        ----------
        text : str
            Already HTML-escaped text.

        Returns
        -------
        rendered : str
            Text with emphasis markers replaced by <strong>/<em> tags.
        """
        # Bold first: a lone '*' rule would otherwise eat the '**' pairs.
        text = re.sub(r'\*\*([^*]+)\*\*', r'<strong>\1</strong>', text)
        return re.sub(r'(?<![*\w])\*([^*\s][^*]*)\*(?!\w)', r'<em>\1</em>', text)

    def _format_examples(self, content: str) -> str:
        """Format Examples/Usage sections with dark code blocks.

        Handles two kinds of code: ``>>>`` doctest blocks (kept with their
        prompts and output lines) and RST literal blocks, i.e. indented lines
        introduced by prose ending in ``::``.
        """
        import textwrap

        lines = content.split('\n')
        output = []
        code_block = []
        mode = None  # None | 'doctest' | 'literal'
        unique_id = f'example-{hash(content) % 10000}'

        def flush():
            nonlocal code_block, mode
            if code_block:
                code = '\n'.join(code_block)
                if mode == 'literal':
                    code = textwrap.dedent(code)
                block_id = f'{unique_id}-{len(output)}'
                output.append(self._create_code_block(code, block_id))
                code_block = []
            mode = None

        for line in lines:
            stripped = line.strip()
            indented = line[:1] in (' ', '\t')

            if stripped.startswith('>>>') or stripped.startswith('$ ') \
                    or (mode == 'doctest' and stripped.startswith('...')):
                if mode == 'literal':
                    flush()
                mode = 'doctest'
                code_block.append(html.escape(line))
            elif mode == 'doctest' and stripped:
                # Doctest output line
                code_block.append(html.escape(line))
            elif not stripped:
                # Blank line ends the current code block
                flush()
            elif indented:
                # Indented literal block (RST ``::`` style)
                if mode == 'doctest':
                    flush()
                mode = 'literal'
                code_block.append(html.escape(line))
            else:
                flush()
                # "Run the server::" reads as "Run the server:" in HTML
                text = stripped[:-1] if stripped.endswith('::') else stripped
                output.append(f'<p class="text-gray-600 my-2">{self._format_inline_code(html.escape(text))}</p>')

        flush()
        return '\n'.join(output)

    #: Commands that mark a code block as shell rather than Python.
    _SHELL_COMMANDS = (
        'tuiml', 'tuiml-mcp', 'pip', 'pipx', 'uv', 'uvx', 'python -m', 'python3 -m',
        'cd', 'ls', 'export', 'git', 'curl', 'wget', 'bash', 'sh ', 'make',
        'pytest', 'docker', 'npm', 'brew', 'sudo', 'chmod', 'mkdir', 'echo',
    )

    def _detect_code_language(self, code: str) -> str:
        """Detect whether a code block holds shell commands or Python.

        CLI docstrings write their examples with ``>>>`` prompts even though
        the content is shell, so the prompt alone cannot decide the language.

        Parameters
        ----------
        code : str
            The (HTML-escaped) code block contents.

        Returns
        -------
        language : str
            Either ``'bash'`` or ``'python'``.
        """
        plain = html.unescape(code)
        for raw in plain.split('\n'):
            line = raw.strip()
            # Strip a doctest or shell prompt before inspecting the command.
            for prompt in ('>>>', '...', '$'):
                if line.startswith(prompt):
                    line = line[len(prompt):].strip()
                    break
            if not line or line.startswith('#'):
                continue
            if line.startswith(self._SHELL_COMMANDS):
                return 'bash'
            if line[:1] in ('{', '['):
                return 'json'
            return 'python'
        return 'python'

    def _create_code_block(self, code: str, block_id: str) -> str:
        """Create a styled code block with copy button matching readme.html style."""
        language = self._detect_code_language(code)
        if language == 'bash':
            # Shell examples read better with a ``$`` prompt than a doctest one.
            code = '\n'.join(
                re.sub(r'^(\s*)&gt;&gt;&gt; ?', r'\1$ ', line)
                for line in code.split('\n')
            )
        return f'''
        <div class="code-block-wrapper !bg-slate-900 !rounded-xl !border-slate-800 !p-0 overflow-hidden shadow-md group !my-2">
            <div class="bg-slate-950/50 px-4 py-2 flex items-center justify-between border-b border-white/5">
                <div class="flex gap-1.5 opacity-60">
                    <div class="w-2.5 h-2.5 rounded-full bg-red-500/80"></div>
                    <div class="w-2.5 h-2.5 rounded-full bg-yellow-500/80"></div>
                    <div class="w-2.5 h-2.5 rounded-full bg-green-500/80"></div>
                </div>
                <div class="flex items-center gap-2">
                    <span class="text-[10px] font-mono text-slate-500 font-bold uppercase tracking-wider">{language}</span>
                    <button class="copy-btn text-slate-500 hover:text-white transition-colors !p-1">
                        <i class="fa-regular fa-copy text-xs"></i>
                    </button>
                </div>
            </div>
            <div class="!p-4 overflow-x-auto">
                <pre class="!m-0 !p-0 !bg-transparent font-mono text-xs text-blue-100 leading-relaxed"><code class="language-{language}">{code}</code></pre>
            </div>
        </div>
        '''

    def _resolve_reference(self, ref: str):
        """Resolve an RST cross-reference target to a URL, or None.

        Tries the full dotted path first, then progressively drops trailing
        components: ``tuiml.workflow.Workflow.fit`` is a method, which has no
        page of its own, so it resolves to the page documenting ``Workflow``.
        The bare final name is the last resort, since it is the most ambiguous
        — several packages define a ``fit``.

        Returning None is meaningful: it says nothing documents this target,
        and the caller renders plain text instead of a link. The previous
        behaviour guessed a path from the name, which produced URLs like
        ``/docs/utils/serialization/load_model.html`` for a function that is
        documented on its module's page, and ``#`` when there was nothing to
        guess from.

        Parameters
        ----------
        ref : str
            Dotted reference target, e.g. ``tuiml.algorithms.svm.SMOReg``.

        Returns
        -------
        url : str or None
            The URL documenting it, or None if it is not documented.
        """
        if not ref:
            return None

        parts = ref.split('.')
        for cut in range(len(parts), 0, -1):
            candidate = '.'.join(parts[:cut])
            href = self._class_map.get(candidate)
            if href:
                return href

        return self._class_map.get(parts[-1])

    def _format_see_also(self, content: str) -> str:
        """Format See Also section with clickable links."""
        # NumPy style wraps a long description onto indented continuation
        # lines. Stitch each entry back together before parsing, or every
        # wrapped line renders as its own orphaned card.
        entries: List[str] = []
        for raw in content.strip('\n').split('\n'):
            if not raw.strip():
                continue
            if raw[:1].isspace() and entries:
                entries[-1] += ' ' + raw.strip()
            else:
                entries.append(raw.strip())

        items = []
        for line in entries:
            if line:
                # Try to extract reference and description from :class:`~path.ClassName` : description
                match = re.match(
                    r':(?:class|func|meth|attr|data|mod|obj|exc):`~?([^`]+)`\s*:?\s*(.*)',
                    line,
                )
                if match:
                    ref, desc = match.groups()
                    parts = ref.split('.')
                    class_name = parts[-1] if parts else ref

                    href = self._resolve_reference(ref)

                    if href:
                        items.append(f'''
                    <a href="{href}" class="flex items-center gap-3 p-3 bg-gray-50 rounded-lg border border-gray-200 hover:bg-indigo-50 hover:border-indigo-200 transition-colors group">
                        <span class="inline-flex items-center justify-center w-8 h-8 rounded-lg bg-indigo-100 text-indigo-600 group-hover:bg-indigo-200">
                            <i class="fa-solid fa-link text-sm"></i>
                        </span>
                        <div>
                            <code class="font-bold text-indigo-600 group-hover:text-indigo-800">{html.escape(class_name)}</code>
                            <p class="text-sm text-gray-600 mt-0.5">{self._format_inline_code(html.escape(desc))}</p>
                        </div>
                    </a>
                    ''')
                    else:
                        # Nothing documents this reference. Render it as text
                        # rather than a link: a dead link looks like a working
                        # one until it is clicked, and the site's 404 is easy
                        # to mistake for a real page.
                        items.append(f'''
                    <div class="flex items-center gap-3 p-3 bg-gray-50 rounded-lg border border-gray-200">
                        <code class="font-bold text-gray-700">{html.escape(class_name)}</code>
                        <span class="text-sm text-gray-600">{self._format_inline_code(html.escape(desc))}</span>
                    </div>
                    ''')
                else:
                    # Simple format: ClassName : description
                    simple_match = re.match(r'(\w+)\s*:\s*(.*)', line)
                    if simple_match:
                        name, desc = simple_match.groups()
                        items.append(f'''
                        <div class="flex items-center gap-3 p-3 bg-gray-50 rounded-lg border border-gray-200">
                            <code class="font-bold text-gray-700">{html.escape(name)}</code>
                            <span class="text-sm text-gray-600">{self._format_inline_code(html.escape(desc))}</span>
                        </div>
                        ''')
                    else:
                        items.append(f'<div class="p-3 bg-gray-50 rounded-lg border border-gray-200 text-gray-600">{self._format_inline_code(html.escape(line))}</div>')

        return f'<div class="space-y-2">{"".join(items)}</div>'

    def _format_references(self, content: str) -> str:
        """Format References section with proper academic citation styling."""
        # Handle reStructuredText citation format: .. [Label] content
        refs = re.split(r'\.\.\s*\[([^\]]+)\]', content)
        items = []
        
        # refs will be: ['', 'Label1', 'content1', 'Label2', 'content2', ...]
        i = 1
        while i < len(refs):
            if i + 1 < len(refs):
                label = refs[i].strip()
                ref_content = refs[i + 1].strip()
                
                # Format the reference content
                # Handle **title** for bold titles
                ref_content = re.sub(r'\*\*([^*]+)\*\*', r'<strong class="text-gray-900">\1</strong>', ref_content)
                # Handle *journal* for italic
                ref_content = re.sub(r'\*([^*]+)\*', r'<em class="text-gray-700">\1</em>', ref_content)
                # Handle DOI links: `text <url>`_
                ref_content = re.sub(
                    r'`([^<]+)\s*<([^>]+)>`_',
                    r'<a href="\2" target="_blank" class="text-indigo-600 hover:text-indigo-800 hover:underline">\1</a>',
                    ref_content
                )
                # Escape remaining HTML but preserve our formatted parts
                # Split by tags, escape non-tags
                parts = re.split(r'(<[^>]+>)', ref_content)
                escaped_parts = []
                for part in parts:
                    if part.startswith('<') and (part.endswith('>') or '>' in part):
                        escaped_parts.append(part)
                    else:
                        escaped_parts.append(html.escape(part))
                ref_content = ''.join(escaped_parts)
                
                items.append(f'''
                <div class="bg-gray-50 rounded-lg border border-gray-200 p-4 mb-3">
                    <div class="flex items-start gap-3">
                        <span class="inline-flex items-center justify-center px-2 py-1 rounded bg-indigo-100 text-indigo-700 text-xs font-bold font-mono flex-shrink-0">{html.escape(label)}</span>
                        <div class="text-sm text-gray-600 leading-relaxed">{ref_content}</div>
                    </div>
                </div>
                ''')
            i += 2

        if not items:
            # Fallback for simple references
            return f'<p class="text-gray-600">{html.escape(content)}</p>'

        return f'<div class="reference-list space-y-2">{"".join(items)}</div>'

    def _format_text(self, text: str) -> str:
        """Format general text with markdown-like handling and LaTeX math."""
        # First, extract and protect math blocks
        math_blocks = []
        
        def save_math(match):
            math_blocks.append(match.group(1).strip())
            return f'{{{{MATH_{len(math_blocks)-1}}}}}'
        
        # Handle block math: .. math::\n    formula
        text = re.sub(r'\.\. math::\s*\n\s*(.+?)(?=\n\n|\n[^\s]|$)', save_math, text, flags=re.DOTALL)
        
        # Handle inline math with :math:`...`
        def save_inline_math(match):
            math_blocks.append(match.group(1).strip())
            return f'{{{{INLINEMATH_{len(math_blocks)-1}}}}}'
        
        text = re.sub(r':math:`([^`]+)`', save_inline_math, text)

        # RST literal blocks: a line ending in "::" introduces indented code.
        # Pull them out as placeholders so list/paragraph handling and HTML
        # escaping cannot mangle them; restored as <pre> blocks at the end.
        literal_blocks = []
        raw_lines = text.split('\n')
        kept_lines = []
        i = 0
        while i < len(raw_lines):
            line = raw_lines[i]
            if line.rstrip().endswith('::') and line.strip() != '::':
                # RST renders "text::" as "text:".
                kept_lines.append(line.rstrip()[:-2] + ':')
                i += 1
                while i < len(raw_lines) and not raw_lines[i].strip():
                    i += 1
                block = []
                while i < len(raw_lines) and (
                    raw_lines[i].startswith((' ', '\t')) or not raw_lines[i].strip()
                ):
                    block.append(raw_lines[i])
                    i += 1
                while block and not block[-1].strip():
                    block.pop()
                if block:
                    indent = min(
                        len(b) - len(b.lstrip()) for b in block if b.strip()
                    )
                    code = '\n'.join(b[indent:] for b in block)
                    kept_lines.append('')
                    kept_lines.append(f'{{{{LITERAL_{len(literal_blocks)}}}}}')
                    kept_lines.append('')
                    literal_blocks.append(code)
            else:
                kept_lines.append(line)
                i += 1
        text = '\n'.join(kept_lines)

        # Process line by line to handle lists
        lines = text.split('\n')
        result_lines = []
        in_numbered_list = False
        in_bullet_list = False
        in_where_block = False
        
        for line in lines:
            stripped = line.strip()
            
            # Check for numbered list item (1. 2. 3. etc.)
            numbered_match = re.match(r'^(\d+)\.\s+(.+)$', stripped)
            # Check for bullet list item (- item)
            bullet_match = re.match(r'^-\s+(.+)$', stripped)
            # Check for where: block start
            where_match = re.match(r'^where:?\s*$', stripped.lower())
            
            if numbered_match:
                if not in_numbered_list:
                    result_lines.append('<ol class="list-decimal list-inside my-3 space-y-1">')
                    in_numbered_list = True
                # Escaped later by the split-by-tags pass; escaping here
                # too would double-encode quotes into visible &quot; text.
                result_lines.append(f'<li class="text-gray-700">{numbered_match.group(2)}</li>')
            elif bullet_match:
                if not in_bullet_list:
                    if in_numbered_list:
                        result_lines.append('</ol>')
                        in_numbered_list = False
                    result_lines.append('<ul class="list-disc list-inside my-3 space-y-1 pl-4">')
                    in_bullet_list = True
                result_lines.append(f'<li class="text-gray-700">{bullet_match.group(1)}</li>')
            elif where_match:
                # Close any open lists
                if in_numbered_list:
                    result_lines.append('</ol>')
                    in_numbered_list = False
                if in_bullet_list:
                    result_lines.append('</ul>')
                    in_bullet_list = False
                result_lines.append('<div class="mt-3 mb-2 font-medium text-gray-800">where:</div>')
                in_where_block = True
            else:
                # Close lists if needed
                if in_numbered_list and stripped:
                    result_lines.append('</ol>')
                    in_numbered_list = False
                if in_bullet_list and stripped and not stripped.startswith('-'):
                    result_lines.append('</ul>')
                    in_bullet_list = False
                
                result_lines.append(line)
        
        # Close any remaining open lists
        if in_numbered_list:
            result_lines.append('</ol>')
        if in_bullet_list:
            result_lines.append('</ul>')
        
        text = '\n'.join(result_lines)
        
        # Now escape HTML for non-list content
        # Split by tags to preserve HTML we just added
        parts = re.split(r'(<[^>]+>)', text)
        escaped_parts = []
        for part in parts:
            if part.startswith('<') and part.endswith('>'):
                escaped_parts.append(part)  # Keep HTML tags
            else:
                escaped_parts.append(html.escape(part))
        text = ''.join(escaped_parts)

        # RST cross-reference roles first, so ':func:' does not leak as text.
        text = re.sub(
            r':(?:func|class|meth|attr|data|mod|obj|exc):`~?([^`]+)`',
            lambda m: '<code class="bg-gray-100 px-1 py-0.5 rounded text-sm font-mono">'
                      + m.group(1).rsplit('.', 1)[-1] + '</code>',
            text,
        )

        # Handle inline code
        text = re.sub(r'``([^`]+)``', r'<code class="bg-gray-100 px-1 py-0.5 rounded text-sm font-mono">\1</code>', text)
        text = re.sub(r'`([^`]+)`', r'<code class="bg-gray-100 px-1 py-0.5 rounded text-sm font-mono">\1</code>', text)

        # Handle bold
        text = re.sub(r'\*\*([^*]+)\*\*', r'<strong>\1</strong>', text)
        
        # Handle italic
        text = re.sub(r'\*([^*]+)\*', r'<em>\1</em>', text)

        # Restore math blocks with KaTeX markup
        for i, math in enumerate(math_blocks):
            # Clean up the LaTeX
            clean_math = math.replace('\\\\', '\\')
            # Block math
            text = text.replace(f'{{{{MATH_{i}}}}}', 
                f'<div class="my-4 p-4 bg-gray-50 rounded-lg border border-gray-200 overflow-x-auto"><span class="katex-display">{html.escape(clean_math)}</span></div>')
            # Inline math
            text = text.replace(f'{{{{INLINEMATH_{i}}}}}',
                f'<span class="katex-inline">{html.escape(clean_math)}</span>')

        # Convert double line breaks to paragraphs (but preserve existing HTML)
        # Only for text not inside tags
        paragraphs = re.split(r'\n\n+', text)
        if len(paragraphs) > 1:
            formatted_paragraphs = []
            for p in paragraphs:
                p = p.strip()
                if p:
                    # Don't wrap if it contains or is a block element. Literal
                    # blocks are still placeholders at this point but expand to
                    # a <div>, so wrapping them would nest a div inside a <p>.
                    if ('{{LITERAL_' in p
                            or any(tag in p for tag in ['<ol', '</ol>', '<ul', '</ul>', '<div', '</div>', '<li'])):
                        formatted_paragraphs.append(p)
                    else:
                        formatted_paragraphs.append(f'<p class="mb-4">{p}</p>')
            text = '\n'.join(formatted_paragraphs)
        
        # Clean up stray newlines within the text
        text = re.sub(r'\n+', ' ', text)
        # But restore newlines around block elements
        text = re.sub(r'\s*(</?(?:ol|ul|li|div|p)[^>]*>)\s*', r'\n\1\n', text)
        # Clean up excessive newlines
        text = re.sub(r'\n{3,}', '\n\n', text)

        # Restore RST literal blocks last: the newline cleanup above must not
        # reach into their <pre> content, or code collapses onto one line.
        for i, code in enumerate(literal_blocks):
            text = text.replace(
                f'{{{{LITERAL_{i}}}}}',
                '<div class="bg-slate-900 rounded-lg p-4 my-3 overflow-x-auto">'
                '<pre class="text-sm font-mono text-slate-100 leading-relaxed">'
                + html.escape(code) + '</pre></div>'
            )

        return text


class PythonDocExtractor:
    """Extracts documentation from Python source files."""

    def __init__(self, source_path: Path):
        self.source_path = source_path

    def extract(self) -> Optional[ModuleDoc]:
        """Extract documentation from the Python file."""
        try:
            with open(self.source_path, 'r', encoding='utf-8') as f:
                source = f.read()
        except Exception as e:
            print(f"Error reading {self.source_path}: {e}")
            return None

        try:
            tree = ast.parse(source)
        except SyntaxError as e:
            print(f"Syntax error in {self.source_path}: {e}")
            return None

        # Always use file stem for module name (e.g., __init__.py -> __init__)
        module_name = self.source_path.stem
        module_docstring = ast.get_docstring(tree)

        doc = ModuleDoc(
            filepath=self.source_path,
            relative_path=self.source_path,
            module_name=module_name,
            docstring=module_docstring
        )

        for node in ast.iter_child_nodes(tree):
            # Underscore-prefixed names are internal helpers, not part of the
            # public API users call, keep them out of the reference.
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) \
                    and node.name.startswith('_'):
                continue
            if isinstance(node, ast.ClassDef):
                class_doc = self._extract_class(node)
                doc.classes.append(class_doc)
            elif isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
                func_doc = self._extract_function(node)
                doc.functions.append(func_doc)
            elif isinstance(node, ast.Assign):
                # Extract module-level constants
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id.isupper():
                        doc.constants.append(target.id)

        return doc

    def _extract_class(self, node: ast.ClassDef) -> DocItem:
        """Extract documentation from a class definition."""
        bases = []
        for base in node.bases:
            if isinstance(base, ast.Name):
                bases.append(base.id)
            elif isinstance(base, ast.Attribute):
                bases.append(f"{self._get_attribute_name(base)}")

        decorators = [self._get_decorator_name(d) for d in node.decorator_list]

        doc_item = DocItem(
            name=node.name,
            docstring=ast.get_docstring(node),
            item_type='class',
            lineno=node.lineno,
            decorators=decorators,
            bases=bases
        )

        # Extract methods
        for child in node.body:
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                method_doc = self._extract_function(child, is_method=True)
                doc_item.methods.append(method_doc)
            elif isinstance(child, ast.ClassDef):
                nested_class = self._extract_class(child)
                doc_item.children.append(nested_class)

        # Pydantic models document their fields via Field(description=...)
        # annotations rather than a docstring section. Synthesize a NumPy
        # "Attributes" section from them so schemas render with the same
        # field tables as hand-written docstrings.
        if 'BaseModel' in bases and 'Attributes' not in (doc_item.docstring or ''):
            fields_section = self._pydantic_fields_section(node)
            if fields_section:
                doc_item.docstring = (
                    (doc_item.docstring or '').rstrip() + '\n\n' + fields_section
                )

        return doc_item

    def _pydantic_fields_section(self, node: ast.ClassDef) -> str:
        """Build a NumPy-style Attributes section from Pydantic field annotations.

        Parameters
        ----------
        node : ast.ClassDef
            The class definition to scan for annotated fields.

        Returns
        -------
        section : str
            An ``Attributes`` docstring section, or ``''`` if the class has
            no documented fields.
        """
        lines = []
        for child in node.body:
            if not isinstance(child, ast.AnnAssign) or not isinstance(child.target, ast.Name):
                continue
            name = child.target.id
            if name.startswith('_'):
                continue

            try:
                type_str = ast.unparse(child.annotation)
            except Exception:
                type_str = ''

            default = None
            required = child.value is None
            description = ''

            value = child.value
            if isinstance(value, ast.Call) and self._get_decorator_name(value.func).endswith('Field'):
                if value.args:
                    first = value.args[0]
                    if isinstance(first, ast.Constant) and first.value is Ellipsis:
                        required = True
                    else:
                        default = ast.unparse(first)
                for kw in value.keywords:
                    if kw.arg == 'default':
                        default = ast.unparse(kw.value)
                    elif kw.arg == 'default_factory':
                        factory = ast.unparse(kw.value)
                        default = {'dict': '{}', 'list': '[]', 'tuple': '()'}.get(
                            factory, f"{factory}()"
                        )
                    elif kw.arg == 'description' and isinstance(kw.value, ast.Constant):
                        description = str(kw.value.value)
            elif value is not None:
                default = ast.unparse(value)

            type_line = f"{name} : {type_str}" if type_str else name
            if default is not None:
                type_line += f", default={default}"
            lines.append(type_line)
            if description:
                lines.append(f"    {description}")

        if not lines:
            return ''
        return 'Attributes\n----------\n' + '\n'.join(lines)

    def _extract_function(self, node: ast.FunctionDef, is_method: bool = False) -> DocItem:
        """Extract documentation from a function/method definition."""
        decorators = [self._get_decorator_name(d) for d in node.decorator_list]
        signature = self._get_function_signature(node)
        cli_usage = self._click_usage(node, decorators)

        # A click command is documented by how it is invoked in a shell; the
        # long @click.option stack is implementation detail, not API surface.
        if cli_usage:
            decorators = [d for d in decorators if not d.startswith('click.')]

        return DocItem(
            name=node.name,
            docstring=ast.get_docstring(node),
            item_type='method' if is_method else 'function',
            lineno=node.lineno,
            signature=signature,
            decorators=decorators,
            cli_usage=cli_usage,
            cli_options=self._click_options(node) if cli_usage else [],
        )

    def _click_options(self, node: ast.FunctionDef) -> List[Dict[str, Any]]:
        """Extract the options and arguments a click command accepts.

        The ``help=`` text on each ``@click.option`` is the real user-facing
        documentation for a command, so it is lifted out of the decorator
        stack and rendered as a table instead of being shown as source.

        Parameters
        ----------
        node : ast.FunctionDef
            The decorated command function.

        Returns
        -------
        options : List[Dict[str, Any]]
            One dict per option with ``flags``, ``type``, ``default``,
            ``required``, ``help`` and ``is_argument`` keys, in declaration
            order.
        """
        options = []
        # Decorators apply bottom-up, so the source order reads top-down.
        for decorator in node.decorator_list:
            if not isinstance(decorator, ast.Call):
                continue
            kind = self._get_decorator_name(decorator.func)
            if kind not in ('click.option', 'click.argument'):
                continue

            flags = []
            for arg in decorator.args:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    flags.append(arg.value)

            entry = {
                'flags': [f for f in flags if f.startswith('-')] or flags,
                'type': '', 'default': None, 'required': False,
                'help': '', 'is_argument': kind == 'click.argument',
                'multiple': False,
            }

            for kw in decorator.keywords:
                if kw.arg == 'help' and isinstance(kw.value, ast.Constant):
                    entry['help'] = str(kw.value.value)
                elif kw.arg == 'required':
                    entry['required'] = bool(getattr(kw.value, 'value', False))
                elif kw.arg == 'multiple':
                    entry['multiple'] = bool(getattr(kw.value, 'value', False))
                elif kw.arg == 'is_flag' and getattr(kw.value, 'value', False):
                    entry['type'] = 'flag'
                elif kw.arg == 'default':
                    try:
                        entry['default'] = ast.unparse(kw.value)
                    except Exception:
                        pass
                elif kw.arg == 'type':
                    entry['type'] = self._click_type(kw.value)

            options.append(entry)
        return options

    def _click_type(self, node) -> str:
        """Describe a click option's ``type=`` argument in words.

        Parameters
        ----------
        node : ast.AST
            The expression assigned to ``type=``.

        Returns
        -------
        type_name : str
            A short type label such as ``int`` or ``choice: a | b``.
        """
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Call):
            name = self._get_decorator_name(node.func)
            if name.endswith('Choice') and node.args:
                try:
                    choices = ast.literal_eval(node.args[0])
                    return 'choice: ' + ' | '.join(str(c) for c in choices)
                except Exception:
                    return 'choice'
            if name.endswith('Path'):
                return 'path'
            if name.endswith('IntRange'):
                return 'int range'
            if name.endswith('FloatRange'):
                return 'float range'
            return name.split('.')[-1].lower()
        try:
            return ast.unparse(node)
        except Exception:
            return ''

    def _click_usage(self, node: ast.FunctionDef, decorators: List[str]) -> str:
        """Build the shell usage line for a click command or group.

        Parameters
        ----------
        node : ast.FunctionDef
            The decorated function definition.
        decorators : List[str]
            Dotted decorator names already resolved for this function.

        Returns
        -------
        usage : str
            A usage string such as ``tuiml benchmark [OPTIONS]``, or ``''``
            when the function is not a click command.
        """
        if not any(d in ('click.command', 'click.group') for d in decorators):
            return ''

        is_group = 'click.group' in decorators
        name = node.name.replace('_', '-')

        # An explicit name wins: @click.command('test-statistics')
        for decorator in node.decorator_list:
            if not isinstance(decorator, ast.Call):
                continue
            if self._get_decorator_name(decorator.func) not in ('click.command', 'click.group'):
                continue
            if decorator.args and isinstance(decorator.args[0], ast.Constant) \
                    and isinstance(decorator.args[0].value, str):
                name = decorator.args[0].value

        if is_group:
            # The root group is invoked as the program itself.
            return 'tuiml [OPTIONS] COMMAND [ARGS]...'
        return f'tuiml {name} [OPTIONS]'

    def _get_function_signature(self, node: ast.FunctionDef) -> str:
        """Get the function signature as a string."""
        args = []

        # Handle positional-only args (Python 3.8+)
        posonlyargs = getattr(node.args, 'posonlyargs', [])
        for arg in posonlyargs:
            args.append(self._format_arg(arg))

        if posonlyargs:
            args.append('/')

        # Regular args
        num_defaults = len(node.args.defaults)
        num_args = len(node.args.args)

        for i, arg in enumerate(node.args.args):
            default_idx = i - (num_args - num_defaults)
            if default_idx >= 0:
                default = node.args.defaults[default_idx]
                args.append(f"{self._format_arg(arg)}={self._get_default_value(default)}")
            else:
                args.append(self._format_arg(arg))

        # *args
        if node.args.vararg:
            args.append(f"*{node.args.vararg.arg}")
        elif node.args.kwonlyargs:
            args.append('*')

        # Keyword-only args
        num_kw_defaults = len(node.args.kw_defaults)
        for i, arg in enumerate(node.args.kwonlyargs):
            default = node.args.kw_defaults[i]
            if default:
                args.append(f"{self._format_arg(arg)}={self._get_default_value(default)}")
            else:
                args.append(self._format_arg(arg))

        # **kwargs
        if node.args.kwarg:
            args.append(f"**{node.args.kwarg.arg}")

        # Return annotation
        returns = ""
        if node.returns:
            returns = f" -> {self._get_annotation(node.returns)}"

        return f"({', '.join(args)}){returns}"

    def _format_arg(self, arg: ast.arg) -> str:
        """Format a function argument."""
        if arg.annotation:
            return f"{arg.arg}: {self._get_annotation(arg.annotation)}"
        return arg.arg

    def _get_annotation(self, node: ast.expr) -> str:
        """Get type annotation as string."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Constant):
            return repr(node.value)
        elif isinstance(node, ast.Subscript):
            value = self._get_annotation(node.value)
            slice_val = self._get_annotation(node.slice)
            return f"{value}[{slice_val}]"
        elif isinstance(node, ast.Attribute):
            return self._get_attribute_name(node)
        elif isinstance(node, ast.Tuple):
            elements = [self._get_annotation(e) for e in node.elts]
            return ', '.join(elements)
        elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
            left = self._get_annotation(node.left)
            right = self._get_annotation(node.right)
            return f"{left} | {right}"
        else:
            return "..."

    def _get_default_value(self, node: ast.expr) -> str:
        """Get default value as string."""
        if isinstance(node, ast.Constant):
            return repr(node.value)
        elif isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.List):
            return "[]"
        elif isinstance(node, ast.Dict):
            return "{}"
        elif isinstance(node, ast.Tuple):
            return "()"
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                return f"{node.func.id}()"
            return "..."
        else:
            return "..."

    def _get_attribute_name(self, node: ast.Attribute) -> str:
        """Get full attribute name (e.g., 'module.Class')."""
        parts = []
        current = node
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        if isinstance(current, ast.Name):
            parts.append(current.id)
        return '.'.join(reversed(parts))

    def _get_decorator_name(self, node) -> str:
        """Get decorator name."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Call):
            return self._get_decorator_name(node.func)
        elif isinstance(node, ast.Attribute):
            return self._get_attribute_name(node)
        return "..."


class HTMLDocGenerator:
    """Generates HTML documentation from extracted docstrings."""

    def __init__(self, output_dir: Path, source_root: Path):
        self.output_dir = output_dir
        self.source_root = source_root
        self.all_modules: List[ModuleDoc] = []
        self.package_docstrings: Dict[Path, str] = {}

    def add_module(self, doc: ModuleDoc):
        """Add a module to the documentation."""
        self.all_modules.append(doc)

    def add_package_docstring(self, dir_path: Path, docstring: str):
        """Record a package's ``__init__.py`` docstring for its index page.

        ``__init__.py`` gets no module card of its own (it mostly re-exports
        names), but its docstring is the package overview, install notes and
        usage guide, so it heads the package's index page instead of being
        dropped.

        Parameters
        ----------
        dir_path : Path
            Package directory, relative to the source root.
        docstring : str
            The raw ``__init__.py`` module docstring.
        """
        self.package_docstrings[dir_path] = docstring

    def _card_summary(self, summary: str) -> str:
        """Escape a module-card summary and render ``inline code`` as <code>.

        Parameters
        ----------
        summary : str
            Raw docstring summary text, possibly containing RST
            double-backtick inline-code spans.

        Returns
        -------
        rendered : str
            HTML-escaped summary with ``spans`` converted to <code> tags.
        """
        escaped = html.escape(summary)
        rendered = re.sub(r'``([^`]+)``', r'<code class="text-indigo-600">\1</code>', escaped)
        return DocstringParser._format_emphasis(rendered)

    def _source_url(self, lineno: int = 0) -> str:
        """Return the GitHub URL for the module being rendered.

        Parameters
        ----------
        lineno : int, default=0
            Line to anchor on. Omit (or 0) to link at the top of the file.

        Returns
        -------
        url : str
            A ``blob`` URL on :data:`GITHUB_BRANCH`, or an empty string when
            no module is currently being rendered.
        """
        rel = getattr(self, "_current_source_rel", None)
        if not rel:
            return ""
        base = DOC_CONFIG["github_url"]
        url = f"{base}/blob/{GITHUB_BRANCH}/{GITHUB_SOURCE_PREFIX}/{rel}"
        return f"{url}#L{lineno}" if lineno else url

    def _source_icon(self, lineno: int = 0, label: str = "View source on GitHub") -> str:
        """Render the GitHub icon that links to an item's implementation.

        Opens in a new tab, so a reader following it does not lose their place
        in the docs. ``rel="noopener"`` goes with ``target="_blank"``: without
        it the opened page gets a handle on this one through ``window.opener``.

        Parameters
        ----------
        lineno : int, default=0
            Line the item starts on.
        label : str, default="View source on GitHub"
            Accessible label and tooltip.

        Returns
        -------
        html : str
            The anchor, or an empty string when there is no source URL.
        """
        url = self._source_url(lineno)
        if not url:
            return ''
        return (
            f'<a href="{url}" target="_blank" rel="noopener noreferrer" '
            f'title="{html.escape(label)}" aria-label="{html.escape(label)}" '
            f'class="inline-flex items-center justify-center w-7 h-7 rounded-md '
            f'text-gray-400 hover:text-gray-900 hover:bg-gray-100 transition-colors '
            f'shrink-0">'
            f'<i class="fa-brands fa-github text-base"></i></a>'
        )

    def _build_class_map(self):
        """Map every documented reference path to the URL that documents it.

        Covers all four things a ``See Also`` entry can name, because a key
        that is missing here sends the entry down a guess-the-path fallback
        that usually produces a 404:

        - a **class**, linking to its module page and its anchor;
        - a **function**, likewise (``:func:`` refs outnumber ``:class:``
          ones in this codebase, and used to resolve to a per-function page
          that has never existed);
        - a **module**, linking to its own page;
        - a **package**, linking to its ``index.html``.

        Each is registered under its fully qualified name, the same name
        without the ``tuiml.`` prefix, and its bare name, so ``:class:``
        entries written in any of those styles all resolve.
        """
        self._class_map = {}

        def register(keys, url):
            """Record ``url`` for each key, without overwriting a better one."""
            for key in keys:
                self._class_map.setdefault(key, url)

        for doc in self.all_modules:
            rel_path = doc.relative_path.relative_to(self.source_root)
            html_url = f"/docs/{str(rel_path.with_suffix('.html'))}"
            parent_dotted = '.'.join(rel_path.parent.parts)
            module_dotted = '.'.join(rel_path.with_suffix('').parts)

            # The module itself, for :mod:`tuiml.evaluation.metrics`.
            register([f"tuiml.{module_dotted}", module_dotted], html_url)

            for cls in doc.classes:
                anchor = f"{html_url}#class-{cls.name}"
                prefix = f"{parent_dotted}." if parent_dotted else ""
                register([
                    f"tuiml.{prefix}{cls.name}",
                    f"{prefix}{cls.name}",
                    f"tuiml.{module_dotted}.{cls.name}",
                    f"{module_dotted}.{cls.name}",
                    cls.name,
                ], anchor)

            # Functions live inside their module's page, not on one of their
            # own. Point at the module, anchored to the function.
            for func in doc.functions:
                anchor = f"{html_url}#func-{func.name}"
                prefix = f"{parent_dotted}." if parent_dotted else ""
                register([
                    f"tuiml.{module_dotted}.{func.name}",
                    f"{module_dotted}.{func.name}",
                    f"tuiml.{prefix}{func.name}",
                    f"{prefix}{func.name}",
                    func.name,
                ], anchor)

        # Packages resolve to their index page, for :mod:`tuiml.sklearn`.
        # Derived from the modules themselves, so a package without an
        # __init__ docstring still resolves.
        packages = set()
        for doc in self.all_modules:
            parent = doc.relative_path.relative_to(self.source_root).parent
            for depth in range(len(parent.parts)):
                packages.add(Path(*parent.parts[:depth + 1]))
        for dir_path in packages | set(self.package_docstrings):
            dotted = '.'.join(dir_path.parts)
            if dotted:
                register([f"tuiml.{dotted}", dotted], f"/docs/{dir_path}/index.html")

    def generate_all(self):
        """Generate all HTML documentation."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Build class-to-URL mapping for See Also links
        self._build_class_map()

        # Generate module pages
        for doc in self.all_modules:
            self._generate_module_page(doc)

        # Generate directory index pages
        self._generate_directory_indexes()

        # Generate the landing page the site serves at /docs/api-reference.html,
        # so its package/module cards can never drift from the real code.
        self._generate_main_index()

    def _generate_module_page(self, doc: ModuleDoc):
        """Generate HTML page for a module."""
        rel_path = doc.relative_path.relative_to(self.source_root)
        output_path = self.output_dir / rel_path.with_suffix('.html')
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Which file the members rendered below come from, so each can link to
        # its own lines on GitHub. Set here rather than threaded through every
        # _render_* signature, and cleared at the end of this method.
        self._current_source_rel = str(rel_path)

        depth = len(output_path.relative_to(self.output_dir).parts) - 1
        index_path = '../' * depth + 'api-reference.html' if depth > 0 else 'api-reference.html'

        # Build breadcrumb
        breadcrumb_parts = list(rel_path.parts[:-1])
        breadcrumb_items = []
        breadcrumb_items.append(f'<a href="{index_path}" class="hover:text-gray-900">API Reference</a>')
        for i, part in enumerate(breadcrumb_parts):
            # Calculate relative path from current module to this breadcrumb level
            up_levels = len(breadcrumb_parts) - i - 1
            current_path = '../' * up_levels + 'index.html' if up_levels > 0 else 'index.html'
            breadcrumb_items.append(f'<a href="{current_path}" class="hover:text-gray-900">{part}</a>')
        breadcrumb_items.append(f'<span class="text-gray-900 font-semibold">{rel_path.name}</span>')
        breadcrumb_html = ' <i class="fa-solid fa-chevron-right text-[10px] text-gray-300 mx-2"></i> '.join(breadcrumb_items)

        # Build content
        content = []

        # Page header: linked breadcrumb, ending in the actual file name. The
        # heading shows the module name without an extension, so the trail is
        # where the reader finds out which file on disk this page documents.
        crumbs = [f'<a href="{index_path}">API Reference</a>']
        for i, part in enumerate(breadcrumb_parts):
            up_levels = len(breadcrumb_parts) - i - 1
            crumbs.append(f'<a href="{"../" * up_levels}index.html">{part}</a>')
        crumbs.append(f'<span class="api-crumb-file">{rel_path.name}</span>')
        content.append(f'<p class="oc-caption api-crumb" style="margin: 0;">{" / ".join(crumbs)}</p>')
        # Module heading, with the GitHub link pushed to the far right of the
        # row. Each class, function and method below carries its own link,
        # anchored to the line it starts on.
        content.append(
            '<div class="flex items-center justify-between gap-3" style="margin-bottom: 32px;">'
            f'<h1 class="oc-display" style="margin: 0;">{doc.module_name}</h1>'
            f'{self._source_icon(0, f"View {rel_path} source on GitHub")}'
            '</div>'
        )

        # On-this-page rail: same oc-toc component as the tutorials/benchmarks
        # pages — fixed beside the column on wide screens, in-flow block on
        # narrow ones, scrollspy-highlighted by oc.js.
        if doc.classes or doc.functions:
            content.append('<nav class="oc-toc oc-toc-flow api-rail" aria-label="On this page">')
            content.append('<div class="oc-toc-label">On this page</div>')
            for cls in doc.classes:
                content.append(f'<a href="#class-{cls.name}">{cls.name}</a>')
            for func in doc.functions:
                content.append(f'<a href="#func-{func.name}">{func.name}</a>')
            content.append('</nav>')

        # Main Content
        content.append('<main>')

        # Module docstring
        if doc.docstring:
            parser = DocstringParser(doc.docstring, self._class_map)
            content.append(f'<div class="prose max-w-none mb-10 text-gray-600 bg-white p-6 rounded-xl border border-gray-200 shadow-sm">{parser.to_html()}</div>')

        # Classes
        if doc.classes:
            content.append('<div class="mb-12">')
            content.append('<h2 class="text-2xl font-bold text-gray-900 mb-6 flex items-center gap-2"><i class="fa-solid fa-cubes text-indigo-500"></i> Classes</h2>')
            # Build module path from relative path
            module_path = str(rel_path.with_suffix('')).replace('/', '.')
            for cls in doc.classes:
                cls.module = module_path  # Set the module path
                content.append(self._render_class(cls))
            content.append('</div>')

        # Functions
        if doc.functions:
            content.append('<div class="mb-12">')
            content.append('<h2 class="text-2xl font-bold text-gray-900 mb-6 flex items-center gap-2"><i class="fa-solid fa-code text-indigo-500"></i> Functions</h2>')
            for func in doc.functions:
                content.append(self._render_function(func))
            content.append('</div>')
            
        content.append('</main>')

        html = self._wrap_page(
            title=f'{doc.module_name} - API Documentation',
            content='\n'.join(content),
            index_path=index_path,
            breadcrumb=breadcrumb_html,
            header_info={
                'name': doc.module_name,
                'path': str(rel_path)
            }
        )

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)

        self._current_source_rel = None

    def _render_class(self, cls: DocItem) -> str:
        """Render a class as HTML with CapyMOA-style design."""
        parts = [f'<div class="mb-12 scroll-mt-24" id="class-{cls.name}">']

        # Class header with name and bases
        bases_html = ''
        if cls.bases:
            bases_str = ', '.join(cls.bases)
            bases_html = f'(<span class="text-cyan-400">{bases_str}</span>)'

        # Decorators
        decorators_html = ''
        if cls.decorators:
            dec_lines = '\n'.join(f'<span class="text-yellow-400">@{d}</span>' for d in cls.decorators)
            decorators_html = f'{dec_lines}\n'

        parts.append(f'''
        <div class="mb-6">
            <div class="flex items-center gap-2">
                <h2 class="text-3xl font-bold text-gray-900">{cls.name}</h2>
                {self._source_icon(cls.lineno, f'View {cls.name} source on GitHub')}
            </div>
            <p class="text-gray-500 font-mono text-sm mt-2">class <span class="text-indigo-600">{cls.module}.{cls.name}</span>{bases_html}</p>
        </div>
        ''')

        # Class docstring summary
        if cls.docstring:
            parser = DocstringParser(cls.docstring, self._class_map)
            if parser.summary:
                parts.append(f'<p class="text-gray-700 text-lg mb-6 leading-relaxed">{parser._format_inline_code(html.escape(parser.summary))}</p>')
            if parser.extended_summary:
                parts.append(f'<div class="text-gray-600 mb-8 leading-relaxed">{parser._format_text(parser.extended_summary)}</div>')

        # Find __init__ method for signature display
        init_method = next((m for m in cls.methods if m.name == '__init__'), None)
        if init_method:
            parts.append(self._render_init_signature(cls.name, init_method))

        # Render docstring sections (Parameters, Examples, etc.)
        class_sections: Dict[str, Any] = {}
        if cls.docstring:
            parser = DocstringParser(cls.docstring, self._class_map)
            class_sections = parser.sections
            for section_name, content in parser.sections.items():
                parts.append(parser._format_section(section_name, content))

        # Methods section (including __init__, __call__, etc. if they have docstrings).
        # __init__ already appears above as the Constructor, alongside the
        # class-level Parameters table, so repeating it here would render the
        # same block twice. Keep it only for the rare class that documents its
        # arguments solely on __init__.
        skip_init = init_method is not None and 'Parameters' in class_sections
        public_methods = [
            m for m in cls.methods
            if (not m.name.startswith('_')
                or m.name in ('__init__', '__call__', '__str__', '__repr__'))
            and not (skip_init and m.name == '__init__')
        ]

        if public_methods:
            parts.append(f'''
            <div class="mt-10">
                <h3 class="text-xl font-bold text-gray-900 mb-6 flex items-center gap-2">
                    <i class="fa-solid fa-gears text-gray-400"></i> Methods
                </h3>
                <div class="space-y-6">
            ''')
            for method in public_methods:
                parts.append(self._render_method(cls.name, method))
            parts.append('</div></div>')

        parts.append('</div>')
        return '\n'.join(parts)

    def _render_init_signature(self, class_name: str, method: DocItem) -> str:
        """Render __init__ signature in CapyMOA style."""
        # Parse signature to colorize types
        sig = method.signature
        
        # Colorize the signature
        colored_sig = self._colorize_signature(sig)
        
        unique_id = f'init-{class_name}'
        
        return f'''
        <div class="mb-8">
            <div class="bg-slate-800 rounded-lg overflow-hidden shadow-lg">
                <div class="flex items-center justify-between px-4 py-2 bg-slate-900/50 border-b border-slate-700">
                    <span class="text-slate-400 text-xs font-mono">Constructor</span>
                    <button onclick="copyCode('{unique_id}')" class="text-slate-400 hover:text-white transition-colors text-sm flex items-center gap-1">
                        <i class="fa-regular fa-copy"></i>
                        <span class="text-xs">Copy</span>
                    </button>
                </div>
                <div class="p-4 overflow-x-auto" id="{unique_id}">
                    <pre class="font-mono text-sm leading-relaxed"><span class="text-purple-400">__init__</span><span class="text-slate-300">(</span>
{colored_sig}
<span class="text-slate-300">)</span></pre>
                </div>
            </div>
        </div>
        '''

    def _colorize_signature(self, sig: str) -> str:
        """Colorize function signature for syntax highlighting."""
        # Remove outer parentheses
        sig = sig.strip()
        if sig.startswith('('):
            sig = sig[1:]
        if sig.endswith(')'):
            sig = sig[:-1]
        
        # Split by comma, handling nested brackets
        params = []
        current = ''
        depth = 0
        for char in sig:
            if char in '([{':
                depth += 1
            elif char in ')]}':
                depth -= 1
            if char == ',' and depth == 0:
                params.append(current.strip())
                current = ''
            else:
                current += char
        if current.strip():
            params.append(current.strip())
        
        colored_params = []
        for param in params:
            if param == 'self':
                colored_params.append('    <span class="text-red-400">self</span><span class="text-slate-400">,</span>')
                continue
                
            # Parse param: name: type = default
            if ':' in param:
                if '=' in param:
                    # name: type = default
                    name_type, default = param.split('=', 1)
                    name, type_hint = name_type.split(':', 1)
                    colored_params.append(
                        f'    <span class="text-orange-300">{name.strip()}</span><span class="text-slate-400">:</span> '
                        f'<span class="text-cyan-400">{html.escape(type_hint.strip())}</span> '
                        f'<span class="text-slate-400">=</span> <span class="text-green-400">{html.escape(default.strip())}</span><span class="text-slate-400">,</span>'
                    )
                else:
                    # name: type
                    name, type_hint = param.split(':', 1)
                    colored_params.append(
                        f'    <span class="text-orange-300">{name.strip()}</span><span class="text-slate-400">:</span> '
                        f'<span class="text-cyan-400">{html.escape(type_hint.strip())}</span><span class="text-slate-400">,</span>'
                    )
            elif '=' in param:
                # name = default
                name, default = param.split('=', 1)
                colored_params.append(
                    f'    <span class="text-orange-300">{name.strip()}</span> '
                    f'<span class="text-slate-400">=</span> <span class="text-green-400">{html.escape(default.strip())}</span><span class="text-slate-400">,</span>'
                )
            else:
                # Just name
                colored_params.append(f'    <span class="text-orange-300">{param.strip()}</span><span class="text-slate-400">,</span>')
        
        return '\n'.join(colored_params)

    def _render_method(self, class_name: str, method: DocItem) -> str:
        """Render a method with full documentation."""
        if not method.docstring:
            # Minimal rendering for undocumented methods
            return f'''
            <div class="border-l-4 border-gray-200 pl-4 py-2 hover:border-indigo-400 transition-colors" id="method-{class_name}-{method.name}">
                <div class="flex items-baseline gap-2">
                    <code class="font-bold text-gray-600 text-base">{method.name}</code>
                    <span class="text-sm text-gray-400 font-mono">{html.escape(method.signature)}</span>
                </div>
            </div>
            '''
        
        parser = DocstringParser(method.docstring, self._class_map)
        
        # Build method signature with syntax highlighting
        sig_parts = method.signature
        highlighted_sig = html.escape(sig_parts)
        
        # Build sections HTML
        sections_html = ''
        for section_name, content in parser.sections.items():
            if section_name in ('Parameters', 'Returns', 'Raises', 'Yields'):
                section_content = parser._format_param_section(content)
                icon = 'fa-sliders' if section_name == 'Parameters' else 'fa-arrow-right-from-bracket' if section_name == 'Returns' else 'fa-triangle-exclamation'
                sections_html += f'''
                <div class="mt-3">
                    <h5 class="text-xs font-bold text-gray-500 uppercase tracking-wide mb-2 flex items-center gap-1">
                        <i class="fa-solid {icon} text-gray-400"></i> {section_name}
                    </h5>
                    <div class="text-sm">{section_content}</div>
                </div>
                '''
        
        return f'''
        <div class="bg-white rounded-lg border border-gray-200 mb-4 overflow-hidden hover:shadow-md transition-shadow" id="method-{class_name}-{method.name}">
            <details class="group">
                <summary class="px-4 py-3 cursor-pointer hover:bg-gray-50 transition-colors flex items-center justify-between">
                    <div class="flex items-center gap-3">
                        <span class="inline-flex items-center justify-center w-8 h-8 rounded-lg bg-indigo-100 text-indigo-600">
                            <i class="fa-solid fa-code text-sm"></i>
                        </span>
                        <div>
                            <code class="font-bold text-indigo-600 text-base">{method.name}</code>
                            <span class="text-sm text-gray-400 font-mono ml-1">{highlighted_sig}</span>
                        </div>
                    </div>
                    <span class="flex items-center gap-1">
                        {self._source_icon(method.lineno, f'View {class_name}.{method.name} source on GitHub')}
                        <i class="fa-solid fa-chevron-down text-gray-400 text-sm group-open:rotate-180 transition-transform"></i>
                    </span>
                </summary>
                <div class="px-4 pb-4 border-t border-gray-100 bg-gray-50/50">
                    <p class="text-sm text-gray-600 mt-3">{parser._format_inline_code(html.escape(parser.summary))}</p>
                    {sections_html}
                </div>
            </details>
        </div>
        '''

    def _render_function(self, func: DocItem) -> str:
        """Render a function as HTML."""
        parts = [f'<div class="bg-white rounded-xl border border-gray-200 overflow-hidden shadow-sm mb-6 scroll-mt-24" id="func-{func.name}">']

        # Header
        decorators_html = ''
        if func.decorators:
            decorators_html = ' '.join(f'<span class="text-amber-600 font-mono text-xs">@{d}</span>' for d in func.decorators)
            decorators_html = f'<div class="mb-2">{decorators_html}</div>'

        if func.cli_usage:
            badge_label, badge_class = 'Command', 'bg-indigo-100 text-indigo-700'
            display_name = func.cli_usage.split(' [')[0]
            usage_line = func.cli_usage
        else:
            badge_label, badge_class = 'Func', 'bg-green-100 text-green-700'
            display_name = func.name
            usage_line = func.name + func.signature

        parts.append(f'''
        <div class="bg-gray-50 px-6 py-4 border-b border-gray-200">
            {decorators_html}
            <div class="flex items-center gap-2">
                <span class="inline-flex items-center justify-center px-2 py-1 rounded text-xs font-bold {badge_class} uppercase tracking-wide">{badge_label}</span>
                <h3 class="text-lg font-bold text-gray-900 font-mono">{display_name}</h3>
                {self._source_icon(func.lineno, f'View {func.name} source on GitHub')}
            </div>
            <div class="text-xs text-gray-400 mt-1 font-mono">Line {func.lineno}</div>
        </div>
        ''')

        # Content
        parts.append('<div class="p-6">')
        parts.append(f'<div class="mb-4 bg-gray-900 text-gray-200 p-3 rounded-lg font-mono text-sm overflow-x-auto shadow-inner">{html.escape(usage_line)}</div>')

        if func.docstring:
            parser = DocstringParser(func.docstring, self._class_map)
            parts.append(f'<div class="prose max-w-none text-gray-600">{parser.to_html()}</div>')

        if func.cli_options:
            parts.append(self._render_cli_options(func.cli_options))

        parts.append('</div></div>')
        return '\n'.join(parts)

    def _render_cli_options(self, options: List[Dict[str, Any]]) -> str:
        """Render a click command's options as a documentation table.

        Parameters
        ----------
        options : List[Dict[str, Any]]
            Option dicts produced by ``PythonDocExtractor._click_options``.

        Returns
        -------
        html_section : str
            An HTML section listing each flag, its type and its help text.
        """
        rows = []
        for opt in options:
            flags = ', '.join(opt['flags'])
            badges = []
            if opt['is_argument']:
                badges.append('<span class="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-gray-100 text-gray-700">argument</span>')
            if opt['type'] == 'flag':
                badges.append('<span class="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-purple-100 text-purple-700">flag</span>')
            elif opt['type']:
                badges.append(f'<span class="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-blue-100 text-blue-700">{html.escape(opt["type"])}</span>')
            if opt['multiple']:
                badges.append('<span class="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-orange-100 text-orange-700">repeatable</span>')
            if opt['required']:
                badges.append('<span class="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-red-100 text-red-700">required</span>')
            elif opt['default'] not in (None, 'None'):
                badges.append(f'<span class="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-blue-50 text-blue-600">= {html.escape(opt["default"])}</span>')

            description = html.escape(opt['help']) or '<span class="italic text-gray-400">No description.</span>'
            rows.append(f'''
            <div class="py-4 grid grid-cols-1 md:grid-cols-12 gap-2 md:gap-4 hover:bg-gray-50 transition-colors px-2 rounded">
                <div class="md:col-span-4">
                    <code class="font-bold text-indigo-600 text-sm">{html.escape(flags)}</code>
                </div>
                <div class="md:col-span-8">
                    <div class="flex flex-wrap gap-1 mb-1">{' '.join(badges)}</div>
                    <div class="text-sm text-gray-600">{description}</div>
                </div>
            </div>
            ''')

        return f'''
        <div class="mb-8 last:mb-0 bg-white rounded-lg border border-gray-200 overflow-hidden mt-6">
            <div class="bg-gray-50 px-4 py-3 border-b border-gray-200">
                <h4 class="font-bold text-gray-800 text-sm uppercase tracking-wide flex items-center">
                    <i class="fa-solid fa-terminal text-indigo-500 mr-2"></i>Options
                </h4>
            </div>
            <div class="p-4"><div class="divide-y divide-gray-100">{''.join(rows)}</div></div>
        </div>
        '''

    def _generate_directory_indexes(self):
        """Generate index.html for each directory."""
        # Group modules by directory
        dir_modules: Dict[Path, List[ModuleDoc]] = {}
        for doc in self.all_modules:
            rel_path = doc.relative_path.relative_to(self.source_root)
            dir_path = rel_path.parent
            if dir_path not in dir_modules:
                dir_modules[dir_path] = []
            dir_modules[dir_path].append(doc)

        # Get all directories (including empty ones for navigation)
        all_dirs = set()
        for doc in self.all_modules:
            rel_path = doc.relative_path.relative_to(self.source_root)
            for i in range(len(rel_path.parts) - 1):
                all_dirs.add(Path(*rel_path.parts[:i+1]))

        for dir_path in all_dirs:
            self._generate_directory_index(dir_path, dir_modules, all_dirs)

    def _generate_directory_index(self, dir_path: Path, dir_modules: Dict[Path, List[ModuleDoc]],
                                  all_dirs: Set[Path]):
        """Generate index.html for a specific directory."""
        output_path = self.output_dir / dir_path / 'index.html'
        output_path.parent.mkdir(parents=True, exist_ok=True)

        depth = len(dir_path.parts)
        root_index = '../' * depth + 'api-reference.html'

        # Build breadcrumb
        breadcrumb_items = []
        breadcrumb_items.append(f'<a href="{root_index}" class="hover:text-gray-900">API Reference</a>')
        for i, part in enumerate(dir_path.parts):
            up_levels = depth - i - 1
            link = '../' * up_levels + 'index.html' if up_levels > 0 else 'index.html'
            if i < len(dir_path.parts) - 1:
                breadcrumb_items.append(f'<a href="{link}" class="hover:text-gray-900">{part}</a>')
            else:
                breadcrumb_items.append(f'<span class="text-gray-900 font-semibold">{part}</span>')
        breadcrumb_html = ' <i class="fa-solid fa-chevron-right text-[10px] text-gray-300 mx-2"></i> '.join(breadcrumb_items)

        # Find subdirectories and modules. Walk every directory that has an
        # index page, not just the ones holding modules directly: a package
        # like datasets/generators only contains subpackages, so keying off
        # dir_modules would hide it from its parent's Packages list.
        subdirs = set()
        for other_dir in all_dirs:
            if len(other_dir.parts) == len(dir_path.parts) + 1:
                if other_dir.parts[:len(dir_path.parts)] == dir_path.parts:
                    subdirs.add(other_dir.parts[-1])

        modules = dir_modules.get(dir_path, [])

        content = []

        # Page header: linked breadcrumb (API Reference / parent dirs) + dir name
        crumbs = [f'<a href="{root_index}">API Reference</a>']
        for i, part in enumerate(dir_path.parts[:-1]):
            up_levels = depth - i - 1
            crumbs.append(f'<a href="{"../" * up_levels}index.html">{part}</a>')
        content.append(f'<p class="oc-caption api-crumb" style="margin: 0;">{" / ".join(crumbs)}</p>')
        content.append(f'<h1 class="oc-display" style="margin-bottom: 48px;">{dir_path.name}/</h1>')

        # Package overview, straight from __init__.py's docstring. This is
        # where a package explains what it is and how to use it (see
        # tuiml/sklearn and tuiml/capymoa), so it goes above the listings.
        package_docstring = self.package_docstrings.get(dir_path)
        if package_docstring:
            overview = DocstringParser(package_docstring, self._class_map).to_html()
            if overview.strip():
                content.append(
                    f'<div class="api-package-overview" style="margin-bottom: 48px;">{overview}</div>'
                )

        # Subdirectories — flat hairline boxes (oc why-card), names only
        if subdirs:
            content.append('<h2 class="oc-h">Packages</h2>')
            content.append('<hr class="oc-rule">')
            content.append('<div class="why-grid" style="margin-top: 0; margin-bottom: 48px;">')
            for subdir in sorted(subdirs):
                content.append(
                    f'<a href="{subdir}/index.html" class="why-card api-card">'
                    f'<div class="why-title">{subdir}/</div></a>'
                )
            content.append('</div>')

        # Modules in this directory — boxes with summary + stats
        if modules:
            content.append('<h2 class="oc-h">Modules</h2>')
            content.append('<hr class="oc-rule">')
            content.append('<div class="why-grid" style="margin-top: 0;">')
            for doc in sorted(modules, key=lambda d: d.module_name):
                summary = ''
                if doc.docstring:
                    summary = DocstringParser(doc.docstring, self._class_map).summary[:100]
                    if len(doc.docstring) > 100:
                        summary += '...'

                n_cls, n_fn = len(doc.classes), len(doc.functions)
                stats = (
                    f'{n_cls} class{"" if n_cls == 1 else "es"} · '
                    f'{n_fn} function{"" if n_fn == 1 else "s"}'
                )
                content.append(
                    f'<a href="{doc.module_name}.html" class="why-card api-card">'
                    f'<div class="why-title">{doc.module_name}</div>'
                    f'<p class="why-text">{self._card_summary(summary)}</p>'
                    f'<div class="oc-caption" style="margin-top: auto;">{stats}</div></a>'
                )
            content.append('</div>')

        html_content = self._wrap_page(
            title=f'{dir_path.name} - API Documentation',
            content='\n'.join(content),
            index_path=root_index,
            breadcrumb=breadcrumb_html,
            header_info={'name': str(dir_path), 'path': str(dir_path)}
        )

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

    # One-line blurbs for the top-level packages on the API-reference landing
    # page (list-row descriptions). Packages missing from this map fall back
    # to a generic line — add new packages here as they appear.
    PACKAGE_BLURBS = {
        'agent': 'MCP server, LLM tool registry, and agent prompts.',
        'algorithms': 'Native ML algorithms across 13 families.',
        'base': 'Classifier/Regressor base classes and decorators.',
        'capymoa': 'Optional CapyMOA streaming wrappers.',
        'cli': 'The tuiml command-line interface.',
        'datasets': 'Loaders, generators, and builtin datasets.',
        'evaluation': 'Metrics, splitting, and cross-validation.',
        'features': 'Feature selection, extraction, and generation.',
        'preprocessing': 'Scalers, encoders, and transforms.',
        'serving': 'Model serving and deployment.',
        'sklearn': 'Optional scikit-learn wrappers.',
        'utils': 'Shared helpers.',
    }

    def _generate_main_index(self):
        """Generate the API-reference landing page.

        Written to ``templates/pages/api-reference.html`` — the template the
        website serves at ``/docs/api-reference.html`` — so the package and
        module rows always reflect the current source tree. Styled with the
        site's oc design system (see website/DESIGN.md): flat list-rows with
        ASCII bracket markers inside .oc-section blocks.
        """
        output_path = self.output_dir.parent / 'pages' / 'api-reference.html'

        def name_box(href: str, name: str, blurb: str) -> str:
            return (
                f'                <a href="{href}" class="why-card api-card">'
                f'<div class="why-title">{name}</div>'
                f'<p class="why-text">{blurb}</p></a>'
            )

        # Top-level packages
        top_dirs = set()
        for doc in self.all_modules:
            rel_path = doc.relative_path.relative_to(self.source_root)
            if len(rel_path.parts) > 1:
                top_dirs.add(rel_path.parts[0])

        package_rows = '\n'.join(
            name_box(f'{d}/index.html', d, self.PACKAGE_BLURBS.get(d, 'Package documentation.'))
            for d in sorted(top_dirs)
        )

        # Root-level modules, described by their docstring summaries
        root_modules = [doc for doc in self.all_modules
                        if len(doc.relative_path.relative_to(self.source_root).parts) == 1]
        module_rows = '\n'.join(
            name_box(
                f'{doc.module_name}.html',
                doc.module_name,
                self._card_summary(
                    DocstringParser(doc.docstring, self._class_map).summary[:100]
                    if doc.docstring else ''
                ),
            )
            for doc in sorted(root_modules, key=lambda d: d.module_name)
        )

        html_content = f'''<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>API Reference, {DOC_CONFIG["project_name"]} Documentation</title>
    <meta name="description" content="Complete API reference for TuiML: algorithms, datasets, preprocessing, evaluation, feature selection, and MCP server modules.">
    <meta name="robots" content="index,follow">
    <link rel="canonical" href="https://tuiml.ai/docs/api-reference.html">

    <meta property="og:type" content="article">
    <meta property="og:site_name" content="{{{{ config.project_name }}}}">
    <meta property="og:title" content="API Reference, {DOC_CONFIG["project_name"]} Documentation">
    <meta property="og:description" content="Complete API reference for all TuiML modules, classes, and functions.">
    <meta property="og:url" content="https://tuiml.ai/docs/api-reference.html">
    <meta property="og:image" content="https://tuiml.ai/static/images/tuiml_logo.png">

    <meta name="twitter:card" content="summary_large_image">
    <meta name="twitter:title" content="API Reference, {DOC_CONFIG["project_name"]} Documentation">
    <meta name="twitter:description" content="Complete API reference for all TuiML modules, classes, and functions.">
    <meta name="twitter:image" content="https://tuiml.ai/static/images/tuiml_logo.png">

    <!-- Favicon -->
    <link rel="icon" type="image/png" sizes="32x32" href="/static/images/favicon-32.png?v=10">
    <link rel="icon" type="image/png" sizes="512x512" href="/static/images/favicon.png?v=10">
    <link rel="apple-touch-icon" sizes="180x180" href="/static/images/apple-touch-icon.png?v=10">

    <!-- Fonts -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">

    <!-- FontAwesome (site nav icons) -->
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css">

    <!-- Tailwind CSS (site nav / footer chrome) -->
    <script src="https://cdn.tailwindcss.com"></script>
</head>

<body class="landing-page antialiased overflow-x-hidden">

    {{% include 'components/_docs_navbar.html' %}}

    <link rel="stylesheet" href="/static/css/oc.css">
    <script src="/static/js/oc.js"></script>
    <link rel="stylesheet" href="/static/css/api-doc.css">

    <div class="oc-wrap">

        <!-- ===================== HEADER ===================== -->
        <section class="oc-section" style="padding-top: 64px;" id="top">
            <h1 class="oc-display">API Reference</h1>
            <p class="oc-body oc-max" style="margin-top: 16px;">
                Complete documentation for every {{{{ config.project_name }}}} module, class, and
                function — generated straight from the source tree, so it can never drift
                from the real code.
            </p>

            <nav class="oc-toc oc-toc-flow" aria-label="On this page">
                <div class="oc-toc-label">On this page</div>
                <a href="#packages">Packages</a>
                <a href="#root-modules">Root modules</a>
            </nav>
        </section>

        <!-- ===================== PACKAGES ===================== -->
        <section class="oc-section" id="packages">
            <h2 class="oc-h">Packages</h2>
            <hr class="oc-rule">
            <div class="why-grid" style="margin-top: 0;">
{package_rows}
            </div>
        </section>

        <!-- ===================== ROOT MODULES ===================== -->
        <section class="oc-section" id="root-modules">
            <h2 class="oc-h">Root modules</h2>
            <hr class="oc-rule">
            <div class="why-grid" style="margin-top: 0;">
{module_rows}
            </div>
        </section>
    </div>

    <!-- Footer -->
    {{% include 'components/_footer.html' %}}

</body>

</html>
'''

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

    def _wrap_page(self, title: str, content: str, index_path: str,
                   breadcrumb: str, header_info: Dict[str, str]) -> str:
        """Wrap content in a Jinja2 extends template.

        Generated pages extend layouts/docs_generated.html which provides
        the navbar, footer, CSS, and JS. Only the content block is filled here.
        The breadcrumb_html variable is set for the navbar component.
        """
        return (
            "{% extends 'layouts/docs_generated.html' %}\n"
            "\n"
            "{% set active_nav = 'api' %}\n"
            "{% set breadcrumb_html = '" + breadcrumb.replace("'", "\\'") + "' %}\n"
            "\n"
            "{% block title %}" + html.escape(title) + "{% endblock %}\n"
            "\n"
            "{% block content %}\n"
            + content + "\n"
            "{% endblock %}"
        )


def main():
    """Main entry point."""
    # Default paths, anchored to the repo root (scripts/ lives one level below it)
    # so the script works regardless of the current working directory.
    repo = Path(__file__).resolve().parent.parent
    source_dir = repo / 'tuiml'
    output_dir = repo / 'website' / 'templates' / '_generated'

    # Parse command line arguments
    if len(sys.argv) >= 2:
        source_dir = Path(sys.argv[1])
    if len(sys.argv) >= 3:
        output_dir = Path(sys.argv[2])

    if not source_dir.exists():
        print(f"Error: Source directory '{source_dir}' does not exist.")
        sys.exit(1)

    print(f"Generating documentation...")
    print(f"  Source: {source_dir}")
    print(f"  Output: {output_dir}")
    print()

    # Find all Python files
    python_files = list(source_dir.rglob('*.py'))
    print(f"Found {len(python_files)} Python files")

    # Create generator
    generator = HTMLDocGenerator(output_dir, source_dir)

    # Process each file
    processed = 0
    skipped = 0

    for filepath in python_files:
        # Skip test files, __pycache__, and package __init__ files, the
        # latter only re-export names and would otherwise show up as a
        # meaningless "__init__" module card in the docs. Only files inside a
        # tests/ directory count as tests: the package itself ships modules
        # such as cli/test_statistics.py that are real public API.
        if (
            '__pycache__' in str(filepath)
            or (filepath.name.startswith('test_') and 'tests' in filepath.parts)
            or filepath.name.startswith('_')
            or any(part.startswith('_') for part in filepath.relative_to(source_dir).parts[:-1])
        ):
            # An __init__.py still carries the package overview. Keep its
            # docstring for the package index page before dropping the file.
            if filepath.name == '__init__.py' and '__pycache__' not in str(filepath):
                rel_dir = filepath.parent.relative_to(source_dir)
                if not any(part.startswith('_') for part in rel_dir.parts):
                    doc = PythonDocExtractor(filepath).extract()
                    if doc and doc.docstring:
                        generator.add_package_docstring(rel_dir, doc.docstring)
            skipped += 1
            continue

        extractor = PythonDocExtractor(filepath)
        doc = extractor.extract()

        if doc:
            doc.relative_path = filepath
            generator.add_module(doc)
            processed += 1
            print(f"  ✓ {filepath.relative_to(source_dir)}")
        else:
            skipped += 1

    print()
    print(f"Processed: {processed} files")
    print(f"Skipped: {skipped} files")

    # Generate HTML
    print()
    print("Generating HTML documentation...")
    generator.generate_all()

    print()
    print(f"Documentation generated successfully!")
    print(f"Open {output_dir / 'index.html'} in your browser to view.")


if __name__ == '__main__':
    main()
