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
from typing import Optional, List, Dict, Any
from datetime import datetime

# ============================================================================
# Global config — single source of truth for generated HTML docs
# ============================================================================
DOC_CONFIG = {
    "project_name": "TuiML",
    "github_url": "https://github.com/tuiml/tuiml",
    "version": "0.1.0",
    "status": "Alpha",
    "copyright_year": datetime.now().year,
}
DOC_CONFIG["version_label"] = f"v{DOC_CONFIG['version']} {DOC_CONFIG['status']}"


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

        # Check if it's a known header
        if line not in self.SECTION_HEADERS:
            return False

        # Check for underline
        if index + 1 < len(lines):
            underline = lines[index + 1].strip()
            if underline and all(c == '-' for c in underline):
                return True

        return False

    def to_html(self) -> str:
        """Convert parsed docstring to HTML."""
        parts = []

        if self.summary:
            parts.append(f'<p class="doc-summary">{html.escape(self.summary)}</p>')

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
        elif name == 'Examples':
            html_content = self._format_examples(content)
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
        elif name == 'Examples':
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
            # Parse option format: ``"value"`` — Description
            option_match = re.match(r'``([^`]+)``\s*[—\-]\s*(.+)', item)
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
        # Replace ``code`` with styled code spans
        return re.sub(
            r'``([^`]+)``',
            r'<code class="bg-slate-100 text-slate-700 px-1 py-0.5 rounded text-xs font-mono">\1</code>',
            text
        )

    def _format_examples(self, content: str) -> str:
        """Format Examples section with dark code blocks."""
        lines = content.split('\n')
        in_code = False
        output = []
        code_block = []
        unique_id = f'example-{hash(content) % 10000}'

        for line in lines:
            if line.strip().startswith('>>>') or line.strip().startswith('...'):
                if not in_code:
                    in_code = True
                # Simply escape and add the code line
                escaped = html.escape(line)
                code_block.append(escaped)
            elif in_code and line.strip() and not line.strip().startswith('>>>'):
                # Output line
                code_block.append(html.escape(line))
            elif in_code and not line.strip():
                if code_block:
                    block_id = f'{unique_id}-{len(output)}'
                    output.append(self._create_code_block('\n'.join(code_block), block_id))
                    code_block = []
                in_code = False
            else:
                if code_block:
                    block_id = f'{unique_id}-{len(output)}'
                    output.append(self._create_code_block('\n'.join(code_block), block_id))
                    code_block = []
                    in_code = False
                if line.strip():
                    output.append(f'<p class="text-gray-600 my-2">{html.escape(line)}</p>')

        if code_block:
            block_id = f'{unique_id}-{len(output)}'
            output.append(self._create_code_block('\n'.join(code_block), block_id))

        return '\n'.join(output)

    def _create_code_block(self, code: str, block_id: str) -> str:
        """Create a styled code block with copy button matching readme.html style."""
        return f'''
        <div class="code-block-wrapper !bg-slate-900 !rounded-xl !border-slate-800 !p-0 overflow-hidden shadow-md group !my-2">
            <div class="bg-slate-950/50 px-4 py-2 flex items-center justify-between border-b border-white/5">
                <div class="flex gap-1.5 opacity-60">
                    <div class="w-2.5 h-2.5 rounded-full bg-red-500/80"></div>
                    <div class="w-2.5 h-2.5 rounded-full bg-yellow-500/80"></div>
                    <div class="w-2.5 h-2.5 rounded-full bg-green-500/80"></div>
                </div>
                <div class="flex items-center gap-2">
                    <span class="text-[10px] font-mono text-slate-500 font-bold uppercase tracking-wider">PYTHON</span>
                    <button class="copy-btn text-slate-500 hover:text-white transition-colors !p-1">
                        <i class="fa-regular fa-copy text-xs"></i>
                    </button>
                </div>
            </div>
            <div class="!p-4 overflow-x-auto">
                <pre class="!m-0 !p-0 !bg-transparent font-mono text-xs text-blue-100 leading-relaxed"><code class="language-python">{code}</code></pre>
            </div>
        </div>
        '''

    def _format_see_also(self, content: str) -> str:
        """Format See Also section with clickable links."""
        lines = content.strip().split('\n')
        items = []
        for line in lines:
            line = line.strip()
            if line:
                # Try to extract reference and description from :class:`~path.ClassName` : description
                match = re.match(r':(?:class|func|meth):`~?([^`]+)`\s*:?\s*(.*)', line)
                if match:
                    ref, desc = match.groups()
                    parts = ref.split('.')
                    class_name = parts[-1] if parts else ref

                    # Look up actual file path in class map
                    href = self._class_map.get(ref) or self._class_map.get(class_name)

                    if not href:
                        # Fall back to snake_case conversion
                        if len(parts) > 1:
                            path_parts = [p for p in parts[:-1] if p != 'tuiml']

                            def to_snake_case(name):
                                result = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
                                return re.sub('([a-z0-9])([A-Z])', r'\1_\2', result).lower()

                            file_name = to_snake_case(class_name)
                            file_path = '/'.join(path_parts) + '/' + file_name
                            href = f"/docs/{file_path}.html"
                        else:
                            href = '#'
                    
                    items.append(f'''
                    <a href="{href}" class="flex items-center gap-3 p-3 bg-gray-50 rounded-lg border border-gray-200 hover:bg-indigo-50 hover:border-indigo-200 transition-colors group">
                        <span class="inline-flex items-center justify-center w-8 h-8 rounded-lg bg-indigo-100 text-indigo-600 group-hover:bg-indigo-200">
                            <i class="fa-solid fa-link text-sm"></i>
                        </span>
                        <div>
                            <code class="font-bold text-indigo-600 group-hover:text-indigo-800">{html.escape(class_name)}</code>
                            <p class="text-sm text-gray-600 mt-0.5">{html.escape(desc)}</p>
                        </div>
                    </a>
                    ''')
                else:
                    # Simple format: ClassName : description
                    simple_match = re.match(r'(\w+)\s*:\s*(.*)', line)
                    if simple_match:
                        name, desc = simple_match.groups()
                        items.append(f'''
                        <div class="flex items-center gap-3 p-3 bg-gray-50 rounded-lg border border-gray-200">
                            <code class="font-bold text-gray-700">{html.escape(name)}</code>
                            <span class="text-sm text-gray-600">{html.escape(desc)}</span>
                        </div>
                        ''')
                    else:
                        items.append(f'<div class="p-3 bg-gray-50 rounded-lg border border-gray-200 text-gray-600">{html.escape(line)}</div>')

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
                result_lines.append(f'<li class="text-gray-700">{html.escape(numbered_match.group(2))}</li>')
            elif bullet_match:
                if not in_bullet_list:
                    if in_numbered_list:
                        result_lines.append('</ol>')
                        in_numbered_list = False
                    result_lines.append('<ul class="list-disc list-inside my-3 space-y-1 pl-4">')
                    in_bullet_list = True
                result_lines.append(f'<li class="text-gray-700">{html.escape(bullet_match.group(1))}</li>')
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
                    # Don't wrap if it contains or is a block element
                    if any(tag in p for tag in ['<ol', '</ol>', '<ul', '</ul>', '<div', '</div>', '<li']):
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

        return doc_item

    def _extract_function(self, node: ast.FunctionDef, is_method: bool = False) -> DocItem:
        """Extract documentation from a function/method definition."""
        decorators = [self._get_decorator_name(d) for d in node.decorator_list]
        signature = self._get_function_signature(node)

        return DocItem(
            name=node.name,
            docstring=ast.get_docstring(node),
            item_type='method' if is_method else 'function',
            lineno=node.lineno,
            signature=signature,
            decorators=decorators
        )

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

    def add_module(self, doc: ModuleDoc):
        """Add a module to the documentation."""
        self.all_modules.append(doc)

    def _build_class_map(self):
        """Build mapping from class reference paths to actual file URLs."""
        self._class_map = {}
        for doc in self.all_modules:
            rel_path = doc.relative_path.relative_to(self.source_root)
            html_url = f"/docs/{str(rel_path.with_suffix('.html'))}"
            parent_parts = rel_path.parent.parts
            parent_dotted = '.'.join(parent_parts)

            for cls in doc.classes:
                # Primary key: tuiml.<package>.<ClassName>
                if parent_dotted:
                    full_ref = f"tuiml.{parent_dotted}.{cls.name}"
                    short_ref = f"{parent_dotted}.{cls.name}"
                else:
                    full_ref = f"tuiml.{cls.name}"
                    short_ref = cls.name

                self._class_map[full_ref] = html_url
                self._class_map[short_ref] = html_url
                # Class name only as fallback
                self._class_map[cls.name] = html_url

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

        # Skip generating api-reference.html — hand-written page exists at that URL

    def _generate_module_page(self, doc: ModuleDoc):
        """Generate HTML page for a module."""
        rel_path = doc.relative_path.relative_to(self.source_root)
        output_path = self.output_dir / rel_path.with_suffix('.html')
        output_path.parent.mkdir(parents=True, exist_ok=True)

        depth = len(output_path.relative_to(self.output_dir).parts) - 1
        index_path = '../' * depth + 'api-reference.html' if depth > 0 else 'api-reference.html'

        # Build breadcrumb
        breadcrumb_parts = list(rel_path.parts[:-1])
        breadcrumb_items = []
        breadcrumb_items.append('<a href="/docs/getting_started.html" class="hover:text-gray-900">Documentation</a>')
        breadcrumb_items.append(f'<a href="{index_path}" class="hover:text-gray-900">API Reference</a>')
        for i, part in enumerate(breadcrumb_parts):
            # Calculate relative path from current module to this breadcrumb level
            up_levels = len(breadcrumb_parts) - i - 1
            current_path = '../' * up_levels + 'index.html' if up_levels > 0 else 'index.html'
            breadcrumb_items.append(f'<a href="{current_path}" class="hover:text-gray-900">{part}</a>')
        breadcrumb_items.append(f'<span class="text-gray-900 font-semibold">{doc.module_name}</span>')
        breadcrumb_html = ' <i class="fa-solid fa-chevron-right text-[10px] text-gray-300 mx-2"></i> '.join(breadcrumb_items)

        # Build content
        content = []

        # Layout: Grid
        content.append('<div class="flex flex-col lg:flex-row gap-8">')
        
        # Sidebar TOC
        content.append('<aside class="w-full lg:w-64 flex-shrink-0">')
        if doc.classes or doc.functions:
            content.append('<div class="sticky top-24 bg-white rounded-xl border border-gray-200 p-6 shadow-sm">')
            content.append('<h3 class="font-bold text-gray-900 mb-4 border-b pb-2">On this page</h3>')
            content.append('<ul class="space-y-2 text-sm">')
            for cls in doc.classes:
                content.append(f'<li><a href="#class-{cls.name}" class="text-indigo-600 hover:text-indigo-800 font-medium block truncate"><i class="fa-solid fa-cube mr-1 text-xs text-gray-400"></i> {cls.name}</a></li>')
                # Optional: Add methods to TOC?
            for func in doc.functions:
                content.append(f'<li><a href="#func-{func.name}" class="text-gray-600 hover:text-gray-900 block truncate"><i class="fa-solid fa-code mr-1 text-xs text-gray-400"></i> {func.name}</a></li>')
            content.append('</ul></div>')
        content.append('</aside>')

        # Main Content
        content.append('<main class="flex-1 min-w-0">')
        
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
        content.append('</div>') # End flex container

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
            <h2 class="text-3xl font-bold text-gray-900 mb-2">{cls.name}</h2>
            <p class="text-gray-500 font-mono text-sm">class <span class="text-indigo-600">{cls.module}.{cls.name}</span>{bases_html}</p>
        </div>
        ''')

        # Class docstring summary
        if cls.docstring:
            parser = DocstringParser(cls.docstring, self._class_map)
            if parser.summary:
                parts.append(f'<p class="text-gray-700 text-lg mb-6 leading-relaxed">{html.escape(parser.summary)}</p>')
            if parser.extended_summary:
                parts.append(f'<div class="text-gray-600 mb-8 leading-relaxed">{parser._format_text(parser.extended_summary)}</div>')

        # Find __init__ method for signature display
        init_method = next((m for m in cls.methods if m.name == '__init__'), None)
        if init_method:
            parts.append(self._render_init_signature(cls.name, init_method))

        # Render docstring sections (Parameters, Examples, etc.)
        if cls.docstring:
            parser = DocstringParser(cls.docstring, self._class_map)
            for section_name, content in parser.sections.items():
                parts.append(parser._format_section(section_name, content))

        # Methods section (including __init__, __call__, etc. if they have docstrings)
        public_methods = [m for m in cls.methods if not m.name.startswith('_') or m.name in ('__init__', '__call__', '__str__', '__repr__')]
        
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
                    <i class="fa-solid fa-chevron-down text-gray-400 text-sm group-open:rotate-180 transition-transform"></i>
                </summary>
                <div class="px-4 pb-4 border-t border-gray-100 bg-gray-50/50">
                    <p class="text-sm text-gray-600 mt-3">{html.escape(parser.summary)}</p>
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

        parts.append(f'''
        <div class="bg-gray-50 px-6 py-4 border-b border-gray-200">
            {decorators_html}
            <div class="flex items-center gap-2">
                <span class="inline-flex items-center justify-center px-2 py-1 rounded text-xs font-bold bg-green-100 text-green-700 uppercase tracking-wide">Func</span>
                <h3 class="text-lg font-bold text-gray-900 font-mono">{func.name}</h3>
            </div>
            <div class="text-xs text-gray-400 mt-1 font-mono">Line {func.lineno}</div>
        </div>
        ''')

        # Content
        parts.append('<div class="p-6">')
        parts.append(f'<div class="mb-4 bg-gray-900 text-gray-200 p-3 rounded-lg font-mono text-sm overflow-x-auto shadow-inner">{html.escape(func.name + func.signature)}</div>')

        if func.docstring:
            parser = DocstringParser(func.docstring, self._class_map)
            parts.append(f'<div class="prose max-w-none text-gray-600">{parser.to_html()}</div>')

        parts.append('</div></div>')
        return '\n'.join(parts)

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
            self._generate_directory_index(dir_path, dir_modules)

    def _generate_directory_index(self, dir_path: Path, dir_modules: Dict[Path, List[ModuleDoc]]):
        """Generate index.html for a specific directory."""
        output_path = self.output_dir / dir_path / 'index.html'
        output_path.parent.mkdir(parents=True, exist_ok=True)

        depth = len(dir_path.parts)
        root_index = '../' * depth + 'api-reference.html'

        # Build breadcrumb
        breadcrumb_items = []
        breadcrumb_items.append('<a href="/docs/getting_started.html" class="hover:text-gray-900">Documentation</a>')
        breadcrumb_items.append(f'<a href="{root_index}" class="hover:text-gray-900">API Reference</a>')
        for i, part in enumerate(dir_path.parts):
            up_levels = depth - i - 1
            link = '../' * up_levels + 'index.html' if up_levels > 0 else 'index.html'
            if i < len(dir_path.parts) - 1:
                breadcrumb_items.append(f'<a href="{link}" class="hover:text-gray-900">{part}</a>')
            else:
                breadcrumb_items.append(f'<span class="text-gray-900 font-semibold">{part}</span>')
        breadcrumb_html = ' <i class="fa-solid fa-chevron-right text-[10px] text-gray-300 mx-2"></i> '.join(breadcrumb_items)

        # Find subdirectories and modules
        subdirs = set()
        for other_dir in dir_modules.keys():
            if len(other_dir.parts) == len(dir_path.parts) + 1:
                if other_dir.parts[:len(dir_path.parts)] == dir_path.parts:
                    subdirs.add(other_dir.parts[-1])

        modules = dir_modules.get(dir_path, [])

        content = []

        # Subdirectories
        if subdirs:
            content.append('<h2 class="text-2xl font-bold text-gray-900 mb-6 flex items-center gap-2 px-1"><i class="fa-regular fa-folder text-yellow-500"></i> Packages</h2>')
            content.append('<div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-12">')
            for subdir in sorted(subdirs):
                content.append(f'''
                <a href="{subdir}/index.html" class="group block p-6 bg-white rounded-xl border border-gray-200 hover:border-indigo-500 hover:shadow-md transition-all">
                    <div class="flex items-center gap-3 mb-2">
                        <span class="w-10 h-10 rounded-lg bg-yellow-50 text-yellow-600 flex items-center justify-center text-xl group-hover:bg-yellow-100 transition-colors">
                            <i class="fa-regular fa-folder"></i>
                        </span>
                        <h3 class="font-bold text-gray-900 group-hover:text-indigo-600 transition-colors">{subdir}</h3>
                    </div>
                </a>
                ''')
            content.append('</div>')

        # Modules in this directory
        if modules:
            content.append('<h2 class="text-2xl font-bold text-gray-900 mb-6 flex items-center gap-2 px-1"><i class="fa-brands fa-python text-blue-500"></i> Modules</h2>')
            content.append('<div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">')
            for doc in sorted(modules, key=lambda d: d.module_name):
                summary = ''
                if doc.docstring:
                    summary = DocstringParser(doc.docstring, self._class_map).summary[:100]
                    if len(doc.docstring) > 100:
                        summary += '...'

                stats = f'{len(doc.classes)} classes, {len(doc.functions)} functions'
                content.append(f'''
                <a href="{doc.module_name}.html" class="group block bg-white rounded-xl border border-gray-200 shadow-sm hover:shadow-md hover:border-indigo-500 transition-all overflow-hidden flex flex-col h-full">
                    <div class="p-6 flex-grow">
                        <div class="flex items-center gap-3 mb-4">
                            <span class="w-10 h-10 rounded-lg bg-blue-50 text-blue-600 flex items-center justify-center text-xl group-hover:bg-blue-100 transition-colors">
                                <i class="fa-brands fa-python"></i>
                            </span>
                            <h3 class="font-bold text-gray-900 group-hover:text-indigo-600 transition-colors">{doc.module_name}</h3>
                        </div>
                        <p class="text-sm text-gray-600 mb-4 line-clamp-2">{html.escape(summary)}</p>
                    </div>
                    <div class="px-6 py-3 bg-gray-50 border-t border-gray-100 text-xs text-gray-500 font-mono">
                        {stats}
                    </div>
                </a>
                ''')
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

    def _generate_main_index(self):
        """Generate the main api-reference.html."""
        output_path = self.output_dir / 'api-reference.html'

        # Get top-level directories
        top_dirs = set()
        for doc in self.all_modules:
            rel_path = doc.relative_path.relative_to(self.source_root)
            if len(rel_path.parts) > 1:
                top_dirs.add(rel_path.parts[0])

        content = []
        
        # Hero section for docs
        content.append(f'''
        <div class="bg-gradient-to-br from-indigo-600 to-violet-700 rounded-2xl p-8 mb-10 text-white shadow-lg">
            <h2 class="text-3xl font-bold mb-4">{DOC_CONFIG["project_name"]} API Reference</h2>
            <div class="flex gap-4 text-indigo-100 text-sm">
                 <span><i class="fa-regular fa-clock mr-1"></i> Generated: {datetime.now().strftime("%Y-%m-%d")}</span>
                 <span><i class="fa-solid fa-layer-group mr-1"></i> Modules: {len(self.all_modules)}</span>
            </div>
        </div>
        ''')

        if top_dirs:
            content.append('<h2 class="text-2xl font-bold text-gray-900 mb-6 flex items-center gap-2 px-1"><i class="fa-solid fa-boxes-stacked text-indigo-500"></i> Packages</h2>')
            content.append('<div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-12">')
            for dir_name in sorted(top_dirs):
                content.append(f'''
                <a href="{dir_name}/index.html" class="group block p-6 bg-white rounded-xl border border-gray-200 hover:border-indigo-500 hover:shadow-md transition-all">
                    <div class="flex items-center gap-4 mb-2">
                        <span class="w-12 h-12 rounded-lg bg-blue-50 text-blue-600 flex items-center justify-center text-2xl group-hover:bg-indigo-100 transition-colors">
                            <i class="fa-regular fa-folder"></i>
                        </span>
                        <h3 class="text-lg font-bold text-gray-900 group-hover:text-indigo-600 transition-colors">{dir_name}</h3>
                    </div>
                    <p class="text-gray-500 text-sm pl-16">Package documentation</p>
                </a>
                ''')
            content.append('</div>')

        # Root-level modules
        root_modules = [doc for doc in self.all_modules
                        if len(doc.relative_path.relative_to(self.source_root).parts) == 1]

        if root_modules:
            content.append('<h2 class="text-2xl font-bold text-gray-900 mb-6 flex items-center gap-2 px-1"><i class="fa-brands fa-python text-blue-500"></i> Root Modules</h2>')
            content.append('<div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">')
            for doc in sorted(root_modules, key=lambda d: d.module_name):
                summary = ''
                if doc.docstring:
                    summary = DocstringParser(doc.docstring, self._class_map).summary[:100]

                content.append(f'''
                <a href="{doc.module_name}.html" class="group block bg-white rounded-xl border border-gray-200 shadow-sm hover:shadow-md hover:border-indigo-500 transition-all overflow-hidden flex flex-col h-full">
                    <div class="p-6 flex-grow">
                        <div class="flex items-center gap-3 mb-4">
                            <span class="w-10 h-10 rounded-lg bg-blue-50 text-blue-600 flex items-center justify-center text-xl group-hover:bg-blue-100 transition-colors">
                                <i class="fa-brands fa-python"></i>
                            </span>
                            <h3 class="font-bold text-gray-900 group-hover:text-indigo-600 transition-colors">{doc.module_name}</h3>
                        </div>
                        <p class="text-sm text-gray-600 mb-4 line-clamp-2">{html.escape(summary)}</p>
                    </div>
                </a>
                ''')
            content.append('</div>')

        html_content = self._wrap_page(
            title=f'{DOC_CONFIG["project_name"]} API Documentation',
            content='\n'.join(content),
            index_path='index.html',
            breadcrumb='<a href="/docs/getting_started.html" class="hover:text-gray-900">Documentation</a> <i class="fa-solid fa-chevron-right text-[10px] text-gray-300 mx-2"></i> <span class="text-gray-900 font-semibold">API Reference</span>',
            header_info={'name': DOC_CONFIG["project_name"], 'path': ''}
        )

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
    output_dir = repo / 'website' / 'templates' / 'docs_api'

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
        # Skip test files and __pycache__
        if '__pycache__' in str(filepath) or filepath.name.startswith('test_'):
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
