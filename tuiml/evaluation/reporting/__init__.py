"""
Result formatters for experiment output.
- ResultMatrix.java
- ResultMatrixPlainText.java
- ResultMatrixCSV.java
- ResultMatrixHTML.java
- ResultMatrixLatex.java

Provides formatters for:
- Plain text tables
- CSV export
- LaTeX tables
- HTML tables
- Markdown tables
"""

from .result_matrix import (
    ResultMatrix,
    ComparisonCell,
    format_results,
)
from .latex import to_latex_table
from .html import to_html_table
from .markdown import to_markdown_table

__all__ = [
    "ResultMatrix",
    "ComparisonCell",
    "format_results",
    "to_latex_table",
    "to_html_table",
    "to_markdown_table",
]
