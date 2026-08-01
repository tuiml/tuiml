"""Turning experiment results into tables you can publish.

A benchmark produces a models-by-datasets grid of scores. These format that
grid for wherever it is going next: a terminal, a paper, a README, or a web
page — without hand-writing the markup each time.

Formatters
----------
- **format_results:** Plain-text table, for a terminal or a log.
- **to_markdown_table:** Markdown, for a README or an issue.
- **to_latex_table:** LaTeX ``tabular``, for a paper.
- **to_html_table:** HTML, for a web page or notebook.

Types
-----
- **ResultMatrix:** The scores themselves, indexed by model and dataset.
- **ComparisonCell:** One cell, carrying its score alongside the outcome of
  any significance test against the baseline, so a table can mark wins and
  losses rather than just printing numbers.

See Also
--------
:mod:`tuiml.evaluation.statistics` : Produces the test outcomes the
    comparison cells report.
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
