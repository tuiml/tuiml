"""
Markdown renderer for benchmark result matrices.

Turns a :class:`~tuiml.evaluation.reporting.ResultMatrix` into a
GitHub-flavoured pipe table. This is the format to reach for when the results
are headed somewhere that renders Markdown but not HTML or LaTeX: a README, a
pull-request comment, an issue, a lab notebook, or a docs page built from
Markdown sources.

Significance against the baseline model is carried by inline emphasis rather
than colour, so the table stays readable as raw text: a significant win is
**bolded** and suffixed with ▲, a significant loss is suffixed with ▼. The
module exposes a single function,
:func:`~tuiml.evaluation.reporting.to_markdown_table`, which is also what
:meth:`ResultMatrix.to_markdown` calls.
"""

from ..statistics import SignificanceLevel

def to_markdown_table(matrix, precision: int = 4) -> str:
    """
    Render a result matrix as a GitHub-flavoured Markdown pipe table.

    The table has a leading ``Dataset`` column, one column per model in
    ``matrix.model_names``, one row per dataset, and a final bolded
    ``**W/L/T**`` row giving each model's win / loss / tie record against the
    baseline::

        | Dataset | RF | SVM |
        |---|---|---|
        | iris | 0.9460 ± 0.0114 | 0.8960 ± 0.0114 ▼ |
        | wine | 0.9780 ± 0.0084 | 0.9620 ± 0.0084 |
        |---|---|---|
        | **W/L/T** | 0/0/1 | 0/1/0 |

    Each cell is ``mean ± std`` (just the mean when ``matrix.show_std`` is
    ``False``). A cell that significantly beats the baseline is wrapped in
    ``**bold**`` and suffixed with ▲; one that significantly loses is suffixed
    with ▼; a tie and the baseline column itself are left unmarked. A
    (dataset, model) pair that was never recorded renders as ``N/A``.

    The function calls ``matrix.compute_statistics()`` itself, so the matrix only
    needs to have been filled with
    :meth:`~tuiml.evaluation.reporting.ResultMatrix.add_result`.

    Parameters
    ----------
    matrix : ResultMatrix
        The filled result matrix to render. Its ``show_std``, ``model_names``
        and ``dataset_names`` attributes drive the layout.
    precision : int, default=4
        Number of digits after the decimal point for means and standard
        deviations.

    Returns
    -------
    markdown : str
        The pipe table, lines joined by ``\\n`` and not terminated by a newline.

    Notes
    -----
    Model and dataset names are inserted as-is. A name containing a pipe
    character would break the table, and one containing Markdown syntax such as
    ``_`` or ``*`` will be interpreted as emphasis by the renderer; rename or
    escape such labels before rendering.

    See Also
    --------
    :class:`~tuiml.evaluation.reporting.ResultMatrix` : The grid being rendered.
    :func:`~tuiml.evaluation.reporting.to_html_table` : Same table with colour-coded HTML cells.
    :func:`~tuiml.evaluation.reporting.to_latex_table` : Same table as a LaTeX float for a paper.

    Examples
    --------
    >>> from tuiml.evaluation.reporting import ResultMatrix, to_markdown_table
    >>> matrix = ResultMatrix(
    ...     model_names=['RF', 'SVM'],
    ...     dataset_names=['iris', 'wine'],
    ...     metric_name='accuracy',
    ... )
    >>> matrix.add_result('iris', 'RF', values=[0.95, 0.93, 0.94, 0.96, 0.95])
    >>> matrix.add_result('iris', 'SVM', values=[0.90, 0.89, 0.91, 0.90, 0.88])
    >>> matrix.add_result('wine', 'RF', values=[0.98, 0.97, 0.99, 0.98, 0.97])
    >>> matrix.add_result('wine', 'SVM', values=[0.96, 0.97, 0.95, 0.96, 0.97])
    >>> print(to_markdown_table(matrix, precision=3))
    | Dataset | RF | SVM |
    |---|---|---|
    | iris | 0.946 ± 0.011 | 0.896 ± 0.011 ▼ |
    | wine | 0.978 ± 0.008 | 0.962 ± 0.008 |
    |---|---|---|
    | **W/L/T** | 0/0/2 | 0/1/1 |
    """
    matrix.compute_statistics()

    lines = []

    header = "| Dataset | " + " | ".join(matrix.model_names) + " |"
    lines.append(header)
    lines.append("|" + "---|" * (matrix.n_models + 1))

    for dataset in matrix.dataset_names:
        row = [f"| {dataset}"]
        for model in matrix.model_names:
            key = (dataset, model)
            if key in matrix._cells:
                cell = matrix._cells[key]
                if matrix.show_std:
                    val = f"{cell.mean:.{precision}f} ± {cell.std:.{precision}f}"
                else:
                    val = f"{cell.mean:.{precision}f}"

                if cell.significance == SignificanceLevel.WIN:
                    val = f"**{val}** ▲"
                elif cell.significance == SignificanceLevel.LOSS:
                    val = f"{val} ▼"
            else:
                val = "N/A"
            row.append(val)
        lines.append(" | ".join(row) + " |")

    lines.append("|" + "---|" * (matrix.n_models + 1))
    row = ["| **W/L/T**"]
    for i in range(matrix.n_models):
        row.append(f"{matrix._wins[i]}/{matrix._losses[i]}/{matrix._ties[i]}")
    lines.append(" | ".join(row) + " |")

    return "\n".join(lines)
