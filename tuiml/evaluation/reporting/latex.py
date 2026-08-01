"""
LaTeX renderer for benchmark result matrices.

Turns a :class:`~tuiml.evaluation.reporting.ResultMatrix` into a complete
``table`` float, so a benchmark can go from an experiment loop straight into a
paper's results section without hand-typing numbers. The output is meant to be
pasted into (or ``\\input``-ed by) a LaTeX document.

The markup deliberately stays close to plain LaTeX: a ``tabular`` with ``\\hline``
rules, values typeset in inline math as ``$mean \\pm std$``, and significance
against the baseline shown by superscript triangles rather than colour, since
results tables are usually printed in black and white. Winning cells also get
``\\textbf``. The triangle superscripts ``\\blacktriangle`` and
``\\blacktriangledown`` come from ``amssymb``, so that package must be loaded in
the document preamble.

The module exposes a single function,
:func:`~tuiml.evaluation.reporting.to_latex_table`, which is also what
:meth:`ResultMatrix.to_latex` calls.
"""

from ..statistics import SignificanceLevel

def to_latex_table(matrix, precision: int = 4) -> str:
    """
    Render a result matrix as a complete LaTeX ``table`` float.

    The output is a self-contained ``table`` environment: a ``\\caption`` naming
    the metric, a centred ``tabular`` with an ``l`` column for the dataset names
    and one ``c`` column per model, and a final ``W/L/T`` row summarising each
    model's record against the baseline::

        \\begin{table}[htbp]
        \\caption{accuracy comparison}
        \\centering
        \\begin{tabular}{lcc}
        \\hline
        Dataset & RF & SVM \\\\
        \\hline
        iris & $0.946 \\pm 0.011$ & $0.896 \\pm 0.011$$^\\blacktriangledown$ \\\\
        \\hline
        W/L/T & 0/0/1 & 0/1/0 \\\\
        \\hline
        \\end{tabular}
        \\end{table}

    Values are typeset in inline math as ``$mean \\pm std$``, or just ``$mean$``
    when ``matrix.show_std`` is ``False``. A cell that significantly beats the
    baseline is wrapped in ``\\textbf`` and marked with a superscript
    ``\\blacktriangle``; a significant loss gets a superscript
    ``\\blacktriangledown``. Both symbols require ``\\usepackage{amssymb}`` in the
    preamble. A (dataset, model) pair that was never recorded renders as ``---``.

    The function calls ``matrix.compute_statistics()`` itself, so the matrix only
    needs to have been filled with
    :meth:`~tuiml.evaluation.reporting.ResultMatrix.add_result`.

    Parameters
    ----------
    matrix : ResultMatrix
        The filled result matrix to render. Its ``metric_name`` becomes the
        caption and ``show_std`` decides whether the ``\\pm`` term appears.
    precision : int, default=4
        Number of digits after the decimal point for means and standard
        deviations. Paper tables usually want 2 or 3 rather than the default.

    Returns
    -------
    latex : str
        The ``table`` block, lines joined by ``\\n`` and not terminated by a
        newline.

    Notes
    -----
    Only dataset names are escaped, and only for underscores (``_`` becomes
    ``\\_``). Model names are emitted verbatim, so a model called ``grad_boost``
    will raise a LaTeX error in math mode; rename such labels, or escape them,
    before rendering. Other special characters (``%``, ``&``, ``#``, ``$``) are
    not escaped in either position.

    See Also
    --------
    :class:`~tuiml.evaluation.reporting.ResultMatrix` : The grid being rendered.
    :func:`~tuiml.evaluation.reporting.to_markdown_table` : Same table for READMEs and PR comments.
    :func:`~tuiml.evaluation.reporting.to_html_table` : Same table with colour-coded HTML cells.

    Examples
    --------
    >>> from tuiml.evaluation.reporting import ResultMatrix, to_latex_table
    >>> matrix = ResultMatrix(
    ...     model_names=['RF', 'SVM'],
    ...     dataset_names=['iris'],
    ...     metric_name='accuracy',
    ... )
    >>> matrix.add_result('iris', 'RF', values=[0.95, 0.93, 0.94, 0.96, 0.95])
    >>> matrix.add_result('iris', 'SVM', values=[0.90, 0.89, 0.91, 0.90, 0.88])
    >>> print(to_latex_table(matrix, precision=3))
    \\begin{table}[htbp]
    \\caption{accuracy comparison}
    \\centering
    \\begin{tabular}{lcc}
    \\hline
    Dataset & RF & SVM \\\\
    \\hline
    iris & $0.946 \\pm 0.011$ & $0.896 \\pm 0.011$$^\\blacktriangledown$ \\\\
    \\hline
    W/L/T & 0/0/1 & 0/1/0 \\\\
    \\hline
    \\end{tabular}
    \\end{table}
    """
    matrix.compute_statistics()

    lines = []

    lines.append("\\begin{table}[htbp]")
    lines.append(f"\\caption{{{matrix.metric_name} comparison}}")
    lines.append("\\centering")
    lines.append("\\begin{tabular}{l" + "c" * matrix.n_models + "}")
    lines.append("\\hline")

    header = "Dataset & " + " & ".join(matrix.model_names) + " \\\\"
    lines.append(header)
    lines.append("\\hline")

    for dataset in matrix.dataset_names:
        row = [dataset.replace("_", "\\_")]
        for model in matrix.model_names:
            key = (dataset, model)
            if key in matrix._cells:
                cell = matrix._cells[key]
                if matrix.show_std:
                    val = f"${cell.mean:.{precision}f} \\pm {cell.std:.{precision}f}$"
                else:
                    val = f"${cell.mean:.{precision}f}$"

                if cell.significance == SignificanceLevel.WIN:
                    val = f"\\textbf{{{val}}}$^\\blacktriangle$"
                elif cell.significance == SignificanceLevel.LOSS:
                    val = f"{val}$^\\blacktriangledown$"
            else:
                val = "---"
            row.append(val)
        lines.append(" & ".join(row) + " \\\\")

    lines.append("\\hline")

    lines.append(" & ".join(["W/L/T"] + [
        f"{matrix._wins[i]}/{matrix._losses[i]}/{matrix._ties[i]}"
        for i in range(matrix.n_models)
    ]) + " \\\\")

    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    return "\n".join(lines)
