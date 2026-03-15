"""
LaTeX table formatter.
"""

from ..statistics import SignificanceLevel

def to_latex_table(matrix, precision: int = 4) -> str:
    """
    Convert ResultMatrix to LaTeX table format.

    Parameters
    ----------
    matrix : ResultMatrix
        The result matrix to format.
    precision : int
        Decimal precision for values.

    Returns
    -------
    latex : str
        LaTeX table string.
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
