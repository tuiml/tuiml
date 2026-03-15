"""
Markdown table formatter.
"""

from ..statistics import SignificanceLevel

def to_markdown_table(matrix, precision: int = 4) -> str:
    """
    Convert ResultMatrix to Markdown table format.

    Parameters
    ----------
    matrix : ResultMatrix
        The result matrix to format.
    precision : int
        Decimal precision for values.

    Returns
    -------
    markdown : str
        Markdown table string.
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
