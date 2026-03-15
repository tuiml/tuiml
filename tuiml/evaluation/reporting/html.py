"""
HTML table formatter.
"""

from ..statistics import SignificanceLevel

def to_html_table(matrix, precision: int = 4) -> str:
    """
    Convert ResultMatrix to HTML table format.

    Parameters
    ----------
    matrix : ResultMatrix
        The result matrix to format.
    precision : int
        Decimal precision for values.

    Returns
    -------
    html : str
        HTML table string.
    """
    matrix.compute_statistics()

    lines = []

    lines.append("<table border='1' style='border-collapse: collapse;'>")
    lines.append(f"<caption>{matrix.metric_name} Comparison</caption>")

    lines.append("<thead><tr>")
    lines.append("<th>Dataset</th>")
    for model in matrix.model_names:
        lines.append(f"<th>{model}</th>")
    lines.append("</tr></thead>")

    lines.append("<tbody>")
    for dataset in matrix.dataset_names:
        lines.append("<tr>")
        lines.append(f"<td>{dataset}</td>")
        for model in matrix.model_names:
            key = (dataset, model)
            if key in matrix._cells:
                cell = matrix._cells[key]
                if matrix.show_std:
                    val = f"{cell.mean:.{precision}f} ± {cell.std:.{precision}f}"
                else:
                    val = f"{cell.mean:.{precision}f}"

                style = ""
                if cell.significance == SignificanceLevel.WIN:
                    style = "background-color: #90EE90; font-weight: bold;"
                    val += " ▲"
                elif cell.significance == SignificanceLevel.LOSS:
                    style = "background-color: #FFB6C1;"
                    val += " ▼"

                lines.append(f"<td style='{style}'>{val}</td>")
            else:
                lines.append("<td>N/A</td>")
        lines.append("</tr>")

    lines.append("<tr style='font-weight: bold;'>")
    lines.append("<td>W/L/T</td>")
    for i in range(matrix.n_models):
        lines.append(
            f"<td>{matrix._wins[i]}/{matrix._losses[i]}/{matrix._ties[i]}</td>"
        )
    lines.append("</tr>")

    lines.append("</tbody></table>")

    return "\n".join(lines)
