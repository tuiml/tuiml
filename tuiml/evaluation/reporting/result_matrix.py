"""
Result matrix for experiment output formatting.
"""

import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

from ..statistics import SignificanceLevel, paired_t_test

@dataclass
class ComparisonCell:
    """
    A cell in the comparison matrix.

    Attributes
    ----------
    mean : float
        Mean value.
    std : float
        Standard deviation.
    significance : SignificanceLevel
        Significance compared to baseline.
    is_baseline : bool
        Whether this is the baseline.
    """
    mean: float
    std: float
    significance: SignificanceLevel = SignificanceLevel.TIE
    is_baseline: bool = False

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Any]:
        """
        Get JSON Schema for ComparisonCell parameters.

        Returns
        -------
        schema : dict
            JSON Schema describing the dataclass fields.
        """
        return {
            "type": "object",
            "properties": {
                "mean": {
                    "type": "number",
                    "description": "Mean value."
                },
                "std": {
                    "type": "number",
                    "description": "Standard deviation."
                },
                "significance": {
                    "type": "string",
                    "enum": ["WIN", "LOSS", "TIE"],
                    "default": "TIE",
                    "description": "Significance compared to baseline."
                },
                "is_baseline": {
                    "type": "boolean",
                    "default": False,
                    "description": "Whether this is the baseline."
                }
            },
            "required": ["mean", "std"],
            "additionalProperties": False
        }

class ResultMatrix:
    """
    Matrix of experiment results with statistical comparisons.

    Parameters
    ----------
    model_names : list of str
        Names of models (columns).
    dataset_names : list of str
        Names of datasets (rows).
    metric_name : str
        Name of the metric being compared.
    higher_better : bool
        Whether higher values are better.
    significance_level : float
        Significance level for statistical tests.
    show_std : bool
        Whether to show standard deviations.

    Examples
    --------
    >>> matrix = ResultMatrix(
    ...     model_names=['RF', 'SVM', 'NB'],
    ...     dataset_names=['iris', 'wine', 'digits'],
    ...     metric_name='accuracy'
    ... )
    >>> matrix.add_result('iris', 'RF', values=[0.95, 0.93, 0.94])
    >>> matrix.add_result('iris', 'SVM', values=[0.92, 0.91, 0.93])
    >>> print(matrix.to_string())
    """

    WIN_SYMBOL = "▲"
    LOSS_SYMBOL = "▼"
    TIE_SYMBOL = ""

    def __init__(
        self,
        model_names: List[str],
        dataset_names: List[str],
        metric_name: str = "metric",
        higher_better: bool = True,
        significance_level: float = 0.05,
        show_std: bool = True,
        baseline_index: int = 0
    ):
        self.model_names = model_names
        self.dataset_names = dataset_names
        self.metric_name = metric_name
        self.higher_better = higher_better
        self.significance_level = significance_level
        self.show_std = show_std
        self.baseline_index = baseline_index

        self.n_models = len(model_names)
        self.n_datasets = len(dataset_names)

        self._values: Dict[Tuple[str, str], np.ndarray] = {}
        self._cells: Dict[Tuple[str, str], ComparisonCell] = {}

        self._wins: np.ndarray = np.zeros(self.n_models, dtype=int)
        self._losses: np.ndarray = np.zeros(self.n_models, dtype=int)
        self._ties: np.ndarray = np.zeros(self.n_models, dtype=int)

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Any]:
        """
        Get JSON Schema for ResultMatrix __init__ parameters.

        Returns
        -------
        schema : dict
            JSON Schema describing the constructor parameters.
        """
        return {
            "type": "object",
            "properties": {
                "model_names": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Names of models (columns)."
                },
                "dataset_names": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Names of datasets (rows)."
                },
                "metric_name": {
                    "type": "string",
                    "default": "metric",
                    "description": "Name of the metric being compared."
                },
                "higher_better": {
                    "type": "boolean",
                    "default": True,
                    "description": "Whether higher values are better."
                },
                "significance_level": {
                    "type": "number",
                    "default": 0.05,
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "Significance level for statistical tests."
                },
                "show_std": {
                    "type": "boolean",
                    "default": True,
                    "description": "Whether to show standard deviations."
                },
                "baseline_index": {
                    "type": "integer",
                    "default": 0,
                    "minimum": 0,
                    "description": "Index of the baseline model for comparisons."
                }
            },
            "required": ["model_names", "dataset_names"],
            "additionalProperties": False
        }

    def add_result(
        self,
        dataset: str,
        model: str,
        values: Union[List[float], np.ndarray]
    ):
        """Add results for a dataset-model combination."""
        self._values[(dataset, model)] = np.asarray(values)

    def compute_statistics(self):
        """Compute all pairwise statistics."""
        baseline_name = self.model_names[self.baseline_index]

        for dataset in self.dataset_names:
            baseline_key = (dataset, baseline_name)
            if baseline_key not in self._values:
                continue

            baseline_values = self._values[baseline_key]

            for i, model in enumerate(self.model_names):
                key = (dataset, model)
                if key not in self._values:
                    continue

                values = self._values[key]
                mean = np.mean(values)
                std = np.std(values, ddof=1) if len(values) > 1 else 0

                if i == self.baseline_index:
                    self._cells[key] = ComparisonCell(
                        mean=mean, std=std,
                        significance=SignificanceLevel.TIE,
                        is_baseline=True
                    )
                    self._ties[i] += 1
                else:
                    try:
                        stats = paired_t_test(
                            values, baseline_values,
                            significance_level=self.significance_level,
                            higher_better=self.higher_better
                        )
                        self._cells[key] = ComparisonCell(
                            mean=mean, std=std,
                            significance=stats.significance,
                            is_baseline=False
                        )

                        if stats.significance == SignificanceLevel.WIN:
                            self._wins[i] += 1
                        elif stats.significance == SignificanceLevel.LOSS:
                            self._losses[i] += 1
                        else:
                            self._ties[i] += 1
                    except Exception:
                        self._cells[key] = ComparisonCell(
                            mean=mean, std=std,
                            significance=SignificanceLevel.TIE,
                            is_baseline=False
                        )
                        self._ties[i] += 1

    def _format_value(self, cell: ComparisonCell, precision: int = 4) -> str:
        """Format a cell value."""
        if self.show_std:
            val = f"{cell.mean:.{precision}f} ± {cell.std:.{precision}f}"
        else:
            val = f"{cell.mean:.{precision}f}"

        if cell.significance == SignificanceLevel.WIN:
            val += f" {self.WIN_SYMBOL}"
        elif cell.significance == SignificanceLevel.LOSS:
            val += f" {self.LOSS_SYMBOL}"

        return val

    def to_string(self, precision: int = 4) -> str:
        """Convert to plain text string."""
        self.compute_statistics()

        col_width = max(15, max(len(m) for m in self.model_names) + 2)
        dataset_width = max(15, max(len(d) for d in self.dataset_names) + 2)

        lines = []

        header = "Dataset".ljust(dataset_width)
        for model in self.model_names:
            header += model.center(col_width)
        lines.append(header)
        lines.append("-" * len(header))

        for dataset in self.dataset_names:
            row = dataset.ljust(dataset_width)
            for model in self.model_names:
                key = (dataset, model)
                if key in self._cells:
                    val = self._format_value(self._cells[key], precision)
                else:
                    val = "N/A"
                row += val.center(col_width)
            lines.append(row)

        lines.append("-" * len(header))

        row = "Wins".ljust(dataset_width)
        for i in range(self.n_models):
            row += str(self._wins[i]).center(col_width)
        lines.append(row)

        row = "Losses".ljust(dataset_width)
        for i in range(self.n_models):
            row += str(self._losses[i]).center(col_width)
        lines.append(row)

        row = "Ties".ljust(dataset_width)
        for i in range(self.n_models):
            row += str(self._ties[i]).center(col_width)
        lines.append(row)

        lines.append("")
        lines.append(f"Baseline: {self.model_names[self.baseline_index]}")
        lines.append(f"Significance level: {self.significance_level}")
        lines.append(f"{self.WIN_SYMBOL} = significantly better, {self.LOSS_SYMBOL} = significantly worse")

        return "\n".join(lines)

    def to_csv(self, precision: int = 4) -> str:
        """Convert to CSV format."""
        self.compute_statistics()

        lines = []

        header = ["Dataset"] + self.model_names
        lines.append(",".join(header))

        for dataset in self.dataset_names:
            row = [dataset]
            for model in self.model_names:
                key = (dataset, model)
                if key in self._cells:
                    cell = self._cells[key]
                    if self.show_std:
                        val = f"{cell.mean:.{precision}f} ± {cell.std:.{precision}f}"
                    else:
                        val = f"{cell.mean:.{precision}f}"
                else:
                    val = ""
                row.append(val)
            lines.append(",".join(row))

        lines.append(",".join(["Wins"] + [str(w) for w in self._wins]))
        lines.append(",".join(["Losses"] + [str(l) for l in self._losses]))
        lines.append(",".join(["Ties"] + [str(t) for t in self._ties]))

        return "\n".join(lines)

    def to_latex(self, precision: int = 4) -> str:
        """Convert to LaTeX format."""
        from .latex import to_latex_table
        return to_latex_table(self, precision)

    def to_html(self, precision: int = 4) -> str:
        """Convert to HTML format."""
        from .html import to_html_table
        return to_html_table(self, precision)

    def to_markdown(self, precision: int = 4) -> str:
        """Convert to Markdown format."""
        from .markdown import to_markdown_table
        return to_markdown_table(self, precision)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        self.compute_statistics()

        return {
            "metric_name": self.metric_name,
            "model_names": self.model_names,
            "dataset_names": self.dataset_names,
            "results": {
                f"{d}_{m}": {
                    "mean": self._cells[(d, m)].mean,
                    "std": self._cells[(d, m)].std,
                    "significance": self._cells[(d, m)].significance.name
                }
                for d in self.dataset_names
                for m in self.model_names
                if (d, m) in self._cells
            },
            "summary": {
                model: {
                    "wins": int(self._wins[i]),
                    "losses": int(self._losses[i]),
                    "ties": int(self._ties[i])
                }
                for i, model in enumerate(self.model_names)
            }
        }

def format_results(
    results: Dict[str, Dict[str, np.ndarray]],
    metric_name: str = "metric",
    format_type: str = "plain",
    **kwargs
) -> str:
    """
    Format experiment results.

    Parameters
    ----------
    results : dict
        Dictionary of {dataset: {model: values}}.
    metric_name : str
        Name of the metric.
    format_type : str
        Output format: 'plain', 'csv', 'latex', 'html', 'markdown'.
    **kwargs
        Additional arguments for ResultMatrix.

    Returns
    -------
    formatted : str
        Formatted results.
    """
    datasets = list(results.keys())
    models = list(set(m for d in results.values() for m in d.keys()))

    matrix = ResultMatrix(
        model_names=models,
        dataset_names=datasets,
        metric_name=metric_name,
        **kwargs
    )

    for dataset, model_results in results.items():
        for model, values in model_results.items():
            matrix.add_result(dataset, model, values)

    if format_type == "csv":
        return matrix.to_csv()
    elif format_type == "latex":
        return matrix.to_latex()
    elif format_type == "html":
        return matrix.to_html()
    elif format_type == "markdown":
        return matrix.to_markdown()
    else:
        return matrix.to_string()
