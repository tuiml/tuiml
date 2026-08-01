"""
Models-by-datasets result grids for benchmark reporting.

This module provides :class:`~tuiml.evaluation.reporting.ResultMatrix`, the data
structure that sits at the end of a benchmark loop: it collects the raw
per-fold scores of every (dataset, model) pair, reduces each pair to a
mean/standard-deviation cell, runs a paired t-test of every model against a
chosen baseline model, and renders the whole grid as a table.

Reach for it once every model has been scored on every dataset and the loose
arrays of fold scores need to become something a reader can look at. The
rendering of the non-plain-text formats is delegated to the sibling exporters
:func:`~tuiml.evaluation.reporting.to_markdown_table`,
:func:`~tuiml.evaluation.reporting.to_latex_table` and
:func:`~tuiml.evaluation.reporting.to_html_table`; plain text and CSV are
produced in this module.

The usual flow is three steps::

    matrix = ResultMatrix(model_names=..., dataset_names=...)
    matrix.add_result(dataset, model, values=[...])   # once per pair
    print(matrix.to_markdown())

:func:`~tuiml.evaluation.reporting.format_results` wraps those three steps for
the common case where the scores already sit in a nested
``{dataset: {model: values}}`` dictionary.
"""

import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

from ..statistics import SignificanceLevel, paired_t_test

@dataclass
class ComparisonCell:
    """
    One aggregated (dataset, model) cell of a result matrix.

    A cell is the *reduced* form of a run: the list of per-fold scores handed to
    :meth:`~tuiml.evaluation.reporting.ResultMatrix.add_result` is collapsed by
    :meth:`~tuiml.evaluation.reporting.ResultMatrix.compute_statistics` into a
    mean, a sample standard deviation and a verdict against the baseline model
    for the same dataset. Every exporter renders a cell as ``mean ± std``
    followed by a win/loss marker.

    Parameters
    ----------
    mean : float
        Arithmetic mean of the per-fold values.
    std : float
        Sample standard deviation (``ddof=1``) of the per-fold values, or ``0``
        when a single fold was recorded.
    significance : SignificanceLevel, default=SignificanceLevel.TIE
        Outcome of the paired t-test of this cell against the baseline model on
        the same dataset. ``WIN`` means significantly better than the baseline,
        ``LOSS`` significantly worse, ``TIE`` no significant difference.
    is_baseline : bool, default=False
        Whether this cell belongs to the baseline model itself. Baseline cells
        are always ``TIE``, since they are compared against themselves.

    Attributes
    ----------
    mean : float
        Arithmetic mean of the per-fold values.
    std : float
        Sample standard deviation of the per-fold values.
    significance : SignificanceLevel
        Verdict against the baseline model.
    is_baseline : bool
        Whether this cell belongs to the baseline model.

    See Also
    --------
    :class:`~tuiml.evaluation.reporting.ResultMatrix` : Grid that owns these cells.

    Examples
    --------
    >>> from tuiml.evaluation.reporting import ComparisonCell
    >>> cell = ComparisonCell(mean=0.946, std=0.0114, is_baseline=True)
    >>> print(f"{cell.mean:.4f} ± {cell.std:.4f}")
    0.9460 ± 0.0114
    >>> cell.significance.name
    'TIE'
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
    Grid of benchmark scores (datasets x models) with baseline significance tests.

    Overview
    --------
    1. Declare the grid: one row per dataset, one column per model, and pick
       which model column is the baseline (``baseline_index``).
    2. Fill it with :meth:`add_result`, once per (dataset, model) pair. What you
       hand over is the **list of raw per-fold scores** from a cross-validated
       run, for example the five accuracies of a 5-fold CV. Nothing is
       aggregated at this point; the array is stored verbatim.
    3. :meth:`compute_statistics` reduces each stored array to a
       :class:`~tuiml.evaluation.reporting.ComparisonCell` holding its mean and
       sample standard deviation, and runs a paired t-test of that array against
       the baseline model's array *on the same dataset*. The per-model
       win/loss/tie tallies shown in the last row are the counts of those
       verdicts across datasets. Every ``to_*`` method calls this for you.
    4. Render with :meth:`to_string`, :meth:`to_csv`, :meth:`to_markdown`,
       :meth:`to_latex`, :meth:`to_html` or :meth:`to_dict`.

    Theory
    ------
    For a dataset :math:`d`, a model :math:`m` and the baseline :math:`b`, the
    per-fold differences :math:`\\delta_k = s_{d,m,k} - s_{d,b,k}` are tested for
    a non-zero mean with a paired t-test at level ``significance_level``. The
    verdict recorded in the cell is

    .. math::
        \\text{WIN if } p < \\alpha \\text{ and the difference favours } m,
        \\quad \\text{LOSS if } p < \\alpha \\text{ and it favours } b,
        \\quad \\text{TIE otherwise,}

    where "favours" is read according to ``higher_better``. A pair that the test
    cannot handle (mismatched fold counts, zero variance) degrades to ``TIE``
    rather than raising.

    Parameters
    ----------
    model_names : list of str
        Names of the models. These become the table columns, in this order.
    dataset_names : list of str
        Names of the datasets. These become the table rows, in this order.
    metric_name : str, default="metric"
        Name of the metric being compared. Used only for captions and in
        :meth:`to_dict`; it does not change how values are treated.
    higher_better : bool, default=True
        Whether a larger value of the metric is a better result. Set to ``False``
        for error-like metrics (RMSE, log loss) so that wins and losses are not
        inverted.
    significance_level : float, default=0.05
        Alpha of the paired t-test against the baseline.
    show_std : bool, default=True
        Whether rendered cells carry the ``± std`` part. When ``False`` only the
        mean is printed.
    baseline_index : int, default=0
        Index into ``model_names`` of the model every other model is compared
        against.

    Attributes
    ----------
    n_models : int
        Number of model columns, ``len(model_names)``.
    n_datasets : int
        Number of dataset rows, ``len(dataset_names)``.
    WIN_SYMBOL : str
        Marker appended to a cell that significantly beats the baseline (``▲``).
    LOSS_SYMBOL : str
        Marker appended to a cell that significantly loses to the baseline (``▼``).
    TIE_SYMBOL : str
        Marker for a non-significant cell (empty string).

    Notes
    -----
    Missing pairs are legal. A (dataset, model) combination that was never
    passed to :meth:`add_result` renders as ``N/A`` (``---`` in LaTeX, empty in
    CSV) and takes no part in the tallies. A dataset whose baseline entry is
    missing is skipped entirely, because there is nothing to compare against.

    The win/loss/tie counters are accumulated inside
    :meth:`compute_statistics`, which every exporter invokes. Rendering the same
    matrix twice therefore doubles the counts in the summary row; build a fresh
    matrix, or hold on to the string from a single render, if you need more than
    one output format.

    See Also
    --------
    :func:`~tuiml.evaluation.reporting.format_results` : Build and render a matrix in one call.
    :class:`~tuiml.evaluation.reporting.ComparisonCell` : The aggregated cell type.
    :func:`~tuiml.evaluation.reporting.to_markdown_table` : Markdown exporter behind :meth:`to_markdown`.
    :func:`~tuiml.evaluation.reporting.to_latex_table` : LaTeX exporter behind :meth:`to_latex`.
    :func:`~tuiml.evaluation.reporting.to_html_table` : HTML exporter behind :meth:`to_html`.

    Examples
    --------
    >>> from tuiml.evaluation.reporting import ResultMatrix
    >>> matrix = ResultMatrix(
    ...     model_names=['RF', 'SVM'],
    ...     dataset_names=['iris', 'wine'],
    ...     metric_name='accuracy',
    ... )
    >>> matrix.add_result('iris', 'RF', values=[0.95, 0.93, 0.94, 0.96, 0.95])
    >>> matrix.add_result('iris', 'SVM', values=[0.90, 0.89, 0.91, 0.90, 0.88])
    >>> matrix.add_result('wine', 'RF', values=[0.98, 0.97, 0.99, 0.98, 0.97])
    >>> matrix.add_result('wine', 'SVM', values=[0.96, 0.97, 0.95, 0.96, 0.97])
    >>> print(matrix.to_markdown())
    | Dataset | RF | SVM |
    |---|---|---|
    | iris | 0.9460 ± 0.0114 | 0.8960 ± 0.0114 ▼ |
    | wine | 0.9780 ± 0.0084 | 0.9620 ± 0.0084 |
    |---|---|---|
    | **W/L/T** | 0/0/2 | 0/1/1 |

    RF is the baseline (column 0), so SVM carries the markers: it is
    significantly worse on ``iris`` and indistinguishable on ``wine``, giving it
    a 0 win / 1 loss / 1 tie record.
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
        """Initialize an empty grid; see the class docstring for the parameters."""
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
        """
        Record the raw per-fold scores of one (dataset, model) pair.

        The values are stored verbatim as a NumPy array; no mean or standard
        deviation is taken here. Aggregation happens later, in
        :meth:`compute_statistics`, which also needs the individual folds to run
        the paired t-test against the baseline. Pass the scores of a single
        cross-validated run, in fold order, and use the *same* number of folds
        for every model on a given dataset so the pairing is meaningful.

        Calling this twice for the same pair replaces the earlier entry.

        Parameters
        ----------
        dataset : str
            Dataset name. Must be one of ``dataset_names``; an unknown name is
            stored but never rendered.
        model : str
            Model name. Must be one of ``model_names``; an unknown name is
            stored but never rendered.
        values : list of float or np.ndarray of shape (n_folds,)
            Per-fold scores of this model on this dataset. A single-element
            sequence is accepted and yields a standard deviation of ``0``.

        Returns
        -------
        None

        See Also
        --------
        :meth:`compute_statistics` : Reduces the recorded values to cells.

        Examples
        --------
        >>> from tuiml.evaluation.reporting import ResultMatrix
        >>> matrix = ResultMatrix(
        ...     model_names=['RF'], dataset_names=['iris'], metric_name='accuracy'
        ... )
        >>> matrix.add_result('iris', 'RF', values=[0.95, 0.93, 0.94])
        >>> matrix.compute_statistics()
        >>> round(float(matrix._cells[('iris', 'RF')].mean), 4)
        0.94
        """
        self._values[(dataset, model)] = np.asarray(values)

    def compute_statistics(self):
        """
        Reduce every recorded value array to a cell and test it against the baseline.

        For each dataset the baseline model's array is looked up first; datasets
        without a baseline entry are skipped. Every other model on that dataset
        is then reduced to a :class:`~tuiml.evaluation.reporting.ComparisonCell`
        (mean, sample standard deviation with ``ddof=1``) and compared to the
        baseline with :func:`~tuiml.evaluation.statistics.paired_t_test` at
        ``significance_level``. The resulting verdict increments that model's
        win, loss or tie counter. A test that raises for any reason is caught and
        recorded as a tie.

        Every ``to_*`` method calls this first, so it rarely needs calling by
        hand; do so only when you want to inspect the cells directly. Note that
        the win/loss/tie counters are incremented, not reset, on each call.

        Returns
        -------
        None

        Examples
        --------
        >>> from tuiml.evaluation.reporting import ResultMatrix
        >>> matrix = ResultMatrix(
        ...     model_names=['RF', 'SVM'],
        ...     dataset_names=['iris'],
        ...     metric_name='accuracy',
        ... )
        >>> matrix.add_result('iris', 'RF', values=[0.95, 0.93, 0.94, 0.96, 0.95])
        >>> matrix.add_result('iris', 'SVM', values=[0.90, 0.89, 0.91, 0.90, 0.88])
        >>> matrix.compute_statistics()
        >>> matrix._cells[('iris', 'RF')].is_baseline
        True
        >>> matrix._cells[('iris', 'SVM')].significance.name
        'LOSS'
        """
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
        """
        Render one cell as the text shared by the plain-text and CSV outputs.

        Produces ``"mean ± std"`` (or just ``"mean"`` when ``show_std`` is
        ``False``), with :attr:`WIN_SYMBOL` or :attr:`LOSS_SYMBOL` appended when
        the cell is significantly better or worse than the baseline.

        Parameters
        ----------
        cell : ComparisonCell
            The aggregated cell to render.
        precision : int, default=4
            Number of digits after the decimal point.

        Returns
        -------
        value : str
            The formatted cell text, for example ``'0.8960 ± 0.0114 ▼'``.
        """
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
        """
        Render the grid as a fixed-width plain-text report.

        The output is a whitespace-aligned table meant for a terminal or a log
        file: a header row of model names, one row per dataset, then ``Wins`` /
        ``Losses`` / ``Ties`` summary rows and a short legend naming the baseline
        and the significance level::

            Dataset               RF            SVM
            ---------------------------------------
            iris                0.946         0.896 ▼
            ---------------------------------------
            Wins                  0              0
            Losses                0              1
            Ties                  1              0

            Baseline: RF
            Significance level: 0.05
            ▲ = significantly better, ▼ = significantly worse

        Column width is ``max(15, longest model name + 2)``, so cells wider than
        that (a long ``mean ± std`` pair at high precision) will run into their
        neighbour. Lower ``precision`` or set ``show_std=False`` if the columns
        collide.

        Parameters
        ----------
        precision : int, default=4
            Number of digits after the decimal point in each cell.

        Returns
        -------
        table : str
            The rendered report, lines joined by ``\\n``. Not terminated by a
            newline.

        See Also
        --------
        :meth:`to_markdown` : Same table as a Markdown pipe table.
        :meth:`to_csv` : Same numbers without the significance markers.

        Examples
        --------
        >>> from tuiml.evaluation.reporting import ResultMatrix
        >>> matrix = ResultMatrix(
        ...     model_names=['RF', 'SVM'],
        ...     dataset_names=['iris'],
        ...     metric_name='accuracy',
        ...     show_std=False,
        ... )
        >>> matrix.add_result('iris', 'RF', values=[0.95, 0.93, 0.94, 0.96, 0.95])
        >>> matrix.add_result('iris', 'SVM', values=[0.90, 0.89, 0.91, 0.90, 0.88])
        >>> print(matrix.to_string(precision=3))
        Dataset               RF            SVM
        ---------------------------------------------
        iris                0.946         0.896 ▼
        ---------------------------------------------
        Wins                  0              0
        Losses                0              1
        Ties                  1              0
        <BLANKLINE>
        Baseline: RF
        Significance level: 0.05
        ▲ = significantly better, ▼ = significantly worse
        """
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
        """
        Render the grid as comma-separated values for a spreadsheet.

        One header line of model names, one line per dataset, then three summary
        lines (``Wins``, ``Losses``, ``Ties``). Unlike the other exporters this
        one drops the ▲/▼ markers, so the file carries the numbers only; a
        missing (dataset, model) pair becomes an empty field rather than
        ``N/A``.

        Cells are written as ``mean ± std`` inside a single field, which keeps
        the file readable but means the value column is text, not a number, when
        ``show_std`` is ``True``. Set ``show_std=False`` for a CSV a spreadsheet
        will parse numerically.

        Parameters
        ----------
        precision : int, default=4
            Number of digits after the decimal point in each cell.

        Returns
        -------
        csv : str
            The rendered CSV, lines joined by ``\\n``. Not terminated by a
            newline and not quoted, so avoid commas in model or dataset names.

        See Also
        --------
        :meth:`to_dict` : Machine-readable output that keeps means and stds apart.

        Examples
        --------
        >>> from tuiml.evaluation.reporting import ResultMatrix
        >>> matrix = ResultMatrix(
        ...     model_names=['RF', 'SVM'],
        ...     dataset_names=['iris'],
        ...     metric_name='accuracy',
        ... )
        >>> matrix.add_result('iris', 'RF', values=[0.95, 0.93, 0.94, 0.96, 0.95])
        >>> matrix.add_result('iris', 'SVM', values=[0.90, 0.89, 0.91, 0.90, 0.88])
        >>> print(matrix.to_csv())
        Dataset,RF,SVM
        iris,0.9460 ± 0.0114,0.8960 ± 0.0114
        Wins,0,0
        Losses,0,1
        Ties,1,0
        """
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
        """
        Render the grid as a LaTeX ``table`` float, ready to paste into a paper.

        Thin wrapper around :func:`~tuiml.evaluation.reporting.to_latex_table`;
        see that function for the exact markup, the required packages and the
        escaping rules.

        Parameters
        ----------
        precision : int, default=4
            Number of digits after the decimal point in each cell.

        Returns
        -------
        latex : str
            A complete ``\\begin{table} ... \\end{table}`` block.

        See Also
        --------
        :func:`~tuiml.evaluation.reporting.to_latex_table` : The exporter this delegates to.

        Examples
        --------
        >>> from tuiml.evaluation.reporting import ResultMatrix
        >>> matrix = ResultMatrix(
        ...     model_names=['RF', 'SVM'],
        ...     dataset_names=['iris'],
        ...     metric_name='accuracy',
        ... )
        >>> matrix.add_result('iris', 'RF', values=[0.95, 0.93, 0.94, 0.96, 0.95])
        >>> matrix.add_result('iris', 'SVM', values=[0.90, 0.89, 0.91, 0.90, 0.88])
        >>> print(matrix.to_latex(precision=3).splitlines()[7])
        iris & $0.946 \\pm 0.011$ & $0.896 \\pm 0.011$$^\\blacktriangledown$ \\\\
        """
        from .latex import to_latex_table
        return to_latex_table(self, precision)

    def to_html(self, precision: int = 4) -> str:
        """
        Render the grid as a standalone, colour-coded HTML ``<table>``.

        Thin wrapper around :func:`~tuiml.evaluation.reporting.to_html_table`;
        see that function for the exact markup and the colour scheme.

        Parameters
        ----------
        precision : int, default=4
            Number of digits after the decimal point in each cell.

        Returns
        -------
        html : str
            A ``<table>`` fragment with inline styles, suitable for embedding in
            a notebook cell, an email or a report page.

        See Also
        --------
        :func:`~tuiml.evaluation.reporting.to_html_table` : The exporter this delegates to.

        Examples
        --------
        >>> from tuiml.evaluation.reporting import ResultMatrix
        >>> matrix = ResultMatrix(
        ...     model_names=['RF', 'SVM'],
        ...     dataset_names=['iris'],
        ...     metric_name='accuracy',
        ... )
        >>> matrix.add_result('iris', 'RF', values=[0.95, 0.93, 0.94, 0.96, 0.95])
        >>> matrix.add_result('iris', 'SVM', values=[0.90, 0.89, 0.91, 0.90, 0.88])
        >>> html = matrix.to_html(precision=3)
        >>> print(html.splitlines()[1])
        <caption>accuracy Comparison</caption>
        >>> print(html.splitlines()[11])
        <td style='background-color: #FFB6C1;'>0.896 ± 0.011 ▼</td>
        """
        from .html import to_html_table
        return to_html_table(self, precision)

    def to_markdown(self, precision: int = 4) -> str:
        """
        Render the grid as a GitHub-flavoured Markdown pipe table.

        Thin wrapper around
        :func:`~tuiml.evaluation.reporting.to_markdown_table`; see that function
        for the exact layout.

        Parameters
        ----------
        precision : int, default=4
            Number of digits after the decimal point in each cell.

        Returns
        -------
        markdown : str
            A pipe table with a header row, one row per dataset and a final
            ``**W/L/T**`` row.

        See Also
        --------
        :func:`~tuiml.evaluation.reporting.to_markdown_table` : The exporter this delegates to.

        Examples
        --------
        >>> from tuiml.evaluation.reporting import ResultMatrix
        >>> matrix = ResultMatrix(
        ...     model_names=['RF', 'SVM'],
        ...     dataset_names=['iris'],
        ...     metric_name='accuracy',
        ... )
        >>> matrix.add_result('iris', 'RF', values=[0.95, 0.93, 0.94, 0.96, 0.95])
        >>> matrix.add_result('iris', 'SVM', values=[0.90, 0.89, 0.91, 0.90, 0.88])
        >>> print(matrix.to_markdown(precision=3))
        | Dataset | RF | SVM |
        |---|---|---|
        | iris | 0.946 ± 0.011 | 0.896 ± 0.011 ▼ |
        |---|---|---|
        | **W/L/T** | 0/0/1 | 0/1/0 |
        """
        from .markdown import to_markdown_table
        return to_markdown_table(self, precision)

    def to_dict(self) -> Dict[str, Any]:
        """
        Export the grid as a plain dictionary for JSON serialization.

        Unlike the text exporters this keeps the mean and the standard deviation
        as separate numbers, which makes it the right choice for storing a
        benchmark run or feeding it to a plotting script. The shape is::

            {
              "metric_name":   "accuracy",
              "model_names":   ["RF", "SVM"],
              "dataset_names": ["iris"],
              "results": {
                "iris_RF":  {"mean": 0.946, "std": 0.0114, "significance": "TIE"},
                "iris_SVM": {"mean": 0.896, "std": 0.0114, "significance": "LOSS"}
              },
              "summary": {
                "RF":  {"wins": 0, "losses": 0, "ties": 1},
                "SVM": {"wins": 0, "losses": 1, "ties": 0}
              }
            }

        Keys in ``results`` are ``f"{dataset}_{model}"``; pairs never passed to
        :meth:`add_result` are simply absent. ``significance`` is the
        :class:`~tuiml.evaluation.statistics.SignificanceLevel` member *name*.

        Returns
        -------
        payload : dict
            Nested dictionary with the keys ``metric_name``, ``model_names``,
            ``dataset_names``, ``results`` and ``summary``. The ``mean`` and
            ``std`` entries are NumPy scalars, so cast them with ``float()``
            before handing the result to :func:`json.dumps`.

        See Also
        --------
        :meth:`to_csv` : Flat, human-readable export of the same numbers.

        Examples
        --------
        >>> from tuiml.evaluation.reporting import ResultMatrix
        >>> matrix = ResultMatrix(
        ...     model_names=['RF', 'SVM'],
        ...     dataset_names=['iris'],
        ...     metric_name='accuracy',
        ... )
        >>> matrix.add_result('iris', 'RF', values=[0.95, 0.93, 0.94, 0.96, 0.95])
        >>> matrix.add_result('iris', 'SVM', values=[0.90, 0.89, 0.91, 0.90, 0.88])
        >>> payload = matrix.to_dict()
        >>> sorted(payload)
        ['dataset_names', 'metric_name', 'model_names', 'results', 'summary']
        >>> round(float(payload['results']['iris_SVM']['mean']), 4)
        0.896
        >>> payload['results']['iris_SVM']['significance']
        'LOSS'
        >>> payload['summary']['SVM']
        {'wins': 0, 'losses': 1, 'ties': 0}
        """
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
    Build a :class:`~tuiml.evaluation.reporting.ResultMatrix` from nested scores and render it.

    A one-call shortcut for the common case where a benchmark loop has already
    collected its per-fold scores into a ``{dataset: {model: values}}``
    dictionary. The dataset rows follow the insertion order of ``results``; the
    model columns are collected from the union of the inner dictionaries. The
    first model column acts as the baseline, exactly as with
    ``baseline_index=0`` on the matrix itself.

    Parameters
    ----------
    results : dict of str to dict of str to array-like
        Nested mapping ``{dataset_name: {model_name: per_fold_values}}``. The
        innermost value is the list or array of raw fold scores, not an already
        averaged number. Inner dictionaries need not all hold the same models;
        absent pairs render as missing cells.
    metric_name : str, default="metric"
        Name of the metric, used for captions.
    format_type : str, default="plain"
        Which renderer to use: ``'plain'`` (fixed-width text), ``'csv'``,
        ``'latex'``, ``'html'`` or ``'markdown'``. Any unrecognised value falls
        back to ``'plain'``.
    **kwargs
        Forwarded verbatim to the
        :class:`~tuiml.evaluation.reporting.ResultMatrix` constructor, for
        example ``higher_better=False``, ``show_std=False`` or
        ``significance_level=0.01``.

    Returns
    -------
    formatted : str
        The rendered table in the requested format.

    Notes
    -----
    Column order is derived from a Python ``set`` of model names, so with two or
    more models it is not stable between runs. Construct a
    :class:`~tuiml.evaluation.reporting.ResultMatrix` directly, passing an
    explicit ``model_names`` list, whenever the column order matters, and note
    that the same mechanism decides which model ends up as the baseline.

    See Also
    --------
    :class:`~tuiml.evaluation.reporting.ResultMatrix` : The underlying grid, with full control over ordering and baseline.

    Examples
    --------
    >>> from tuiml.evaluation.reporting import format_results
    >>> scores = {
    ...     'iris': {'RF': [0.95, 0.93, 0.94]},
    ...     'wine': {'RF': [0.98, 0.97, 0.99]},
    ... }
    >>> print(format_results(scores, metric_name='accuracy', format_type='markdown'))
    | Dataset | RF |
    |---|---|
    | iris | 0.9400 ± 0.0100 |
    | wine | 0.9800 ± 0.0100 |
    |---|---|
    | **W/L/T** | 0/0/2 |
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
