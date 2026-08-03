"""
Critical Difference (CD) diagrams: the standard way to report a benchmark of
many algorithms over many datasets.

Averaging accuracy across datasets is misleading — the scale of a metric differs
per dataset, and one easy dataset can dominate the mean. Demšar (2006) instead
recommends **ranking** the algorithms *within* each dataset, averaging those
ranks, running a Friedman test for "are any of them different at all?", and
following up with a Nemenyi post-hoc test that yields a single *critical
difference* (CD): the smallest gap between two average ranks that counts as
significant.

This module provides the three pieces of that recipe plus the plot:

- :func:`compute_ranks` — per-dataset ranks with ties averaged.
- :func:`critical_difference` — the CD value for ``k`` algorithms over ``n``
  datasets at a significance level.
- :func:`plot_critical_difference` — the diagram itself, returning a
  :class:`CDDiagramResult` with ranks, CD, cliques, and the Friedman statistic.

**Reading the diagram.** Algorithms sit on a horizontal *rank* axis running from
1 (best) on the left to ``k`` (worst) on the right; each one is connected by a
line to its average rank. A ``CD``-wide reference bar is drawn at the top for
scale. Thick horizontal bars underneath join algorithms whose average ranks
differ by **less than** the CD — those are the groups that are *not*
statistically distinguishable. So the claim "A beats B" is only supported when A
is to the left of B **and** no thick bar joins them.

Notes
-----
The tabulated Studentised-range critical values used here cover 2-20
algorithms; beyond that the value is extrapolated crudely (see
:func:`critical_difference`).

See Also
--------
:func:`~tuiml.evaluation.statistics.friedman_test` : The omnibus test the
    diagram summarises; check it is significant before interpreting the ranks.
:func:`~tuiml.evaluation.statistics.nemenyi_post_hoc` : Pairwise p-values behind
    the "not significantly different" bars.
:func:`~tuiml.evaluation.visualization.plot_ranking_table` : The same score
    matrix as a table of raw values and ranks.

References
----------
.. [Demsar2006] Demšar, J. (2006). "Statistical comparisons of classifiers over
   multiple data sets." *Journal of Machine Learning Research*, 7, 1-30.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Literal
from dataclasses import dataclass

# Try to import plotting libraries
try:
    import matplotlib.pyplot as plt
    import matplotlib.lines as mlines
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from ._style import apply_style, setup_figure, SEMANTIC_COLORS

@dataclass
class CDDiagramResult:
    """Numbers behind a Critical Difference diagram.

    Returned by :func:`plot_critical_difference` so the statistics shown in the
    figure can be reported in text or asserted on in tests.

    Attributes
    ----------
    avg_ranks : dict of {str: float}
        Average rank of each algorithm across datasets, keyed by the name given
        to :func:`plot_critical_difference`. Lower is better; 1.0 means the
        algorithm won on every dataset. Keys follow the *input* order, not the
        sorted order drawn in the figure.
    critical_difference : float
        The Nemenyi critical difference. Two algorithms are declared
        significantly different only if their average ranks differ by at least
        this much.
    groups : list of list of str
        Cliques of algorithm names that are **not** significantly different from
        one another — each list corresponds to one thick bar in the diagram.
        Groups may overlap, and an algorithm distinguishable from all others
        appears in none of them.
    p_value : float
        P-value of the Friedman omnibus test. If this is above your
        :math:`\\alpha`, the ranking as a whole is not significant and the
        pairwise reading of the diagram should not be trusted.
    test_statistic : float
        The Friedman :math:`\\chi^2` statistic, with ``n_algorithms - 1``
        degrees of freedom.

    See Also
    --------
    :func:`~tuiml.evaluation.visualization.plot_critical_difference` : Produces
        this result.
    :func:`~tuiml.evaluation.statistics.friedman_test` : Standalone Friedman
        test with the same statistic.

    Examples
    --------
    >>> from tuiml.evaluation.visualization import CDDiagramResult
    >>> res = CDDiagramResult(
    ...     avg_ranks={'A': 1.0, 'B': 2.0, 'C': 3.0},
    ...     critical_difference=1.5,
    ...     groups=[['A', 'B'], ['B', 'C']],
    ...     p_value=0.03,
    ...     test_statistic=6.0,
    ... )
    >>> min(res.avg_ranks, key=res.avg_ranks.get)
    'A'
    """
    avg_ranks: Dict[str, float]
    critical_difference: float
    groups: List[List[str]]
    p_value: float
    test_statistic: float

    @classmethod
    def get_parameter_schema(cls) -> Dict:
        """Return JSON Schema for the dataclass fields.

        Returns
        -------
        schema : dict
            JSON Schema describing the fields of :class:`CDDiagramResult`.
        """
        return {
            "type": "object",
            "properties": {
                "avg_ranks": {
                    "type": "object",
                    "additionalProperties": {"type": "number"},
                    "description": "Dictionary mapping algorithm names to their average ranks."
                },
                "critical_difference": {
                    "type": "number",
                    "description": "The critical difference value for statistical significance."
                },
                "groups": {
                    "type": "array",
                    "items": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "description": "List of groups of algorithms that are not significantly different."
                },
                "p_value": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                    "description": "P-value from the Friedman test."
                },
                "test_statistic": {
                    "type": "number",
                    "description": "Chi-squared test statistic from the Friedman test."
                }
            },
            "required": ["avg_ranks", "critical_difference", "groups", "p_value", "test_statistic"],
            "additionalProperties": False
        }

def compute_ranks(
    scores: np.ndarray,
    lower_better: bool = False
) -> np.ndarray:
    """
    Rank algorithms **within each dataset**, averaging tied ranks.

    This is the first step of the Demšar (2006) comparison protocol: ranking
    per dataset removes the effect of datasets having wildly different score
    scales, which is what makes a plain average across datasets untrustworthy.

    Each row of ``scores`` is ranked independently. The best algorithm on a row
    gets rank 1, the worst gets rank ``n_algorithms``. Ties share the mean of
    the ranks they span, so two algorithms tying for first both get 1.5 and the
    next one gets 3 — rank sums stay comparable across rows.

    Parameters
    ----------
    scores : ndarray of shape (n_datasets, n_algorithms)
        Performance scores; one row per dataset, one column per algorithm.
    lower_better : bool, default=False
        Direction of the metric. Leave False for accuracy, F1, AUC, …; set True
        for error rates, RMSE, log-loss and other "smaller is better" metrics.

    Returns
    -------
    ranks : ndarray of shape (n_datasets, n_algorithms)
        Rank matrix aligned with ``scores``; 1 is best. Fractional values
        indicate ties.

    Notes
    -----
    Column ``j`` of the result always refers to the same algorithm as column
    ``j`` of the input — nothing is sorted. The dtype follows ``scores`` (via
    ``np.zeros_like``), so pass a float array: an integer score matrix cannot
    represent the fractional ranks produced by ties.

    See Also
    --------
    :func:`~tuiml.evaluation.visualization.plot_critical_difference` : Consumes
        these ranks and tests which differences are significant.

    Examples
    --------
    >>> import numpy as np
    >>> from tuiml.evaluation.visualization import compute_ranks
    >>> scores = np.array([[0.90, 0.85, 0.80],
    ...                    [0.70, 0.70, 0.65]])
    >>> compute_ranks(scores)
    array([[1. , 2. , 3. ],
           [1.5, 1.5, 3. ]])

    With an error metric, flip the direction:

    >>> compute_ranks(np.array([[0.10, 0.15, 0.20]]), lower_better=True)
    array([[1., 2., 3.]])
    """
    n_datasets, n_algorithms = scores.shape
    ranks = np.zeros_like(scores)

    for i in range(n_datasets):
        if lower_better:
            order = np.argsort(scores[i])
        else:
            order = np.argsort(-scores[i])

        rank_values = np.zeros(n_algorithms)
        j = 0
        while j < n_algorithms:
            tied_start = j
            while j < n_algorithms - 1 and scores[i, order[j]] == scores[i, order[j + 1]]:
                j += 1
            avg_rank = (tied_start + j + 2) / 2
            for k in range(tied_start, j + 1):
                rank_values[order[k]] = avg_rank
            j += 1

        ranks[i] = rank_values

    return ranks

def critical_difference(
    n_datasets: int,
    n_algorithms: int,
    alpha: float = 0.05,
    test: str = 'nemenyi'
) -> float:
    """
    Smallest gap between two average ranks that counts as significant.

    Implements the Nemenyi critical difference of Demšar (2006). For :math:`k`
    algorithms compared over :math:`N` datasets,

    .. math::
        CD = q_{\\alpha} \\sqrt{\\frac{k(k+1)}{6N}}

    where :math:`q_{\\alpha}` is the Studentised range statistic at level
    :math:`\\alpha` divided by :math:`\\sqrt{2}`, tabulated below. Two
    algorithms whose average ranks differ by at least ``CD`` are declared
    significantly different; anything closer is joined by a bar in the diagram.

    Two consequences worth internalising: the CD *shrinks* as you add datasets
    (:math:`\\propto 1/\\sqrt{N}`) and *grows* as you add algorithms — throwing
    extra baselines into a benchmark makes every comparison harder to call.

    Parameters
    ----------
    n_datasets : int
        Number of datasets :math:`N` (rows of the score matrix).
    n_algorithms : int
        Number of algorithms :math:`k` (columns of the score matrix).
    alpha : float, default=0.05
        Significance level. Only two tables exist: ``alpha <= 0.05`` selects the
        0.05 critical values, anything larger selects the 0.10 values. Values
        such as 0.01 therefore behave exactly like 0.05.
    test : str, default='nemenyi'
        Accepted for interface compatibility. The Nemenyi critical values are
        used regardless of what is passed; this argument does not currently
        change the result.

    Returns
    -------
    cd : float
        The critical difference, in units of average rank.

    Notes
    -----
    Tabulated :math:`q_{\\alpha}` values cover ``n_algorithms`` from 2 to 20.
    Above 20 the value is approximated by a linear extrapolation, and an
    unlisted count falls back to 3.5 — treat results for very large ``k`` as
    indicative only.

    See Also
    --------
    :func:`~tuiml.evaluation.visualization.plot_critical_difference` : Draws the
        diagram this value scales.
    :func:`~tuiml.evaluation.statistics.nemenyi_post_hoc` : Exact pairwise
        p-values instead of a single threshold.

    References
    ----------
    .. [Demsar2006] Demšar, J. (2006). "Statistical comparisons of classifiers
       over multiple data sets." *Journal of Machine Learning Research*, 7, 1-30.

    Examples
    --------
    >>> from tuiml.evaluation.visualization import critical_difference
    >>> round(float(critical_difference(n_datasets=20, n_algorithms=5)), 3)
    1.364

    More datasets tighten the threshold:

    >>> round(float(critical_difference(n_datasets=80, n_algorithms=5)), 3)
    0.682
    """
    q_alpha_005 = {
        2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850,
        7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164, 11: 3.219,
        12: 3.268, 13: 3.313, 14: 3.354, 15: 3.391, 16: 3.426,
        17: 3.458, 18: 3.489, 19: 3.517, 20: 3.544
    }

    q_alpha_010 = {
        2: 1.645, 3: 2.052, 4: 2.291, 5: 2.459, 6: 2.589,
        7: 2.693, 8: 2.780, 9: 2.855, 10: 2.920, 11: 2.978,
        12: 3.030, 13: 3.077, 14: 3.120, 15: 3.159, 16: 3.196,
        17: 3.230, 18: 3.261, 19: 3.291, 20: 3.319
    }

    if alpha <= 0.05:
        q_table = q_alpha_005
    else:
        q_table = q_alpha_010

    k = n_algorithms
    if k > 20:
        q = 2.576 + 0.1 * (k - 20)
    else:
        q = q_table.get(k, 3.5)

    cd = q * np.sqrt(k * (k + 1) / (6 * n_datasets))

    return cd

def _find_cliques(adj_matrix: np.ndarray) -> List[List[int]]:
    """Group mutually-indistinguishable algorithms into cliques.

    An edge means "these two are not significantly different". Each returned
    clique becomes one thick bar in the diagram, so a bar may only span
    algorithms that are *all* pairwise indistinguishable.

    Parameters
    ----------
    adj_matrix : ndarray of shape (n_algorithms, n_algorithms) of bool
        Symmetric adjacency matrix indexed by rank order, where
        ``adj_matrix[i, j]`` is True when the average ranks of ``i`` and ``j``
        differ by less than the critical difference.

    Returns
    -------
    cliques : list of list of int
        Cliques of size two or more, each a sorted list of indices into the
        rank-sorted algorithm order. Cliques that are subsets of another clique
        are dropped; singletons are never returned.

    Notes
    -----
    Uses a greedy sweep (seed at each vertex, absorb every later vertex adjacent
    to the whole current clique) rather than full Bron-Kerbosch enumeration, so
    for general graphs it can miss a maximal clique. On the interval graph
    produced by a one-dimensional rank axis the greedy sweep is adequate.
    """
    n = len(adj_matrix)
    cliques = []

    # Find all maximal cliques using Bron-Kerbosch-like approach
    for i in range(n):
        clique = [i]
        for j in range(i + 1, n):
            # Check if j is connected to all members of current clique
            if all(adj_matrix[j, k] for k in clique):
                clique.append(j)
        if len(clique) > 1:
            # Check if this clique is maximal (not subset of existing)
            clique_set = set(clique)
            is_subset = False
            for existing in cliques:
                if clique_set.issubset(set(existing)):
                    is_subset = True
                    break
            if not is_subset:
                # Remove any existing cliques that are subsets of this one
                cliques = [c for c in cliques if not set(c).issubset(clique_set)]
                cliques.append(clique)

    return cliques


def plot_critical_difference(
    scores: Union[np.ndarray, Dict[str, np.ndarray]],
    names: List[str] = None,
    lower_better: bool = False,
    alpha: float = 0.05,
    test: Literal['nemenyi', 'wilcoxon'] = 'nemenyi',
    correction: Literal['holm', 'bonferroni', 'none'] = 'holm',
    title: str = None,
    figsize: Tuple[int, int] = None,
    save_path: str = None,
) -> Optional[CDDiagramResult]:
    """
    Draw a Critical Difference diagram comparing algorithms over many datasets.

    The plot answers one question: *which of these algorithms can I actually
    claim are different?* Each algorithm is ranked within every dataset (see
    :func:`compute_ranks`), the ranks are averaged, and a Nemenyi critical
    difference is computed from the number of algorithms and datasets.

    **How to read it.** The horizontal axis is average rank, best (1) on the
    left, worst (``k``) on the right. Every algorithm hangs off the axis by a
    connector line ending at its average rank, with names printed on the left
    for the better half and on the right for the worse half. The short bar
    labelled ``CD`` at the top is a ruler showing how wide the critical
    difference is in rank units. The **thick horizontal bars underneath join
    algorithms that are NOT significantly different** — if a bar spans two
    algorithms, the data does not support preferring one over the other, no
    matter how their average ranks are ordered::

                            |------ CD ------|

        1        2        3        4        5        6
        |--------|--------|--------|--------|--------|
        |        |        |             |        |
      SVM      Forest   Boosting      kNN    NaiveBayes
                 |________|             |________|
                  (tied)                 (tied)

    Here SVM is significantly better than everything to the right of Forest,
    Forest and Boosting are indistinguishable, and so are kNN and NaiveBayes.

    Parameters
    ----------
    scores : ndarray of shape (n_datasets, n_algorithms) or dict of {str: ndarray}
        Performance scores; one row per dataset, one column per algorithm. If a
        dict is given, keys become the algorithm names and each value is that
        algorithm's score across the datasets (all values must be the same
        length), overriding ``names``.
    names : list of str, optional
        Algorithm names in column order. Required when ``scores`` is an array.
    lower_better : bool, default=False
        Set True when smaller scores are better (error rate, RMSE, log-loss).
    alpha : float, default=0.05
        Significance level driving the critical difference. Note the
        implementation only distinguishes ``<= 0.05`` from larger values — see
        :func:`critical_difference`.
    test : {'nemenyi', 'wilcoxon'}, default='nemenyi'
        Accepted for interface compatibility; the critical difference is always
        computed from the Nemenyi statistic. Passing ``'wilcoxon'`` does not
        currently change the figure.
    correction : {'holm', 'bonferroni', 'none'}, default='holm'
        Accepted for interface compatibility; multiple-comparison correction is
        implicit in the Nemenyi critical value and this argument is not
        currently applied.
    title : str, optional
        Figure title. Title-cased before rendering. No title is drawn if None.
    figsize : tuple of (float, float), optional
        Figure size in inches. When None, the width scales with both the number
        of algorithms and the longest labels on each side; the height scales
        with the number of algorithms. This leaves enough room for long model
        names without compressing the rank axis.
    save_path : str, optional
        If given, the figure is also written to this path as a 300-dpi PNG with
        a white background and tight bounding box, before being shown.

    Returns
    -------
    result : CDDiagramResult
        Average ranks, the critical difference, the cliques drawn as bars, and
        the Friedman test statistic and p-value. Check ``result.p_value`` first:
        if the Friedman test is not significant, the ordering shown carries no
        statistical weight.

    Raises
    ------
    ImportError
        If matplotlib is not installed (it is imported lazily).
    ValueError
        If ``names`` is None while ``scores`` is an array, or if ``len(names)``
        does not match the number of columns in ``scores``.

    Notes
    -----
    Side effects: this function mutates the global matplotlib style (see
    ``apply_style``), calls ``matplotlib.pyplot.show()`` before returning — so
    it blocks in a GUI backend and renders inline in a notebook — and writes a
    file when ``save_path`` is given. Use a non-interactive backend such as
    ``Agg`` to render headlessly. The figure object itself is not returned; grab
    it with ``matplotlib.pyplot.gcf()`` before ``show()`` clears the display if
    you need to post-process it.

    Bars are computed from the pairwise rank gaps directly, so with many
    algorithms the bars can overlap; each one is drawn on its own row between
    the rank axis and the labels. The method connector lines pass through the
    bars, making group membership explicit.

    See Also
    --------
    :func:`~tuiml.evaluation.visualization.compute_ranks` : The per-dataset
        ranking step.
    :func:`~tuiml.evaluation.visualization.critical_difference` : The CD value
        that sets the width of the bars.
    :func:`~tuiml.evaluation.statistics.friedman_test` : The omnibus test whose
        statistic and p-value are reported in the result.
    :func:`~tuiml.evaluation.statistics.nemenyi_post_hoc` : Pairwise p-values
        for a numeric write-up of the same comparison.
    :func:`~tuiml.evaluation.visualization.plot_boxplot_comparison` : Score
        spread per algorithm, a useful companion figure.

    References
    ----------
    .. [Demsar2006] Demšar, J. (2006). "Statistical comparisons of classifiers
       over multiple data sets." *Journal of Machine Learning Research*, 7, 1-30.

    Examples
    --------
    >>> import numpy as np
    >>> from tuiml.evaluation.visualization import plot_critical_difference
    >>> scores = np.array([
    ...     [0.85, 0.82, 0.78],
    ...     [0.87, 0.84, 0.80],
    ...     [0.83, 0.81, 0.79],
    ... ])
    >>> names = ['Algorithm A', 'Algorithm B', 'Algorithm C']
    >>> result = plot_critical_difference(scores, names)   # doctest: +SKIP
    >>> result.avg_ranks                                   # doctest: +SKIP
    {'Algorithm A': 1.0, 'Algorithm B': 2.0, 'Algorithm C': 3.0}

    Passing a dict names the algorithms for you, and saves the figure:

    >>> results = {
    ...     'SVM': np.array([0.91, 0.88, 0.93]),
    ...     'Forest': np.array([0.89, 0.90, 0.88]),
    ...     'kNN': np.array([0.80, 0.79, 0.84]),
    ... }
    >>> res = plot_critical_difference(
    ...     results, title='Accuracy over 3 datasets',
    ...     save_path='cd.png')                            # doctest: +SKIP

    For an error metric, flip the direction so rank 1 is the smallest value:

    >>> errors = np.array([[0.15, 0.18, 0.22], [0.13, 0.16, 0.21]])
    >>> res = plot_critical_difference(
    ...     errors, names=['A', 'B', 'C'], lower_better=True)   # doctest: +SKIP
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for plotting")

    # Apply modern styling
    apply_style()

    # Convert dict to array if needed
    if isinstance(scores, dict):
        names = list(scores.keys())
        scores = np.array([scores[name] for name in names]).T

    if names is None:
        raise ValueError("names must be provided when scores is an array")

    scores = np.asarray(scores)
    n_datasets, n_algorithms = scores.shape

    if len(names) != n_algorithms:
        raise ValueError(
            f"Number of names ({len(names)}) must match "
            f"number of algorithms ({n_algorithms})"
        )

    # Compute ranks
    ranks = compute_ranks(scores, lower_better=lower_better)
    avg_ranks = np.mean(ranks, axis=0)

    # Compute critical difference
    cd = critical_difference(n_datasets, n_algorithms, alpha, test)

    # Sort by average rank
    sorted_indices = np.argsort(avg_ranks)
    sorted_names = [names[i] for i in sorted_indices]
    sorted_ranks = avg_ranks[sorted_indices]

    # Build adjacency matrix for cliques (algorithms not significantly different)
    adj_matrix = np.zeros((n_algorithms, n_algorithms), dtype=bool)
    for i in range(n_algorithms):
        for j in range(i + 1, n_algorithms):
            if abs(sorted_ranks[j] - sorted_ranks[i]) < cd:
                adj_matrix[i, j] = True
                adj_matrix[j, i] = True

    # Find cliques (groups of algorithms not significantly different)
    clique_indices = _find_cliques(adj_matrix)
    groups = [[sorted_names[i] for i in clique] for clique in clique_indices]

    # Friedman test statistic
    chi2 = 12 * n_datasets / (n_algorithms * (n_algorithms + 1)) * \
           (np.sum(avg_ranks ** 2) - n_algorithms * (n_algorithms + 1) ** 2 / 4)

    # P-value approximation
    from ..statistics.nonparametric import _chi2_cdf
    p_value = 1 - _chi2_cdf(chi2, n_algorithms - 1)

    # === PLOTTING (aeon style) ===
    # Calculate figure size from both the rank axis and the model labels. Long
    # names need real horizontal space; expanding only the data-coordinate
    # limits compresses the rank axis and produces awkward connector lines.
    if figsize is None:
        half = (n_algorithms + 1) // 2
        left_label_width = max(len(name) for name in sorted_names[:half])
        right_names_for_width = sorted_names[half:]
        right_label_width = (
            max(len(name) for name in right_names_for_width)
            if right_names_for_width else 0
        )
        width = max(
            8,
            n_algorithms * 1.0
            + (left_label_width + right_label_width) * 0.12,
        )
        height = max(3, n_algorithms * 0.4)
        figsize = (width, height)

    fig, ax = setup_figure(figsize=figsize)
    # Inline notebook output is commonly displayed on high-density screens.
    # A higher raster DPI keeps labels and connector lines crisp there, while
    # explicit save_path output continues to use 300 DPI below.
    fig.set_dpi(160)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    # Layout parameters
    lowv = 1
    highv = n_algorithms

    # Split algorithms: half on left (better), half on right (worse)
    half = (n_algorithms + 1) // 2
    left_names = sorted_names[:half]
    left_ranks = sorted_ranks[:half]
    # Reverse right side so worst rank is at top (standard convention, minimizes crossings)
    right_names = sorted_names[half:][::-1]
    right_ranks = sorted_ranks[half:][::-1]

    # Calculate text space needed
    max_name_len = max(len(name) for name in sorted_names)
    textspace = max(1.2, max_name_len * 0.1)

    # Number of cliques for spacing
    n_cliques = len([c for c in clique_indices if len(c) >= 2])

    # Vertical layout parameters (in data coordinates)
    line_height = 0.3
    bar_gap = 0.12
    n_left = len(left_names)
    n_right = len(right_names)
    max_lines = max(n_left, n_right)

    # Calculate total height
    cd_space = 0.5  # Space for CD indicator at top
    axis_space = 0.3  # Space for axis labels
    algo_space = max_lines * line_height + 0.2
    clique_space = max(0.22, n_cliques * bar_gap + 0.12)

    total_height = cd_space + axis_space + algo_space + clique_space

    # Set plot limits
    ax.set_xlim(lowv - textspace, highv + textspace)
    ax.set_ylim(0, total_height)

    # Y-positions
    axis_y = total_height - cd_space - axis_space
    cd_y = total_height - 0.25

    # === Draw CD indicator at top ===
    cd_x_start = (lowv + highv) / 2 - cd / 2
    cd_x_end = cd_x_start + cd

    # CD bar
    axis_color = SEMANTIC_COLORS['text']
    connector_color = SEMANTIC_COLORS['muted_text']
    ax.hlines(cd_y, cd_x_start, cd_x_end, color=axis_color, linewidth=4)
    # CD endpoints
    ax.vlines(cd_x_start, cd_y - 0.06, cd_y + 0.06, color=axis_color, linewidth=2.5)
    ax.vlines(cd_x_end, cd_y - 0.06, cd_y + 0.06, color=axis_color, linewidth=2.5)
    # CD label
    ax.text((cd_x_start + cd_x_end) / 2, cd_y + 0.12, 'CD',
            ha='center', va='bottom', fontsize=13, fontweight='bold', color=axis_color)

    # === Draw horizontal axis ===
    ax.hlines(axis_y, lowv, highv, color=axis_color, linewidth=2.5)

    # Draw tick marks and labels on axis
    tick_size = 0.05
    for i in range(lowv, highv + 1):
        ax.vlines(i, axis_y - tick_size, axis_y + tick_size, color=axis_color, linewidth=2)
        ax.text(i, axis_y + tick_size + 0.08, str(i), ha='center', va='bottom',
                fontsize=13, color=connector_color, fontweight='bold')

    # Remove all spines and ticks
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

    # === Draw left side algorithms (better ranks) ===
    left_text_x = lowv - 0.15
    line_color = connector_color
    line_width = 1.5
    label_offset = clique_space + 0.1

    for i, (name, rank) in enumerate(zip(left_names, left_ranks)):
        y_pos = axis_y - label_offset - (i * line_height)

        # Algorithm name on the left
        ax.text(left_text_x, y_pos, name, ha='right', va='center',
                fontsize=13, fontweight='bold', color=axis_color)

        # Horizontal line from name to rank position
        ax.hlines(y_pos, left_text_x + 0.08, rank, color=line_color, linewidth=line_width)

        # Vertical line up to axis
        ax.vlines(rank, y_pos, axis_y, color=line_color, linewidth=line_width)

        # Small dot at rank position on axis
        ax.plot(rank, axis_y, 'o', color=axis_color, markersize=8, zorder=5)

    # === Draw right side algorithms (worse ranks) ===
    right_text_x = highv + 0.15

    for i, (name, rank) in enumerate(zip(right_names, right_ranks)):
        y_pos = axis_y - label_offset - (i * line_height)

        # Algorithm name on the right
        ax.text(right_text_x, y_pos, name, ha='left', va='center',
                fontsize=13, fontweight='bold', color=axis_color)

        # Horizontal line from rank position to name
        ax.hlines(y_pos, rank, right_text_x - 0.08, color=line_color, linewidth=line_width)

        # Vertical line up to axis
        ax.vlines(rank, y_pos, axis_y, color=line_color, linewidth=line_width)

        # Small dot at rank position on axis
        ax.plot(rank, axis_y, 'o', color=axis_color, markersize=8, zorder=5)

    # === Draw clique bars (algorithms not significantly different) ===
    # Put clique bars immediately below the rank axis. Because each method's
    # vertical connector continues down to its label, the bar visibly crosses
    # every connector belonging to that non-significant group.
    bar_start_y = axis_y - 0.1

    for i, clique in enumerate(clique_indices):
        if len(clique) < 2:
            continue

        clique_ranks = [sorted_ranks[j] for j in clique]
        left_rank = min(clique_ranks)
        right_rank = max(clique_ranks)

        y_bar = bar_start_y - (i * bar_gap)

        # Draw thick horizontal bar
        ax.plot([left_rank, right_rank], [y_bar, y_bar], color=axis_color,
                linewidth=6, solid_capstyle='round', zorder=6)

    # Title
    if title:
        ax.set_title(title.title(), fontsize=16, fontweight='bold', pad=15, color=axis_color)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format='png', bbox_inches='tight', dpi=300,
                    facecolor='white', edgecolor='none')

    plt.show()

    return CDDiagramResult(
        avg_ranks={name: rank for name, rank in zip(names, avg_ranks)},
        critical_difference=cd,
        groups=groups,
        p_value=p_value,
        test_statistic=chi2
    )
