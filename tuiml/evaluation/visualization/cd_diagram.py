"""
Critical Difference diagram visualization.

References
----------
Demsar, J. (2006). Statistical comparisons of classifiers over multiple data sets.
Journal of Machine Learning Research, 7, 1-30.
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
    """Result from Critical Difference diagram computation."""
    avg_ranks: Dict[str, float]
    critical_difference: float
    groups: List[List[str]]
    p_value: float
    test_statistic: float

    @classmethod
    def get_parameter_schema(cls) -> Dict:
        """
        Get JSON Schema for the dataclass fields.

        Returns
        -------
        dict
            JSON Schema describing the dataclass fields.
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
    Compute ranks for each algorithm across datasets.

    Parameters
    ----------
    scores : ndarray of shape (n_datasets, n_algorithms)
        Performance scores matrix.
    lower_better : bool, default=False
        If True, lower scores are better (e.g., error rates).

    Returns
    -------
    ranks : ndarray of shape (n_datasets, n_algorithms)
        Rank matrix (1 = best).
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
    Compute critical difference for Nemenyi or Bonferroni-Dunn test.

    Parameters
    ----------
    n_datasets : int
        Number of datasets.
    n_algorithms : int
        Number of algorithms.
    alpha : float
        Significance level.
    test : str
        'nemenyi' or 'bonferroni-dunn'.

    Returns
    -------
    cd : float
        Critical difference value.
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
    """
    Find maximal cliques in the adjacency matrix using a greedy approach.

    Algorithms that are not significantly different are connected.
    Returns groups (cliques) of algorithms that form connected components.
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
    Plot Critical Difference diagram for comparing multiple classifiers.

    Uses the aeon/Demšar (2006) style with algorithms listed on left and right
    sides, connected by horizontal lines to their rank positions on the axis.
    Thick bars connect algorithms that are NOT significantly different.

    Parameters
    ----------
    scores : ndarray of shape (n_datasets, n_algorithms) or dict
        Performance scores matrix, or dict of {algorithm: scores_array}.
    names : list of str, optional
        Algorithm names (required if scores is ndarray).
    lower_better : bool, default=False
        If True, lower scores are better (e.g., error rates).
    alpha : float, default=0.05
        Significance level.
    test : {'nemenyi', 'wilcoxon'}, default='nemenyi'
        Statistical test to use.
    correction : {'holm', 'bonferroni', 'none'}, default='holm'
        Multiple comparison correction method.
    title : str, optional
        Plot title.
    figsize : tuple, optional
        Figure size. Auto-calculated based on number of algorithms if not provided.
    save_path : str, optional
        Path to save the figure.

    Returns
    -------
    result : CDDiagramResult or None
        Computation results.

    Examples
    --------
    >>> from tuiml.evaluation.visualization import plot_critical_difference
    >>> import numpy as np
    >>> scores = np.array([
    ...     [0.85, 0.82, 0.78],
    ...     [0.87, 0.84, 0.80],
    ...     [0.83, 0.81, 0.79],
    ... ])
    >>> names = ['Algorithm A', 'Algorithm B', 'Algorithm C']
    >>> plot_critical_difference(scores, names, lower_better=False)
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
    # Calculate figure size based on number of algorithms
    if figsize is None:
        width = max(8, n_algorithms * 1.0)
        height = max(3, n_algorithms * 0.4)
        figsize = (width, height)

    fig, ax = setup_figure(figsize=figsize)
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
    n_left = len(left_names)
    n_right = len(right_names)
    max_lines = max(n_left, n_right)

    # Calculate total height
    cd_space = 0.5  # Space for CD indicator at top
    axis_space = 0.3  # Space for axis labels
    algo_space = max_lines * line_height + 0.2
    clique_space = max(0.3, n_cliques * 0.12 + 0.1)

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

    for i, (name, rank) in enumerate(zip(left_names, left_ranks)):
        y_pos = axis_y - 0.2 - (i * line_height)

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
        y_pos = axis_y - 0.2 - (i * line_height)

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
    # Position clique bars below all algorithm labels
    lowest_label_y = axis_y - 0.2 - (max_lines - 1) * line_height
    bar_start_y = lowest_label_y - 0.25
    bar_gap = 0.12

    for i, clique in enumerate(clique_indices):
        if len(clique) < 2:
            continue

        clique_ranks = [sorted_ranks[j] for j in clique]
        left_rank = min(clique_ranks)
        right_rank = max(clique_ranks)

        y_bar = bar_start_y - (i * bar_gap)

        # Draw thick horizontal bar
        ax.plot([left_rank, right_rank], [y_bar, y_bar], color=axis_color,
                linewidth=10, solid_capstyle='butt')

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
