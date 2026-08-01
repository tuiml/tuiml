"""Statistical significance tests."""

from typing import Any, Dict

from .._spec import ToolSpec


def execute_statistical_test(**kwargs) -> Dict[str, Any]:
    """Run statistical significance tests on experiment results.

    Backs the ``tuiml_test_statistics`` tool. Supported tests: friedman,
    nemenyi, wilcoxon, paired_t, anova, friedman_aligned, quade. The
    pairwise tests (wilcoxon, paired_t) compare the first two algorithms
    in ``results``.

    Parameters
    ----------
    test : str
        Name of the test to run (arrives via ``**kwargs``, like all
        parameters below).
    results : dict
        ``{algorithm_name: [scores...]}`` mapping of per-fold scores.
    significance_level : float, default=0.05
        Alpha level for significance.
    higher_better : bool, default=True
        Whether higher scores are better (pairwise tests only).

    Returns
    -------
    result : dict
        On success: ``status`` (``'success'``), ``test`` and
        ``significant``; most tests add ``statistic`` and ``p_value``;
        pairwise tests add ``algorithms`` and a ``details`` dict of
        means/stds; nemenyi returns per-pair significance in
        ``details``. On failure: ``status`` (``'error'``), ``error`` and
        optionally ``suggestion`` / ``error_type``.
    """
    import numpy as np

    try:
        test_name = kwargs['test']
        raw_results = kwargs['results']
        alpha = kwargs.get('significance_level', 0.05)
        higher_better = kwargs.get('higher_better', True)

        # Convert results to numpy arrays
        results = {name: np.array(scores, dtype=float) for name, scores in raw_results.items()}

        from tuiml.evaluation.statistics import (
            friedman_test, nemenyi_post_hoc, wilcoxon_signed_rank_test,
            paired_t_test, one_way_anova, friedman_aligned_ranks_test, quade_test,
        )

        if test_name == 'friedman':
            statistic, p_value, significant = friedman_test(results, significance_level=alpha)
            return {
                'status': 'success',
                'test': 'friedman',
                'statistic': float(statistic),
                'p_value': float(p_value),
                'significant': bool(significant),
            }

        elif test_name == 'nemenyi':
            pairwise = nemenyi_post_hoc(results, significance_level=alpha)
            details = {f"{k[0]} vs {k[1]}": bool(v) for k, v in pairwise.items()}
            return {
                'status': 'success',
                'test': 'nemenyi',
                'significant': any(pairwise.values()),
                'details': details,
            }

        elif test_name in ('wilcoxon', 'paired_t'):
            # Pairwise tests: use first two algorithms
            names = list(results.keys())
            if len(names) < 2:
                return {
                    'status': 'error',
                    'error': 'Pairwise tests require at least 2 algorithms in results.',
                }
            x, y = results[names[0]], results[names[1]]

            if test_name == 'wilcoxon':
                stats = wilcoxon_signed_rank_test(x, y, significance_level=alpha, higher_better=higher_better)
            else:
                stats = paired_t_test(x, y, significance_level=alpha, higher_better=higher_better)

            return {
                'status': 'success',
                'test': test_name,
                'algorithms': [names[0], names[1]],
                'statistic': float(stats.t_statistic),
                'p_value': float(stats.p_value),
                'significant': stats.is_significant(),
                'details': {
                    f'{names[0]}_mean': float(stats.x_mean),
                    f'{names[1]}_mean': float(stats.y_mean),
                    f'{names[0]}_std': float(stats.x_std),
                    f'{names[1]}_std': float(stats.y_std),
                    'diff_mean': float(stats.diff_mean),
                    'significance': stats.significance.name,
                }
            }

        elif test_name == 'anova':
            groups = list(results.values())
            f_stat, p_value, significant = one_way_anova(*groups, significance_level=alpha)
            return {
                'status': 'success',
                'test': 'anova',
                'statistic': float(f_stat),
                'p_value': float(p_value),
                'significant': bool(significant),
            }

        elif test_name == 'friedman_aligned':
            statistic, p_value, significant = friedman_aligned_ranks_test(results, significance_level=alpha)
            return {
                'status': 'success',
                'test': 'friedman_aligned',
                'statistic': float(statistic),
                'p_value': float(p_value),
                'significant': bool(significant),
            }

        elif test_name == 'quade':
            statistic, p_value, significant = quade_test(results, significance_level=alpha)
            return {
                'status': 'success',
                'test': 'quade',
                'statistic': float(statistic),
                'p_value': float(p_value),
                'significant': bool(significant),
            }

        else:
            return {
                'status': 'error',
                'error': f"Unknown test: '{test_name}'",
                'suggestion': "Available tests: friedman, nemenyi, wilcoxon, paired_t, anova, friedman_aligned, quade"
            }
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e),
            'error_type': type(e).__name__
        }


SPEC = ToolSpec(
    name='tuiml_test_statistics',
    description="Run statistical significance tests on experiment results (cross-validation scores). "
        "Supports Friedman test, Nemenyi post-hoc, Wilcoxon signed-rank, paired t-test, "
        "one-way ANOVA, Friedman aligned ranks, and Quade test.",
    input_schema={
            "type": "object",
            "properties": {
                "test": {
                    "type": "string",
                    "enum": [
                        "friedman", "nemenyi", "wilcoxon",
                        "paired_t", "anova", "friedman_aligned", "quade"
                    ],
                    "description": (
                        "Statistical test to run:\n"
                        "- friedman: Non-parametric test for 3+ algorithms\n"
                        "- nemenyi: Post-hoc pairwise test after Friedman\n"
                        "- wilcoxon: Non-parametric pairwise test (2 algorithms)\n"
                        "- paired_t: Parametric pairwise test (2 algorithms)\n"
                        "- anova: Parametric test for 3+ groups\n"
                        "- friedman_aligned: More powerful variant of Friedman\n"
                        "- quade: Non-parametric test accounting for dataset difficulty"
                    )
                },
                "results": {
                    "type": "object",
                    "description": "Algorithm CV scores: { 'AlgorithmName': [score1, score2, ...], ... }"
                },
                "significance_level": {
                    "type": "number",
                    "default": 0.05,
                    "description": "Significance level (alpha), default 0.05"
                },
                "higher_better": {
                    "type": "boolean",
                    "default": True,
                    "description": "Whether higher scores are better (default True)"
                }
            },
            "required": ["test", "results"]
        },
    output_schema={
            "type": "object",
            "properties": {
                "status": {"type": "string", "enum": ["success", "error"]},
                "test": {"type": "string"},
                "statistic": {"type": "number"},
                "p_value": {"type": "number"},
                "significant": {"type": "boolean"},
                "details": {"type": "object"},
                "error": {"type": "string"}
            },
            "required": ["status"]
        },
    execute=execute_statistical_test,
    group='workflow',
    read_only=True, destructive=False,
    idempotent=True, open_world=False,
)
