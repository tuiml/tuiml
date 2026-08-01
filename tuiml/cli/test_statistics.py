"""Test Statistics Command - Significance tests over experiment results."""

import click
import json
from tuiml.agent.tools import execute_tool

@click.command('test-statistics')
@click.option('--test', type=str, required=True, help="Test to run. Across 3+ algorithms: 'friedman', 'friedman_aligned' (more powerful variant), 'quade' (accounts for dataset difficulty), 'anova' (parametric). Post-hoc after Friedman: 'nemenyi'. Between 2 algorithms: 'wilcoxon', 'paired_t' (parametric)")
@click.option('--results', type=str, required=True, help='Cross-validation scores as a JSON object mapping each algorithm name to its list of fold scores')
@click.option('--significance-level', type=float, help='Significance level alpha below which a difference counts as real (default: 0.05)')
@click.option('--higher-better/--no-higher-better', default=True, help='Whether a higher score means a better model, which sets the ranking direction (default: higher is better)')
@click.option('--json-output', is_flag=True, help='Output raw JSON')
def test_statistics(test, results, significance_level, higher_better, json_output):
    """Run statistical significance tests over experiment results.

    Compares algorithms using their cross-validation scores and reports the
    test statistic, the p-value and whether the difference is significant,
    so a benchmark result can be defended rather than eyeballed. The
    Friedman test, Friedman aligned ranks, the Quade test, one-way ANOVA,
    the Nemenyi post-hoc test, the Wilcoxon signed-rank test and the paired
    t-test are all available. Scores are given as a JSON object mapping
    each algorithm name to its list of fold scores, and --significance-level
    moves alpha away from its 0.05 default.

    Examples
    --------
    Compare three or more algorithms:

    $ tuiml test-statistics --test friedman --results "$SCORES"

    Find which pairs actually differ, once Friedman is significant:

    $ tuiml test-statistics --test nemenyi --results "$SCORES"

    Compare exactly two algorithms without assuming normality:

    $ tuiml test-statistics --test wilcoxon --results "$SCORES"
    """
    kwargs = {
            'test': test,
            'results': json.loads(results) if results else None,
            'significance_level': significance_level,
            'higher_better': higher_better,
    }
    # Remove None values
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    result = execute_tool('tuiml_test_statistics', **kwargs)
    if json_output:
        click.echo(json.dumps(result, indent=2, default=str))
    else:
        if result.get('status') == 'error':
            click.echo(f"Error: {result.get('error')}", err=True)
        else:
            # For now, print pretty JSON if not explicitly requested otherwise
            click.echo(json.dumps(result, indent=2, default=str))
