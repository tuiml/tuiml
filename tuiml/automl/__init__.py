"""TuiML's own AutoML layer: pick the model, tune it, hand back a spec.

Everything here is built from parts the library already owns -- the registry,
each algorithm's published parameter schema, TuiML's splitters and metrics --
so an AutoML run introduces no new dependency and no second modelling stack.

The pieces are usable separately:

:mod:`~tuiml.automl.search_space`
    Turns any algorithm's ``get_parameter_schema()`` into a
    :class:`~tuiml.base.tuning.ParameterDistribution`. Useful on its own with
    :class:`~tuiml.evaluation.tuning.RandomSearchCV`.
:mod:`~tuiml.automl.portfolio`
    Queries the registry for the algorithms that fit a task and ranks them
    cheap-and-strong first.
:mod:`~tuiml.automl.automl`
    :class:`AutoMLClassifier` and :class:`AutoMLRegressor`: the search itself.
:mod:`~tuiml.automl.ensembling`
    Greedy ensemble selection (Caruana et al., 2004) over the trial pool.

The distinguishing output is ``best_spec_``: the winning configuration written
as a :func:`tuiml.train` spec, so the result of a search is portable data
rather than a pickled object.

Examples
--------
>>> from tuiml.automl import AutoMLClassifier
>>> from tuiml.datasets import load_iris
>>> data = load_iris()
>>> automl = AutoMLClassifier(time_budget=5, cv=3, random_state=0)
>>> _ = automl.fit(data.X, data.y)
>>> sorted(automl.best_spec_)
['evaluation', 'model', 'pipeline']
"""

from tuiml.automl.automl import AutoMLClassifier, AutoMLRegressor
from tuiml.automl.ensembling import GreedyEnsemble, greedy_selection
from tuiml.automl.portfolio import Candidate, build_portfolio, describe_portfolio
from tuiml.automl.search_space import schema_to_distribution, search_space_for

__all__ = [
    "AutoMLClassifier",
    "AutoMLRegressor",
    "GreedyEnsemble",
    "greedy_selection",
    "Candidate",
    "build_portfolio",
    "describe_portfolio",
    "schema_to_distribution",
    "search_space_for",
]
