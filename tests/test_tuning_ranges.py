"""Numeric choices must remain distinct from continuous search ranges."""

import numpy as np
import pytest

from tuiml.base.tuning import ParameterDistribution
from tuiml.algorithms.trees import DecisionTreeClassifier
from tuiml.evaluation.tuning import BayesianSearchCV


@pytest.mark.parametrize("choices", [[2, 8], [0.01, 0.3], [True, False]])
def test_random_search_preserves_two_value_choices(choices):
    distribution = ParameterDistribution({"value": choices})
    samples = [distribution.sample(random_state=seed)["value"] for seed in range(20)]
    observed = set(samples)
    assert observed == set(choices)
    assert all(type(value) is type(choices[0]) for value in samples)


def test_python_tuple_ranges_still_sample_interior_values():
    distribution = ParameterDistribution({"depth": (2, 8, "int"), "rate": (0.01, 0.3)})
    samples = [distribution.sample(random_state=seed) for seed in range(20)]
    assert all(2 <= s["depth"] <= 8 and 0.01 <= s["rate"] <= 0.3 for s in samples)
    assert any(2 < s["depth"] < 8 for s in samples)
    assert any(0.01 < s["rate"] < 0.3 for s in samples)


def test_bayesian_two_value_choices_use_index_bounds():
    tuner = BayesianSearchCV(
        DecisionTreeClassifier(),
        param_space={"criterion": ["gini", "entropy"], "max_depth": [2, 8]},
    )
    assert tuner._bounds == [(0.0, 1.0), (0.0, 1.0)]
    for index in (0, 1):
        vector = np.array([index, index], dtype=float)
        params = tuner._vector_to_params(vector)
        assert params == {"criterion": ["gini", "entropy"][index], "max_depth": [2, 8][index]}
        np.testing.assert_array_equal(tuner._params_to_vector(params), vector)
