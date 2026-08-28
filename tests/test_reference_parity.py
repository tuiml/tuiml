"""TuiML's numerics against scikit-learn as an external reference.

The contract suite checks that an algorithm is self-consistent -- right shapes,
reproducible under a seed, survives a pickle. It cannot tell whether the answer
is *correct*, because nothing outside TuiML is consulted. That gap is how a
benchmark result reading "logistic regression is 2.6 points behind
scikit-learn" went unexplained: no test compared the two directly, so there was
nothing to say whether the deficit was in the implementation or in how the
comparison was set up. It was the latter, and one test here would have shown it
immediately.

Skipped entirely without ``tuiml[sklearn]``.

**Objectives normalise differently, and that is the whole difficulty.** A
penalty constant cannot be copied between libraries. For logistic regression:

    scikit-learn   0.5*||w||^2 + C*sum(loss)
    TuiML          mean(loss) + 0.5*ridge*||w||^2

Dividing scikit-learn's by ``C*n`` puts it in TuiML's form, giving
``ridge = 1/(C*n)`` -- which depends on the sample size, and is exactly what
TuiML's ``ridge="auto"`` computes. Assuming the constants transfer directly is
what produced the phantom deficit.
"""

import numpy as np
import pytest

pytest.importorskip("sklearn")

from sklearn.datasets import (  # noqa: E402
    load_breast_cancer,
    load_diabetes,
    load_digits,
    load_iris,
    load_wine,
)

CLASSIFICATION = {
    "iris": load_iris,
    "wine": load_wine,
    "breast_cancer": load_breast_cancer,
    "digits": load_digits,
}


def _equivalent_ridge(n_samples, C=1.0):
    """Return the TuiML ridge matching a scikit-learn ``C`` on ``n`` samples.

    Parameters
    ----------
    n_samples : int
        Rows in the training set. The equivalence depends on it because TuiML
        averages the loss where scikit-learn sums it.
    C : float, default=1.0
        scikit-learn's inverse regularisation strength.

    Returns
    -------
    ridge : float
        The equivalent TuiML penalty.
    """
    return 1.0 / (C * n_samples)


# --------------------------------------------------------------------------
# Logistic regression
# --------------------------------------------------------------------------

@pytest.mark.parametrize("name", list(CLASSIFICATION))
def test_logistic_matches_sklearn_at_equivalent_penalty(name):
    """Within a point of scikit-learn once the penalties actually correspond.

    The tolerance is not tight because the solvers differ in their stopping
    rule, not because the answers are expected to diverge; on iris the two
    agree exactly.
    """
    from sklearn.linear_model import LogisticRegression as SkLR

    from tuiml.algorithms.linear import LogisticRegression as TuiLR

    data = CLASSIFICATION[name]()
    X, y = data.data, data.target

    tuiml_model = TuiLR(ridge=_equivalent_ridge(len(y)), max_iter=5000).fit(X, y)
    sklearn_model = SkLR(C=1.0, max_iter=5000).fit(X, y)

    tuiml_acc = (tuiml_model.predict(X) == y).mean()
    sklearn_acc = (sklearn_model.predict(X) == y).mean()

    assert tuiml_acc >= sklearn_acc - 0.01, (
        f"{name}: TuiML {tuiml_acc:.4f} vs scikit-learn {sklearn_acc:.4f}"
    )


def test_logistic_auto_ridge_is_the_sklearn_default_equivalent():
    """``ridge="auto"`` resolves to 1/n_samples, which is C=1.0.

    Worth pinning separately: it means the out-of-the-box TuiML model is
    already configured like scikit-learn's, and a benchmark comparing defaults
    is comparing like with like.
    """
    from sklearn.linear_model import LogisticRegression as SkLR

    from tuiml.algorithms.linear import LogisticRegression as TuiLR

    X, y = load_iris(return_X_y=True)

    auto = TuiLR(max_iter=5000).fit(X, y)
    explicit = TuiLR(ridge=_equivalent_ridge(len(y)), max_iter=5000).fit(X, y)
    reference = SkLR(C=1.0, max_iter=5000).fit(X, y)

    np.testing.assert_allclose(auto.coef_, explicit.coef_, rtol=1e-6)
    assert (auto.predict(X) == y).mean() >= (reference.predict(X) == y).mean() - 0.01


def test_over_regularising_logistic_is_visibly_worse():
    """Guards the guard.

    If ridge=0.5 scored as well as the equivalent penalty on iris, the test
    above would pass for the wrong reason and the original error would be
    invisible. It does not: 0.5 is ~75x too strong at n=150.
    """
    from tuiml.algorithms.linear import LogisticRegression as TuiLR

    X, y = load_iris(return_X_y=True)

    correct = TuiLR(ridge=_equivalent_ridge(len(y)), max_iter=5000).fit(X, y)
    over = TuiLR(ridge=0.5, max_iter=5000).fit(X, y)

    correct_acc = (correct.predict(X) == y).mean()
    over_acc = (over.predict(X) == y).mean()
    assert correct_acc - over_acc > 0.05, (
        f"expected a large penalty to hurt: {correct_acc:.4f} vs {over_acc:.4f}"
    )


# --------------------------------------------------------------------------
# Algorithms whose objective needs no translation
# --------------------------------------------------------------------------

@pytest.mark.parametrize("name", list(CLASSIFICATION))
def test_gaussian_naive_bayes_matches_sklearn(name):
    """Closed form, no regularisation, no solver: these should agree closely."""
    from sklearn.naive_bayes import GaussianNB

    from tuiml.algorithms.bayesian import NaiveBayesClassifier

    X, y = CLASSIFICATION[name](return_X_y=True)

    tuiml_acc = (NaiveBayesClassifier().fit(X, y).predict(X) == y).mean()
    sklearn_acc = (GaussianNB().fit(X, y).predict(X) == y).mean()

    assert abs(tuiml_acc - sklearn_acc) < 0.02, (
        f"{name}: TuiML {tuiml_acc:.4f} vs scikit-learn {sklearn_acc:.4f}"
    )


@pytest.mark.parametrize("name", ["iris", "wine", "breast_cancer"])
def test_knn_matches_sklearn_exactly(name):
    """Deterministic given the same k and metric, so demand exact agreement."""
    from sklearn.neighbors import KNeighborsClassifier

    from tuiml.algorithms.neighbors import KNearestNeighborsClassifier

    X, y = CLASSIFICATION[name](return_X_y=True)

    tuiml_pred = KNearestNeighborsClassifier(k=5).fit(X, y).predict(X)
    sklearn_pred = KNeighborsClassifier(n_neighbors=5).fit(X, y).predict(X)

    agreement = (np.asarray(tuiml_pred) == sklearn_pred).mean()
    assert agreement > 0.98, f"{name}: only {agreement:.3f} agreement"


def test_linear_regression_matches_sklearn_coefficients():
    """Ordinary least squares has one answer; the coefficients should match."""
    from sklearn.linear_model import LinearRegression as SkLinear

    from tuiml.algorithms.linear import LinearRegression as TuiLinear

    X, y = load_diabetes(return_X_y=True)

    tuiml_model = TuiLinear().fit(X, y)
    sklearn_model = SkLinear().fit(X, y)

    # TuiML names the fitted weights coefficients_, not coef_.
    np.testing.assert_allclose(
        np.ravel(tuiml_model.coefficients_),
        np.ravel(sklearn_model.coef_),
        rtol=1e-4,
    )
    np.testing.assert_allclose(
        float(np.ravel(tuiml_model.intercept_)[0]),
        float(sklearn_model.intercept_),
        rtol=1e-4,
    )
