"""The contract sweeps must pick up new components without being edited.

The whole point of the registry-driven design is that adding an algorithm
subscribes it to the full battery automatically. That guarantee is easy to
break by accident -- a filter tightened in ``_registered_algorithms`` or an
import that stops firing would silently shrink coverage while every existing
test still passes. These tests assert the guarantee directly.
"""

import numpy as np

from tuiml.base.algorithms import Classifier, classifier

from tuiml.registry import ComponentType, registry

from .contract.algorithms import ALL_CHECKS, check_algorithm
from .test_common import ALGORITHMS, SKIP_ALGORITHMS


@classifier(tags=["test"], version="1.0.0")
class _ContractProbeClassifier(Classifier):
    """A minimal, deliberately correct classifier used to prove auto-pickup.

    It exists only so the sweep has something newly registered to find. It
    predicts the majority class, which is enough to satisfy every contract
    check without depending on any real algorithm staying green.
    """

    def __init__(self, random_state=None):
        super().__init__()
        self.random_state = random_state

    @classmethod
    def get_parameter_schema(cls):
        """Return JSON Schema for constructor parameters."""
        return {
            "random_state": {
                "type": ["integer", "null"],
                "default": None,
                "description": "Random seed for reproducibility",
            },
        }

    @classmethod
    def get_capabilities(cls):
        """Return the capability strings this classifier declares."""
        return ["numeric", "binary_class", "multiclass"]

    def fit(self, X, y=None):
        """Record the majority class.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data, used only for its shape.
        y : np.ndarray of shape (n_samples,)
            Target values.

        Returns
        -------
        self : _ContractProbeClassifier
            The fitted estimator.
        """
        values, counts = np.unique(np.asarray(y), return_counts=True)
        self.classes_ = values
        self.majority_ = values[int(np.argmax(counts))]
        self._is_fitted = True
        return self

    def predict(self, X):
        """Predict the majority class for every row.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Samples to predict.

        Returns
        -------
        y : np.ndarray of shape (n_samples,)
            The majority class, repeated.
        """
        self._check_is_fitted()
        return np.full(len(X), self.majority_)

    def predict_proba(self, X):
        """Return a degenerate distribution on the majority class.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Samples to predict.

        Returns
        -------
        proba : np.ndarray of shape (n_samples, n_classes)
            One-hot rows, each summing to 1.
        """
        self._check_is_fitted()
        proba = np.zeros((len(X), len(self.classes_)))
        proba[:, int(np.argmax(self.classes_ == self.majority_))] = 1.0
        return proba


def test_the_decorator_registers_without_any_wiring():
    """Defining a decorated class is all it takes to enter the registry."""
    assert registry.get("_ContractProbeClassifier") is _ContractProbeClassifier, (
        "the @classifier decorator no longer registers on class definition, so "
        "a new algorithm would never reach the registry the sweep reads"
    )


def test_the_sweep_covers_every_native_algorithm():
    """No algorithm shipped in the package escapes the contract sweep.

    This is the guarantee that makes the suite self-maintaining: the sweep
    reads the registry, so an algorithm added under ``tuiml/`` is tested
    without anyone editing a test. A filter tightened by accident would show
    up here as a shortfall, rather than as silently missing coverage.
    """
    swept = {name for name, _ in ALGORITHMS}
    native = set()
    for component_type in (ComponentType.CLASSIFIER, ComponentType.REGRESSOR,
                           ComponentType.CLUSTERER):
        native.update(registry.list_names(component_type))

    missed = {
        name for name in native
        if "." not in name and "_v" not in name
        and getattr(registry.get(name), "__module__", "").startswith("tuiml.")
        and name not in swept
        and name not in SKIP_ALGORITHMS
    }
    assert not missed, (
        f"{len(missed)} algorithm(s) live in the package but are not swept by "
        f"the contract suite, so they are untested: {sorted(missed)[:10]}. "
        f"Either fix the discovery filter or add an entry to SKIP_ALGORITHMS "
        f"with a reason."
    )


def test_skips_stay_few_and_documented():
    """Skipping is the escape hatch of last resort, so keep it visible.

    Every skipped algorithm is one nothing tests at all -- unlike an xfail,
    which still runs the remaining checks. Each entry carries a written
    reason, and the list is meant to shrink.
    """
    assert len(SKIP_ALGORITHMS) <= 3, (
        f"{len(SKIP_ALGORITHMS)} algorithms are skipped entirely; the sweep is "
        f"drifting back towards opt-in coverage"
    )
    for name, reason in SKIP_ALGORITHMS.items():
        assert isinstance(reason, str) and len(reason) > 15, (
            f"{name} is skipped without a usable reason"
        )


def test_the_probe_satisfies_every_check():
    """The probe passes the battery, so a failure means the checks broke.

    If this fails while real algorithms still pass, the fault is in the
    checks or fixtures rather than in any algorithm.
    """
    results = check_algorithm(_ContractProbeClassifier)
    failures = {k: v for k, v in results.items() if v is not None}
    assert not failures, f"contract checks reject a correct algorithm: {failures}"


def test_every_check_ran():
    """The battery reports on all of its checks, none silently skipped."""
    results = check_algorithm(_ContractProbeClassifier)
    assert set(results) == {c.__name__ for c in ALL_CHECKS}
