"""Starter templates for new user algorithms."""

from __future__ import annotations

from typing import Any, Dict

_CLASSIFIER_TEMPLATE = '''"""{description}"""

import numpy as np
from typing import Dict, Any

from tuiml.base.algorithms import Classifier, classifier


@classifier(tags=["custom"], version="{version}")
class {class_name}(Classifier):
    """{description}

    Parameters
    ----------
    n_neighbors : int, default=5
        Placeholder hyperparameter, replace with your own.
    """

    def __init__(self, n_neighbors: int = 5):
        super().__init__()
        self.n_neighbors = n_neighbors

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Any]:
        return {{
            "n_neighbors": {{"type": int, "default": 5, "range": (1, 100)}},
        }}

    def fit(self, X: np.ndarray, y: np.ndarray) -> "{class_name}":
        X = np.asarray(X)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        # TODO: implement training logic
        self._most_common_ = self.classes_[np.argmax(np.bincount(y.astype(int)))] if self.n_classes_ else 0
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        # TODO: implement prediction logic
        return np.full(len(X), self._most_common_)
'''


_REGRESSOR_TEMPLATE = '''"""{description}"""

import numpy as np
from typing import Dict, Any

from tuiml.base.algorithms import Regressor, regressor


@regressor(tags=["custom"], version="{version}")
class {class_name}(Regressor):
    """{description}

    Parameters
    ----------
    alpha : float, default=1.0
        Placeholder hyperparameter, replace with your own.
    """

    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Any]:
        return {{
            "alpha": {{"type": float, "default": 1.0, "range": (0.0, 10.0)}},
        }}

    def fit(self, X: np.ndarray, y: np.ndarray) -> "{class_name}":
        X = np.asarray(X)
        y = np.asarray(y, dtype=float)
        # TODO: implement training logic
        self.mean_ = float(y.mean())
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        # TODO: implement prediction logic
        return np.full(len(X), self.mean_)
'''


def skeleton(kind: str, class_name: str = "MyAlgorithm",
             version: str = "1.0.0",
             description: str = "Describe what your algorithm does.") -> Dict[str, Any]:
    """Return a ready-to-fill template for a new algorithm.

    Parameters
    ----------
    kind : str
        Either ``'classifier'`` or ``'regressor'`` (case-insensitive).
    class_name : str, default="MyAlgorithm"
        Name of the generated class.
    version : str, default="1.0.0"
        Semver version baked into the decorator.
    description : str, default="Describe what your algorithm does."
        One-line description used as the module and class docstring.

    Returns
    -------
    result : Dict[str, Any]
        On success: keys ``status``, ``kind``, ``class_name``, ``version``,
        ``code`` (the filled-in template source), and ``notes``. On failure:
        keys ``status``, ``error_type``, ``error``.
    """
    kind = kind.lower()
    if kind not in {"classifier", "regressor"}:
        return {
            "status": "error",
            "error_type": "ValueError",
            "error": f"kind must be 'classifier' or 'regressor', got {kind!r}",
        }
    template = _CLASSIFIER_TEMPLATE if kind == "classifier" else _REGRESSOR_TEMPLATE
    return {
        "status": "success",
        "kind": kind,
        "class_name": class_name,
        "version": version,
        "code": template.format(
            class_name=class_name,
            version=version,
            description=description,
        ),
        "notes": (
            "Fill in fit() and predict(), adjust __init__ hyperparameters and "
            "get_parameter_schema(), then pass the completed source to "
            "tuiml_create_algorithm."
        ),
    }
