"""Neural network algorithms.

This module provides artificial neural network implementations for binary
and multiclass classification.

Algorithms
----------
- **PerceptronClassifier:** The classic single-layer neural network.
- **VotedPerceptronClassifier:** Robust perceptron that uses weighted voting
  across all weight vectors encountered during training.
- **AveragedPerceptronClassifier:** Perceptron that averages all weight vectors
  for improved stability and generalization.
- **MultilayerPerceptronClassifier:** Feedforward neural network with
  configurable hidden layers and backpropagation training.
"""

from tuiml.algorithms.neural.perceptron import (
    PerceptronClassifier,
    VotedPerceptronClassifier,
    AveragedPerceptronClassifier,
)
from tuiml.algorithms.neural.multilayer_perceptron import MultilayerPerceptronClassifier, MultilayerPerceptronRegressor

try:
    import sklearn
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

if SKLEARN_AVAILABLE:
    from tuiml.algorithms.neural.sklearn_mlp_clf import SklearnMLPClassifier
    from tuiml.algorithms.neural.sklearn_mlp_reg import SklearnMLPRegressor

__all__ = [
    "PerceptronClassifier",
    "VotedPerceptronClassifier",
    "AveragedPerceptronClassifier",
    "MultilayerPerceptronClassifier",
    "MultilayerPerceptronRegressor",
]

if SKLEARN_AVAILABLE:
    __all__.extend([
        "SklearnMLPClassifier",
        "SklearnMLPRegressor",
    ])

