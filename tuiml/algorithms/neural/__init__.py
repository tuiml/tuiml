"""Neural network algorithms.

This module provides artificial neural network implementations for binary
and multiclass classification.

Algorithms
----------
- **PerceptronClassifier:** The classic single-layer neural network.
  across all weight vectors encountered during training.
  for improved stability and generalization.
- **MultilayerPerceptronClassifier:** Feedforward neural network with
  configurable hidden layers and backpropagation training.
"""

from tuiml.algorithms.neural.perceptron import (
    PerceptronClassifier,
)
from tuiml.algorithms.neural.multilayer_perceptron import MultilayerPerceptronClassifier, MultilayerPerceptronRegressor

__all__ = [
    "PerceptronClassifier",
    "MultilayerPerceptronClassifier",
    "MultilayerPerceptronRegressor",
]
