"""Synthetic regression datasets.

Continuous targets generated from a function you know, so you can measure how
much of it a model recovered. All three are non-linear on purpose: a linear
model should visibly struggle where a tree ensemble does not.

Generators
----------
- **Friedman:** The Friedman benchmarks. The target combines products, a
  squared term and linear terms, and the data carries extra features that do
  not affect it at all — so it tests feature selection as well as fit.
- **MexicanHat:** The radially symmetric ``sinc`` surface: smooth, with
  concentric ridges no linear model can follow.
- **Sine:** A sine wave with configurable noise. The simplest case, and a
  quick check that a regressor can fit any curve.

Examples
--------
>>> from tuiml.datasets.generators.regression import Friedman
>>> data = Friedman(n_samples=500, noise=0.1, random_state=0).generate()
>>> data.X.shape[0]
500
"""

from tuiml.datasets.generators.regression.friedman import Friedman
from tuiml.datasets.generators.regression.mexican_hat import MexicanHat
from tuiml.datasets.generators.regression.sine import Sine

__all__ = [
    "Friedman",
    "MexicanHat",
    "Sine",
]
