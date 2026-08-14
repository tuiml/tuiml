"""Deep learning architectures for tabular data, backed by PyTorch.

Three families, each with a classifier and a regressor:

- **FT-Transformer** (:class:`FTTransformerClassifier`,
  :class:`FTTransformerRegressor`) -- a feature tokenizer turns every column
  into its own embedding, and a Transformer attends across those feature
  tokens.
- **SAINT** (:class:`SAINTClassifier`, :class:`SAINTRegressor`) -- the same,
  plus **intersample attention**: a second attention stage that runs across
  the rows of the batch, so a sample's representation depends on its
  neighbours.
- **NODE** (:class:`NODEClassifier`, :class:`NODERegressor`) -- ensembles of
  oblivious decision trees made differentiable with
  :func:`~tuiml.algorithms.tabular_foundation.node.entmax15` and stacked with
  DenseNet-style connectivity.

PyTorch is an **optional** dependency:

.. code-block:: bash

    pip install 'tuiml[torch]'

Importing this package never imports torch, and constructing a model never
imports torch either -- the classes are registered and introspectable on any
install. The dependency is required only by :meth:`fit`, which raises a clear
``ImportError`` naming the install command. See
:mod:`tuiml.utils.torch_backend`.

Examples
--------
>>> from tuiml.algorithms.tabular_foundation import FTTransformerClassifier
>>> model = FTTransformerClassifier(d_token=8, n_blocks=1, random_state=0)
>>> model.n_blocks
1
"""

from tuiml.algorithms.tabular_foundation.ft_transformer import (
    FTTransformerClassifier,
    FTTransformerRegressor,
)
from tuiml.algorithms.tabular_foundation.saint import SAINTClassifier, SAINTRegressor
from tuiml.algorithms.tabular_foundation.node import (
    entmax15,
    NODEClassifier,
    NODERegressor,
)

__all__ = [
    "FTTransformerClassifier",
    "FTTransformerRegressor",
    "SAINTClassifier",
    "SAINTRegressor",
    "NODEClassifier",
    "NODERegressor",
    "entmax15",
]
