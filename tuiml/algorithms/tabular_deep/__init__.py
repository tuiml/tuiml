"""Native deep learning architectures for tabular data, backed by PyTorch.

Three supervised architectures, each with a classifier and a regressor. They
are TuiML's own implementations: torch is used only as a tensor and autograd
library, and every model is **trained from scratch on your data**. There is no
pretrained checkpoint and nothing is downloaded.

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
  :func:`~tuiml.algorithms.tabular_deep.node.entmax15` and stacked with
  DenseNet-style connectivity.

Being native implementations, all six register in the TuiML hub under their
**bare** class names, alongside every other native algorithm.

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
>>> from tuiml.algorithms.tabular_deep import FTTransformerClassifier
>>> model = FTTransformerClassifier(d_token=8, n_blocks=1, random_state=0)
>>> model.n_blocks
1

See Also
--------
:mod:`tuiml.foundation` : Pretrained tabular foundation models (TabICL), which
    run a downloaded checkpoint instead of training on your data.
"""

from tuiml.algorithms.tabular_deep.ft_transformer import (
    FTTransformerClassifier,
    FTTransformerRegressor,
)
from tuiml.algorithms.tabular_deep.saint import SAINTClassifier, SAINTRegressor
from tuiml.algorithms.tabular_deep.node import (
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
