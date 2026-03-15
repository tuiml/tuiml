"""Association rule mining algorithms.

This module provides algorithms for discovering interesting relationships
(rules) between variables in large transaction datasets. It follows the
standard Weka approach for associators.

Algorithms
----------
- **AprioriAssociator:** The classic candidate-generation based association miner.
- **FPGrowthAssociator:** Efficient tree-based miner without candidate generation.
- **ECLATAssociator:** Vertical data format miner using depth-first search.
"""

# Base classes and data structures (from algorithms/base - single source of truth)
from tuiml.base.algorithms import (
    Associator,
    associator,
    FrequentItemset,
    AssociationRule,
)

# Association algorithms
from tuiml.algorithms.associations.apriori import AprioriAssociator
from tuiml.algorithms.associations.fpgrowth import FPGrowthAssociator
from tuiml.algorithms.associations.eclat import ECLATAssociator

__all__ = [
    # Base classes
    "Associator",
    "associator",
    # Data structures
    "FrequentItemset",
    "AssociationRule",
    # Algorithms
    "AprioriAssociator",
    "FPGrowthAssociator",
    "ECLATAssociator",
]
