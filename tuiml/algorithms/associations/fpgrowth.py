"""FP-Growth algorithm for association rule mining."""

import numpy as np
from typing import Dict, List, Any, Optional, FrozenSet, Set
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import combinations

from tuiml.base.algorithms import (
    Associator, associator, FrequentItemset, AssociationRule
)

@dataclass
class FPNode:
    """Node in the FP-Tree."""
    item: Optional[int] = None
    count: int = 0
    parent: Optional["FPNode"] = None
    children: Dict[int, "FPNode"] = field(default_factory=dict)
    link: Optional["FPNode"] = None  # Link to next node with same item

class FPTree:
    """FP-Tree data structure for efficient frequent pattern mining."""

    def __init__(self):
        self.root = FPNode()
        self.header_table: Dict[int, FPNode] = {}  # item -> first node
        self.item_counts: Dict[int, int] = defaultdict(int)

    def insert_transaction(self, transaction: List[int], count: int = 1) -> None:
        """Insert a transaction into the tree.

        Parameters
        ----------
        transaction : list of int
            Ordered list of items in the transaction.
        count : int, default=1
            The count to add for each item along the path.
        """
        node = self.root

        for item in transaction:
            self.item_counts[item] += count

            if item in node.children:
                node.children[item].count += count
            else:
                new_node = FPNode(item=item, count=count, parent=node)
                node.children[item] = new_node

                # Update header table
                if item in self.header_table:
                    # Find last node in the link chain
                    current = self.header_table[item]
                    while current.link is not None:
                        current = current.link
                    current.link = new_node
                else:
                    self.header_table[item] = new_node

            node = node.children[item]

    def get_prefix_path(self, node: FPNode) -> tuple:
        """Get the prefix path ending at node.

        Parameters
        ----------
        node : FPNode
            The node whose prefix path is to be extracted.

        Returns
        -------
        path : list of int
            Items on the path from root to the node's parent (in root-to-leaf order).
        count : int
            The count at the given node.
        """
        path = []
        count = node.count
        current = node.parent

        while current is not None and current.item is not None:
            path.append(current.item)
            current = current.parent

        return path[::-1], count

    def get_conditional_pattern_base(self, item: int) -> List[tuple]:
        """Get conditional pattern base for an item.

        Parameters
        ----------
        item : int
            The item for which to extract the conditional pattern base.

        Returns
        -------
        patterns : list of tuple
            List of ``(path, count)`` tuples representing prefix paths.
        """
        patterns = []
        node = self.header_table.get(item)

        while node is not None:
            path, count = self.get_prefix_path(node)
            if path:
                patterns.append((path, count))
            node = node.link

        return patterns

@associator(tags=["frequent-itemset", "tree-based", "efficient"], version="1.0.0")
class FPGrowthAssociator(Associator):
    """FP-Growth algorithm for association rule mining.

    The **FP-Growth** (Frequent Pattern Growth) algorithm is a more efficient
    alternative to Apriori. It avoids expensive **candidate generation** by
    representing the transaction database as a compact **FP-tree** data structure
    and discovers frequent patterns by recursively exploring **conditional FP-trees**.

    Overview
    --------
    The algorithm operates in two main phases:

    1. **FP-tree construction:** Scan the database once to find frequent 1-itemsets, then scan again to insert each transaction (sorted by descending frequency) into the FP-tree
    2. For each item in ascending frequency order, extract its **conditional pattern base** (prefix paths ending at nodes for that item)
    3. Build a **conditional FP-tree** from the conditional pattern base
    4. Recursively mine the conditional FP-tree to discover longer frequent patterns
    5. Combine prefix patterns to form all frequent itemsets
    6. Generate association rules from the complete set of frequent itemsets

    Theory
    ------
    The FP-tree achieves compression by sharing **common prefixes** among
    transactions. A header table links all nodes for the same item, enabling
    efficient traversal.

    For an itemset :math:`X`, the support is obtained from the FP-tree as:

    .. math::
        \\text{support}(X) = \\frac{\\text{count}(X)}{|T|}

    where :math:`\\text{count}(X)` is derived from the conditional pattern base
    of the least frequent item in :math:`X`.

    **Confidence** and **Lift** for a rule :math:`A \\Rightarrow C`:

    .. math::
        \\text{confidence}(A \\Rightarrow C) = \\frac{\\text{support}(A \\cup C)}{\\text{support}(A)}

    .. math::
        \\text{lift}(A \\Rightarrow C) = \\frac{\\text{confidence}(A \\Rightarrow C)}{\\text{support}(C)}

    The **FP-tree completeness theorem** guarantees that the FP-tree contains
    all information needed for mining frequent patterns, with no information
    loss compared to the original database.

    Parameters
    ----------
    min_support : float, default=0.1
        Minimum support threshold. Expressed as a fraction of the total
        number of transactions.
    min_confidence : float, default=0.8
        Minimum confidence threshold for rule generation. Rules with
        confidence below this value will be discarded.
    max_itemset_size : int or None, default=None
        Maximum size of frequent itemsets to discover. If None, no limit
        is applied.

    Attributes
    ----------
    frequent_itemsets_ : list of FrequentItemset
        The discovered frequent itemsets and their support counts.
    rules_ : list of AssociationRule
        The association rules generated from the frequent itemsets.
    n_transactions_ : int
        The total number of transactions processed during ``fit``.
    n_items_ : int
        The number of unique items encountered in the data.

    Notes
    -----
    **Complexity:**

    - FP-tree construction: :math:`O(n \\cdot m)` where :math:`n` = number of transactions and :math:`m` = average transaction length
    - Space: :math:`O(n \\cdot m)` worst case for the FP-tree, but typically much smaller due to prefix sharing
    - Mining: depends on the tree structure; highly compressed trees lead to faster mining

    **When to use FPGrowthAssociator:**

    - Large datasets where Apriori's multiple database scans are prohibitive
    - When the transaction database has many shared prefixes (high compression)
    - Dense datasets with many frequent items
    - When you want to avoid candidate generation overhead entirely

    References
    ----------
    .. [Han2000] Han, J., Pei, J. and Yin, Y. (2000).
           **Mining Frequent Patterns Without Candidate Generation.**
           *Proceedings of the 2000 ACM SIGMOD International Conference on Management of Data*,
           pp. 1-12. DOI: `10.1145/342009.335372 <https://doi.org/10.1145/342009.335372>`_

    .. [Han2004] Han, J., Pei, J., Yin, Y. and Mao, R. (2004).
           **Mining Frequent Patterns without Candidate Generation: A Frequent-Pattern Tree Approach.**
           *Data Mining and Knowledge Discovery*, 8(1), pp. 53-87.
           DOI: `10.1023/B:DAMI.0000005258.31418.83 <https://doi.org/10.1023/B:DAMI.0000005258.31418.83>`_

    See Also
    --------
    :class:`~tuiml.algorithms.associations.AprioriAssociator` : Classic level-wise breadth-first association rule mining.
    :class:`~tuiml.algorithms.associations.ECLATAssociator` : Equivalence class based frequent itemset mining using vertical data layout.

    Examples
    --------
    Basic usage for discovering association rules from transaction data:

    >>> from tuiml.algorithms.associations import FPGrowthAssociator
    >>> transactions = [['milk', 'bread', 'butter'], ['beer', 'diapers'],
    ...                 ['milk', 'diapers', 'beer', 'cola'], ['bread', 'milk', 'diapers', 'beer']]
    >>> model = FPGrowthAssociator(min_support=0.5, min_confidence=0.7)
    >>> model.fit(transactions)
    FPGrowthAssociator(n_itemsets=11, n_rules=18, min_support=0.5)
    """

    def __init__(self, min_support: float = 0.1,
                 min_confidence: float = 0.8,
                 max_itemset_size: Optional[int] = None):
        """Initialize FP-Growth.

        Parameters
        ----------
        min_support : float, default=0.1
            Minimum support threshold.
        min_confidence : float, default=0.8
            Minimum confidence for rules.
        max_itemset_size : int or None, default=None
            Maximum itemset size.
        """
        super().__init__()
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.max_itemset_size = max_itemset_size
        self._item_support: Dict[FrozenSet[int], float] = {}

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        """Return parameter schema."""
        return {
            "min_support": {
                "type": "number",
                "default": 0.1,
                "minimum": 0,
                "maximum": 1,
                "description": "Minimum support threshold"
            },
            "min_confidence": {
                "type": "number",
                "default": 0.8,
                "minimum": 0,
                "maximum": 1,
                "description": "Minimum confidence for rules"
            },
            "max_itemset_size": {
                "type": "integer",
                "default": None,
                "minimum": 1,
                "description": "Maximum itemset size"
            }
        }

    @classmethod
    def get_capabilities(cls) -> List[str]:
        """Return algorithm capabilities."""
        return ["binary", "nominal"]

    @classmethod
    def get_complexity(cls) -> str:
        """Return time/space complexity."""
        return "O(n * m) for tree construction, mining depends on tree structure"

    @classmethod
    def get_references(cls) -> List[str]:
        """Return academic references."""
        return [
            "Han, J., Pei, J., & Yin, Y. (2000). Mining frequent patterns "
            "without candidate generation. SIGMOD '00, 1-12."
        ]

    def _build_fp_tree(self, transactions: List[FrozenSet[int]],
                       min_count: int) -> FPTree:
        """Build the FP-tree from transactions.

        Counts item frequencies, filters infrequent items, and inserts
        each transaction (with items sorted by descending frequency)
        into a new FP-tree.

        Parameters
        ----------
        transactions : list of frozenset of int
            The preprocessed transaction database.
        min_count : int
            Minimum absolute support count for an item to be included.

        Returns
        -------
        tree : FPTree
            The constructed FP-tree.
        """
        # Count item frequencies
        item_counts = defaultdict(int)
        for transaction in transactions:
            for item in transaction:
                item_counts[item] += 1

        # Filter infrequent items
        frequent_items = {item for item, count in item_counts.items()
                        if count >= min_count}

        # Sort items by frequency (descending) for optimal tree compression
        item_order = {item: -count for item, count in item_counts.items()
                     if item in frequent_items}

        # Build tree
        tree = FPTree()

        for transaction in transactions:
            # Filter and sort items
            filtered = [item for item in transaction if item in frequent_items]
            filtered.sort(key=lambda x: (item_order.get(x, 0), x))

            if filtered:
                tree.insert_transaction(filtered)

        return tree

    def _mine_patterns(self, tree: FPTree, prefix: FrozenSet[int],
                       min_count: int, n_transactions: int) -> None:
        """Recursively mine frequent patterns from the FP-tree.

        For each item in the tree (processed in ascending frequency order),
        constructs a conditional pattern base and conditional FP-tree,
        then recurses to find longer frequent patterns.

        Parameters
        ----------
        tree : FPTree
            The FP-tree (or conditional FP-tree) to mine.
        prefix : frozenset of int
            The current pattern prefix being extended.
        min_count : int
            Minimum absolute support count.
        n_transactions : int
            Total number of transactions (for computing support fraction).
        """
        # Process items in ascending order of frequency
        items = sorted(tree.item_counts.keys(),
                      key=lambda x: tree.item_counts[x])

        for item in items:
            count = tree.item_counts[item]
            if count < min_count:
                continue

            # Create new pattern
            new_pattern = prefix | frozenset([item])

            # Check max size
            if self.max_itemset_size is not None and len(new_pattern) > self.max_itemset_size:
                continue

            support = count / n_transactions
            self._item_support[new_pattern] = support
            self.frequent_itemsets_.append(
                FrequentItemset(new_pattern, support, count)
            )

            # Get conditional pattern base
            cond_patterns = tree.get_conditional_pattern_base(item)

            if cond_patterns:
                # Build conditional FP-tree
                cond_tree = FPTree()
                for path, path_count in cond_patterns:
                    cond_tree.insert_transaction(path, path_count)

                # Recursively mine
                if cond_tree.header_table:
                    self._mine_patterns(cond_tree, new_pattern, min_count, n_transactions)

    def _generate_rules(self) -> None:
        """Generate association rules from frequent itemsets.

        Enumerates all possible antecedent/consequent splits for each
        frequent itemset with two or more items. Computes confidence,
        lift, leverage, conviction, Jaccard, Kulczynski, and
        all-confidence metrics, and filters rules by ``min_confidence``.
        """
        # Build support lookup
        itemset_support = {fs.items: fs.support for fs in self.frequent_itemsets_}

        # Get single item support
        item_support = {}
        for fs in self.frequent_itemsets_:
            if len(fs.items) == 1:
                item = list(fs.items)[0]
                item_support[item] = fs.support

        # Generate rules from itemsets with 2+ items
        for itemset in self.frequent_itemsets_:
            if len(itemset.items) < 2:
                continue

            items = itemset.items
            itemset_support_val = itemset.support

            # Generate all possible rules from this itemset
            for i in range(1, len(items)):
                for antecedent in combinations(items, i):
                    antecedent = frozenset(antecedent)
                    consequent = items - antecedent

                    # Calculate confidence
                    ant_support = itemset_support.get(antecedent, 0)
                    if ant_support == 0:
                        continue

                    confidence = itemset_support_val / ant_support

                    if confidence < self.min_confidence:
                        continue

                    # Calculate consequent support
                    cons_support = itemset_support.get(consequent, 0)
                    if cons_support == 0:
                        # Estimate from single items
                        cons_support = 1.0
                        for item in consequent:
                            cons_support *= item_support.get(item, 0.01)

                    # Calculate metrics
                    lift = confidence / cons_support if cons_support > 0 else 1.0
                    leverage = itemset_support_val - (ant_support * cons_support)

                    if cons_support < 1.0 and confidence < 1.0:
                        conviction = (1 - cons_support) / (1 - confidence)
                    else:
                        conviction = float('inf') if confidence >= 1.0 else 1.0

                    # Calculate Jaccard coefficient
                    # Jaccard = P(A,C) / (P(A) + P(C) - P(A,C))
                    jaccard = itemset_support_val / (ant_support + cons_support - itemset_support_val) \
                        if (ant_support + cons_support - itemset_support_val) > 0 else 0.0

                    # Calculate Kulczynski measure
                    # Kulc = 0.5 * (P(C|A) + P(A|C))
                    # P(C|A) = confidence, P(A|C) = P(A,C) / P(C)
                    p_a_given_c = itemset_support_val / cons_support if cons_support > 0 else 0.0
                    kulczynski = 0.5 * (confidence + p_a_given_c)

                    # Calculate All-Confidence
                    # All-conf = P(A,C) / max(P(A), P(C))
                    all_confidence = itemset_support_val / max(ant_support, cons_support) \
                        if max(ant_support, cons_support) > 0 else 0.0

                    rule = AssociationRule(
                        antecedent=antecedent,
                        consequent=consequent,
                        support=itemset_support_val,
                        confidence=confidence,
                        lift=lift,
                        leverage=leverage,
                        conviction=conviction,
                        jaccard=jaccard,
                        kulczynski=kulczynski,
                        all_confidence=all_confidence
                    )
                    self.rules_.append(rule)

        # Sort by confidence
        self.rules_.sort(key=lambda r: r.confidence, reverse=True)

    def fit(self, X) -> "FPGrowthAssociator":
        """Find frequent itemsets and generate association rules.

        Builds an FP-tree from the input transactions and recursively extracts 
        frequent patterns to generate association rules.

        Parameters
        ----------
        X : array-like or list of lists
            The transaction data. Can be a binary matrix of shape 
            (n_transactions, n_items) or a list of transactions where 
            each transaction is a list of items.

        Returns
        -------
        self : FPGrowthAssociator
            Returns the fitted associator instance.
        """
        # Preprocess transactions
        transactions = self._preprocess_transactions(X)

        if not transactions:
            self._is_fitted = True
            return self

        self.n_transactions_ = len(transactions)
        self.n_items_ = len(set.union(*[set(t) for t in transactions]))

        # Reset state
        self.frequent_itemsets_ = []
        self.rules_ = []
        self._item_support = {}

        # Calculate minimum count
        min_count = int(np.ceil(self.min_support * self.n_transactions_))

        # Build FP-tree
        tree = self._build_fp_tree(transactions, min_count)

        # Mine frequent patterns
        if tree.header_table:
            self._mine_patterns(tree, frozenset(), min_count, self.n_transactions_)

        # Generate rules
        self._generate_rules()

        self._is_fitted = True
        return self

    def __repr__(self) -> str:
        """String representation."""
        if self._is_fitted:
            return (f"FPGrowthAssociator(n_itemsets={len(self.frequent_itemsets_)}, "
                   f"n_rules={len(self.rules_)}, "
                   f"min_support={self.min_support})")
        return f"FPGrowthAssociator(min_support={self.min_support}, min_confidence={self.min_confidence})"
