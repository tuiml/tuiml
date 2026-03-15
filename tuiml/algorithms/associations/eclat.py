"""ECLATAssociator algorithm for association rule mining."""

import numpy as np
from typing import Dict, List, Any, Optional, FrozenSet, Set, Tuple
from itertools import combinations

from tuiml.base.algorithms import (
    Associator, associator, FrequentItemset, AssociationRule
)

@associator(tags=["frequent-itemset", "vertical", "depth-first"], version="1.0.0")
class ECLATAssociator(Associator):
    """ECLATAssociator algorithm for association rule mining.

    ECLATAssociator (**Equivalence CLAss Transformation**) uses a **vertical data
    layout** (storing transaction IDs for each item) and **depth-first search** to
    find frequent itemsets. It is often faster than Apriori for dense datasets
    because it avoids expensive candidate generation by **intersecting
    transaction ID sets** (tidsets).

    Overview
    --------
    The algorithm operates on a vertical representation of the database:

    1. Convert the horizontal transaction database to a vertical format where each item maps to its set of transaction IDs (tidset)
    2. Filter items whose tidset size is below the minimum support count
    3. Sort frequent items by ascending support (heuristic for faster intersections)
    4. For each frequent item, recursively extend the itemset by intersecting tidsets with remaining items
    5. If the intersection meets the minimum support, record the new frequent itemset and recurse deeper
    6. Generate association rules from all discovered frequent itemsets

    Theory
    ------
    ECLAT exploits the **equivalence class** property of itemsets. Two itemsets
    belong to the same equivalence class if they share the same :math:`(k{-}1)`-prefix.

    The support of an itemset :math:`X` is computed directly from its tidset:

    .. math::
        \\text{support}(X) = \\frac{|\\text{tidset}(X)|}{|T|}

    where :math:`T` is the set of all transactions. To compute the tidset of
    :math:`X \\cup Y`:

    .. math::
        \\text{tidset}(X \\cup Y) = \\text{tidset}(X) \\cap \\text{tidset}(Y)

    **Confidence** and **Lift** for a rule :math:`A \\Rightarrow C` are:

    .. math::
        \\text{confidence}(A \\Rightarrow C) = \\frac{\\text{support}(A \\cup C)}{\\text{support}(A)}

    .. math::
        \\text{lift}(A \\Rightarrow C) = \\frac{\\text{confidence}(A \\Rightarrow C)}{\\text{support}(C)}

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
    metric : str, default='confidence'
        The metric used to rank and filter the discovered rules.
        Options include: ``'confidence'``, ``'lift'``, ``'leverage'``,
        ``'conviction'``, ``'jaccard'``, ``'kulczynski'``, ``'all_confidence'``.

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

    - Space: :math:`O(n \\cdot m)` for the vertical database where :math:`n` = number of transactions and :math:`m` = number of items
    - Time: :math:`O(2^m)` worst case, but typically much faster due to tidset intersection pruning
    - Tidset intersection: :math:`O(\\min(|t_A|, |t_B|))` per pair using sorted sets

    **When to use ECLATAssociator:**

    - Dense datasets where many items co-occur frequently
    - When you want to avoid the multiple database scans required by Apriori
    - Datasets that fit in memory in vertical format
    - When depth-first exploration is preferred over breadth-first

    References
    ----------
    .. [Zaki2000] Zaki, M.J. (2000).
           **Scalable Algorithms for Association Mining.**
           *IEEE Transactions on Knowledge and Data Engineering*, 12(3), pp. 372-390.
           DOI: `10.1109/69.846291 <https://doi.org/10.1109/69.846291>`_

    .. [ZakiHsiao2002] Zaki, M.J. and Hsiao, C.J. (2002).
           **CHARM: An Efficient Algorithm for Closed Itemset Mining.**
           *Proceedings of the 2002 SIAM International Conference on Data Mining*,
           pp. 457-473. DOI: `10.1137/1.9781611972726.27 <https://doi.org/10.1137/1.9781611972726.27>`_

    See Also
    --------
    :class:`~tuiml.algorithms.associations.AprioriAssociator` : Classic level-wise breadth-first association rule mining.
    :class:`~tuiml.algorithms.associations.FPGrowthAssociator` : FP-tree based frequent pattern mining without candidate generation.

    Examples
    --------
    Basic usage for discovering association rules from transaction data:

    >>> from tuiml.algorithms.associations import ECLATAssociator
    >>> transactions = [['bread', 'milk'], ['bread', 'diaper', 'beer', 'egg'],
    ...                 ['milk', 'diaper', 'beer', 'cola'], ['bread', 'milk', 'diaper', 'beer']]
    >>> model = ECLATAssociator(min_support=0.5, min_confidence=0.7)
    >>> model.fit(transactions)
    ECLATAssociator(n_itemsets=11, n_rules=18, min_support=0.5)
    """

    def __init__(self, min_support: float = 0.1,
                 min_confidence: float = 0.8,
                 max_itemset_size: Optional[int] = None,
                 metric: str = 'confidence'):
        """Initialize ECLATAssociator.

        Parameters
        ----------
        min_support : float, default=0.1
            Minimum support threshold.
        min_confidence : float, default=0.8
            Minimum confidence for rules.
        max_itemset_size : int or None, default=None
            Maximum itemset size.
        metric : str, default='confidence'
            Rule ranking metric.
        """
        super().__init__()
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.max_itemset_size = max_itemset_size
        self.metric = metric
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
            },
            "metric": {
                "type": "string",
                "default": "confidence",
                "enum": ["confidence", "lift", "leverage", "conviction", 
                         "jaccard", "kulczynski", "all_confidence"],
                "description": "Rule ranking metric"
            }
        }

    @classmethod
    def get_capabilities(cls) -> List[str]:
        """Return algorithm capabilities."""
        return ["binary", "nominal"]

    @classmethod
    def get_complexity(cls) -> str:
        """Return time/space complexity."""
        return "O(n * m) space for vertical DB, O(2^m) worst case time"

    @classmethod
    def get_references(cls) -> List[str]:
        """Return academic references."""
        return [
            "Zaki, M. J. (2000). Scalable algorithms for association mining. "
            "IEEE Transactions on Knowledge and Data Engineering, 12(3), 372-390."
        ]

    def _mine_frequent_itemsets(self, 
                               prefix: FrozenSet[int], 
                               item_tids: List[Tuple[int, Set[int]]], 
                               min_support_count: int) -> None:
        """Recursive depth-first search for frequent itemsets.

        Extends the current prefix itemset by intersecting tidsets of
        candidate items. For each valid extension (meeting minimum
        support), the method recurses to explore deeper combinations.

        Parameters
        ----------
        prefix : frozenset of int
            Current itemset prefix (e.g., ``frozenset({A})``).
        item_tids : list of tuple (int, set of int)
            List of ``(item, tids)`` tuples for items that can extend the prefix.
        min_support_count : int
            Minimum absolute support count for an itemset to be frequent.
        """
        # Iterate through remaining items
        for i, (item, tids) in enumerate(item_tids):
            # Form new frequent itemset
            new_itemset = prefix | {item}
            support_count = len(tids)
            support = support_count / self.n_transactions_
            
            # Store frequent itemset
            self._item_support[new_itemset] = support
            self.frequent_itemsets_.append(
                FrequentItemset(new_itemset, support, support_count)
            )
            
            # Check if we should go deeper
            if self.max_itemset_size is not None and len(new_itemset) >= self.max_itemset_size:
                continue
                
            # Form new candidates for next level
            # Intersection of TIDs is the key operation in ECLATAssociator
            new_item_tids = []
            for j in range(i + 1, len(item_tids)):
                next_item, next_tids = item_tids[j]
                
                # Intersect transaction IDs
                intersect_tids = tids.intersection(next_tids)
                
                if len(intersect_tids) >= min_support_count:
                    new_item_tids.append((next_item, intersect_tids))
            
            # Recursive call if there are valid extensions
            if new_item_tids:
                self._mine_frequent_itemsets(new_itemset, new_item_tids, min_support_count)

    def _generate_rules(self) -> None:
        """Generate association rules from frequent itemsets.

        Enumerates all possible antecedent/consequent splits for each
        frequent itemset with two or more items. Computes confidence,
        lift, leverage, conviction, Jaccard, Kulczynski, and
        all-confidence metrics, and filters rules by ``min_confidence``.
        """
        # Get support for single items (needed for lift calculation)
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
            itemset_support = itemset.support

            # Generate all possible rules from this itemset
            for i in range(1, len(items)):
                for antecedent in combinations(items, i):
                    antecedent = frozenset(antecedent)
                    consequent = items - antecedent

                    # Calculate confidence
                    ant_support = self._item_support.get(antecedent, 0)
                    if ant_support == 0:
                        continue

                    confidence = itemset_support / ant_support

                    if confidence < self.min_confidence:
                        continue

                    # Calculate consequent support
                    cons_support = self._item_support.get(consequent, 0)
                    if cons_support == 0:
                        # Try to compute from single items (fallback)
                        cons_support = 1.0
                        for item in consequent:
                            cons_support *= item_support.get(item, 0.01)

                    # Calculate metrics
                    lift = confidence / cons_support if cons_support > 0 else 1.0
                    leverage = itemset_support - (ant_support * cons_support)

                    if cons_support < 1.0 and confidence < 1.0:
                        conviction = (1 - cons_support) / (1 - confidence)
                    else:
                        conviction = float('inf') if confidence >= 1.0 else 1.0

                    # Calculate Jaccard coefficient
                    # Jaccard = P(A,C) / (P(A) + P(C) - P(A,C))
                    jaccard = itemset_support / (ant_support + cons_support - itemset_support) \
                        if (ant_support + cons_support - itemset_support) > 0 else 0.0

                    # Calculate Kulczynski measure
                    # Kulc = 0.5 * (P(C|A) + P(A|C))
                    # P(C|A) = confidence, P(A|C) = P(A,C) / P(C)
                    p_a_given_c = itemset_support / cons_support if cons_support > 0 else 0.0
                    kulczynski = 0.5 * (confidence + p_a_given_c)

                    # Calculate All-Confidence
                    # All-conf = P(A,C) / max(P(A), P(C))
                    all_confidence = itemset_support / max(ant_support, cons_support) \
                        if max(ant_support, cons_support) > 0 else 0.0

                    rule = AssociationRule(
                        antecedent=antecedent,
                        consequent=consequent,
                        support=itemset_support,
                        confidence=confidence,
                        lift=lift,
                        leverage=leverage,
                        conviction=conviction,
                        jaccard=jaccard,
                        kulczynski=kulczynski,
                        all_confidence=all_confidence
                    )
                    self.rules_.append(rule)

        # Sort rules by selected metric
        if self.metric == 'confidence':
            self.rules_.sort(key=lambda r: r.confidence, reverse=True)
        elif self.metric == 'lift':
            self.rules_.sort(key=lambda r: r.lift, reverse=True)
        elif self.metric == 'leverage':
            self.rules_.sort(key=lambda r: r.leverage, reverse=True)
        elif self.metric == 'conviction':
            self.rules_.sort(key=lambda r: r.conviction, reverse=True)
        elif self.metric == 'jaccard':
            self.rules_.sort(key=lambda r: r.jaccard, reverse=True)
        elif self.metric == 'kulczynski':
            self.rules_.sort(key=lambda r: r.kulczynski, reverse=True)
        elif self.metric == 'all_confidence':
            self.rules_.sort(key=lambda r: r.all_confidence, reverse=True)

    def fit(self, X) -> "ECLATAssociator":
        """Find frequent itemsets and generate association rules.

        Converts input data to a vertical format and performs recursive 
        intersection of transaction IDs to find frequent itemsets.

        Parameters
        ----------
        X : array-like or list of lists
            The transaction data. Can be a binary matrix of shape 
            (n_transactions, n_items) or a list of transactions where 
            each transaction is a list of items.

        Returns
        -------
        self : ECLATAssociator
            Returns the fitted associator instance.
        """
        # Preprocess transactions
        transactions = self._preprocess_transactions(X)

        if not transactions:
            self._is_fitted = True
            return self

        self.n_transactions_ = len(transactions)
        
        # Reset state
        self.frequent_itemsets_ = []
        self.rules_ = []
        self._item_support = {}

        # 1. Convert to Vertical Format (Item -> {TIDs})
        item_tids: Dict[int, Set[int]] = {}
        all_items = set()
        
        for tid, transaction in enumerate(transactions):
            for item in transaction:
                all_items.add(item)
                if item not in item_tids:
                    item_tids[item] = set()
                item_tids[item].add(tid)
        
        self.n_items_ = len(all_items)
        min_support_count = int(np.ceil(self.min_support * self.n_transactions_))
        
        # 2. Find frequent 1-itemsets
        # Convert map to list of (item, tids) sorted by support (heuristic for faster intersection)
        frequent_1_item_tids = []
        
        for item, tids in item_tids.items():
            if len(tids) >= min_support_count:
                frequent_1_item_tids.append((item, tids))
        
        # Sort by support count ascending (Eclat optimization: reducing intersection cost)
        frequent_1_item_tids.sort(key=lambda x: len(x[1]))
        
        # 3. Mine frequent itemsets recursively (DFS)
        self._mine_frequent_itemsets(frozenset(), frequent_1_item_tids, min_support_count)
        
        # 4. Generate rules
        self._generate_rules()

        self._is_fitted = True
        return self

    def __repr__(self) -> str:
        """String representation."""
        if self._is_fitted:
            return (f"ECLATAssociator(n_itemsets={len(self.frequent_itemsets_)}, "
                   f"n_rules={len(self.rules_)}, "
                   f"min_support={self.min_support})")
        return f"ECLATAssociator(min_support={self.min_support}, min_confidence={self.min_confidence})"
