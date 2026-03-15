"""AprioriAssociator algorithm for association rule mining."""

import numpy as np
from typing import Dict, List, Any, Optional, FrozenSet, Set
from itertools import combinations

from tuiml.base.algorithms import (
    Associator, associator, FrequentItemset, AssociationRule
)

@associator(tags=["frequent-itemset", "classic"], version="1.0.0")
class AprioriAssociator(Associator):
    """AprioriAssociator algorithm for association rule mining.

    AprioriAssociator is the classic algorithm for discovering **frequent itemsets** and
    generating **association rules**. It uses the **apriori property** -- that all
    subsets of a frequent itemset must also be frequent -- to efficiently prune
    the search space via a **level-wise breadth-first** candidate generation strategy.

    Overview
    --------
    The algorithm operates in iterative passes over the transaction database:

    1. Scan the database to find all frequent 1-itemsets (single items meeting ``min_support``)
    2. Generate candidate :math:`k`-itemsets by joining frequent :math:`(k{-}1)`-itemsets
    3. Prune candidates whose :math:`(k{-}1)`-subsets are not all frequent (apriori property)
    4. Scan the database again to count support for remaining candidates
    5. Repeat steps 2--4 until no new frequent itemsets are found
    6. Generate association rules from all discovered frequent itemsets

    Theory
    ------
    The key metrics for association rule mining are defined as follows.

    For a rule :math:`A \\Rightarrow C`:

    **Support** measures how frequently the itemset appears in the database:

    .. math::
        \\text{support}(A \\Rightarrow C) = P(A \\cup C) = \\frac{|\\{t \\in T : A \\cup C \\subseteq t\\}|}{|T|}

    **Confidence** measures reliability of the rule:

    .. math::
        \\text{confidence}(A \\Rightarrow C) = P(C | A) = \\frac{\\text{support}(A \\cup C)}{\\text{support}(A)}

    **Lift** measures the degree to which :math:`A` and :math:`C` are independent:

    .. math::
        \\text{lift}(A \\Rightarrow C) = \\frac{\\text{confidence}(A \\Rightarrow C)}{\\text{support}(C)}

    A lift value of 1 indicates independence; values greater than 1 indicate
    positive correlation.

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

    - Training: :math:`O(2^m)` worst case where :math:`m` is the number of items, though the apriori pruning typically reduces this dramatically
    - Each pass requires :math:`O(n \\cdot c_k)` where :math:`n` = number of transactions and :math:`c_k` = number of candidates at level :math:`k`
    - Space: :math:`O(c_k)` for storing candidate itemsets at each level

    **When to use AprioriAssociator:**

    - When you need a well-understood, transparent algorithm for mining association rules
    - Sparse datasets where the number of frequent itemsets is manageable
    - When interpretability of the mining process is important
    - Smaller to medium datasets where multiple database scans are acceptable

    References
    ----------
    .. [Agrawal1994] Agrawal, R. and Srikant, R. (1994).
           **Fast Algorithms for Mining Association Rules in Large Databases.**
           *Proceedings of the 20th International Conference on Very Large Data Bases (VLDB)*,
           pp. 478-499.

    .. [Agrawal1993] Agrawal, R., Imielinski, T. and Swami, A. (1993).
           **Mining Association Rules Between Sets of Items in Large Databases.**
           *Proceedings of the 1993 ACM SIGMOD International Conference on Management of Data*,
           pp. 207-216. DOI: `10.1145/170035.170072 <https://doi.org/10.1145/170035.170072>`_

    See Also
    --------
    :class:`~tuiml.algorithms.associations.ECLATAssociator` : Equivalence class based frequent itemset mining using vertical data layout.
    :class:`~tuiml.algorithms.associations.FPGrowthAssociator` : FP-tree based frequent pattern mining without candidate generation.

    Examples
    --------
    Basic usage for discovering association rules from transaction data:

    >>> from tuiml.algorithms.associations import AprioriAssociator
    >>> transactions = [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]
    >>> apriori = AprioriAssociator(min_support=0.5, min_confidence=0.7)
    >>> apriori.fit(transactions)
    AprioriAssociator(n_itemsets=6, n_rules=4, min_support=0.5)
    >>> rules = apriori.get_rules()
    """

    def __init__(self, min_support: float = 0.1,
                 min_confidence: float = 0.8,
                 max_itemset_size: Optional[int] = None,
                 metric: str = 'confidence'):
        """Initialize AprioriAssociator.

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
                "enum": ["confidence", "lift", "leverage", "conviction", "jaccard", "kulczynski", "all_confidence"],
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
        return "O(2^m) worst case, where m is number of items"

    @classmethod
    def get_references(cls) -> List[str]:
        """Return academic references."""
        return [
            "Agrawal, R. & Srikant, R. (1994). Fast Algorithms for Mining "
            "Association Rules in Large Databases. VLDB '94, 478-499."
        ]

    def _count_support(self, transactions: List[FrozenSet[int]],
                       candidates: List[FrozenSet[int]]) -> Dict[FrozenSet[int], int]:
        """Count support for candidate itemsets.

        Parameters
        ----------
        transactions : list of frozenset of int
            The preprocessed transaction database.
        candidates : list of frozenset of int
            Candidate itemsets to count support for.

        Returns
        -------
        counts : dict
            Mapping from candidate itemset to its absolute support count.
        """
        counts = {c: 0 for c in candidates}

        for transaction in transactions:
            for candidate in candidates:
                if candidate.issubset(transaction):
                    counts[candidate] += 1

        return counts

    def _generate_candidates(self, prev_frequent: List[FrozenSet[int]],
                             k: int) -> List[FrozenSet[int]]:
        """Generate candidate k-itemsets from frequent (k-1)-itemsets.

        Uses the self-join and pruning strategy of the Apriori algorithm:
        two :math:`(k{-}1)`-itemsets are joined if their union has exactly
        :math:`k` items, and the resulting candidate is pruned if any of its
        :math:`(k{-}1)`-subsets is not frequent.

        Parameters
        ----------
        prev_frequent : list of frozenset of int
            Frequent :math:`(k{-}1)`-itemsets from the previous pass.
        k : int
            Target itemset size for the generated candidates.

        Returns
        -------
        candidates : list of frozenset of int
            List of candidate :math:`k`-itemsets.
        """
        candidates = set()

        # Self-join: combine itemsets that share k-2 items
        prev_list = list(prev_frequent)
        for i, itemset1 in enumerate(prev_list):
            for itemset2 in prev_list[i + 1:]:
                # Union should have exactly k items
                union = itemset1 | itemset2
                if len(union) == k:
                    # Pruning: all (k-1) subsets must be frequent
                    is_valid = True
                    for item in union:
                        subset = union - {item}
                        if subset not in prev_frequent:
                            is_valid = False
                            break
                    if is_valid:
                        candidates.add(union)

        return list(candidates)

    def _find_frequent_itemsets(self, transactions: List[FrozenSet[int]]) -> None:
        """Find all frequent itemsets using the AprioriAssociator algorithm.

        Performs iterative level-wise search starting from frequent
        1-itemsets, generating and pruning candidates at each level,
        until no more frequent itemsets can be found.

        Parameters
        ----------
        transactions : list of frozenset of int
            The preprocessed transaction database.
        """
        n_transactions = len(transactions)
        min_count = int(np.ceil(self.min_support * n_transactions))

        # Find all unique items
        all_items = set()
        for transaction in transactions:
            all_items.update(transaction)

        # Find frequent 1-itemsets
        candidates_1 = [frozenset([item]) for item in all_items]
        counts_1 = self._count_support(transactions, candidates_1)

        frequent_1 = []
        for itemset, count in counts_1.items():
            if count >= min_count:
                support = count / n_transactions
                self._item_support[itemset] = support
                frequent_1.append(itemset)
                self.frequent_itemsets_.append(
                    FrequentItemset(itemset, support, count)
                )

        # Iteratively find larger itemsets
        prev_frequent = set(frequent_1)
        k = 2

        while prev_frequent:
            if self.max_itemset_size is not None and k > self.max_itemset_size:
                break

            # Generate candidates
            candidates = self._generate_candidates(prev_frequent, k)
            if not candidates:
                break

            # Count support
            counts = self._count_support(transactions, candidates)

            # Filter frequent itemsets
            current_frequent = set()
            for itemset, count in counts.items():
                if count >= min_count:
                    support = count / n_transactions
                    self._item_support[itemset] = support
                    current_frequent.add(itemset)
                    self.frequent_itemsets_.append(
                        FrequentItemset(itemset, support, count)
                    )

            prev_frequent = current_frequent
            k += 1

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
                        # Try to compute from single items
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

    def fit(self, X) -> "AprioriAssociator":
        """Find frequent itemsets and generate association rules.

        Processes the input transaction data to discover itemsets that exceed 
        the ``min_support`` threshold and generates rules meeting the 
        ``min_confidence`` requirement.

        Parameters
        ----------
        X : array-like or list of lists
            The transaction data. Can be a binary matrix of shape 
            (n_transactions, n_items) where 1 indicates presence, or a 
            list of transactions where each transaction is a list/set 
            of item indices or names.

        Returns
        -------
        self : AprioriAssociator
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

        # Find frequent itemsets
        self._find_frequent_itemsets(transactions)

        # Generate rules
        self._generate_rules()

        self._is_fitted = True
        return self

    def __repr__(self) -> str:
        """String representation."""
        if self._is_fitted:
            return (f"AprioriAssociator(n_itemsets={len(self.frequent_itemsets_)}, "
                   f"n_rules={len(self.rules_)}, "
                   f"min_support={self.min_support})")
        return f"AprioriAssociator(min_support={self.min_support}, min_confidence={self.min_confidence})"
