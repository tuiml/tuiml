"""
Statistical tests for model comparison.
- PairedTTester.java
- PairedStats.java

This module provides:
- Parametric tests: t-test, corrected t-test
- Non-parametric tests: Wilcoxon, Friedman, Nemenyi
- Multiple comparison corrections: Bonferroni, Holm
"""

from .parametric import (
    paired_t_test,
    corrected_paired_t_test,
    one_way_anova,
    PairedStats,
    SignificanceLevel,
)
from .nonparametric import (
    wilcoxon_signed_rank_test,
    friedman_test,
    nemenyi_post_hoc,
    friedman_aligned_ranks_test,
    quade_test,
)
from .corrections import (
    bonferroni_correction,
    holm_correction,
    hochberg_correction,
    hommel_correction,
    benjamini_hochberg,
)

__all__ = [
    # Parametric
    "paired_t_test",
    "corrected_paired_t_test",
    "one_way_anova",
    "PairedStats",
    "SignificanceLevel",
    # Non-parametric
    "wilcoxon_signed_rank_test",
    "friedman_test",
    "nemenyi_post_hoc",
    "friedman_aligned_ranks_test",
    "quade_test",
    # Corrections
    "bonferroni_correction",
    "holm_correction",
    "hochberg_correction",
    "hommel_correction",
    "benjamini_hochberg",
]
