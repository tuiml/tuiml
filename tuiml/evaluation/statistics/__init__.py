"""Statistical tests for model comparison.

Deciding whether one model actually beat another. A difference in mean
cross-validation score is not evidence on its own; these tests say whether it
survives the variance across folds.

Comparing two models
--------------------
- **paired_t_test:** Paired t-test over per-fold scores.
- **corrected_paired_t_test:** The Nadeau-Bengio correction, which accounts
  for the training data that cross-validation folds share. Prefer it over the
  plain t-test for CV results, whose independence assumption k-fold violates.
- **wilcoxon_signed_rank_test:** Non-parametric alternative, for scores that
  are not normally distributed.

Comparing several models
------------------------
- **friedman_test:** Rank-based omnibus test across models and datasets.
- **friedman_aligned_ranks_test:** More powerful when datasets are few.
- **quade_test:** Weights each dataset by how much the models differ on it.
- **one_way_anova:** Parametric alternative.
- **nemenyi_post_hoc:** Which pairs differ, once an omnibus test is
  significant.

Multiple-comparison corrections
-------------------------------
**bonferroni_correction**, **holm_correction**, **hochberg_correction**,
**hommel_correction** and **benjamini_hochberg**. Comparing many models
multiplies the chance of a false positive; these adjust for it.

Notes
-----
Run an omnibus test before the post-hoc. Testing every pair without one
inflates the error rate, which is exactly what the corrections above control.
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
