"""Statistical testing over benchmark results.

Deciding whether one algorithm actually beat another, rather than eyeballing
a table of means. Exposes :mod:`tuiml.evaluation.statistics` as a tool so an
agent can reach for a significance test on the results it just produced.

Tools
-----
- **tuiml_test_statistics:** Run a significance test over per-fold scores,
  keyed by algorithm name. Paired t-test and Wilcoxon for two algorithms;
  Friedman, Quade and ANOVA with post-hoc Nemenyi for more than two, plus
  the usual multiple-comparison corrections.

Notes
-----
The results it takes are the per-fold scores ``tuiml_benchmark`` returns, so
the two chain directly.
"""
