# Autonomous ML research with TuiML

Use this when the request is a research question ("does X beat Y", "is
bagging shallow trees competitive", "write a paper on ..."), an algorithm
idea to implement and test, or a reproducibility study. The output is a
set of experiments, a statistics table, and a paper or report that a
reviewer could check line by line.

## Contents
- Protocol checklist
- Step 1: frame the question
- Step 2: design the experiment
- Step 3: run it
- Step 4: analyse
- Step 5: write the paper
- Step 6: self-review and iterate
- Reproducibility manifest

## Protocol checklist

Copy this into the response and tick items as they complete:

```
Research progress:
- [ ] 1. Question, hypothesis, and what result would falsify it
- [ ] 2. Design: baselines, datasets, metric, CV scheme, seeds, budget
- [ ] 3. Run: every configuration with the same seeds and folds
- [ ] 4. Analyse: ranking, Friedman + post-hoc, effect sizes, failure cases
- [ ] 5. Paper: draft every section from the tables, not from memory
- [ ] 6. Self-review against the rubric, fix, re-review
- [ ] 7. Export the notebook and list every number's source
```

## Step 1: frame the question

State in one sentence each: the question, the hypothesis, the claim that
would appear in the abstract if true, and the observation that would
refute it. Say which of these the experiment can actually decide. If the
user gave an algorithm idea, restate it as a testable difference from the
nearest existing algorithm (`tuiml_describe` the neighbour first).

## Step 2: design the experiment

- **Baselines**: at least one trivial (`ZeroRuleClassifier`), one strong classical (`RandomForestClassifier`, `SVC` with `StandardScaler`), one boosting (`XGBoostClassifier`). Regression: `LinearRegression`, `RandomForestRegressor`, `XGBoostRegressor`.
- **Datasets**: five or more with different sizes and class balances. Built-ins that work for classification: `iris`, `glass`, `diabetes`, `breast-cancer`, `credit-g`, `ionosphere`, `vote`, `soybean`, `hypothyroid`. Regression: `cpu`, or upload files. Never claim generality from one dataset.
- **Evaluation**: `cv=10`, repeated with three seeds (`random_seed` 1, 2, 3) when the budget allows. Report mean ± std over folds.
- **Metric**: one primary metric decided before running (`accuracy_score` for balanced, `f1_score` or `balanced_accuracy_score` for imbalanced, `root_mean_squared_error` for regression). Secondary metrics are reported, not used to pick winners.
- **Tuning**: if any method is tuned, tune every method with the same budget via `tuiml_tune` on training folds only. Untuned-vs-tuned comparisons are not a result.
- **Budget**: cap total runs; write the number down before starting.

Record the design as a table in the response before running anything.

## Step 3: run it

```
tuiml_profile_data(data=<each dataset>)                 # sizes, class balance, missing values
tuiml_get_skeleton(kind="classifier")                   # only if implementing a new idea
tuiml_create_algorithm(name=..., kind=..., code=..., version="1.0.0")
tuiml_benchmark(algorithms=[...all methods...], data=[...all datasets...],
                cv=10, metrics=[primary, secondary...], random_seed=<seed>)
```

One `tuiml_benchmark` per seed keeps folds identical across methods. Save
each result dict in the response verbatim; the paper's tables are built
from them, not re-run. If a method fails on a dataset, record the failure
as a cell, do not drop the dataset.

Iterating on a custom algorithm: `tuiml_read_algorithm` → `tuiml_edit_algorithm(..., bump_version=True)`
→ re-benchmark with both pinned versions (`Name_v1_0_0`, `Name_v1_0_1`) in
the same call, so the improvement is measured on identical folds.

## Step 4: analyse

```
tuiml_test_statistics(results=<benchmark result>, test="friedman", metric="accuracy_score")
tuiml_test_statistics(results=<benchmark result>, test="nemenyi", metric="accuracy_score")
tuiml_plot(plot_type="cd_diagram", benchmark_results=<benchmark result>, metric="accuracy_score")
tuiml_plot(plot_type="boxplot_comparison", benchmark_results=<benchmark result>, metric="accuracy_score")
```

Both tools accept the `tuiml_benchmark` result as returned. They reduce it
to `{algorithm: [scores]}`: per-fold scores for one dataset, one mean per
dataset for several (Friedman blocks = datasets). To build that dict by
hand: `results[dataset][algorithm][metric]["scores"]`.

Report: mean rank per method, Friedman p-value, which pairwise differences
survive Nemenyi at α = 0.05, and the largest absolute gap in the primary
metric. Then look for the pattern in where the new method loses; that
paragraph is usually the most useful one in the paper. A difference that
is not significant is reported as "no evidence of a difference", never
as "comparable" or "competitive".

## Step 5: write the paper

Structure (NeurIPS/ICML style, 6 to 8 pages):

1. **Abstract**: problem, method in one sentence, the primary result with a number, the caveat.
2. **Introduction**: the question, why existing answers are insufficient, contributions as a bulleted list where each bullet maps to a section and a table.
3. **Related work**: nearest three or four families of methods; state what is different, not that they exist.
4. **Method**: the algorithm as pseudocode, the hyperparameters, complexity. If the source was created with `tuiml_create_algorithm`, the pseudocode must match that source.
5. **Experimental setup**: the Step 2 table, verbatim. Seeds, folds, budget, hardware, TuiML version from `tuiml_system_info`.
6. **Results**: one main table (datasets × methods, mean ± std, best in bold), the CD diagram, the statistics paragraph from Step 4.
7. **Analysis**: where it wins, where it loses, why, one ablation.
8. **Limitations** and **Conclusion**: what the evidence does not show.
9. **Reproducibility statement**: the notebook path and the seeds.

Writing rules: every number in the text appears in a table; every claim in
the abstract appears in Results; use the exact algorithm names; past tense
for what was done, present tense for what the results show; no adjectives
in place of numbers ("substantially better" → "+3.1 points mean accuracy,
p = 0.004").

## Step 6: self-review and iterate

Review the draft as a sceptical reviewer, answering each item with
evidence or a fix:

```
Review rubric:
- [ ] Is every abstract claim backed by a table cell and a p-value?
- [ ] Were all methods given the same tuning budget and the same folds?
- [ ] Are there at least five datasets, and are the losses discussed?
- [ ] Is the primary metric the one chosen before running?
- [ ] Do seeds, folds and TuiML version appear in the setup?
- [ ] Could a reader re-run every table from the exported notebook alone?
- [ ] Is anything described that was not actually run?
- [ ] Does the limitations section name the weakest result?
```

Any "no" is a required change. Fix it, then re-run the rubric. Two clean
passes end the loop. If a fix needs more experiments, go back to Step 3
with the same seeds and extend the tables rather than replacing them.

## Reproducibility manifest

Finish with `tuiml_export_notebook(path="<study>.ipynb", title=...)` and a
list mapping each table and figure to the tool call that produced it
(tool, arguments, seed). The manifest goes in the appendix of the paper.
