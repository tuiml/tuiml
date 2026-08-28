#!/usr/bin/env python3
"""Recompute the published headline benchmark numbers from ``summary_cv10.csv``.

Every figure quoted in ``README.md`` and on the homepage should be derivable
from the committed results file by anyone, without trusting a screenshot.
Running this script prints them.

Two methodological choices matter, and both are deliberate.

**Pairing.** A mean over all rows silently compares frameworks on different
subsets: one that crashes on the hard datasets looks better than one that
finishes them. Every number here is restricted to cells -- an
``(algorithm, dataset, seed, fold)`` tuple -- where *all three* frameworks
completed. Accuracy pairs separately from timing, because a run can produce a
fit time without a usable accuracy.

**Configuration arm.** The harness runs each algorithm twice. Under
``defaults`` every framework uses its own library defaults, which is what a
user gets on install and what the published figure reports. Under ``matched``
hyperparameters are forced to agree across frameworks, which isolates the
implementation from its default settings. The two answer different questions,
so this script prints both rather than picking one -- see the module docstring
of :mod:`figure` for which arm the published chart uses.

Usage
-----
::

    uv run python benchmarks/tools/headline.py
"""

from __future__ import annotations

import os

import pandas as pd
from scipy.stats import wilcoxon

CSV = os.path.join(os.path.dirname(__file__), "..", "summary_cv10.csv")

# One benchmark observation. Pairing on all four compares like with like:
# the same algorithm on the same fold of the same dataset.
CELL = ["algorithm", "dataset", "seed", "fold"]
FRAMEWORKS = ["tuiml", "sklearn", "weka"]


def paired(df: pd.DataFrame, value: str, config: str) -> pd.DataFrame:
    """Pivot one metric to framework columns, keeping only fully-paired cells.

    Parameters
    ----------
    df : pd.DataFrame
        Raw rows from ``summary_cv10.csv``.
    value : str
        Column to compare across frameworks, e.g. ``"metric_accuracy"``.
    config : str
        Configuration arm, ``"defaults"`` or ``"matched"``.

    Returns
    -------
    paired : pd.DataFrame
        One row per cell, one column per framework, no missing values.
    """
    sub = df[df.config == config]
    ok = sub[(sub.status == "ok") & sub[value].notna()]
    return ok.pivot_table(index=CELL, columns="framework", values=value).dropna()


def _report(df: pd.DataFrame, config: str) -> None:
    """Print accuracy and timing headlines for one configuration arm.

    Parameters
    ----------
    df : pd.DataFrame
        Raw rows from ``summary_cv10.csv``.
    config : str
        Configuration arm to summarise.
    """
    acc = paired(df, "metric_accuracy", config)
    print(f"\n=== config = {config} ===")
    print(f"ACCURACY  ({len(acc):,} paired cells)")
    for fw in FRAMEWORKS:
        lead = "  <-- best" if fw == acc.mean().idxmax() else ""
        print(f"  {fw:<9} {acc[fw].mean() * 100:.2f}%{lead}")
    gap = (acc["tuiml"].mean() - acc["sklearn"].mean()) * 100
    _, p = wilcoxon(acc["tuiml"], acc["sklearn"])
    print(f"  tuiml - sklearn = {gap:+.2f} pt   (Wilcoxon p = {p:.2g})")

    fit = paired(df, "fit_s", config)
    per_alg = fit.reset_index().groupby("algorithm")[FRAMEWORKS].sum()
    per_alg["weka/tuiml"] = per_alg.weka / per_alg.tuiml
    wins = int((per_alg["weka/tuiml"] > 1).sum())

    print(f"TRAINING TIME  ({len(fit):,} paired cells)")
    print(f"  TuiML faster than Weka on {wins}/{len(per_alg)} algorithms")
    print(f"  pooled weka/tuiml    = {per_alg.weka.sum() / per_alg.tuiml.sum():.2f}x")
    print(f"  pooled tuiml/sklearn = {per_alg.tuiml.sum() / per_alg.sklearn.sum():.2f}x slower")

    print("  per-algorithm weka/tuiml (<1 means TuiML is slower):")
    for alg, row in per_alg.sort_values("weka/tuiml", ascending=False).iterrows():
        flag = "   <-- TuiML slower" if row["weka/tuiml"] < 1 else ""
        print(f"    {alg:<20} {row['weka/tuiml']:>8.2f}x{flag}")


def main() -> None:
    """Print both configuration arms, and where they disagree."""
    df = pd.read_csv(CSV)
    for config in ("defaults", "matched"):
        _report(df, config)

    # The gap between the arms is the actionable part: an algorithm that only
    # loses once hyperparameters are forced to agree has an implementation
    # problem, not a defaults problem.
    acc = paired(df, "metric_accuracy", "matched").reset_index()
    acc["gap"] = (acc.tuiml - acc.sklearn) * 100
    by_alg = acc.groupby("algorithm").gap.mean().sort_values()
    print("\n=== matched: TuiML - scikit-learn accuracy, by algorithm (pt) ===")
    for alg, gap in by_alg.items():
        print(f"  {alg:<20} {gap:+6.2f}")


if __name__ == "__main__":
    main()
