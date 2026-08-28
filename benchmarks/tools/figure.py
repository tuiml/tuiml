#!/usr/bin/env python3
"""Render the four-panel benchmark summary figure from ``summary_cv10.csv``.

The published figure has to be regenerable, or its numbers drift away from the
committed data with nothing to catch it. This script is the only source of
``assets/benchmark_summary_{light,dark}.svg``.

The chart reports the **defaults** arm: every framework configured as it ships,
which is what a user gets on install. The ``matched`` arm -- hyperparameters
forced to agree across frameworks -- answers a different question and is not
plotted here; ``headline.py`` prints both, and the per-algorithm breakdown it
ends with is the one to read before quoting any single number.

Pairing follows ``headline.py``: every panel is restricted to cells where all
three frameworks completed, so no framework benefits from having crashed on the
hard datasets.

Usage
-----
::

    uv run python benchmarks/tools/figure.py
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from headline import CELL, paired  # noqa: E402

HERE = os.path.dirname(__file__)
CSV = os.path.join(HERE, "..", "summary_cv10.csv")
ASSETS = os.path.join(HERE, "..", "..", "assets")

CONFIG = "defaults"
FRAMEWORKS = ["tuiml", "sklearn", "weka"]
LABELS = {"tuiml": "TuiML", "sklearn": "scikit-learn", "weka": "Weka"}

# TuiML blue for the subject, neutral greys for the comparators: the figure
# should not flatter by colour.
BARS = {"tuiml": "#3b82f6", "sklearn": "#94a3b8", "weka": "#cbd5e1"}

THEMES = {
    "light": {"fg": "#0f172a", "muted": "#64748b"},
    "dark": {"fg": "#e2e8f0", "muted": "#94a3b8"},
}

PANELS = [
    # Two decimals on accuracy: the frameworks are within 0.1 pt of each other
    # here, and one decimal rounds two of them to the same printed number.
    ("metric_accuracy", "Accuracy", "higher is better", 100.0, "{:.2f} %"),
    ("fit_s", "Training time", "lower is better", 1.0, "{:.1f} s"),
    ("predict_s", "Inference time", "lower is better", 1.0, "{:.2f} s"),
    ("peak_rss_mb", "Peak memory", "lower is better", 1.0, "{:.0f} MB"),
]


def _render(df: pd.DataFrame, theme: str, caption: str) -> str:
    """Draw all four panels for one theme and write the SVG.

    Parameters
    ----------
    df : pd.DataFrame
        Raw rows from ``summary_cv10.csv``.
    theme : str
        Key into :data:`THEMES`.
    caption : str
        Footer line describing the sample.

    Returns
    -------
    path : str
        Path of the SVG written.
    """
    c = THEMES[theme]
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.4))
    fig.patch.set_alpha(0.0)

    for ax, (col, title, direction, scale, fmt) in zip(axes, PANELS):
        means = paired(df, col, CONFIG).mean() * scale
        ys = list(range(len(FRAMEWORKS)))
        vals = [means[f] for f in FRAMEWORKS]

        ax.barh(ys, vals, color=[BARS[f] for f in FRAMEWORKS], height=0.62)
        ax.set_yticks(ys)
        ax.set_yticklabels([LABELS[f] for f in FRAMEWORKS], fontsize=15, color=c["fg"])
        ax.invert_yaxis()
        ax.set_title(f"{title}  ·  {direction}", fontsize=16, color=c["fg"], pad=18)

        # The axis carries no ticks, so the printed value is the only quantity
        # the reader has to trust; put it just past the end of each bar.
        span = max(vals)
        for y, v in zip(ys, vals):
            ax.text(v + span * 0.03, y, fmt.format(v), va="center",
                    fontsize=14, color=c["muted"])
        ax.set_xlim(0, span * 1.28)

        ax.set_xticks([])
        ax.patch.set_alpha(0.0)
        for side in ax.spines.values():
            side.set_visible(False)
        ax.tick_params(axis="y", length=0)

    fig.text(0.5, 0.02, caption, ha="center", fontsize=13, color=c["muted"])
    fig.tight_layout(rect=(0, 0.06, 1, 1))

    path = os.path.join(ASSETS, f"benchmark_summary_{theme}.svg")
    fig.savefig(path, format="svg", transparent=True)
    plt.close(fig)
    return path


def main() -> None:
    """Write the light and dark SVGs, captioned with the sample they describe."""
    df = pd.read_csv(CSV)
    n_acc = len(paired(df, "metric_accuracy", CONFIG))
    n_time = len(paired(df, "fit_s", CONFIG))
    caption = (
        f"Library defaults · 13 algorithms × 51 TabArena datasets · 10-fold CV · "
        f"mean over cells all three frameworks completed "
        f"({n_time:,} timing, {n_acc:,} accuracy) · Weka memory includes its JVM baseline"
    )
    for theme in THEMES:
        print("wrote", os.path.relpath(_render(df, theme, caption)))


if __name__ == "__main__":
    main()
