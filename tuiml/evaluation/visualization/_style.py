"""
Shared matplotlib stylesheet backing every plot in ``tuiml.evaluation.visualization``.

Private module (maintainer-facing): the public API re-exports ``apply_style``,
``reset_style``, ``get_colors``, ``setup_figure``, ``style_axis``, ``PALETTES``
and ``SEMANTIC_COLORS`` from :mod:`tuiml.evaluation.visualization`.

What lives here:

- ``PALETTES`` — four named categorical colour cycles (``'default'``,
  ``'vibrant'``, ``'muted'``, ``'scientific'``), each a list of hex strings.
- ``SEMANTIC_COLORS`` — role-based tokens (``'primary'``, ``'danger'``,
  ``'text'``, ``'grid'``, …) so plots reference *meaning*, not a hex literal.
- ``STYLE_CONFIG`` — the rcParams dict applied by :func:`apply_style`
  (sans-serif fonts, bold labels, top/right spines removed, 300-dpi savefig).
- Helpers to build a styled figure (:func:`setup_figure`) and to finish an axis
  consistently (:func:`style_axis`, :func:`annotate_bars`).

Side effects to be aware of when maintaining this module:

1. Importing it calls :func:`apply_style` once, mutating the **global**
   ``matplotlib.rcParams``. Anything that imports ``tuiml.evaluation.visualization``
   therefore inherits TuiML styling. :func:`reset_style` undoes it.
2. matplotlib is imported lazily behind ``try/except ImportError``; the module
   still imports without it (``HAS_MATPLOTLIB is False``). :func:`apply_style`
   and :func:`reset_style` then no-op, while :func:`setup_figure` raises
   :exc:`ImportError`.

Notes
-----
Every plotting function in the package calls :func:`apply_style` (directly or
via :func:`setup_figure`) on entry, so palette changes made by a caller are
overwritten unless passed through the function's own arguments.
"""

import numpy as np
from typing import Optional

try:
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# Modern color palettes
PALETTES = {
    'default': [
        '#4C72B0',  # Steel blue
        '#DD8452',  # Coral
        '#55A868',  # Sage green
        '#C44E52',  # Brick red
        '#8172B3',  # Lavender
        '#937860',  # Taupe
        '#DA8BC3',  # Pink
        '#8C8C8C',  # Gray
        '#CCB974',  # Olive
        '#64B5CD',  # Sky blue
    ],
    'vibrant': [
        '#0077B6',  # Deep blue
        '#E63946',  # Red
        '#2A9D8F',  # Teal
        '#E9C46A',  # Yellow
        '#9B2335',  # Burgundy
        '#264653',  # Dark teal
        '#F4A261',  # Orange
        '#A855F7',  # Purple
    ],
    'muted': [
        '#6B7280',  # Gray
        '#3B82F6',  # Blue
        '#10B981',  # Emerald
        '#F59E0B',  # Amber
        '#EF4444',  # Red
        '#8B5CF6',  # Violet
        '#EC4899',  # Pink
        '#14B8A6',  # Teal
    ],
    'scientific': [
        '#1f77b4',  # Blue
        '#ff7f0e',  # Orange
        '#2ca02c',  # Green
        '#d62728',  # Red
        '#9467bd',  # Purple
        '#8c564b',  # Brown
        '#e377c2',  # Pink
        '#7f7f7f',  # Gray
    ],
}

# Semantic color tokens for consistent, professional visuals
SEMANTIC_COLORS = {
    'primary': '#2563EB',
    'secondary': '#0F766E',
    'success': '#16A34A',
    'warning': '#D97706',
    'danger': '#DC2626',
    'neutral': '#6B7280',
    'text': '#1F2937',
    'muted_text': '#6B7280',
    'grid': '#E5E7EB',
    'border': '#D1D5DB',
}

# Default style configuration
STYLE_CONFIG = {
    # Figure
    'figure.facecolor': 'white',
    'figure.edgecolor': 'white',
    'figure.dpi': 100,
    'figure.autolayout': True,

    # Font
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica Neue', 'DejaVu Sans', 'sans-serif'],
    'font.size': 12,
    'font.weight': 'medium',

    # Axes
    'axes.facecolor': 'white',
    'axes.edgecolor': '#111111',
    'axes.linewidth': 1.5,
    'axes.grid': False,
    'axes.titlesize': 16,
    'axes.titleweight': 'bold',
    'axes.titlepad': 14,
    'axes.labelsize': 14,
    'axes.labelweight': 'bold',
    'axes.labelpad': 10,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.labelcolor': SEMANTIC_COLORS['text'],
    'axes.titlecolor': SEMANTIC_COLORS['text'],
    'axes.prop_cycle': None,  # Set dynamically

    # Grid
    'grid.color': SEMANTIC_COLORS['grid'],
    'grid.linewidth': 0.8,
    'grid.alpha': 0.8,
    'grid.linestyle': '-',

    # Ticks
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'xtick.color': SEMANTIC_COLORS['text'],
    'ytick.color': SEMANTIC_COLORS['text'],
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'xtick.major.size': 6,
    'ytick.major.size': 6,
    'xtick.major.width': 1.2,
    'ytick.major.width': 1.2,
    'xtick.minor.visible': False,
    'ytick.minor.visible': False,

    # Legend
    'legend.fontsize': 12,
    'legend.frameon': True,
    'legend.framealpha': 0.9,
    'legend.facecolor': 'white',
    'legend.edgecolor': '#999999',
    'legend.borderpad': 0.5,
    'legend.labelspacing': 0.4,

    # Lines
    'lines.linewidth': 2.5,
    'lines.markersize': 8,
    'lines.markeredgewidth': 0,

    # Patches (bars, etc.)
    'patch.linewidth': 1.0,
    'patch.edgecolor': 'white',

    # Savefig
    'savefig.dpi': 300,
    'savefig.facecolor': 'white',
    'savefig.edgecolor': 'white',
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
}


def apply_style(palette: str = 'default', dark_mode: bool = False) -> None:
    """
    Install the TuiML stylesheet into the global matplotlib rcParams.

    Writes every entry of ``STYLE_CONFIG`` into ``matplotlib.rcParams``, then
    sets ``axes.prop_cycle`` from the requested palette. Unknown or rejected
    rcParams keys are skipped silently, so the call is safe across matplotlib
    versions.

    Parameters
    ----------
    palette : {'default', 'vibrant', 'muted', 'scientific'}, default='default'
        Categorical colour cycle to install. An unrecognised name falls back to
        ``'default'`` rather than raising.
    dark_mode : bool, default=False
        If True, overlay dark figure/axes/text/grid colours on top of the base
        configuration.

    Returns
    -------
    None
        The function mutates global state and returns nothing.

    Notes
    -----
    This is a **global, process-wide side effect** — every subsequent matplotlib
    figure is affected, not just TuiML plots. Call :func:`reset_style` to
    restore matplotlib's own defaults. If matplotlib is not installed the call
    returns immediately without error.

    See Also
    --------
    reset_style : Undo this, restoring matplotlib defaults.
    setup_figure : Apply the style *and* create a figure in one call.

    Examples
    --------
    >>> from tuiml.evaluation.visualization import apply_style, reset_style
    >>> apply_style(palette='vibrant', dark_mode=True)
    >>> reset_style()
    """
    if not HAS_MATPLOTLIB:
        return

    # Get color palette
    colors = PALETTES.get(palette, PALETTES['default'])

    # Apply base configuration
    for key, value in STYLE_CONFIG.items():
        if value is not None and key != 'axes.prop_cycle':
            try:
                mpl.rcParams[key] = value
            except (KeyError, ValueError):
                pass

    # Set color cycle
    mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=colors)

    # Dark mode adjustments
    if dark_mode:
        dark_config = {
            'figure.facecolor': '#1F2937',
            'figure.edgecolor': '#1F2937',
            'axes.facecolor': '#1F2937',
            'axes.edgecolor': '#9CA3AF',
            'axes.labelcolor': '#F3F4F6',
            'text.color': '#F3F4F6',
            'xtick.color': '#9CA3AF',
            'ytick.color': '#9CA3AF',
            'grid.color': '#374151',
            'legend.facecolor': '#374151',
            'legend.edgecolor': '#4B5563',
            'savefig.facecolor': '#1F2937',
            'savefig.edgecolor': '#1F2937',
        }
        for key, value in dark_config.items():
            try:
                mpl.rcParams[key] = value
            except (KeyError, ValueError):
                pass


def reset_style() -> None:
    """Restore matplotlib's factory rcParams, discarding the TuiML style.

    Returns
    -------
    None
        Mutates global ``matplotlib.rcParams``; no-ops if matplotlib is absent.

    See Also
    --------
    apply_style : Re-install the TuiML stylesheet.
    """
    if HAS_MATPLOTLIB:
        mpl.rcdefaults()


def get_colors(n: int = None, palette: str = 'default') -> list:
    """
    Draw ``n`` colours from a named palette, cycling when more are requested.

    Parameters
    ----------
    n : int, optional
        Number of colours to return. If None, the whole palette is returned
        (the caller gets the module's own list object, so do not mutate it).
        When ``n`` exceeds the palette length the colours repeat from the start.
    palette : {'default', 'vibrant', 'muted', 'scientific'}, default='default'
        Palette name; an unknown name falls back to ``'default'``.

    Returns
    -------
    colors : list of str
        Hex colour strings such as ``'#4C72B0'``, of length ``n``.

    Examples
    --------
    >>> from tuiml.evaluation.visualization import get_colors
    >>> get_colors(3, palette='scientific')
    ['#1f77b4', '#ff7f0e', '#2ca02c']
    """
    colors = PALETTES.get(palette, PALETTES['default'])
    if n is None:
        return colors
    if n <= len(colors):
        return colors[:n]
    # Cycle colors if more are needed
    return [colors[i % len(colors)] for i in range(n)]


def setup_figure(
    figsize: tuple = (10, 6),
    palette: str = 'default',
    dark_mode: bool = False,
    style: str = None
) -> tuple:
    """
    Apply the TuiML style and create a single-axes figure in one call.

    Equivalent to :func:`apply_style` followed by ``plt.subplots(figsize=...)``.
    This is the entry point every TuiML plotting function uses, which is why all
    of them share the same fonts, colour cycle and spine treatment.

    Parameters
    ----------
    figsize : tuple of (float, float), default=(10, 6)
        Figure size ``(width, height)`` in inches.
    palette : {'default', 'vibrant', 'muted', 'scientific'}, default='default'
        Categorical colour cycle installed before the figure is created.
    dark_mode : bool, default=False
        Use the dark colour overlay.
    style : str, optional
        Name of an extra matplotlib style sheet layered on top (e.g.
        ``'seaborn-v0_8-whitegrid'``). An unknown name is ignored silently.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The newly created figure.
    ax : matplotlib.axes.Axes
        Its single axes.

    Raises
    ------
    ImportError
        If matplotlib is not installed.

    Notes
    -----
    Also mutates global rcParams (see :func:`apply_style`). The figure is
    created but never shown or closed — the caller owns it.

    Examples
    --------
    >>> from tuiml.evaluation.visualization import setup_figure
    >>> fig, ax = setup_figure(figsize=(6, 4))          # doctest: +SKIP
    >>> _ = ax.plot([1, 2, 3], [1, 4, 9])               # doctest: +SKIP
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for plotting")

    # Apply base style
    apply_style(palette=palette, dark_mode=dark_mode)

    # Apply additional style if specified
    if style:
        try:
            plt.style.use(style)
        except (OSError, ValueError):
            pass

    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


def style_axis(
    ax,
    title: str = None,
    xlabel: str = None,
    ylabel: str = None,
    legend: bool = True,
    legend_loc: str = 'best',
    grid: bool = False,
    despine: bool = True,
) -> None:
    """
    Finish an axis: titles, grid, spines and legend, all in the house style.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to restyle, modified in place.
    title : str, optional
        Axis title. **Title-cased before display** (``str.title()``), so
        acronyms such as ``'ROC Curve'`` come out as ``'Roc Curve'``.
    xlabel : str, optional
        X-axis label; also title-cased.
    ylabel : str, optional
        Y-axis label; also title-cased.
    legend : bool, default=True
        Draw a legend, but only if the axis already has labelled artists.
    legend_loc : str, default='best'
        Any matplotlib legend location string, e.g. ``'lower right'``.
    grid : bool, default=False
        Show grid lines using ``STYLE_CONFIG``'s grid colour, width and alpha.
        Explicitly turned off when False.
    despine : bool, default=True
        Hide the top and right spines.

    Returns
    -------
    None
        ``ax`` is modified in place.
    """
    if title:
        ax.set_title(title.title(), fontsize=16, fontweight='bold', pad=14)
    if xlabel:
        ax.set_xlabel(xlabel.title(), fontsize=14, fontweight='bold')
    if ylabel:
        ax.set_ylabel(ylabel.title(), fontsize=14, fontweight='bold')

    if grid:
        ax.grid(
            True,
            alpha=STYLE_CONFIG['grid.alpha'],
            linewidth=STYLE_CONFIG['grid.linewidth'],
            color=STYLE_CONFIG['grid.color'],
        )
    else:
        ax.grid(False)

    if despine:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    if legend:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc=legend_loc, framealpha=0.95, edgecolor='#999999')


def annotate_bars(
    ax,
    bars,
    fmt: str = '.2f',
    offset: float = 3,
    fontsize: int = 11,
    color: str = '#111111',
) -> None:
    """
    Write each bar's height as a text label just above the bar.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes holding the bars; annotations are added in place.
    bars : matplotlib.container.BarContainer
        The container returned by ``ax.bar(...)``. Any iterable of
        ``Rectangle`` patches works.
    fmt : str, default='.2f'
        Format spec applied to each bar height, e.g. ``'.1%'`` or ``'d'``.
    offset : float, default=3
        Vertical offset in typographic points between the bar top and the text.
    fontsize : int, default=11
        Font size of the annotation text.
    color : str, default='#111111'
        Text colour.

    Returns
    -------
    None
        ``ax`` is modified in place.

    Notes
    -----
    Labels are anchored at the top of the bar, so negative bars get their label
    inside the bar rather than below it.
    """
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f'{height:{fmt}}',
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, offset),
            textcoords='offset points',
            ha='center',
            va='bottom',
            fontsize=fontsize,
            color=color,
        )


# Auto-apply style when module is imported
if HAS_MATPLOTLIB:
    apply_style()
