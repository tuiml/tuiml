"""Convenience loaders for bundled datasets outside classification and regression.

Covers the two remaining task types shipped with TuiML: market-basket data for
association rule mining (:func:`load_supermarket`) and the Reuters text
collections (:func:`load_reuters_corn`, :func:`load_reuters_grain`), which come
as separate train and test files.

Every loader here is also reachable by name through
:func:`~tuiml.datasets.builtin.catalog.load_dataset`.

Examples
--------
>>> from tuiml.datasets import load_supermarket, load_reuters_corn
>>> transactions = load_supermarket()
>>> test = load_reuters_corn(split='test')
"""

from tuiml.datasets.builtin._paths import _get_path
from tuiml.datasets.loaders import Dataset, load_arff


def load_supermarket() -> Dataset:
    """Load the Supermarket dataset.

    4627 samples, 217 features (for association rule mining).

    Returns
    -------
    dataset : Dataset
        Supermarket transaction dataset for association rules.
    """
    return load_arff(_get_path("other", "supermarket.arff"))


def load_reuters_corn(split: str = 'train') -> Dataset:
    """Load the Reuters Corn dataset.

    Parameters
    ----------
    split : str, default='train'
        Which split to load: ``'train'`` or ``'test'``.

    Returns
    -------
    dataset : Dataset
        Reuters corn text classification dataset.
    """
    return load_arff(_get_path("other", f"ReutersCorn-{split}.arff"))


def load_reuters_grain(split: str = 'train') -> Dataset:
    """Load the Reuters Grain dataset.

    Parameters
    ----------
    split : str, default='train'
        Which split to load: ``'train'`` or ``'test'``.

    Returns
    -------
    dataset : Dataset
        Reuters grain text classification dataset.
    """
    return load_arff(_get_path("other", f"ReutersGrain-{split}.arff"))


__all__ = [
    "load_supermarket",
    "load_reuters_corn",
    "load_reuters_grain",
]
