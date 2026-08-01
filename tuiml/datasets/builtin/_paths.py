"""Filesystem layout of the bundled dataset files.

The ARFF files ship inside the package under ``builtin/data/``, split into one
directory per task type. Keeping the data under ``data/`` leaves the module
names ``classification``, ``regression`` and ``other`` free for the loader
modules that sit beside this one.
"""

from pathlib import Path

_DATA_DIR = Path(__file__).parent / "data"

_CLASSIFICATION_DIR = _DATA_DIR / "classification"
_REGRESSION_DIR = _DATA_DIR / "regression"
_OTHER_DIR = _DATA_DIR / "other"

#: Category name to the directory holding that category's ARFF files.
_CATEGORY_DIRS = {
    "classification": _CLASSIFICATION_DIR,
    "regression": _REGRESSION_DIR,
    "other": _OTHER_DIR,
}


def _get_path(category: str, filename: str) -> Path:
    """Get path to a built-in dataset file.

    Parameters
    ----------
    category : str
        Dataset category: ``"classification"``, ``"regression"``, or ``"other"``.
    filename : str
        Name of the ARFF file (e.g., ``"iris.arff"``).

    Returns
    -------
    path : Path
        Resolved path to the dataset file.

    Raises
    ------
    FileNotFoundError
        If the file is not bundled with the installed package.
    """
    path = _CATEGORY_DIRS[category] / filename
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {filename}")
    return path
