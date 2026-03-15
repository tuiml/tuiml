"""
ARFF (Attribute-Relation File Format) loader.

WEKA's native file format for datasets.
"""

import re
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field
from pathlib import Path

@dataclass
class Dataset:
    """Container for loaded datasets with support for features and targets.

    Provides a standardized way to access data and metadata across different 
    file formats and data sources.

    Attributes
    ----------
    X : np.ndarray
        Feature matrix of shape ``(n_samples, n_features)``.
    y : np.ndarray or None, default=None
        Target values (labels) of shape ``(n_samples,)``.
    feature_names : List[str]
        List of attribute names for the features.
    target_names : List[str] or None
        List of class names for the target (for classification tasks).
    name : str, default="dataset"
        Name of the dataset, often derived from filename or @relation tag.
    description : str, default=""
        Textual description or comments included in the data file.

    Examples
    --------
    >>> from tuiml.datasets import load_iris
    >>> data = load_iris()
    >>> data.X.shape
    (150, 4)
    >>> data.feature_names
    ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    >>> df = data.to_pandas()
    >>> df.shape
    (150, 5)
    """
    X: np.ndarray
    y: Optional[np.ndarray] = None
    feature_names: List[str] = field(default_factory=list)
    target_names: Optional[List[str]] = None
    name: str = "dataset"
    description: str = ""

    @property
    def n_samples(self) -> int:
        """Total number of observations in the dataset."""
        return self.X.shape[0]

    @property
    def n_features(self) -> int:
        """Total number of input features (attributes)."""
        return self.X.shape[1]

    @property
    def shape(self) -> Tuple[int, int]:
        """Dimensionality of the feature matrix (n_samples, n_features)."""
        return self.X.shape

    def __iter__(self):
        """Allow unpacking of the dataset object.

        Allows usage like: ``X, y = dataset``
        """
        yield self.X
        yield self.y

    def __repr__(self) -> str:
        return (
            f"Dataset(name='{self.name}', n_samples={self.n_samples}, "
            f"n_features={self.n_features})"
        )

    def to_pandas(self, include_target: bool = True):
        """Convert Dataset to pandas DataFrame."""
        from tuiml.datasets.loaders.pandas import to_pandas
        return to_pandas(self, include_target=include_target)

def load_arff(
    filepath: Union[str, Path],
    target_column: int = -1
) -> Dataset:
    """Load data from ARFF (Attribute-Relation File Format) files.

    ARFF is the native format for WEKA, supporting rich metadata, dense 
    and sparse data, and explicit type declarations.

    Parameters
    ----------
    filepath : Union[str, Path]
        Path to the ARFF file to be loaded.
    target_column : int, default=-1
        The index of the column to treat as the target variable:
        
        - ``-1`` — Use the last column (standard for most ARFFs)
        - ``int`` — Use specific zero-based index
        - ``None`` — Do not extract a target (X will contain all columns)

    Returns
    -------
    Dataset
        Standardized dataset object containing data and metadata.

    Examples
    --------
    >>> from tuiml.datasets.loaders import load_arff
    >>> data = load_arff('iris.arff')
    >>> X, y = data
    >>> print(X.shape)
    (150, 4)
    """
    filepath = Path(filepath)

    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()

    return _parse_arff(content, target_column, filepath.stem)

def _parse_arff(
    content: str,
    target_column: int = -1,
    name: str = "dataset"
) -> Dataset:
    """Parse raw ARFF text content into a Dataset object.

    Parameters
    ----------
    content : str
        Raw text content of an ARFF file.
    target_column : int, default=-1
        Index of the column to use as target. Use -1 for the last column,
        or None to skip target extraction.
    name : str, default="dataset"
        Fallback name for the dataset if no @relation is found.

    Returns
    -------
    Dataset
        Parsed dataset with features, target, and metadata.
    """
    lines = content.split('\n')

    relation = name
    attributes = []  # [(name, type, values)]
    data_lines = []
    in_data = False
    description = []

    for line in lines:
        line_stripped = line.strip()

        if not line_stripped:
            continue

        if line_stripped.startswith('%'):
            if not in_data:
                description.append(line_stripped[1:].strip())
            continue

        lower = line_stripped.lower()

        if lower.startswith('@relation'):
            relation = _extract_value(line_stripped, '@relation')
        elif lower.startswith('@attribute'):
            attr = _parse_attribute(line_stripped)
            attributes.append(attr)
        elif lower.startswith('@data'):
            in_data = True
        elif in_data:
            data_lines.append(line_stripped)

    # Parse data
    data, nominal_maps = _parse_data(data_lines, attributes)

    # Handle target
    if target_column is not None and len(attributes) > 0:
        if target_column < 0:
            target_column = len(attributes) + target_column

        target_attr = attributes[target_column]
        target = data[:, target_column].astype(int)

        # Get target names for nominal
        target_names = target_attr[2] if target_attr[1] == 'nominal' else None

        # Remove target from features
        feature_idx = [i for i in range(len(attributes)) if i != target_column]
        features = data[:, feature_idx]
        feature_names = [attributes[i][0] for i in feature_idx]
    else:
        features = data
        target = None
        feature_names = [a[0] for a in attributes]
        target_names = None

    return Dataset(
        X=features,
        y=target,
        feature_names=feature_names,
        target_names=target_names,
        name=relation,
        description='\n'.join(description)
    )

def _extract_value(line: str, keyword: str) -> str:
    """Extract the value following an ARFF keyword on a header line.

    Parameters
    ----------
    line : str
        Full ARFF header line (e.g., ``@relation iris``).
    keyword : str
        The ARFF keyword to strip (e.g., ``'@relation'``).

    Returns
    -------
    str
        The extracted value, with surrounding quotes removed if present.
    """
    rest = line[len(keyword):].strip()
    if rest.startswith("'") or rest.startswith('"'):
        quote = rest[0]
        end = rest.find(quote, 1)
        return rest[1:end]
    return rest.split()[0] if rest else ""

def _parse_attribute(line: str) -> Tuple[str, str, Optional[List[str]]]:
    """Parse an @attribute line into name, type, and nominal values.

    Parameters
    ----------
    line : str
        A single @attribute declaration line from an ARFF file.

    Returns
    -------
    tuple of (str, str, list of str or None)
        A tuple of ``(name, type, values)`` where *type* is one of
        ``'numeric'``, ``'string'``, or ``'nominal'``, and *values* is a
        list of nominal class labels or None.
    """
    rest = line[len('@attribute'):].strip()

    # Extract name (may be quoted)
    if rest.startswith("'") or rest.startswith('"'):
        quote = rest[0]
        end = rest.find(quote, 1)
        name = rest[1:end]
        rest = rest[end + 1:].strip()
    else:
        parts = rest.split(None, 1)
        name = parts[0]
        rest = parts[1] if len(parts) > 1 else ""

    rest_lower = rest.lower()

    if rest_lower in ('numeric', 'real', 'integer'):
        return (name, 'numeric', None)
    elif rest_lower == 'string':
        return (name, 'string', None)
    elif rest.startswith('{'):
        values_str = rest[1:rest.rfind('}')]
        values = [v.strip().strip('"\'') for v in values_str.split(',')]
        return (name, 'nominal', values)
    else:
        return (name, 'numeric', None)

def _parse_data(
    data_lines: List[str],
    attributes: List[Tuple]
) -> Tuple[np.ndarray, Dict[int, Dict[str, int]]]:
    """Parse the @data section of an ARFF file into a numeric array.

    Parameters
    ----------
    data_lines : list of str
        Lines from the @data section (dense or sparse format).
    attributes : list of tuple
        Attribute definitions as returned by ``_parse_attribute``.

    Returns
    -------
    data : np.ndarray
        Numeric array of shape ``(n_samples, n_attributes)``.
    nominal_maps : dict of {int: dict of {str: int}}
        Mapping from attribute index to nominal-value-to-integer encoding.
    """
    n_attrs = len(attributes)
    rows = []

    # Create nominal mappings
    nominal_maps = {}
    for i, (name, atype, values) in enumerate(attributes):
        if atype == 'nominal' and values:
            nominal_maps[i] = {v: j for j, v in enumerate(values)}

    for line in data_lines:
        if not line or line.startswith('%'):
            continue

        if line.startswith('{'):
            row = _parse_sparse(line, n_attrs, nominal_maps)
        else:
            row = _parse_dense(line, nominal_maps)

        if len(row) == n_attrs:
            rows.append(row)

    return np.array(rows, dtype=float), nominal_maps

def _parse_dense(line: str, nominal_maps: Dict) -> List[float]:
    """Parse a single dense-format data line into a list of floats.

    Parameters
    ----------
    line : str
        Comma-separated data line from the @data section.
    nominal_maps : dict
        Nominal-value-to-integer mappings keyed by attribute index.

    Returns
    -------
    list of float
        Numeric values for each attribute, with nominals encoded and
        missing values represented as ``np.nan``.
    """
    values = []
    current = ""
    in_quotes = False

    for char in line:
        if char in ('"', "'") and not in_quotes:
            in_quotes = True
        elif char in ('"', "'") and in_quotes:
            in_quotes = False
        elif char == ',' and not in_quotes:
            values.append(current.strip())
            current = ""
        else:
            current += char
    if current:
        values.append(current.strip())

    row = []
    for i, val in enumerate(values):
        val = val.strip().strip('"\'')
        if val == '?' or val == '':
            row.append(np.nan)
        elif i in nominal_maps:
            row.append(nominal_maps[i].get(val, np.nan))
        else:
            try:
                row.append(float(val))
            except ValueError:
                row.append(np.nan)
    return row

def _parse_sparse(line: str, n_attrs: int, nominal_maps: Dict) -> List[float]:
    """Parse a single sparse-format data line into a list of floats.

    Parameters
    ----------
    line : str
        Sparse data line in the form ``{index value, ...}``.
    n_attrs : int
        Total number of attributes (determines output length).
    nominal_maps : dict
        Nominal-value-to-integer mappings keyed by attribute index.

    Returns
    -------
    list of float
        Numeric values for each attribute, defaulting to ``0.0`` for
        indices not present in the sparse representation.
    """
    row = [0.0] * n_attrs
    content = line[1:line.rfind('}')]

    for item in content.split(','):
        item = item.strip()
        if not item:
            continue
        parts = item.split(None, 1)
        if len(parts) == 2:
            idx = int(parts[0])
            val = parts[1].strip().strip('"\'')
            if val == '?':
                row[idx] = np.nan
            elif idx in nominal_maps:
                row[idx] = nominal_maps[idx].get(val, np.nan)
            else:
                try:
                    row[idx] = float(val)
                except ValueError:
                    row[idx] = np.nan
    return row

def save_arff(
    filepath: Union[str, Path],
    data: np.ndarray,
    feature_names: Optional[List[str]] = None,
    target: Optional[np.ndarray] = None,
    target_names: Optional[List[str]] = None,
    relation: str = "data"
):
    """Save data to ARFF (Attribute-Relation File Format).

    Parameters
    ----------
    filepath : Union[str, Path]
        Output path where the ARFF file will be saved.
    data : np.ndarray
        Feature matrix to save.
    feature_names : List[str] or None, default=None
        Names for the features. If None, generic names like ``attr0``, ``attr1`` 
        will be used.
    target : np.ndarray or None, default=None
        Target values to include in the file.
    target_names : List[str] or None, default=None
        Names for target classes (for nominal attributes).
    relation : str, default="data"
        Name of the relation (dataset name) in the ARFF header.

    Examples
    --------
    >>> import numpy as np
    >>> from tuiml.datasets.loaders import save_arff
    >>> X = np.random.rand(10, 2)
    >>> save_arff('output.arff', X, feature_names=['x1', 'x2'])
    """
    filepath = Path(filepath)

    if feature_names is None:
        feature_names = [f"attr{i}" for i in range(data.shape[1])]

    with open(filepath, 'w') as f:
        f.write(f"@relation {relation}\n\n")

        for name in feature_names:
            f.write(f"@attribute {name} numeric\n")

        if target is not None:
            if target_names:
                values = ','.join(target_names)
                f.write(f"@attribute class {{{values}}}\n")
            else:
                f.write("@attribute target numeric\n")

        f.write("\n@data\n")

        for i in range(len(data)):
            row = ','.join(str(v) for v in data[i])
            if target is not None:
                if target_names:
                    row += f",{target_names[int(target[i])]}"
                else:
                    row += f",{target[i]}"
            f.write(row + "\n")
