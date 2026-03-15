"""
ARFF (Attribute-Relation File Format) parser.

Parses WEKA's ARFF file format into numpy arrays and metadata.
"""

import re
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field

@dataclass
class Attribute:
    """Represents an ARFF attribute."""
    name: str
    type: str  # 'numeric', 'nominal', 'string', 'date'
    values: Optional[List[str]] = None  # For nominal attributes
    date_format: Optional[str] = None  # For date attributes

    @property
    def is_numeric(self) -> bool:
        return self.type in ('numeric', 'real', 'integer')

    @property
    def is_nominal(self) -> bool:
        return self.type == 'nominal'

@dataclass
class ARFFData:
    """Container for parsed ARFF data."""
    relation: str
    attributes: List[Attribute]
    data: np.ndarray
    target: Optional[np.ndarray] = None
    feature_names: List[str] = field(default_factory=list)
    target_name: Optional[str] = None
    target_names: Optional[List[str]] = None

    @property
    def n_samples(self) -> int:
        return self.data.shape[0]

    @property
    def n_features(self) -> int:
        return self.data.shape[1]

def parse_arff(filepath: str, target_column: int = -1) -> ARFFData:
    """
    Parse an ARFF file.

    Parameters
    ----------
    filepath : str
        Path to the ARFF file.
    target_column : int
        Index of target column (-1 for last column, None for no target).

    Returns
    -------
    ARFFData
        ARFFData object containing the parsed data.
    """
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()

    return parse_arff_string(content, target_column)

def parse_arff_string(content: str, target_column: int = -1) -> ARFFData:
    """
    Parse ARFF content from a string.

    Parameters
    ----------
    content : str
        ARFF file content as string.
    target_column : int
        Index of target column (-1 for last, None for no target).

    Returns
    -------
    ARFFData
        ARFFData object containing the parsed data.
    """
    lines = content.split('\n')

    relation = ""
    attributes: List[Attribute] = []
    data_lines: List[str] = []
    in_data_section = False

    for line in lines:
        line = line.strip()

        # Skip empty lines and comments
        if not line or line.startswith('%'):
            continue

        line_lower = line.lower()

        if line_lower.startswith('@relation'):
            relation = _parse_relation(line)
        elif line_lower.startswith('@attribute'):
            attr = _parse_attribute(line)
            attributes.append(attr)
        elif line_lower.startswith('@data'):
            in_data_section = True
        elif in_data_section:
            data_lines.append(line)

    # Parse data
    data = _parse_data(data_lines, attributes)

    # Handle target column
    if target_column is not None and len(attributes) > 0:
        if target_column < 0:
            target_column = len(attributes) + target_column

        # Extract target
        target_attr = attributes[target_column]
        target = data[:, target_column]

        # Convert nominal target to integer labels
        if target_attr.is_nominal:
            target_names = target_attr.values
            # Create mapping
            value_to_idx = {v: i for i, v in enumerate(target_names)}
            target = np.array([
                value_to_idx.get(str(int(t)) if not np.isnan(t) else '?', -1)
                if isinstance(t, (int, float, np.floating, np.integer)) else
                value_to_idx.get(t, -1)
                for t in target
            ])
        else:
            target_names = None

        # Remove target from features
        feature_indices = [i for i in range(len(attributes)) if i != target_column]
        features = data[:, feature_indices]
        feature_names = [attributes[i].name for i in feature_indices]
        target_name = target_attr.name
    else:
        features = data
        target = None
        feature_names = [attr.name for attr in attributes]
        target_name = None
        target_names = None

    return ARFFData(
        relation=relation,
        attributes=attributes,
        data=features,
        target=target,
        feature_names=feature_names,
        target_name=target_name,
        target_names=target_names,
    )

def _parse_relation(line: str) -> str:
    """Parse @relation line.

    Returns
    -------
    str
        The relation name extracted from the line.
    """
    match = re.match(r'@relation\s+["\']?([^"\']+)["\']?', line, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return line.split(None, 1)[1] if len(line.split()) > 1 else ""

def _parse_attribute(line: str) -> Attribute:
    """Parse @attribute line.

    Returns
    -------
    Attribute
        Parsed attribute with name, type, and optional values.
    """
    # Remove @attribute prefix
    line = re.sub(r'^@attribute\s+', '', line, flags=re.IGNORECASE)

    # Handle quoted attribute names
    if line.startswith("'") or line.startswith('"'):
        quote = line[0]
        end_quote = line.find(quote, 1)
        name = line[1:end_quote]
        rest = line[end_quote + 1:].strip()
    else:
        parts = line.split(None, 1)
        name = parts[0]
        rest = parts[1] if len(parts) > 1 else ""

    rest_lower = rest.lower()

    # Determine type
    if rest_lower in ('numeric', 'real', 'integer'):
        return Attribute(name=name, type='numeric')
    elif rest_lower == 'string':
        return Attribute(name=name, type='string')
    elif rest_lower.startswith('date'):
        date_format = None
        if ' ' in rest:
            date_format = rest.split(None, 1)[1].strip('"\'')
        return Attribute(name=name, type='date', date_format=date_format)
    elif rest.startswith('{'):
        # Nominal attribute
        values_str = rest[1:rest.rfind('}')]
        values = [v.strip().strip('"\'') for v in values_str.split(',')]
        return Attribute(name=name, type='nominal', values=values)
    else:
        # Default to numeric
        return Attribute(name=name, type='numeric')

def _parse_data(data_lines: List[str], attributes: List[Attribute]) -> np.ndarray:
    """Parse data section.

    Parameters
    ----------
    data_lines : list of str
        Raw data lines from the ARFF file.
    attributes : list of Attribute
        Parsed attribute definitions used for type conversion.

    Returns
    -------
    np.ndarray
        2-D float array of shape ``(n_samples, n_attributes)``.
    """
    n_attrs = len(attributes)
    rows = []

    # Create value mappings for nominal attributes
    nominal_maps = {}
    for i, attr in enumerate(attributes):
        if attr.is_nominal and attr.values:
            nominal_maps[i] = {v: j for j, v in enumerate(attr.values)}

    for line in data_lines:
        if not line or line.startswith('%'):
            continue

        # Handle sparse format
        if line.startswith('{'):
            row = _parse_sparse_row(line, n_attrs, attributes, nominal_maps)
        else:
            row = _parse_dense_row(line, attributes, nominal_maps)

        if len(row) == n_attrs:
            rows.append(row)

    return np.array(rows, dtype=float)

def _parse_dense_row(line: str, attributes: List[Attribute],
                     nominal_maps: Dict[int, Dict[str, int]]) -> List[float]:
    """Parse a dense data row.

    Parameters
    ----------
    line : str
        Comma-separated data row.
    attributes : list of Attribute
        Attribute definitions for type lookup.
    nominal_maps : dict
        Mapping of attribute index to {value: int} for nominal attributes.

    Returns
    -------
    list of float
        Numeric values for one row, with nominals mapped to ints.
    """
    # Handle quoted values and commas
    values = []
    current = ""
    in_quotes = False
    quote_char = None

    for char in line:
        if char in ('"', "'") and not in_quotes:
            in_quotes = True
            quote_char = char
        elif char == quote_char and in_quotes:
            in_quotes = False
            quote_char = None
        elif char == ',' and not in_quotes:
            values.append(current.strip())
            current = ""
        else:
            current += char

    if current:
        values.append(current.strip())

    # Convert to numeric
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

def _parse_sparse_row(line: str, n_attrs: int, attributes: List[Attribute],
                      nominal_maps: Dict[int, Dict[str, int]]) -> List[float]:
    """Parse a sparse data row.

    Parameters
    ----------
    line : str
        Sparse-format row, e.g. ``{0 val, 3 val}``.
    n_attrs : int
        Total number of attributes (determines row length).
    attributes : list of Attribute
        Attribute definitions for type lookup.
    nominal_maps : dict
        Mapping of attribute index to {value: int} for nominal attributes.

    Returns
    -------
    list of float
        Numeric values for one row, with unspecified indices set to 0.
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

def write_arff(filepath: str, data: np.ndarray, feature_names: List[str],
               relation: str = "data", target: Optional[np.ndarray] = None,
               target_name: str = "class", target_names: Optional[List[str]] = None):
    """
    Write data to ARFF format.

    Parameters
    ----------
    filepath : str
        Output file path.
    data : np.ndarray
        Feature data of shape ``(n_samples, n_features)``.
    feature_names : list of str
        List of feature names.
    relation : str
        Relation name.
    target : np.ndarray or None
        Target values (optional).
    target_name : str
        Name of target attribute.
    target_names : list of str or None
        Names of target classes (for classification).
    """
    with open(filepath, 'w') as f:
        f.write(f"@relation {relation}\n\n")

        # Write attributes
        for name in feature_names:
            f.write(f"@attribute {name} numeric\n")

        # Write target attribute if present
        if target is not None:
            if target_names:
                values = ','.join(target_names)
                f.write(f"@attribute {target_name} {{{values}}}\n")
            else:
                f.write(f"@attribute {target_name} numeric\n")

        # Write data
        f.write("\n@data\n")
        for i in range(len(data)):
            row = ','.join(str(v) for v in data[i])
            if target is not None:
                if target_names:
                    row += f",{target_names[int(target[i])]}"
                else:
                    row += f",{target[i]}"
            f.write(row + "\n")
