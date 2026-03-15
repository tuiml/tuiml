"""Tests for the ARFF parser (tuiml.datasets._core.arff_parser)."""

import importlib
import sys
import numpy as np
import pytest
from pathlib import Path

# The _core/__init__.py has a broken import (ArffParser does not exist).
# We import the arff_parser module directly to bypass the __init__.py.
_arff_parser_path = (
    Path(__file__).resolve().parents[2]
    / "tuiml"
    / "datasets"
    / "_core"
    / "arff_parser.py"
)
_spec = importlib.util.spec_from_file_location(
    "arff_parser", str(_arff_parser_path)
)
_arff_parser = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_arff_parser)

parse_arff_string = _arff_parser.parse_arff_string
parse_arff = _arff_parser.parse_arff
Attribute = _arff_parser.Attribute
ARFFData = _arff_parser.ARFFData
_parse_relation = _arff_parser._parse_relation
_parse_attribute = _arff_parser._parse_attribute
write_arff = _arff_parser.write_arff


# ===================================================================
# parse_arff_string tests
# ===================================================================

class TestParseArffString:
    """Tests for parsing ARFF content from strings."""

    def test_parse_simple_numeric(self):
        """Parse a minimal ARFF with numeric attributes only."""
        content = """\
@relation test_data

@attribute x1 numeric
@attribute x2 numeric

@data
1.0,2.0
3.0,4.0
5.0,6.0
"""
        result = parse_arff_string(content, target_column=None)
        assert isinstance(result, ARFFData)
        assert result.relation == "test_data"
        assert result.n_samples == 3
        assert result.n_features == 2
        assert result.target is None
        np.testing.assert_array_almost_equal(
            result.data, [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        )

    def test_parse_with_nominal_target(self):
        """Parse ARFF with a nominal target (last column)."""
        content = """\
@relation flowers

@attribute sepal_len numeric
@attribute sepal_wid numeric
@attribute class {setosa,versicolor,virginica}

@data
5.1,3.5,setosa
7.0,3.2,versicolor
6.3,3.3,virginica
"""
        result = parse_arff_string(content, target_column=-1)
        assert result.relation == "flowers"
        assert result.n_samples == 3
        assert result.n_features == 2
        assert result.target is not None
        assert result.target_names == ["setosa", "versicolor", "virginica"]
        assert result.target_name == "class"
        assert len(result.feature_names) == 2
        assert "sepal_len" in result.feature_names
        assert "sepal_wid" in result.feature_names

    def test_parse_with_comments(self):
        """Comments (% lines) should be skipped gracefully."""
        content = """\
% This is a comment
% Another comment line
@relation commented

@attribute a numeric

@data
% comment in data section
1.0
2.0
"""
        result = parse_arff_string(content, target_column=None)
        assert result.n_samples == 2
        assert result.data[0, 0] == 1.0
        assert result.data[1, 0] == 2.0

    def test_parse_missing_values(self):
        """Missing values (?) should be parsed as NaN."""
        content = """\
@relation missing

@attribute x1 numeric
@attribute x2 numeric

@data
1.0,2.0
?,4.0
3.0,?
"""
        result = parse_arff_string(content, target_column=None)
        assert np.isnan(result.data[1, 0])
        assert np.isnan(result.data[2, 1])
        assert result.data[0, 0] == 1.0
        assert result.data[0, 1] == 2.0

    def test_parse_nominal_features(self):
        """Nominal attributes should be encoded as integers."""
        content = """\
@relation nominal

@attribute color {red,green,blue}
@attribute size numeric

@data
red,1.0
green,2.0
blue,3.0
red,4.0
"""
        result = parse_arff_string(content, target_column=None)
        assert result.n_samples == 4
        # red=0, green=1, blue=2
        assert result.data[0, 0] == 0.0
        assert result.data[1, 0] == 1.0
        assert result.data[2, 0] == 2.0
        assert result.data[3, 0] == 0.0

    def test_parse_sparse_format(self):
        """Sparse ARFF format should be parsed correctly."""
        content = """\
@relation sparse

@attribute x1 numeric
@attribute x2 numeric
@attribute x3 numeric
@attribute x4 numeric

@data
{0 1.0, 3 4.0}
{1 2.0, 2 3.0}
"""
        result = parse_arff_string(content, target_column=None)
        assert result.n_samples == 2
        assert result.n_features == 4
        np.testing.assert_array_almost_equal(
            result.data[0], [1.0, 0.0, 0.0, 4.0]
        )
        np.testing.assert_array_almost_equal(
            result.data[1], [0.0, 2.0, 3.0, 0.0]
        )

    def test_parse_empty_data_section(self):
        """Empty data section should produce 0 samples."""
        content = """\
@relation empty

@attribute x1 numeric

@data
"""
        result = parse_arff_string(content, target_column=None)
        assert result.n_samples == 0

    def test_target_column_specific_index(self):
        """target_column=0 should use first column as target."""
        content = """\
@relation target_first

@attribute target {a,b}
@attribute x1 numeric
@attribute x2 numeric

@data
a,1.0,2.0
b,3.0,4.0
"""
        result = parse_arff_string(content, target_column=0)
        assert result.target is not None
        assert result.target_name == "target"
        assert result.n_features == 2
        assert "x1" in result.feature_names
        assert "x2" in result.feature_names


# ===================================================================
# Attribute parsing tests
# ===================================================================

class TestParseAttribute:
    """Tests for _parse_attribute line parsing."""

    def test_numeric_attribute(self):
        """Parse a numeric attribute line."""
        attr = _parse_attribute("@attribute temperature numeric")
        assert attr.name == "temperature"
        assert attr.type == "numeric"
        assert attr.is_numeric is True
        assert attr.is_nominal is False

    def test_real_attribute(self):
        """Parse a 'real' type attribute (synonym for numeric)."""
        attr = _parse_attribute("@attribute weight real")
        assert attr.name == "weight"
        assert attr.type == "numeric"
        assert attr.is_numeric is True

    def test_integer_attribute(self):
        """Parse an 'integer' type attribute."""
        attr = _parse_attribute("@attribute count integer")
        assert attr.name == "count"
        assert attr.type == "numeric"

    def test_nominal_attribute(self):
        """Parse a nominal attribute with values."""
        attr = _parse_attribute("@attribute color {red,green,blue}")
        assert attr.name == "color"
        assert attr.type == "nominal"
        assert attr.is_nominal is True
        assert attr.values == ["red", "green", "blue"]

    def test_string_attribute(self):
        """Parse a string attribute."""
        attr = _parse_attribute("@attribute name string")
        assert attr.name == "name"
        assert attr.type == "string"

    def test_quoted_attribute_name(self):
        """Parse an attribute with a quoted name."""
        attr = _parse_attribute("@attribute 'my attribute' numeric")
        assert attr.name == "my attribute"
        assert attr.type == "numeric"

    def test_date_attribute(self):
        """Parse a date type attribute."""
        attr = _parse_attribute("@attribute timestamp date \"yyyy-MM-dd\"")
        assert attr.name == "timestamp"
        assert attr.type == "date"


# ===================================================================
# Relation parsing tests
# ===================================================================

class TestParseRelation:
    """Tests for _parse_relation line parsing."""

    def test_simple_relation(self):
        """Parse a simple @relation line."""
        result = _parse_relation("@relation my_dataset")
        assert result == "my_dataset"

    def test_quoted_relation(self):
        """Parse a quoted @relation name."""
        result = _parse_relation("@relation 'my complex dataset'")
        assert result == "my complex dataset"

    def test_double_quoted_relation(self):
        """Parse a double-quoted @relation name."""
        result = _parse_relation('@relation "another dataset"')
        assert result == "another dataset"


# ===================================================================
# ARFFData dataclass tests
# ===================================================================

class TestARFFData:
    """Tests for the ARFFData container."""

    def test_properties(self):
        """n_samples and n_features should reflect data shape."""
        data = ARFFData(
            relation="test",
            attributes=[
                Attribute(name="x1", type="numeric"),
                Attribute(name="x2", type="numeric"),
            ],
            data=np.array([[1, 2], [3, 4], [5, 6]], dtype=float),
            feature_names=["x1", "x2"],
        )
        assert data.n_samples == 3
        assert data.n_features == 2


# ===================================================================
# write_arff tests
# ===================================================================

class TestWriteArff:
    """Tests for the write_arff function."""

    def test_write_and_parse_roundtrip(self, tmp_path):
        """Writing and re-parsing ARFF should preserve data."""
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        filepath = str(tmp_path / "roundtrip.arff")

        write_arff(filepath, X, feature_names=["a", "b"], relation="roundtrip")

        result = parse_arff(filepath, target_column=None)

        assert result.relation == "roundtrip"
        assert result.n_samples == 3
        assert result.n_features == 2
        np.testing.assert_array_almost_equal(result.data, X)

    def test_write_with_nominal_target(self, tmp_path):
        """Writing with target names should produce a parseable nominal class."""
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        y = np.array([0, 1])
        filepath = str(tmp_path / "classified.arff")

        write_arff(
            filepath, X,
            feature_names=["f1", "f2"],
            relation="clf",
            target=y,
            target_name="species",
            target_names=["cat", "dog"],
        )

        result = parse_arff(filepath, target_column=-1)

        assert result.target_name == "species"
        assert result.target_names == ["cat", "dog"]
        assert result.n_features == 2

    def test_write_with_numeric_target(self, tmp_path):
        """Writing with numeric target (no target_names) should work."""
        X = np.array([[10.0, 20.0], [30.0, 40.0]])
        y = np.array([100.0, 200.0])
        filepath = str(tmp_path / "regression.arff")

        write_arff(
            filepath, X,
            feature_names=["x1", "x2"],
            relation="reg",
            target=y,
            target_name="price",
        )

        result = parse_arff(filepath, target_column=-1)

        assert result.n_samples == 2
        assert result.target_name == "price"

    def test_write_no_target(self, tmp_path):
        """Writing without target should produce a valid ARFF."""
        X = np.array([[1.0, 2.0, 3.0]])
        filepath = str(tmp_path / "no_target.arff")

        write_arff(filepath, X, feature_names=["a", "b", "c"], relation="test")

        result = parse_arff(filepath, target_column=None)

        assert result.n_samples == 1
        assert result.n_features == 3
