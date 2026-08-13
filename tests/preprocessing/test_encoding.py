"""Categorical encoding filters.

Covers one-hot, ordinal, label, and rare-category encoders.
"""

import numpy as np
from tuiml.preprocessing.encoding.one_hot import OneHotEncoder
from tuiml.preprocessing.encoding.ordinal import OrdinalEncoder
from tuiml.preprocessing.encoding.label import LabelEncoder
from tuiml.preprocessing.encoding.rare_category import RareCategoryEncoder


# --------------------------------------------------------------------------
# Tests for OneHotEncoder transformer.
# --------------------------------------------------------------------------

class TestOneHotEncoder:
    """Tests for OneHotEncoder transformer."""

    def test_init_defaults(self):
        encoder = OneHotEncoder()
        assert encoder.categories is None
        assert encoder.drop is None
        assert encoder.columns is None

    def test_basic_encoding(self):
        X = np.array([[0], [1], [2], [0]])
        encoder = OneHotEncoder()
        X_encoded = encoder.fit_transform(X)
        assert X_encoded.shape == (4, 3)
        # Row 0: category 0 -> [1, 0, 0]
        np.testing.assert_allclose(X_encoded[0], [1, 0, 0])
        # Row 1: category 1 -> [0, 1, 0]
        np.testing.assert_allclose(X_encoded[1], [0, 1, 0])
        # Row 2: category 2 -> [0, 0, 1]
        np.testing.assert_allclose(X_encoded[2], [0, 0, 1])

    def test_drop_first(self):
        X = np.array([[0], [1], [2], [0]])
        encoder = OneHotEncoder(drop="first")
        X_encoded = encoder.fit_transform(X)
        assert X_encoded.shape == (4, 2)  # 3 categories minus 1

    def test_drop_if_binary(self):
        X = np.array([[0], [1], [0], [1]])
        encoder = OneHotEncoder(drop="if_binary")
        X_encoded = encoder.fit_transform(X)
        assert X_encoded.shape == (4, 1)  # Binary: drop one

    def test_columns_parameter(self):
        X = np.array([[1.0, 0], [2.0, 1], [3.0, 2]])
        encoder = OneHotEncoder(columns=[1])
        X_encoded = encoder.fit_transform(X)
        # Column 0 passed through, column 1 one-hot encoded to 3 columns
        assert X_encoded.shape == (3, 4)
        np.testing.assert_allclose(X_encoded[:, 0], [1.0, 2.0, 3.0])

    def test_get_parameter_schema(self):
        schema = OneHotEncoder.get_parameter_schema()
        assert "categories" in schema
        assert "drop" in schema
        assert "columns" in schema

    def test_get_feature_names_out(self):
        X = np.array([[0], [1], [2]])
        encoder = OneHotEncoder()
        encoder.fit(X, feature_names=["color"])
        names = encoder.get_feature_names_out()
        assert len(names) == 3
        assert "color_0.0" in names or "color_0" in names


# --------------------------------------------------------------------------
# Tests for OrdinalEncoder transformer.
# --------------------------------------------------------------------------

class TestOrdinalEncoder:
    """Tests for OrdinalEncoder transformer."""

    def test_init_defaults(self):
        encoder = OrdinalEncoder()
        assert encoder.categories is None
        assert encoder.columns is None

    def test_explicit_order(self):
        X = np.array([["low"], ["medium"], ["high"]], dtype=object)
        encoder = OrdinalEncoder(categories=[["low", "medium", "high"]])
        X_encoded = encoder.fit_transform(X)
        np.testing.assert_allclose(X_encoded.flatten(), [0.0, 1.0, 2.0])

    def test_inferred_order(self):
        X = np.array([["cat"], ["dog"], ["cat"], ["bird"]], dtype=object)
        encoder = OrdinalEncoder()
        X_encoded = encoder.fit_transform(X)
        # Order inferred from first occurrence: cat=0, dog=1, bird=2
        assert X_encoded[0, 0] == 0.0
        assert X_encoded[1, 0] == 1.0
        assert X_encoded[3, 0] == 2.0

    def test_get_parameter_schema(self):
        schema = OrdinalEncoder.get_parameter_schema()
        assert "categories" in schema
        assert "columns" in schema

    def test_unknown_category_maps_to_negative(self):
        X_train = np.array([["a"], ["b"]], dtype=object)
        X_test = np.array([["a"], ["c"]], dtype=object)
        encoder = OrdinalEncoder()
        encoder.fit(X_train)
        X_encoded = encoder.transform(X_test)
        assert X_encoded[0, 0] == 0.0
        assert X_encoded[1, 0] == -1.0

    def test_multiple_columns(self):
        X = np.array([["low", "red"], ["high", "blue"]], dtype=object)
        encoder = OrdinalEncoder(
            categories=[["low", "high"], ["red", "blue"]]
        )
        X_encoded = encoder.fit_transform(X)
        assert X_encoded.shape == (2, 2)
        np.testing.assert_allclose(X_encoded[0], [0.0, 0.0])
        np.testing.assert_allclose(X_encoded[1], [1.0, 1.0])


# --------------------------------------------------------------------------
# Tests for LabelEncoder transformer.
# --------------------------------------------------------------------------

class TestLabelEncoder:
    """Tests for LabelEncoder transformer."""

    def test_init_defaults(self):
        encoder = LabelEncoder()
        assert encoder.columns is None

    def test_basic_encoding(self):
        X = np.array([["cat"], ["dog"], ["cat"]], dtype=object)
        encoder = LabelEncoder()
        X_encoded = encoder.fit_transform(X)
        # cat=0, dog=1 (order of first occurrence)
        assert X_encoded[0, 0] == 0.0
        assert X_encoded[1, 0] == 1.0
        assert X_encoded[2, 0] == 0.0

    def test_auto_detect_string_columns(self):
        X = np.array([[1.0, "cat"], [2.0, "dog"]], dtype=object)
        encoder = LabelEncoder()
        X_encoded = encoder.fit_transform(X)
        # Column 0 should stay numeric, column 1 should be encoded
        np.testing.assert_allclose(X_encoded[:, 0], [1.0, 2.0])
        assert X_encoded[0, 1] == 0.0
        assert X_encoded[1, 1] == 1.0

    def test_unseen_category_maps_to_negative(self):
        X_train = np.array([["cat"], ["dog"]], dtype=object)
        X_test = np.array([["cat"], ["bird"]], dtype=object)
        encoder = LabelEncoder()
        encoder.fit(X_train)
        X_encoded = encoder.transform(X_test)
        assert X_encoded[0, 0] == 0.0
        assert X_encoded[1, 0] == -1.0

    def test_categories_property(self):
        X = np.array([["a"], ["b"], ["c"]], dtype=object)
        encoder = LabelEncoder()
        encoder.fit(X)
        cats = encoder.categories_
        assert 0 in cats
        assert len(cats[0]) == 3

    def test_get_parameter_schema(self):
        schema = LabelEncoder.get_parameter_schema()
        assert "columns" in schema


# --------------------------------------------------------------------------
# Tests for RareCategoryEncoder transformer.
# --------------------------------------------------------------------------

class TestRareCategoryEncoder:
    """Tests for RareCategoryEncoder transformer."""

    def test_init_defaults(self):
        encoder = RareCategoryEncoder()
        assert encoder.min_frequency == 5
        assert encoder.merged_value == -1
        assert encoder.columns is None

    def test_merge_rare_values(self):
        # 0 appears 5 times, 1 appears 3 times, 2 appears 1 time
        X = np.array([[0], [0], [0], [0], [0], [1], [1], [1], [2]])
        encoder = RareCategoryEncoder(min_frequency=3, merged_value=-1)
        X_merged = encoder.fit_transform(X)
        # Value 2 (count 1) should be merged to -1
        np.testing.assert_allclose(X_merged[8, 0], -1.0)
        # Value 0 (count 5) should remain
        np.testing.assert_allclose(X_merged[0, 0], 0.0)
        # Value 1 (count 3) should remain
        np.testing.assert_allclose(X_merged[5, 0], 1.0)

    def test_proportion_threshold(self):
        X = np.arange(10).reshape(-1, 1).astype(float)
        # Each value appears once out of 10 = 0.1
        encoder = RareCategoryEncoder(min_frequency=0.2)
        X_merged = encoder.fit_transform(X)
        # All values appear less than 20%, so all should be merged
        for val in X_merged.flatten():
            assert val == -1.0

    def test_columns_parameter(self):
        X = np.array([[0, 0], [0, 1], [0, 1], [1, 1]])
        encoder = RareCategoryEncoder(min_frequency=2, columns=[0])
        X_merged = encoder.fit_transform(X)
        # Column 0: 0 appears 3 times, 1 appears 1 time -> 1 merged
        np.testing.assert_allclose(X_merged[3, 0], -1.0)
        # Column 1 should be unchanged
        np.testing.assert_allclose(X_merged[:, 1], X[:, 1])

    def test_get_parameter_schema(self):
        schema = RareCategoryEncoder.get_parameter_schema()
        assert "min_frequency" in schema
        assert "merged_value" in schema
        assert "columns" in schema

    def test_value_maps_property(self):
        X = np.array([[0], [0], [0], [1]])
        encoder = RareCategoryEncoder(min_frequency=2)
        encoder.fit(X)
        maps = encoder.value_maps_
        assert isinstance(maps, dict)
