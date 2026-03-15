"""Tests for text vectorizers."""

import numpy as np
import pytest

from tuiml.preprocessing.text.vectorizers import (
    CountVectorizer,
    TfidfTransformer,
    TfidfVectorizer,
    HashingVectorizer,
)


# ---------------------------------------------------------------------------
# CountVectorizer
# ---------------------------------------------------------------------------

class TestCountVectorizer:

    def test_init_defaults(self):
        vec = CountVectorizer()
        assert vec.max_features is None
        assert vec.binary is False
        assert vec.lowercase is True
        assert vec.ngram_range == (1, 1)

    def test_fit_returns_self(self):
        vec = CountVectorizer()
        result = vec.fit(["hello world"])
        assert result is vec

    def test_basic_vectorization(self):
        docs = ["the cat", "the dog"]
        vec = CountVectorizer()
        X = vec.fit_transform(docs)
        assert X.shape[0] == 2
        assert X.shape[1] == len(vec.vocabulary_)

    def test_vocabulary_built(self):
        docs = ["cat dog", "dog bird"]
        vec = CountVectorizer()
        vec.fit(docs)
        assert "cat" in vec.vocabulary_
        assert "dog" in vec.vocabulary_
        assert "bird" in vec.vocabulary_

    def test_binary_mode(self):
        docs = ["cat cat cat dog"]
        vec = CountVectorizer(binary=True)
        X = vec.fit_transform(docs)
        # All non-zero counts should be 1
        assert np.all(X[X > 0] == 1)

    def test_max_features(self):
        docs = ["cat dog bird fish", "cat dog bird", "cat dog"]
        vec = CountVectorizer(max_features=2)
        X = vec.fit_transform(docs)
        assert X.shape[1] == 2

    def test_stop_words_english(self):
        docs = ["the cat in the hat"]
        vec = CountVectorizer(stop_words="english")
        vec.fit(docs)
        assert "the" not in vec.vocabulary_
        assert "in" not in vec.vocabulary_
        assert "cat" in vec.vocabulary_
        assert "hat" in vec.vocabulary_

    def test_stop_words_custom(self):
        docs = ["foo bar baz"]
        vec = CountVectorizer(stop_words=["foo"])
        vec.fit(docs)
        assert "foo" not in vec.vocabulary_
        assert "bar" in vec.vocabulary_

    def test_feature_names_out(self):
        docs = ["cat dog", "dog bird"]
        vec = CountVectorizer()
        vec.fit(docs)
        names = vec.get_feature_names_out()
        assert isinstance(names, list)
        assert len(names) == len(vec.vocabulary_)

    def test_transform_before_fit_raises(self):
        vec = CountVectorizer()
        with pytest.raises(RuntimeError):
            vec.transform(["hello"])

    def test_get_parameter_schema(self):
        schema = CountVectorizer.get_parameter_schema()
        assert "max_features" in schema
        assert "binary" in schema
        assert "lowercase" in schema
        assert "stop_words" in schema
        assert "ngram_range" in schema


# ---------------------------------------------------------------------------
# TfidfTransformer
# ---------------------------------------------------------------------------

class TestTfidfTransformer:

    def test_init_defaults(self):
        t = TfidfTransformer()
        assert t.norm == "l2"
        assert t.use_idf is True
        assert t.smooth_idf is True
        assert t.sublinear_tf is False

    def test_fit_returns_self(self):
        X = np.array([[3, 0, 1], [2, 1, 0]])
        t = TfidfTransformer()
        result = t.fit(X)
        assert result is t

    def test_basic_transform(self):
        X = np.array([[3, 0, 1], [2, 1, 0]])
        t = TfidfTransformer()
        X_tfidf = t.fit_transform(X)
        assert X_tfidf.shape == X.shape

    def test_l2_normalization(self):
        X = np.array([[3, 0, 1], [2, 1, 0]])
        t = TfidfTransformer(norm="l2")
        X_tfidf = t.fit_transform(X)
        # Each row should have L2 norm = 1
        norms = np.sqrt(np.sum(X_tfidf ** 2, axis=1))
        np.testing.assert_allclose(norms, [1.0, 1.0], atol=1e-10)

    def test_l1_normalization(self):
        X = np.array([[3, 0, 1], [2, 1, 0]])
        t = TfidfTransformer(norm="l1")
        X_tfidf = t.fit_transform(X)
        # Each row should have L1 norm = 1
        norms = np.sum(np.abs(X_tfidf), axis=1)
        np.testing.assert_allclose(norms, [1.0, 1.0], atol=1e-10)

    def test_no_idf(self):
        X = np.array([[3, 0, 1], [2, 1, 0]])
        t = TfidfTransformer(use_idf=False, norm=None)
        X_tfidf = t.fit_transform(X)
        # Without IDF and norm, should just be the original counts as floats
        np.testing.assert_allclose(X_tfidf, X.astype(float))

    def test_idf_stored(self):
        X = np.array([[3, 0, 1], [2, 1, 0]])
        t = TfidfTransformer()
        t.fit(X)
        assert t.idf_ is not None
        assert len(t.idf_) == 3

    def test_smooth_idf(self):
        X = np.array([[1, 0], [1, 1]])
        t = TfidfTransformer(smooth_idf=True)
        t.fit(X)
        # smooth: log((n+1)/(df+1)) + 1
        n = 2
        df = np.array([2, 1])
        expected_idf = np.log((n + 1) / (df + 1)) + 1
        np.testing.assert_allclose(t.idf_, expected_idf)

    def test_transform_before_fit_raises(self):
        t = TfidfTransformer()
        X = np.array([[1, 2]])
        with pytest.raises(RuntimeError):
            t.transform(X)

    def test_get_parameter_schema(self):
        schema = TfidfTransformer.get_parameter_schema()
        assert "norm" in schema
        assert "use_idf" in schema
        assert "smooth_idf" in schema
        assert "sublinear_tf" in schema


# ---------------------------------------------------------------------------
# TfidfVectorizer
# ---------------------------------------------------------------------------

class TestTfidfVectorizer:

    def test_init_defaults(self):
        vec = TfidfVectorizer()
        assert vec.norm == "l2"
        assert vec.use_idf is True

    def test_fit_returns_self(self):
        vec = TfidfVectorizer()
        result = vec.fit(["hello world", "foo bar"])
        assert result is vec

    def test_basic_vectorization(self):
        docs = ["the cat", "the dog", "a bird"]
        vec = TfidfVectorizer()
        X = vec.fit_transform(docs)
        assert X.shape[0] == 3
        assert X.shape[1] > 0

    def test_l2_normalized(self):
        docs = ["cat dog", "dog bird", "cat bird"]
        vec = TfidfVectorizer(norm="l2")
        X = vec.fit_transform(docs)
        norms = np.sqrt(np.sum(X ** 2, axis=1))
        np.testing.assert_allclose(norms, np.ones(3), atol=1e-10)

    def test_vocabulary_accessible(self):
        docs = ["cat dog", "bird fish"]
        vec = TfidfVectorizer()
        vec.fit(docs)
        assert "cat" in vec.vocabulary_
        assert "dog" in vec.vocabulary_

    def test_idf_accessible(self):
        docs = ["cat dog", "bird fish"]
        vec = TfidfVectorizer()
        vec.fit(docs)
        assert vec.idf_ is not None

    def test_feature_names_out(self):
        docs = ["cat dog", "bird fish"]
        vec = TfidfVectorizer()
        vec.fit(docs)
        names = vec.get_feature_names_out()
        assert len(names) == len(vec.vocabulary_)

    def test_transform_before_fit_raises(self):
        vec = TfidfVectorizer()
        with pytest.raises(RuntimeError):
            vec.transform(["hello"])

    def test_get_parameter_schema(self):
        schema = TfidfVectorizer.get_parameter_schema()
        assert "max_features" in schema
        assert "norm" in schema
        assert "use_idf" in schema


# ---------------------------------------------------------------------------
# HashingVectorizer
# ---------------------------------------------------------------------------

class TestHashingVectorizer:

    def test_init_defaults(self):
        vec = HashingVectorizer()
        assert vec.n_features == 2 ** 20
        assert vec.binary is False
        assert vec.norm == "l2"

    def test_fit_returns_self(self):
        vec = HashingVectorizer(n_features=128)
        result = vec.fit(["hello world"])
        assert result is vec

    def test_basic_vectorization(self):
        docs = ["cat in the hat"]
        vec = HashingVectorizer(n_features=128)
        X = vec.fit_transform(docs)
        assert X.shape == (1, 128)

    def test_fixed_size_output(self):
        docs = ["cat dog bird fish", "hello world"]
        vec = HashingVectorizer(n_features=64)
        X = vec.fit_transform(docs)
        assert X.shape == (2, 64)

    def test_l2_normalization(self):
        docs = ["cat dog bird"]
        vec = HashingVectorizer(n_features=128, norm="l2")
        X = vec.fit_transform(docs)
        norm = np.sqrt(np.sum(X ** 2, axis=1))
        np.testing.assert_allclose(norm, [1.0], atol=1e-10)

    def test_binary_mode(self):
        docs = ["cat cat cat dog"]
        vec = HashingVectorizer(n_features=128, binary=True, norm=None)
        X = vec.fit_transform(docs)
        assert np.all(X[X > 0] == 1)

    def test_no_vocabulary_stored(self):
        vec = HashingVectorizer(n_features=128)
        vec.fit(["hello world"])
        # HashingVectorizer should not store vocabulary
        assert not hasattr(vec, "vocabulary_") or vec.vocabulary_ is None or len(getattr(vec, "vocabulary_", {})) == 0

    def test_get_parameter_schema(self):
        schema = HashingVectorizer.get_parameter_schema()
        assert "n_features" in schema
        assert "binary" in schema
        assert "norm" in schema
