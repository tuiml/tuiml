"""Tests for text cleaner transformers."""

import pytest

from tuiml.preprocessing.text.cleaners import (
    TextCleaner,
    StopWordRemover,
    Stemmer,
    TextNormalizer,
)


# ---------------------------------------------------------------------------
# TextCleaner
# ---------------------------------------------------------------------------

class TestTextCleaner:

    def test_init_defaults(self):
        cleaner = TextCleaner()
        assert cleaner.lowercase is True
        assert cleaner.remove_punctuation is False
        assert cleaner.remove_numbers is False
        assert cleaner.remove_whitespace is True
        assert cleaner.remove_html is True
        assert cleaner.remove_urls is True
        assert cleaner.remove_emails is True
        assert cleaner.remove_special_chars is False
        assert cleaner.strip_accents is False
        assert cleaner.min_word_length == 1

    def test_fit_returns_self(self):
        cleaner = TextCleaner()
        result = cleaner.fit(["hello"])
        assert result is cleaner

    def test_lowercase(self):
        cleaner = TextCleaner(lowercase=True)
        cleaner.fit([])
        result = cleaner.transform(["Hello WORLD"])
        assert result == ["hello world"]

    def test_no_lowercase(self):
        cleaner = TextCleaner(lowercase=False)
        cleaner.fit([])
        result = cleaner.transform(["Hello WORLD"])
        assert result == ["Hello WORLD"]

    def test_remove_html(self):
        cleaner = TextCleaner(remove_html=True, lowercase=False)
        cleaner.fit([])
        result = cleaner.transform(["<p>Hello</p> <b>World</b>"])
        assert "<p>" not in result[0]
        assert "<b>" not in result[0]
        assert "Hello" in result[0]
        assert "World" in result[0]

    def test_remove_urls(self):
        cleaner = TextCleaner(remove_urls=True, lowercase=False)
        cleaner.fit([])
        result = cleaner.transform(["Visit https://example.com today"])
        assert "https://example.com" not in result[0]
        assert "Visit" in result[0]
        assert "today" in result[0]

    def test_remove_emails(self):
        cleaner = TextCleaner(remove_emails=True, lowercase=False)
        cleaner.fit([])
        result = cleaner.transform(["Contact user@example.com for info"])
        assert "user@example.com" not in result[0]
        assert "Contact" in result[0]

    def test_remove_punctuation(self):
        cleaner = TextCleaner(remove_punctuation=True, lowercase=False)
        cleaner.fit([])
        result = cleaner.transform(["Hello, World! How's it?"])
        assert "," not in result[0]
        assert "!" not in result[0]
        assert "?" not in result[0]

    def test_remove_numbers(self):
        cleaner = TextCleaner(remove_numbers=True, lowercase=False)
        cleaner.fit([])
        result = cleaner.transform(["There are 42 cats and 7 dogs"])
        assert "42" not in result[0]
        assert "7" not in result[0]
        assert "cats" in result[0]

    def test_remove_whitespace(self):
        cleaner = TextCleaner(remove_whitespace=True, lowercase=False)
        cleaner.fit([])
        result = cleaner.transform(["Hello   World    Test"])
        assert "   " not in result[0]
        assert "Hello World Test" == result[0]

    def test_strip_accents(self):
        cleaner = TextCleaner(strip_accents=True, lowercase=False)
        cleaner.fit([])
        result = cleaner.transform(["cafe\u0301 re\u0301sume\u0301"])
        assert "\u0301" not in result[0]

    def test_min_word_length(self):
        cleaner = TextCleaner(min_word_length=3, lowercase=True)
        cleaner.fit([])
        result = cleaner.transform(["I am a big cat in the hat"])
        words = result[0].split()
        for w in words:
            assert len(w) >= 3

    def test_multiple_documents(self):
        cleaner = TextCleaner(lowercase=True)
        cleaner.fit([])
        texts = ["Hello World", "FOO BAR", "Test"]
        result = cleaner.transform(texts)
        assert len(result) == 3
        assert result[0] == "hello world"
        assert result[1] == "foo bar"
        assert result[2] == "test"

    def test_get_parameter_schema(self):
        schema = TextCleaner.get_parameter_schema()
        assert "lowercase" in schema
        assert "remove_punctuation" in schema
        assert "remove_html" in schema
        assert "min_word_length" in schema


# ---------------------------------------------------------------------------
# StopWordRemover
# ---------------------------------------------------------------------------

class TestStopWordRemover:

    def test_init_defaults(self):
        remover = StopWordRemover()
        assert remover.case_sensitive is False
        assert len(remover.stop_words) > 0

    def test_english_stop_words(self):
        remover = StopWordRemover(stop_words="english")
        remover.fit([])
        result = remover.transform(["the cat sat on the mat"])
        assert "the" not in result[0].split()
        assert "on" not in result[0].split()
        assert "cat" in result[0].split()
        assert "sat" in result[0].split()
        assert "mat" in result[0].split()

    def test_custom_stop_words(self):
        remover = StopWordRemover(stop_words=["foo", "bar"])
        remover.fit([])
        result = remover.transform(["foo hello bar world"])
        assert result == ["hello world"]

    def test_case_insensitive(self):
        remover = StopWordRemover(stop_words=["the"], case_sensitive=False)
        remover.fit([])
        result = remover.transform(["The cat THE mat the"])
        # "The", "THE", "the" should all be removed
        words = result[0].split()
        for w in words:
            assert w.lower() != "the"

    def test_case_sensitive(self):
        remover = StopWordRemover(stop_words=["the"], case_sensitive=True)
        remover.fit([])
        result = remover.transform(["The cat the mat"])
        words = result[0].split()
        # "The" (uppercase T) should remain, "the" should be removed
        assert "The" in words
        assert "the" not in words

    def test_fit_returns_self(self):
        remover = StopWordRemover()
        result = remover.fit([])
        assert result is remover

    def test_get_parameter_schema(self):
        schema = StopWordRemover.get_parameter_schema()
        assert "stop_words" in schema
        assert "case_sensitive" in schema


# ---------------------------------------------------------------------------
# Stemmer
# ---------------------------------------------------------------------------

class TestStemmer:

    def test_init_defaults(self):
        stemmer = Stemmer()
        assert stemmer.lowercase is True

    def test_fit_returns_self(self):
        stemmer = Stemmer()
        result = stemmer.fit([])
        assert result is stemmer

    def test_basic_stemming(self):
        stemmer = Stemmer()
        stemmer.fit([])
        result = stemmer.transform(["running cats playing"])
        # The porter stemmer should reduce words
        words = result[0].split()
        assert len(words) == 3

    def test_plural_removal(self):
        stemmer = Stemmer()
        stemmer.fit([])
        result = stemmer.transform(["cats dogs"])
        words = result[0].split()
        assert "cat" in words
        assert "dog" in words

    def test_lowercase_option(self):
        stemmer = Stemmer(lowercase=True)
        stemmer.fit([])
        result = stemmer.transform(["RUNNING CATS"])
        assert result[0] == result[0].lower()

    def test_no_lowercase(self):
        stemmer = Stemmer(lowercase=False)
        stemmer.fit([])
        result = stemmer.transform(["RUNNING"])
        # Without lowercase, "RUNNING" won't match suffix rules and stays as-is
        assert result[0] == "RUNNING"

    def test_multiple_documents(self):
        stemmer = Stemmer()
        stemmer.fit([])
        result = stemmer.transform(["cats", "dogs", "running"])
        assert len(result) == 3

    def test_get_parameter_schema(self):
        schema = Stemmer.get_parameter_schema()
        assert "lowercase" in schema


# ---------------------------------------------------------------------------
# TextNormalizer
# ---------------------------------------------------------------------------

class TestTextNormalizer:

    def test_init_defaults(self):
        normalizer = TextNormalizer()
        assert normalizer.form == "NFKC"
        assert normalizer.lowercase is True
        assert normalizer.strip is True
        assert normalizer.collapse_whitespace is True

    def test_fit_returns_self(self):
        normalizer = TextNormalizer()
        result = normalizer.fit([])
        assert result is normalizer

    def test_lowercase(self):
        normalizer = TextNormalizer(lowercase=True)
        normalizer.fit([])
        result = normalizer.transform(["Hello WORLD"])
        assert result == ["hello world"]

    def test_strip_whitespace(self):
        normalizer = TextNormalizer(strip=True)
        normalizer.fit([])
        result = normalizer.transform(["  hello world  "])
        assert result[0] == "hello world"

    def test_collapse_whitespace(self):
        normalizer = TextNormalizer(collapse_whitespace=True)
        normalizer.fit([])
        result = normalizer.transform(["hello    world   test"])
        assert result[0] == "hello world test"

    def test_combined_operations(self):
        normalizer = TextNormalizer(
            lowercase=True, strip=True, collapse_whitespace=True
        )
        normalizer.fit([])
        result = normalizer.transform(["  Hello   World  "])
        assert result == ["hello world"]

    def test_no_operations(self):
        normalizer = TextNormalizer(
            lowercase=False, strip=False, collapse_whitespace=False
        )
        normalizer.fit([])
        text = "  Hello   World  "
        result = normalizer.transform([text])
        # Unicode normalization still applies but should not change ASCII
        assert "Hello" in result[0]
        assert "World" in result[0]

    def test_multiple_documents(self):
        normalizer = TextNormalizer()
        normalizer.fit([])
        texts = ["  Hello  ", "  WORLD  ", "  Test  "]
        result = normalizer.transform(texts)
        assert len(result) == 3
        assert result[0] == "hello"
        assert result[1] == "world"
        assert result[2] == "test"

    def test_get_parameter_schema(self):
        schema = TextNormalizer.get_parameter_schema()
        assert "form" in schema
        assert "lowercase" in schema
        assert "strip" in schema
        assert "collapse_whitespace" in schema
