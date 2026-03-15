"""Tests for text tokenizers."""

import pytest

from tuiml.preprocessing.text.tokenizers import (
    WordTokenizer,
    NGramTokenizer,
    RegexTokenizer,
    SentenceTokenizer,
    WhitespaceTokenizer,
    TreebankTokenizer,
)


# ---------------------------------------------------------------------------
# WordTokenizer
# ---------------------------------------------------------------------------

class TestWordTokenizer:

    def test_init_defaults(self):
        tokenizer = WordTokenizer()
        assert tokenizer.lowercase is True
        assert tokenizer.remove_punctuation is True
        assert tokenizer.min_length == 1

    def test_basic_tokenization(self):
        tokenizer = WordTokenizer()
        tokens = tokenizer.tokenize("Hello, World! This is a test.")
        assert "hello" in tokens
        assert "world" in tokens
        assert "test" in tokens

    def test_lowercase(self):
        tokenizer = WordTokenizer(lowercase=True)
        tokens = tokenizer.tokenize("Hello WORLD")
        assert tokens == ["hello", "world"]

    def test_no_lowercase(self):
        tokenizer = WordTokenizer(lowercase=False)
        tokens = tokenizer.tokenize("Hello WORLD")
        assert tokens == ["Hello", "WORLD"]

    def test_min_length(self):
        tokenizer = WordTokenizer(min_length=3)
        tokens = tokenizer.tokenize("I am a big cat")
        for t in tokens:
            assert len(t) >= 3

    def test_callable(self):
        tokenizer = WordTokenizer()
        tokens = tokenizer("Hello World")
        assert tokens == ["hello", "world"]

    def test_get_parameter_schema(self):
        schema = WordTokenizer.get_parameter_schema()
        assert "lowercase" in schema
        assert "remove_punctuation" in schema
        assert "min_length" in schema


# ---------------------------------------------------------------------------
# NGramTokenizer
# ---------------------------------------------------------------------------

class TestNGramTokenizer:

    def test_word_bigrams(self):
        tokenizer = NGramTokenizer(n=2, level="word")
        tokens = tokenizer.tokenize("the quick brown fox")
        assert "the quick" in tokens
        assert "quick brown" in tokens
        assert "brown fox" in tokens
        assert len(tokens) == 3

    def test_word_trigrams(self):
        tokenizer = NGramTokenizer(n=3, level="word")
        tokens = tokenizer.tokenize("the quick brown fox")
        assert "the quick brown" in tokens
        assert "quick brown fox" in tokens
        assert len(tokens) == 2

    def test_char_trigrams(self):
        tokenizer = NGramTokenizer(n=3, level="char")
        tokens = tokenizer.tokenize("hello")
        assert tokens == ["hel", "ell", "llo"]

    def test_char_bigrams(self):
        tokenizer = NGramTokenizer(n=2, level="char")
        tokens = tokenizer.tokenize("abc")
        assert tokens == ["ab", "bc"]

    def test_range_ngrams(self):
        tokenizer = NGramTokenizer(n=1, max_n=2, level="word")
        tokens = tokenizer.tokenize("the quick brown")
        # Should include unigrams and bigrams
        assert "the" in tokens
        assert "quick" in tokens
        assert "brown" in tokens
        assert "the quick" in tokens
        assert "quick brown" in tokens

    def test_invalid_level_raises(self):
        with pytest.raises(ValueError):
            NGramTokenizer(level="invalid")

    def test_lowercase(self):
        tokenizer = NGramTokenizer(n=2, level="word", lowercase=True)
        tokens = tokenizer.tokenize("Hello World")
        assert "hello world" in tokens

    def test_get_parameter_schema(self):
        schema = NGramTokenizer.get_parameter_schema()
        assert "n" in schema
        assert "max_n" in schema
        assert "level" in schema
        assert "lowercase" in schema


# ---------------------------------------------------------------------------
# RegexTokenizer
# ---------------------------------------------------------------------------

class TestRegexTokenizer:

    def test_default_pattern(self):
        tokenizer = RegexTokenizer()
        tokens = tokenizer.tokenize("Hello, World!")
        assert tokens == ["hello", "world"]

    def test_gaps_mode(self):
        tokenizer = RegexTokenizer(pattern=r"\s+", gaps=True, lowercase=False)
        tokens = tokenizer.tokenize("Hello World Test")
        assert tokens == ["Hello", "World", "Test"]

    def test_custom_pattern(self):
        tokenizer = RegexTokenizer(pattern=r"[a-z]+", lowercase=True)
        tokens = tokenizer.tokenize("Hello 123 World")
        assert "hello" in tokens
        assert "world" in tokens

    def test_lowercase(self):
        tokenizer = RegexTokenizer(lowercase=True)
        tokens = tokenizer.tokenize("HELLO WORLD")
        assert all(t == t.lower() for t in tokens)

    def test_no_lowercase(self):
        tokenizer = RegexTokenizer(lowercase=False)
        tokens = tokenizer.tokenize("HELLO WORLD")
        assert "HELLO" in tokens
        assert "WORLD" in tokens

    def test_get_parameter_schema(self):
        schema = RegexTokenizer.get_parameter_schema()
        assert "pattern" in schema
        assert "gaps" in schema
        assert "lowercase" in schema


# ---------------------------------------------------------------------------
# SentenceTokenizer
# ---------------------------------------------------------------------------

class TestSentenceTokenizer:

    def test_basic_sentences(self):
        tokenizer = SentenceTokenizer()
        sentences = tokenizer.tokenize(
            "Hello world. How are you? I am fine!"
        )
        assert len(sentences) >= 2

    def test_single_sentence(self):
        tokenizer = SentenceTokenizer()
        sentences = tokenizer.tokenize("Hello world")
        assert len(sentences) == 1
        assert sentences[0] == "Hello world"

    def test_custom_abbreviations(self):
        tokenizer = SentenceTokenizer(abbreviations=["dr.", "mr."])
        assert "dr." in tokenizer.abbreviations
        assert "mr." in tokenizer.abbreviations

    def test_get_parameter_schema(self):
        schema = SentenceTokenizer.get_parameter_schema()
        assert "abbreviations" in schema


# ---------------------------------------------------------------------------
# WhitespaceTokenizer
# ---------------------------------------------------------------------------

class TestWhitespaceTokenizer:

    def test_basic_split(self):
        tokenizer = WhitespaceTokenizer()
        tokens = tokenizer.tokenize("Hello, World!")
        assert tokens == ["Hello,", "World!"]

    def test_lowercase(self):
        tokenizer = WhitespaceTokenizer(lowercase=True)
        tokens = tokenizer.tokenize("Hello World")
        assert tokens == ["hello", "world"]

    def test_no_lowercase(self):
        tokenizer = WhitespaceTokenizer(lowercase=False)
        tokens = tokenizer.tokenize("Hello World")
        assert tokens == ["Hello", "World"]

    def test_multiple_spaces(self):
        tokenizer = WhitespaceTokenizer()
        tokens = tokenizer.tokenize("Hello   World")
        assert tokens == ["Hello", "World"]

    def test_get_parameter_schema(self):
        schema = WhitespaceTokenizer.get_parameter_schema()
        assert "lowercase" in schema


# ---------------------------------------------------------------------------
# TreebankTokenizer
# ---------------------------------------------------------------------------

class TestTreebankTokenizer:

    def test_contractions(self):
        tokenizer = TreebankTokenizer()
        tokens = tokenizer.tokenize("They'll save and invest more.")
        assert "'ll" in tokens
        assert "They" in tokens

    def test_punctuation_separation(self):
        tokenizer = TreebankTokenizer()
        tokens = tokenizer.tokenize("Hello, world.")
        assert "," in tokens
        assert "." in tokens

    def test_wont_contraction(self):
        tokenizer = TreebankTokenizer()
        tokens = tokenizer.tokenize("I won't go")
        assert "n't" in tokens

    def test_simple_text(self):
        tokenizer = TreebankTokenizer()
        tokens = tokenizer.tokenize("Hello world")
        assert "Hello" in tokens
        assert "world" in tokens

    def test_get_parameter_schema(self):
        schema = TreebankTokenizer.get_parameter_schema()
        assert isinstance(schema, dict)
