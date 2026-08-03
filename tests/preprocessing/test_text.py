"""Text preprocessing: cleaners, tokenizers and vectorizers.

Merged from: test_text_cleaners.py, test_text_tokenizers.py, test_text_vectorizers.py
"""

from tuiml.preprocessing.text.cleaners import (
    TextCleaner,
    StopWordRemover,
    Stemmer,
    TextNormalizer,
)
import pytest
from tuiml.preprocessing.text.tokenizers import (
    WordTokenizer,
    NGramTokenizer,
    RegexTokenizer,
    SentenceTokenizer,
    WhitespaceTokenizer,
    TreebankTokenizer,
)
import numpy as np
from tuiml.preprocessing.text.vectorizers import (
    CountVectorizer,
    TfidfTransformer,
    TfidfVectorizer,
    HashingVectorizer,
)


# --------------------------------------------------------------------------
# Tests for text cleaner transformers.
# --------------------------------------------------------------------------

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

    def test_get_parameter_schema(self):
        schema = StopWordRemover.get_parameter_schema()
        assert "stop_words" in schema
        assert "case_sensitive" in schema


class TestStemmer:

    def test_init_defaults(self):
        stemmer = Stemmer()
        assert stemmer.lowercase is True

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


class TestTextNormalizer:

    def test_init_defaults(self):
        normalizer = TextNormalizer()
        assert normalizer.form == "NFKC"
        assert normalizer.lowercase is True
        assert normalizer.strip is True
        assert normalizer.collapse_whitespace is True

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


# --------------------------------------------------------------------------
# Tests for text tokenizers.
# --------------------------------------------------------------------------

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


# --------------------------------------------------------------------------
# Tests for text vectorizers.
# --------------------------------------------------------------------------

class TestCountVectorizer:

    def test_init_defaults(self):
        vec = CountVectorizer()
        assert vec.max_features is None
        assert vec.binary is False
        assert vec.lowercase is True
        assert vec.ngram_range == (1, 1)

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

    def test_get_parameter_schema(self):
        schema = CountVectorizer.get_parameter_schema()
        assert "max_features" in schema
        assert "binary" in schema
        assert "lowercase" in schema
        assert "stop_words" in schema
        assert "ngram_range" in schema


class TestTfidfTransformer:

    def test_init_defaults(self):
        t = TfidfTransformer()
        assert t.norm == "l2"
        assert t.use_idf is True
        assert t.smooth_idf is True
        assert t.sublinear_tf is False

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

    def test_get_parameter_schema(self):
        schema = TfidfTransformer.get_parameter_schema()
        assert "norm" in schema
        assert "use_idf" in schema
        assert "smooth_idf" in schema
        assert "sublinear_tf" in schema


class TestTfidfVectorizer:

    def test_init_defaults(self):
        vec = TfidfVectorizer()
        assert vec.norm == "l2"
        assert vec.use_idf is True

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

    def test_get_parameter_schema(self):
        schema = TfidfVectorizer.get_parameter_schema()
        assert "max_features" in schema
        assert "norm" in schema
        assert "use_idf" in schema


class TestHashingVectorizer:

    def test_init_defaults(self):
        vec = HashingVectorizer()
        assert vec.n_features == 2 ** 20
        assert vec.binary is False
        assert vec.norm == "l2"

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
