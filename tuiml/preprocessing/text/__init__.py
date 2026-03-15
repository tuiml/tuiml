"""
Text preprocessing utilities.
                 weka.core.tokenizers.*

This module provides:
- Tokenizers: Word, N-gram, Regex, Sentence tokenization
- Vectorizers: CountVectorizer, TfidfVectorizer, HashingVectorizer
- Cleaners: Text cleaning, stop word removal, stemming

Examples
--------
>>> from tuiml.preprocessing.text import TfidfVectorizer, WordTokenizer
>>>
>>> # Tokenize text
>>> tokenizer = WordTokenizer()
>>> tokens = tokenizer.tokenize("Hello, World!")
>>> print(tokens)  # ['hello', 'world']
>>>
>>> # Convert to TF-IDF
>>> vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
>>> X = vectorizer.fit_transform(documents)
"""

# Tokenizers
from .tokenizers import (
    BaseTokenizer,
    WordTokenizer,
    NGramTokenizer,
    RegexTokenizer,
    SentenceTokenizer,
    WhitespaceTokenizer,
    TreebankTokenizer,
)

# Vectorizers
from .vectorizers import (
    CountVectorizer,
    TfidfTransformer,
    TfidfVectorizer,
    HashingVectorizer,
)

# Cleaners
from .cleaners import (
    TextCleaner,
    StopWordRemover,
    Stemmer,
    TextNormalizer,
)

__all__ = [
    # Tokenizers
    "BaseTokenizer",
    "WordTokenizer",
    "NGramTokenizer",
    "RegexTokenizer",
    "SentenceTokenizer",
    "WhitespaceTokenizer",
    "TreebankTokenizer",
    # Vectorizers
    "CountVectorizer",
    "TfidfTransformer",
    "TfidfVectorizer",
    "HashingVectorizer",
    # Cleaners
    "TextCleaner",
    "StopWordRemover",
    "Stemmer",
    "TextNormalizer",
]
