"""
Text tokenization utilities.
"""

import re
from abc import ABC, abstractmethod
from typing import List, Optional, Iterator, Pattern
import string

class BaseTokenizer(ABC):
    """
    Base class for text tokenizers.
    """

    @abstractmethod
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize a text string.

        Parameters
        ----------
        text : str
            Input text to tokenize.

        Returns
        -------
        tokens : list of str
            List of tokens.
        """
        pass

    def __call__(self, text: str) -> List[str]:
        """Allow tokenizer to be called directly."""
        return self.tokenize(text)

class WordTokenizer(BaseTokenizer):
    """
    Simple word tokenizer that splits on whitespace and punctuation.

    Parameters
    ----------
    lowercase : bool, default=True
        Convert tokens to lowercase.
    remove_punctuation : bool, default=True
        Remove punctuation from tokens.
    min_length : int, default=1
        Minimum token length to keep.

    Examples
    --------
    >>> from tuiml.preprocessing.text import WordTokenizer
    >>> tokenizer = WordTokenizer()
    >>> tokenizer.tokenize("Hello, World! This is a test.")
    ['hello', 'world', 'this', 'is', 'a', 'test']
    """

    def __init__(
        self,
        lowercase: bool = True,
        remove_punctuation: bool = True,
        min_length: int = 1
    ):
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.min_length = min_length

        # Compile regex for efficiency
        self._word_pattern = re.compile(r'\b\w+\b')

    @classmethod
    def get_parameter_schema(cls) -> dict:
        """Return JSON Schema for parameters."""
        return {
            "lowercase": {
                "type": "boolean",
                "default": True,
                "description": "Convert tokens to lowercase"
            },
            "remove_punctuation": {
                "type": "boolean",
                "default": True,
                "description": "Remove punctuation from tokens"
            },
            "min_length": {
                "type": "integer",
                "default": 1,
                "minimum": 1,
                "description": "Minimum token length to keep"
            }
        }

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into words."""
        if self.lowercase:
            text = text.lower()

        # Extract words
        tokens = self._word_pattern.findall(text)

        # Filter by length
        if self.min_length > 1:
            tokens = [t for t in tokens if len(t) >= self.min_length]

        return tokens

    def __repr__(self) -> str:
        return (
            f"WordTokenizer(lowercase={self.lowercase}, "
            f"remove_punctuation={self.remove_punctuation}, "
            f"min_length={self.min_length})"
        )

class NGramTokenizer(BaseTokenizer):
    """
    N-gram tokenizer that generates character or word n-grams.

    Parameters
    ----------
    n : int, default=2
        N-gram size.
    max_n : int, optional
        Maximum n-gram size (for generating range of n-grams).
    level : str, default='word'
        'word' for word n-grams, 'char' for character n-grams.
    lowercase : bool, default=True
        Convert to lowercase first.

    Examples
    --------
    >>> from tuiml.preprocessing.text import NGramTokenizer
    >>>
    >>> # Word bigrams
    >>> tokenizer = NGramTokenizer(n=2, level='word')
    >>> tokenizer.tokenize("the quick brown fox")
    ['the quick', 'quick brown', 'brown fox']
    >>>
    >>> # Character trigrams
    >>> tokenizer = NGramTokenizer(n=3, level='char')
    >>> tokenizer.tokenize("hello")
    ['hel', 'ell', 'llo']
    """

    def __init__(
        self,
        n: int = 2,
        max_n: int = None,
        level: str = 'word',
        lowercase: bool = True
    ):
        self.n = n
        self.max_n = max_n or n
        self.level = level
        self.lowercase = lowercase

        if level not in ['word', 'char']:
            raise ValueError("level must be 'word' or 'char'")

        self._word_tokenizer = WordTokenizer(lowercase=lowercase)

    @classmethod
    def get_parameter_schema(cls) -> dict:
        """Return JSON Schema for parameters."""
        return {
            "n": {
                "type": "integer",
                "default": 2,
                "minimum": 1,
                "description": "N-gram size"
            },
            "max_n": {
                "type": ["integer", "null"],
                "default": None,
                "description": "Maximum n-gram size (for range of n-grams)"
            },
            "level": {
                "type": "string",
                "default": "word",
                "enum": ["word", "char"],
                "description": "'word' for word n-grams, 'char' for character n-grams"
            },
            "lowercase": {
                "type": "boolean",
                "default": True,
                "description": "Convert to lowercase first"
            }
        }

    def tokenize(self, text: str) -> List[str]:
        """Generate n-grams from text."""
        if self.lowercase:
            text = text.lower()

        ngrams = []

        if self.level == 'word':
            # Word n-grams
            words = self._word_tokenizer.tokenize(text)
            for n in range(self.n, self.max_n + 1):
                for i in range(len(words) - n + 1):
                    ngrams.append(' '.join(words[i:i + n]))
        else:
            # Character n-grams
            text = text.replace(' ', '')  # Remove spaces for char n-grams
            for n in range(self.n, self.max_n + 1):
                for i in range(len(text) - n + 1):
                    ngrams.append(text[i:i + n])

        return ngrams

    def __repr__(self) -> str:
        return (
            f"NGramTokenizer(n={self.n}, max_n={self.max_n}, "
            f"level='{self.level}')"
        )

class RegexTokenizer(BaseTokenizer):
    """
    Tokenizer using regular expression patterns.

    Parameters
    ----------
    pattern : str, default=r'\\w+'
        Regex pattern for matching tokens.
    gaps : bool, default=False
        If True, pattern matches gaps between tokens.
        If False, pattern matches tokens themselves.
    lowercase : bool, default=True
        Convert tokens to lowercase.

    Examples
    --------
    >>> from tuiml.preprocessing.text import RegexTokenizer
    >>>
    >>> # Match words
    >>> tokenizer = RegexTokenizer(pattern=r'\\w+')
    >>> tokenizer.tokenize("Hello, World!")
    ['hello', 'world']
    >>>
    >>> # Split on whitespace (gaps=True)
    >>> tokenizer = RegexTokenizer(pattern=r'\\s+', gaps=True)
    >>> tokenizer.tokenize("Hello World")
    ['Hello', 'World']
    """

    @classmethod
    def get_parameter_schema(cls) -> dict:
        """Return JSON Schema for parameters."""
        return {
            "pattern": {
                "type": "string",
                "default": r"\w+",
                "description": "Regex pattern for matching tokens"
            },
            "gaps": {
                "type": "boolean",
                "default": False,
                "description": "If true, pattern matches gaps between tokens"
            },
            "lowercase": {
                "type": "boolean",
                "default": True,
                "description": "Convert tokens to lowercase"
            }
        }

    def __init__(
        self,
        pattern: str = r'\w+',
        gaps: bool = False,
        lowercase: bool = True
    ):
        self.pattern = pattern
        self.gaps = gaps
        self.lowercase = lowercase
        self._compiled = re.compile(pattern)

    def tokenize(self, text: str) -> List[str]:
        """Tokenize using regex pattern."""
        if self.lowercase:
            text = text.lower()

        if self.gaps:
            tokens = self._compiled.split(text)
            tokens = [t for t in tokens if t]  # Remove empty strings
        else:
            tokens = self._compiled.findall(text)

        return tokens

    def __repr__(self) -> str:
        return (
            f"RegexTokenizer(pattern={self.pattern!r}, gaps={self.gaps})"
        )

class SentenceTokenizer(BaseTokenizer):
    """
    Tokenizer that splits text into sentences.

    Parameters
    ----------
    abbreviations : list of str, optional
        Custom abbreviations to not split on (e.g., ['mr.', 'dr.']).

    Examples
    --------
    >>> from tuiml.preprocessing.text import SentenceTokenizer
    >>> tokenizer = SentenceTokenizer()
    >>> tokenizer.tokenize("Hello world. How are you? I'm fine!")
    ['Hello world.', 'How are you?', "I'm fine!"]
    """

    # Common abbreviations
    DEFAULT_ABBREVIATIONS = [
        'mr.', 'mrs.', 'ms.', 'dr.', 'prof.', 'sr.', 'jr.',
        'vs.', 'etc.', 'e.g.', 'i.e.', 'inc.', 'ltd.', 'co.',
        'st.', 'ave.', 'blvd.', 'rd.', 'ft.', 'mt.',
        'jan.', 'feb.', 'mar.', 'apr.', 'jun.', 'jul.',
        'aug.', 'sep.', 'oct.', 'nov.', 'dec.'
    ]

    @classmethod
    def get_parameter_schema(cls) -> dict:
        """Return JSON Schema for parameters."""
        return {
            "abbreviations": {
                "type": ["array", "null"],
                "default": None,
                "description": "Custom abbreviations to not split on (e.g., ['mr.', 'dr.'])"
            }
        }

    def __init__(self, abbreviations: List[str] = None):
        self.abbreviations = abbreviations or self.DEFAULT_ABBREVIATIONS

        # Build sentence-ending pattern
        self._sentence_pattern = re.compile(
            r'(?<=[.!?])\s+(?=[A-Z])'
        )

    def tokenize(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple split on sentence boundaries
        sentences = self._sentence_pattern.split(text)

        # Clean up
        sentences = [s.strip() for s in sentences if s.strip()]

        return sentences

    def __repr__(self) -> str:
        return "SentenceTokenizer()"

class WhitespaceTokenizer(BaseTokenizer):
    """
    Simple tokenizer that splits on whitespace only.

    Parameters
    ----------
    lowercase : bool, default=False
        Convert tokens to lowercase.

    Examples
    --------
    >>> from tuiml.preprocessing.text import WhitespaceTokenizer
    >>> tokenizer = WhitespaceTokenizer()
    >>> tokenizer.tokenize("Hello, World!")
    ['Hello,', 'World!']
    """

    @classmethod
    def get_parameter_schema(cls) -> dict:
        """Return JSON Schema for parameters."""
        return {
            "lowercase": {
                "type": "boolean",
                "default": False,
                "description": "Convert tokens to lowercase"
            }
        }

    def __init__(self, lowercase: bool = False):
        self.lowercase = lowercase

    def tokenize(self, text: str) -> List[str]:
        """Split on whitespace."""
        if self.lowercase:
            text = text.lower()
        return text.split()

    def __repr__(self) -> str:
        return f"WhitespaceTokenizer(lowercase={self.lowercase})"

class TreebankTokenizer(BaseTokenizer):
    """
    Penn Treebank style tokenizer.

    Handles contractions, punctuation, and special cases
    following Penn Treebank conventions.

    Examples
    --------
    >>> from tuiml.preprocessing.text import TreebankTokenizer
    >>> tokenizer = TreebankTokenizer()
    >>> tokenizer.tokenize("They'll save and invest more.")
    ['They', "'ll", 'save', 'and', 'invest', 'more', '.']
    """

    @classmethod
    def get_parameter_schema(cls) -> dict:
        """Return JSON Schema for parameters."""
        return {}  # No configurable parameters

    # Contraction patterns
    CONTRACTIONS = [
        (r"(\w+)'ll", r"\1 'll"),
        (r"(\w+)'re", r"\1 're"),
        (r"(\w+)'ve", r"\1 've"),
        (r"(\w+)n't", r"\1 n't"),
        (r"(\w+)'d", r"\1 'd"),
        (r"(\w+)'s", r"\1 's"),
        (r"(\w+)'m", r"\1 'm"),
    ]

    def __init__(self):
        self._contractions = [
            (re.compile(pattern, re.IGNORECASE), repl)
            for pattern, repl in self.CONTRACTIONS
        ]

    def tokenize(self, text: str) -> List[str]:
        """Tokenize using Treebank conventions."""
        # Handle contractions
        for pattern, repl in self._contractions:
            text = pattern.sub(repl, text)

        # Separate punctuation
        text = re.sub(r'([^\w\s\'])', r' \1 ', text)

        # Clean up whitespace
        tokens = text.split()

        return tokens

    def __repr__(self) -> str:
        return "TreebankTokenizer()"
