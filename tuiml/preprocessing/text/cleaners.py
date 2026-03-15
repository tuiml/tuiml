"""
Text cleaning and normalization utilities.

Provides preprocessing functions to clean and normalize text
before tokenization and vectorization.
"""

import re
import unicodedata
from typing import List, Optional, Set, Callable
from tuiml.base.preprocessing import Transformer

class TextCleaner(Transformer):
    """
    Comprehensive text cleaning transformer.

    Applies multiple cleaning operations in sequence.

    Parameters
    ----------
    lowercase : bool, default=True
        Convert to lowercase.
    remove_punctuation : bool, default=False
        Remove all punctuation.
    remove_numbers : bool, default=False
        Remove all numbers.
    remove_whitespace : bool, default=True
        Normalize whitespace (multiple spaces to single).
    remove_html : bool, default=True
        Remove HTML tags.
    remove_urls : bool, default=True
        Remove URLs.
    remove_emails : bool, default=True
        Remove email addresses.
    remove_special_chars : bool, default=False
        Remove special characters.
    strip_accents : bool, default=False
        Remove accent marks from characters.
    min_word_length : int, default=1
        Remove words shorter than this.

    Examples
    --------
    >>> from tuiml.preprocessing.text import TextCleaner
    >>>
    >>> cleaner = TextCleaner(
    ...     lowercase=True,
    ...     remove_html=True,
    ...     remove_urls=True
    ... )
    >>> clean_text = cleaner.transform(["<p>Visit https://example.com!</p>"])
    >>> print(clean_text[0])  # 'visit'
    """

    def __init__(
        self,
        lowercase: bool = True,
        remove_punctuation: bool = False,
        remove_numbers: bool = False,
        remove_whitespace: bool = True,
        remove_html: bool = True,
        remove_urls: bool = True,
        remove_emails: bool = True,
        remove_special_chars: bool = False,
        strip_accents: bool = False,
        min_word_length: int = 1
    ):
        super().__init__()
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.remove_numbers = remove_numbers
        self.remove_whitespace = remove_whitespace
        self.remove_html = remove_html
        self.remove_urls = remove_urls
        self.remove_emails = remove_emails
        self.remove_special_chars = remove_special_chars
        self.strip_accents = strip_accents
        self.min_word_length = min_word_length

        # Compile regex patterns
        self._html_pattern = re.compile(r'<[^>]+>')
        self._url_pattern = re.compile(
            r'https?://\S+|www\.\S+|ftp://\S+'
        )
        self._email_pattern = re.compile(
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        )
        self._whitespace_pattern = re.compile(r'\s+')
        self._number_pattern = re.compile(r'\d+')
        self._punct_pattern = re.compile(r'[^\w\s]')
        self._special_pattern = re.compile(r'[^a-zA-Z0-9\s]')

    @classmethod
    def get_parameter_schema(cls) -> dict:
        """Return JSON Schema for parameters."""
        return {
            "lowercase": {
                "type": "boolean",
                "default": True,
                "description": "Convert to lowercase"
            },
            "remove_punctuation": {
                "type": "boolean",
                "default": False,
                "description": "Remove all punctuation"
            },
            "remove_numbers": {
                "type": "boolean",
                "default": False,
                "description": "Remove all numbers"
            },
            "remove_whitespace": {
                "type": "boolean",
                "default": True,
                "description": "Normalize whitespace (multiple spaces to single)"
            },
            "remove_html": {
                "type": "boolean",
                "default": True,
                "description": "Remove HTML tags"
            },
            "remove_urls": {
                "type": "boolean",
                "default": True,
                "description": "Remove URLs"
            },
            "remove_emails": {
                "type": "boolean",
                "default": True,
                "description": "Remove email addresses"
            },
            "remove_special_chars": {
                "type": "boolean",
                "default": False,
                "description": "Remove special characters"
            },
            "strip_accents": {
                "type": "boolean",
                "default": False,
                "description": "Remove accent marks from characters"
            },
            "min_word_length": {
                "type": "integer",
                "default": 1,
                "minimum": 1,
                "description": "Remove words shorter than this"
            }
        }

    def fit(self, X, y=None) -> "TextCleaner":
        """Fit (no-op for cleaner)."""
        self._is_fitted = True
        return self

    def transform(self, texts: List[str]) -> List[str]:
        """
        Clean a list of text documents.

        Parameters
        ----------
        texts : list of str
            Input texts.

        Returns
        -------
        cleaned : list of str
            Cleaned texts.
        """
        return [self._clean_text(text) for text in texts]

    def _clean_text(self, text: str) -> str:
        """Apply all cleaning operations."""
        # Remove HTML
        if self.remove_html:
            text = self._html_pattern.sub(' ', text)

        # Remove URLs
        if self.remove_urls:
            text = self._url_pattern.sub(' ', text)

        # Remove emails
        if self.remove_emails:
            text = self._email_pattern.sub(' ', text)

        # Lowercase
        if self.lowercase:
            text = text.lower()

        # Strip accents
        if self.strip_accents:
            text = self._strip_accents(text)

        # Remove numbers
        if self.remove_numbers:
            text = self._number_pattern.sub(' ', text)

        # Remove punctuation
        if self.remove_punctuation:
            text = self._punct_pattern.sub(' ', text)

        # Remove special characters
        if self.remove_special_chars:
            text = self._special_pattern.sub(' ', text)

        # Normalize whitespace
        if self.remove_whitespace:
            text = self._whitespace_pattern.sub(' ', text).strip()

        # Filter by word length
        if self.min_word_length > 1:
            words = text.split()
            words = [w for w in words if len(w) >= self.min_word_length]
            text = ' '.join(words)

        return text

    def _strip_accents(self, text: str) -> str:
        """Remove accent marks from characters."""
        # Normalize to NFD form (decomposed)
        text = unicodedata.normalize('NFD', text)
        # Remove combining characters (accents)
        text = ''.join(
            char for char in text
            if unicodedata.category(char) != 'Mn'
        )
        return text

    def __repr__(self) -> str:
        return f"TextCleaner(lowercase={self.lowercase})"

class StopWordRemover(Transformer):
    """
    Remove stop words from text.

    Parameters
    ----------
    stop_words : list of str or 'english', default='english'
        Stop words to remove.
    case_sensitive : bool, default=False
        Whether matching is case-sensitive.

    Examples
    --------
    >>> from tuiml.preprocessing.text import StopWordRemover
    >>> remover = StopWordRemover(stop_words='english')
    >>> remover.transform(["the cat sat on the mat"])
    ['cat sat mat']
    """

    ENGLISH_STOP_WORDS = {
        'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
        'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
        'to', 'was', 'were', 'will', 'with', 'this', 'but', 'they',
        'have', 'had', 'what', 'when', 'where', 'who', 'which', 'why', 'how',
        'all', 'each', 'every', 'both', 'few', 'more', 'most', 'other',
        'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
        'than', 'too', 'very', 'can', 'just', 'should', 'now', 'i', 'me',
        'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your',
        'yours', 'yourself', 'yourselves', 'him', 'his', 'himself', 'she',
        'her', 'hers', 'herself', 'them', 'their', 'theirs', 'themselves',
        'been', 'being', 'having', 'do', 'does', 'did', 'doing', 'would',
        'could', 'ought', 'am', 'about', 'against', 'between', 'into',
        'through', 'during', 'before', 'after', 'above', 'below', 'up',
        'down', 'out', 'off', 'over', 'under', 'again', 'further', 'then',
        'once', 'here', 'there', 'any', 'until', 'while', 'if', 'or',
        'because', 'these', 'those'
    }

    @classmethod
    def get_parameter_schema(cls) -> dict:
        """Return JSON Schema for parameters."""
        return {
            "stop_words": {
                "type": ["string", "array"],
                "default": "english",
                "description": "Stop words to remove: 'english' or list of words"
            },
            "case_sensitive": {
                "type": "boolean",
                "default": False,
                "description": "Whether matching is case-sensitive"
            }
        }

    def __init__(
        self,
        stop_words: str | List[str] = 'english',
        case_sensitive: bool = False
    ):
        super().__init__()
        self.case_sensitive = case_sensitive

        if stop_words == 'english':
            self.stop_words = self.ENGLISH_STOP_WORDS
        else:
            self.stop_words = set(stop_words)

        if not case_sensitive:
            self.stop_words = {w.lower() for w in self.stop_words}

    def fit(self, X, y=None) -> "StopWordRemover":
        """Fit (no-op)."""
        self._is_fitted = True
        return self

    def transform(self, texts: List[str]) -> List[str]:
        """Remove stop words from texts."""
        result = []
        for text in texts:
            words = text.split()
            if self.case_sensitive:
                filtered = [w for w in words if w not in self.stop_words]
            else:
                filtered = [w for w in words if w.lower() not in self.stop_words]
            result.append(' '.join(filtered))
        return result

    def __repr__(self) -> str:
        return f"StopWordRemover(n_words={len(self.stop_words)})"

class Stemmer(Transformer):
    """
    Apply stemming to reduce words to their root form.

    Implements Porter Stemmer algorithm.

    Parameters
    ----------
    lowercase : bool, default=True
        Convert to lowercase before stemming.

    Examples
    --------
    >>> from tuiml.preprocessing.text import Stemmer
    >>> stemmer = Stemmer()
    >>> stemmer.transform(["running cats are playing"])
    ['run cat are play']
    """

    @classmethod
    def get_parameter_schema(cls) -> dict:
        """Return JSON Schema for parameters."""
        return {
            "lowercase": {
                "type": "boolean",
                "default": True,
                "description": "Convert to lowercase before stemming"
            }
        }

    def __init__(self, lowercase: bool = True):
        super().__init__()
        self.lowercase = lowercase

    def fit(self, X, y=None) -> "Stemmer":
        """Fit (no-op)."""
        self._is_fitted = True
        return self

    def transform(self, texts: List[str]) -> List[str]:
        """Stem all words in texts."""
        result = []
        for text in texts:
            if self.lowercase:
                text = text.lower()
            words = text.split()
            stemmed = [self._porter_stem(word) for word in words]
            result.append(' '.join(stemmed))
        return result

    def _porter_stem(self, word: str) -> str:
        """Apply Porter stemming algorithm (simplified version)."""
        # Step 1a: Remove common suffixes
        if word.endswith('sses'):
            word = word[:-2]
        elif word.endswith('ies'):
            word = word[:-2]
        elif word.endswith('ss'):
            pass
        elif word.endswith('s'):
            word = word[:-1]

        # Step 1b: Remove -ed, -ing
        if word.endswith('eed'):
            if len(word) > 4:
                word = word[:-1]
        elif word.endswith('ed'):
            if self._has_vowel(word[:-2]):
                word = word[:-2]
                word = self._step1b_fix(word)
        elif word.endswith('ing'):
            if self._has_vowel(word[:-3]):
                word = word[:-3]
                word = self._step1b_fix(word)

        # Step 2: Remove common suffixes
        suffixes = [
            ('ational', 'ate'), ('tional', 'tion'), ('enci', 'ence'),
            ('anci', 'ance'), ('izer', 'ize'), ('abli', 'able'),
            ('alli', 'al'), ('entli', 'ent'), ('eli', 'e'),
            ('ousli', 'ous'), ('ization', 'ize'), ('ation', 'ate'),
            ('ator', 'ate'), ('alism', 'al'), ('iveness', 'ive'),
            ('fulness', 'ful'), ('ousness', 'ous'), ('aliti', 'al'),
            ('iviti', 'ive'), ('biliti', 'ble')
        ]
        for suffix, replacement in suffixes:
            if word.endswith(suffix):
                stem = word[:-len(suffix)]
                if len(stem) > 1:
                    word = stem + replacement
                break

        # Step 3: Remove more suffixes
        suffixes3 = [
            ('icate', 'ic'), ('ative', ''), ('alize', 'al'),
            ('iciti', 'ic'), ('ical', 'ic'), ('ful', ''), ('ness', '')
        ]
        for suffix, replacement in suffixes3:
            if word.endswith(suffix):
                stem = word[:-len(suffix)]
                if len(stem) > 1:
                    word = stem + replacement
                break

        return word

    def _has_vowel(self, word: str) -> bool:
        """Check if word contains a vowel."""
        return any(c in 'aeiou' for c in word)

    def _step1b_fix(self, word: str) -> str:
        """Fix word after removing -ed or -ing."""
        if word.endswith(('at', 'bl', 'iz')):
            word += 'e'
        elif len(word) >= 2 and word[-1] == word[-2] and word[-1] not in 'lsz':
            word = word[:-1]
        return word

    def __repr__(self) -> str:
        return "Stemmer()"

class TextNormalizer(Transformer):
    """
    Normalize text with multiple operations.

    Parameters
    ----------
    form : str, default='NFKC'
        Unicode normalization form ('NFC', 'NFD', 'NFKC', 'NFKD').
    lowercase : bool, default=True
        Convert to lowercase.
    strip : bool, default=True
        Strip leading/trailing whitespace.
    collapse_whitespace : bool, default=True
        Replace multiple whitespace with single space.

    Examples
    --------
    >>> from tuiml.preprocessing.text import TextNormalizer
    >>> normalizer = TextNormalizer()
    >>> normalizer.transform(["  Hello   World  "])
    ['hello world']
    """

    @classmethod
    def get_parameter_schema(cls) -> dict:
        """Return JSON Schema for parameters."""
        return {
            "form": {
                "type": "string",
                "default": "NFKC",
                "enum": ["NFC", "NFD", "NFKC", "NFKD"],
                "description": "Unicode normalization form"
            },
            "lowercase": {
                "type": "boolean",
                "default": True,
                "description": "Convert to lowercase"
            },
            "strip": {
                "type": "boolean",
                "default": True,
                "description": "Strip leading/trailing whitespace"
            },
            "collapse_whitespace": {
                "type": "boolean",
                "default": True,
                "description": "Replace multiple whitespace with single space"
            }
        }

    def __init__(
        self,
        form: str = 'NFKC',
        lowercase: bool = True,
        strip: bool = True,
        collapse_whitespace: bool = True
    ):
        super().__init__()
        self.form = form
        self.lowercase = lowercase
        self.strip = strip
        self.collapse_whitespace = collapse_whitespace

    def fit(self, X, y=None) -> "TextNormalizer":
        """Fit (no-op)."""
        self._is_fitted = True
        return self

    def transform(self, texts: List[str]) -> List[str]:
        """Normalize texts."""
        result = []
        for text in texts:
            # Unicode normalization
            text = unicodedata.normalize(self.form, text)

            # Lowercase
            if self.lowercase:
                text = text.lower()

            # Collapse whitespace
            if self.collapse_whitespace:
                text = re.sub(r'\s+', ' ', text)

            # Strip
            if self.strip:
                text = text.strip()

            result.append(text)
        return result

    def __repr__(self) -> str:
        return f"TextNormalizer(form='{self.form}')"
