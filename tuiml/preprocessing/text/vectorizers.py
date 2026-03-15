"""
Text vectorization utilities - Convert text to numeric feature vectors.
"""

import numpy as np
from collections import Counter
from typing import Dict, List, Optional, Union, Callable
from scipy import sparse
import math

from tuiml.base.preprocessing import Transformer
from .tokenizers import BaseTokenizer, WordTokenizer

class CountVectorizer(Transformer):
    """Convert a collection of text documents to a matrix of token counts.

    Implements a "bag-of-words" representation where each document is 
    represented by the frequency of terms in the vocabulary.

    Parameters
    ----------
    tokenizer : BaseTokenizer, optional
        The strategy for splitting text into tokens. If ``None``, a 
        :class:`WordTokenizer` is used.

    max_features : int, optional
        The maximum number of terms to include in the vocabulary. If provided, 
        it keeps only the top ``max_features`` ordered by term frequency across 
        the corpus.

    min_df : int or float, default=1
        Minimum document frequency to include a term.
        - ``Int``: Minimum absolute count.
        - ``Float``: Minimum proportion of documents.

    max_df : float, default=1.0
        Maximum proportion of documents a term can appear in to be included. 
        Useful for filtering out corpus-specific stop words.

    binary : bool, default=False
        If ``True``, all non-zero counts are set to 1. This is useful for 
        discrete probabilistic models that only care about presence/absence.

    lowercase : bool, default=True
        If ``True``, converts all text to lowercase before tokenization.

    stop_words : list of str or 'english', optional
        A list of words that will be filtered out. If ``"english"``, a 
        built-in list is used.

    ngram_range : tuple (min_n, max_n), default=(1, 1)
        The lower and upper boundary of the range of n-values for different 
        n-grams to be extracted.

    Attributes
    ----------
    vocabulary_ : dict
        Mapping of terms to feature indices.

    feature_names_ : list of str
        Ordered list of terms corresponding to the column indices.

    See Also
    --------
    :class:`~tuiml.preprocessing.text.TfidfVectorizer` : Counts weighted by importance.
    :class:`~tuiml.preprocessing.text.HashingVectorizer` : Memory-efficient counting.

    Examples
    --------
    Vectorize a small corpus:

    >>> from tuiml.preprocessing.text import CountVectorizer
    >>> docs = ["cat in the hat", "hat on the mat"]
    >>> vectorizer = CountVectorizer(stop_words='english')
    >>> X = vectorizer.fit_transform(docs)
    >>> print(vectorizer.get_feature_names_out())
    ['cat', 'hat', 'mat']
    """

    # English stop words
    ENGLISH_STOP_WORDS = {
        'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
        'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
        'to', 'was', 'were', 'will', 'with', 'the', 'this', 'but', 'they',
        'have', 'had', 'what', 'when', 'where', 'who', 'which', 'why', 'how',
        'all', 'each', 'every', 'both', 'few', 'more', 'most', 'other',
        'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
        'than', 'too', 'very', 'can', 'just', 'should', 'now'
    }

    def __init__(
        self,
        tokenizer: BaseTokenizer = None,
        max_features: int = None,
        min_df: Union[int, float] = 1,
        max_df: float = 1.0,
        binary: bool = False,
        lowercase: bool = True,
        stop_words: Union[List[str], str] = None,
        ngram_range: tuple = (1, 1),
        vocabulary: Dict[str, int] = None
    ):
        super().__init__()
        self.tokenizer = tokenizer or WordTokenizer(lowercase=lowercase)
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.binary = binary
        self.lowercase = lowercase
        self.ngram_range = ngram_range
        self.vocabulary = vocabulary

        # Handle stop words
        if stop_words == 'english':
            self.stop_words = self.ENGLISH_STOP_WORDS
        elif stop_words is not None:
            self.stop_words = set(stop_words)
        else:
            self.stop_words = None

        self.vocabulary_: Dict[str, int] = {}
        self.feature_names_: List[str] = []

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict]:
        """Return JSON Schema for parameters."""
        return {
            "max_features": {
                "type": ["integer", "null"],
                "default": None,
                "minimum": 1,
                "description": "Maximum number of features (vocabulary size)"
            },
            "min_df": {
                "type": ["integer", "number"],
                "default": 1,
                "description": "Minimum document frequency (int=count, float=proportion)"
            },
            "max_df": {
                "type": "number",
                "default": 1.0,
                "minimum": 0.0,
                "maximum": 1.0,
                "description": "Maximum document frequency (proportion)"
            },
            "binary": {
                "type": "boolean",
                "default": False,
                "description": "If True, all non-zero counts are set to 1"
            },
            "lowercase": {
                "type": "boolean",
                "default": True,
                "description": "Convert to lowercase before tokenizing"
            },
            "stop_words": {
                "type": ["array", "string", "null"],
                "default": None,
                "description": "Stop words to remove ('english' or list)"
            },
            "ngram_range": {
                "type": "array",
                "default": [1, 1],
                "description": "Range of n-grams to extract [min_n, max_n]"
            }
        }

    def fit(self, documents: List[str], y=None) -> "CountVectorizer":
        """
        Learn vocabulary from documents.

        Parameters
        ----------
        documents : list of str
            List of text documents.
        y : ignored

        Returns
        -------
        self
        """
        # Tokenize all documents
        tokenized_docs = [self._tokenize(doc) for doc in documents]
        n_docs = len(documents)

        # Count document frequencies
        doc_freqs = Counter()
        for tokens in tokenized_docs:
            unique_tokens = set(tokens)
            for token in unique_tokens:
                doc_freqs[token] += 1

        # Apply document frequency filters
        min_df = self.min_df
        if isinstance(min_df, float):
            min_df = int(min_df * n_docs)

        max_df = int(self.max_df * n_docs)

        # Filter vocabulary
        filtered_vocab = {
            token: freq for token, freq in doc_freqs.items()
            if min_df <= freq <= max_df
        }

        # Sort by frequency (descending) then alphabetically
        sorted_vocab = sorted(
            filtered_vocab.items(),
            key=lambda x: (-x[1], x[0])
        )

        # Apply max_features limit
        if self.max_features:
            sorted_vocab = sorted_vocab[:self.max_features]

        # Build vocabulary mapping
        if self.vocabulary is not None:
            self.vocabulary_ = self.vocabulary
        else:
            self.vocabulary_ = {
                token: idx for idx, (token, _) in enumerate(sorted_vocab)
            }

        self.feature_names_ = sorted(
            self.vocabulary_.keys(),
            key=lambda x: self.vocabulary_[x]
        )

        self._is_fitted = True
        return self

    def transform(self, documents: List[str]) -> np.ndarray:
        """
        Transform documents to count matrix.

        Parameters
        ----------
        documents : list of str
            Documents to transform.

        Returns
        -------
        X : ndarray of shape (n_documents, n_features)
            Document-term matrix.
        """
        self._check_is_fitted()

        n_docs = len(documents)
        n_features = len(self.vocabulary_)

        # Build sparse matrix
        rows, cols, data = [], [], []

        for doc_idx, doc in enumerate(documents):
            tokens = self._tokenize(doc)
            token_counts = Counter(tokens)

            for token, count in token_counts.items():
                if token in self.vocabulary_:
                    rows.append(doc_idx)
                    cols.append(self.vocabulary_[token])
                    data.append(1 if self.binary else count)

        # Create sparse matrix
        X = sparse.csr_matrix(
            (data, (rows, cols)),
            shape=(n_docs, n_features),
            dtype=np.float64
        )

        return X.toarray()

    def fit_transform(self, documents: List[str], y=None) -> np.ndarray:
        """Fit and transform documents."""
        return self.fit(documents, y).transform(documents)

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize and generate n-grams."""
        if self.lowercase:
            text = text.lower()

        tokens = self.tokenizer.tokenize(text)

        # Remove stop words
        if self.stop_words:
            tokens = [t for t in tokens if t not in self.stop_words]

        # Generate n-grams
        min_n, max_n = self.ngram_range
        if min_n == 1 and max_n == 1:
            return tokens

        ngrams = []
        for n in range(min_n, max_n + 1):
            for i in range(len(tokens) - n + 1):
                ngrams.append(' '.join(tokens[i:i + n]))

        return ngrams

    def get_feature_names_out(self) -> List[str]:
        """Get feature names."""
        return self.feature_names_

    def __repr__(self) -> str:
        return (
            f"CountVectorizer(max_features={self.max_features}, "
            f"ngram_range={self.ngram_range})"
        )

class TfidfTransformer(Transformer):
    """Transform a count matrix to a TF-IDF representation.

    Overview
    --------
    TF-IDF (Term Frequency-Inverse Document Frequency) balances the local 
    importance of a term (TF) with its global rarity (IDF).

    Theory
    ------
    The IDF weight for a term :math:`t` is calculated as:

    .. math::
        \\text{idf}(t) = \\log \\frac{n}{df(t)} + 1

    where :math:`n` is the total number of documents and :math:`df(t)` is the 
    number of documents containing term :math:`t`.

    Parameters
    ----------
    norm : {'l1', 'l2', None}, default='l2'
        Normalization strategy for each row:
        - ``"l2"``: Sum of squares is 1 (Euclidean norm).
        - ``"l1"``: Sum of absolute values is 1.

    use_idf : bool, default=True
        If ``True``, enables inverse document frequency reweighting.

    smooth_idf : bool, default=True
        If ``True``, adds 1 to document frequencies to prevent zero division: 
        :math:`\\log \\frac{n+1}{df+1} + 1`.

    sublinear_tf : bool, default=False
        If ``True``, applies sublinear scaling to term frequency:
        :math:`1 + \\log(\\text{tf})`.

    Attributes
    ----------
    idf_ : ndarray
        The learned inverse document frequency vector.

    See Also
    --------
    :class:`~tuiml.preprocessing.text.TfidfVectorizer` : Combined counting and weighting.

    Examples
    --------
    Weight a count matrix:

    >>> from tuiml.preprocessing.text import TfidfTransformer
    >>> import numpy as np
    >>> counts = np.array([[3, 0, 1], [2, 1, 0]])
    >>> transformer = TfidfTransformer()
    >>> X_tfidf = transformer.fit_transform(counts)
    """

    def __init__(
        self,
        norm: str = 'l2',
        use_idf: bool = True,
        smooth_idf: bool = True,
        sublinear_tf: bool = False
    ):
        super().__init__()
        self.norm = norm
        self.use_idf = use_idf
        self.smooth_idf = smooth_idf
        self.sublinear_tf = sublinear_tf

        self.idf_: np.ndarray = None

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict]:
        """Return JSON Schema for parameters."""
        return {
            "norm": {
                "type": ["string", "null"],
                "default": "l2",
                "enum": ["l1", "l2", None],
                "description": "Normalization method"
            },
            "use_idf": {
                "type": "boolean",
                "default": True,
                "description": "Enable inverse document frequency weighting"
            },
            "smooth_idf": {
                "type": "boolean",
                "default": True,
                "description": "Add 1 to document frequencies to prevent zero divisions"
            },
            "sublinear_tf": {
                "type": "boolean",
                "default": False,
                "description": "Apply sublinear TF scaling (1 + log(tf))"
            }
        }

    def fit(self, X: np.ndarray, y=None) -> "TfidfTransformer":
        """
        Learn IDF weights from count matrix.

        Parameters
        ----------
        X : ndarray of shape (n_documents, n_features)
            Document-term count matrix.

        Returns
        -------
        self
        """
        X = np.asarray(X)
        n_docs, n_features = X.shape

        if self.use_idf:
            # Count documents containing each term
            df = np.sum(X > 0, axis=0)

            # Compute IDF
            if self.smooth_idf:
                idf = np.log((n_docs + 1) / (df + 1)) + 1
            else:
                idf = np.log(n_docs / df) + 1

            self.idf_ = idf

        self._is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform count matrix to TF-IDF.

        Parameters
        ----------
        X : ndarray of shape (n_documents, n_features)
            Document-term count matrix.

        Returns
        -------
        X_tfidf : ndarray
            TF-IDF weighted matrix.
        """
        self._check_is_fitted()
        X = np.asarray(X, dtype=np.float64)

        # Apply sublinear TF
        if self.sublinear_tf:
            X = np.where(X > 0, 1 + np.log(X), 0)

        # Apply IDF
        if self.use_idf:
            X = X * self.idf_

        # Normalize
        if self.norm == 'l2':
            norms = np.sqrt(np.sum(X ** 2, axis=1, keepdims=True))
            norms[norms == 0] = 1
            X = X / norms
        elif self.norm == 'l1':
            norms = np.sum(np.abs(X), axis=1, keepdims=True)
            norms[norms == 0] = 1
            X = X / norms

        return X

    def fit_transform(self, X: np.ndarray, y=None) -> np.ndarray:
        """Fit and transform."""
        return self.fit(X, y).transform(X)

    def __repr__(self) -> str:
        return (
            f"TfidfTransformer(norm='{self.norm}', use_idf={self.use_idf})"
        )

class TfidfVectorizer(Transformer):
    """Convert a collection of raw documents to a matrix of TF-IDF features.

    Equivalent to :class:`CountVectorizer` followed by :class:`TfidfTransformer`.

    Parameters
    ----------
    tokenizer : BaseTokenizer, optional
        The strategy for splitting text into tokens.

    max_features : int, optional
        The maximum number of terms to include in the vocabulary.

    min_df : int or float, default=1
        Minimum document frequency (see :class:`CountVectorizer`).

    max_df : float, default=1.0
        Maximum document frequency (see :class:`CountVectorizer`).

    lowercase : bool, default=True
        If ``True``, converts text to lowercase.

    ngram_range : tuple, default=(1, 1)
        The range of n-grams to extract.

    norm : {'l1', 'l2', None}, default='l2'
        Normalization strategy (see :class:`TfidfTransformer`).

    use_idf : bool, default=True
        Enable IDF weighting.

    smooth_idf : bool, default=True
        Smooth IDF weights.

    sublinear_tf : bool, default=False
        Use sublinear TF scaling.

    Attributes
    ----------
    vocabulary_ : dict
        Mapping of terms to feature indices.

    idf_ : ndarray
        The learned inverse document frequency vector.

    Examples
    --------
    Directly compute TF-IDF features:

    >>> from tuiml.preprocessing.text import TfidfVectorizer
    >>> docs = ["the cat", "the dog"]
    >>> vectorizer = TfidfVectorizer()
    >>> X = vectorizer.fit_transform(docs)
    >>> print(X.shape)
    (2, 3)
    """

    def __init__(
        self,
        tokenizer: BaseTokenizer = None,
        max_features: int = None,
        min_df: Union[int, float] = 1,
        max_df: float = 1.0,
        lowercase: bool = True,
        stop_words: Union[List[str], str] = None,
        ngram_range: tuple = (1, 1),
        norm: str = 'l2',
        use_idf: bool = True,
        smooth_idf: bool = True,
        sublinear_tf: bool = False
    ):
        super().__init__()

        self._count_vectorizer = CountVectorizer(
            tokenizer=tokenizer,
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            lowercase=lowercase,
            stop_words=stop_words,
            ngram_range=ngram_range
        )

        self._tfidf_transformer = TfidfTransformer(
            norm=norm,
            use_idf=use_idf,
            smooth_idf=smooth_idf,
            sublinear_tf=sublinear_tf
        )

        # Store parameters
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.lowercase = lowercase
        self.stop_words = stop_words
        self.ngram_range = ngram_range
        self.norm = norm
        self.use_idf = use_idf
        self.smooth_idf = smooth_idf
        self.sublinear_tf = sublinear_tf

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict]:
        """Return JSON Schema for parameters."""
        return {
            "max_features": {
                "type": ["integer", "null"],
                "default": None,
                "description": "Maximum vocabulary size"
            },
            "min_df": {
                "type": ["integer", "number"],
                "default": 1,
                "description": "Minimum document frequency"
            },
            "max_df": {
                "type": "number",
                "default": 1.0,
                "description": "Maximum document frequency"
            },
            "lowercase": {
                "type": "boolean",
                "default": True,
                "description": "Convert to lowercase"
            },
            "stop_words": {
                "type": ["array", "string", "null"],
                "default": None,
                "description": "Stop words to remove"
            },
            "ngram_range": {
                "type": "array",
                "default": [1, 1],
                "description": "N-gram range [min, max]"
            },
            "norm": {
                "type": "string",
                "default": "l2",
                "enum": ["l1", "l2"],
                "description": "Normalization method"
            },
            "use_idf": {
                "type": "boolean",
                "default": True,
                "description": "Use IDF weighting"
            },
            "smooth_idf": {
                "type": "boolean",
                "default": True,
                "description": "Smooth IDF weights"
            },
            "sublinear_tf": {
                "type": "boolean",
                "default": False,
                "description": "Use sublinear TF"
            }
        }

    def fit(self, documents: List[str], y=None) -> "TfidfVectorizer":
        """
        Learn vocabulary and IDF weights.

        Parameters
        ----------
        documents : list of str
            Training documents.

        Returns
        -------
        self
        """
        X_counts = self._count_vectorizer.fit_transform(documents)
        self._tfidf_transformer.fit(X_counts)
        self._is_fitted = True
        return self

    def transform(self, documents: List[str]) -> np.ndarray:
        """
        Transform documents to TF-IDF matrix.

        Parameters
        ----------
        documents : list of str
            Documents to transform.

        Returns
        -------
        X : ndarray of shape (n_documents, n_features)
            TF-IDF matrix.
        """
        self._check_is_fitted()
        X_counts = self._count_vectorizer.transform(documents)
        return self._tfidf_transformer.transform(X_counts)

    def fit_transform(self, documents: List[str], y=None) -> np.ndarray:
        """Fit and transform documents."""
        X_counts = self._count_vectorizer.fit_transform(documents)
        X_tfidf = self._tfidf_transformer.fit_transform(X_counts)
        self._is_fitted = True
        return X_tfidf

    @property
    def vocabulary_(self) -> Dict[str, int]:
        """Get vocabulary mapping."""
        return self._count_vectorizer.vocabulary_

    @property
    def idf_(self) -> np.ndarray:
        """Get IDF weights."""
        return self._tfidf_transformer.idf_

    def get_feature_names_out(self) -> List[str]:
        """Get feature names."""
        return self._count_vectorizer.get_feature_names_out()

    def __repr__(self) -> str:
        return (
            f"TfidfVectorizer(max_features={self.max_features}, "
            f"ngram_range={self.ngram_range})"
        )

class HashingVectorizer(Transformer):
    """Convert text to a fixed-size feature matrix using the hashing trick.

    Overview
    --------
    HashingVectorizer is a memory-efficient alternative to CountVectorizer. 
    It doesn't store a vocabulary in memory, but instead hashes terms directly 
    to a fixed number of buckets.

    Notes
    -----
    **Pros:**
    - Very low memory footprint (does not store vocabulary).
    - Can handle large, out-of-core datasets.
    
    **Cons:**
    - No inverse mapping (cannot retrieve the original words from indices).
    - Potential for hash collisions.

    Parameters
    ----------
    n_features : int, default=2**20
        The number of bins in the hash table. Larger values reduce collisions 
        but increase memory usage of output matrices.

    tokenizer : BaseTokenizer, optional
        The strategy for splitting text into tokens.

    lowercase : bool, default=True
        If ``True``, converts text to lowercase.

    ngram_range : tuple, default=(1, 1)
        The range of n-grams to extract.

    binary : bool, default=False
        If ``True``, all non-zero counts are set to 1.

    norm : {'l1', 'l2', None}, default='l2'
        Normalization strategy for output vectors.

    Examples
    --------
    Memory-efficient vectorization:

    >>> from tuiml.preprocessing.text import HashingVectorizer
    >>> docs = ["cat in the hat"]
    >>> vectorizer = HashingVectorizer(n_features=128)
    >>> X = vectorizer.fit_transform(docs)
    >>> print(X.shape)
    (1, 128)
    """

    def __init__(
        self,
        n_features: int = 2**20,
        tokenizer: BaseTokenizer = None,
        lowercase: bool = True,
        stop_words: Union[List[str], str] = None,
        ngram_range: tuple = (1, 1),
        binary: bool = False,
        norm: str = 'l2'
    ):
        super().__init__()
        self.n_features = n_features
        self.tokenizer = tokenizer or WordTokenizer(lowercase=lowercase)
        self.lowercase = lowercase
        self.ngram_range = ngram_range
        self.binary = binary
        self.norm = norm

        if stop_words == 'english':
            self.stop_words = CountVectorizer.ENGLISH_STOP_WORDS
        elif stop_words is not None:
            self.stop_words = set(stop_words)
        else:
            self.stop_words = None

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict]:
        """Return JSON Schema for parameters."""
        return {
            "n_features": {
                "type": "integer",
                "default": 1048576,
                "minimum": 1,
                "description": "Number of features (hash buckets)"
            },
            "lowercase": {
                "type": "boolean",
                "default": True,
                "description": "Convert to lowercase"
            },
            "stop_words": {
                "type": ["array", "string", "null"],
                "default": None,
                "description": "Stop words to remove"
            },
            "ngram_range": {
                "type": "array",
                "default": [1, 1],
                "description": "N-gram range [min, max]"
            },
            "binary": {
                "type": "boolean",
                "default": False,
                "description": "If True, use binary counts"
            },
            "norm": {
                "type": "string",
                "default": "l2",
                "enum": ["l1", "l2"],
                "description": "Normalization method"
            }
        }

    def fit(self, documents: List[str], y=None) -> "HashingVectorizer":
        """Fit (no-op for hashing vectorizer)."""
        self._is_fitted = True
        return self

    def transform(self, documents: List[str]) -> np.ndarray:
        """
        Transform documents to hashed feature matrix.

        Parameters
        ----------
        documents : list of str
            Documents to transform.

        Returns
        -------
        X : ndarray of shape (n_documents, n_features)
            Hashed feature matrix.
        """
        n_docs = len(documents)
        X = np.zeros((n_docs, self.n_features))

        for doc_idx, doc in enumerate(documents):
            tokens = self._tokenize(doc)
            token_counts = Counter(tokens)

            for token, count in token_counts.items():
                # Hash token to feature index
                feature_idx = hash(token) % self.n_features
                X[doc_idx, feature_idx] += 1 if self.binary else count

        # Normalize
        if self.norm == 'l2':
            norms = np.sqrt(np.sum(X ** 2, axis=1, keepdims=True))
            norms[norms == 0] = 1
            X = X / norms
        elif self.norm == 'l1':
            norms = np.sum(np.abs(X), axis=1, keepdims=True)
            norms[norms == 0] = 1
            X = X / norms

        return X

    def fit_transform(self, documents: List[str], y=None) -> np.ndarray:
        """Fit and transform."""
        return self.fit(documents, y).transform(documents)

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize and generate n-grams."""
        if self.lowercase:
            text = text.lower()

        tokens = self.tokenizer.tokenize(text)

        if self.stop_words:
            tokens = [t for t in tokens if t not in self.stop_words]

        min_n, max_n = self.ngram_range
        if min_n == 1 and max_n == 1:
            return tokens

        ngrams = []
        for n in range(min_n, max_n + 1):
            for i in range(len(tokens) - n + 1):
                ngrams.append(' '.join(tokens[i:i + n]))

        return ngrams

    def __repr__(self) -> str:
        return f"HashingVectorizer(n_features={self.n_features})"
