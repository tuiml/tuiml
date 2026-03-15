"""String Kernel implementation."""

import numpy as np
from typing import Dict, Any, List, Optional

from tuiml.base.kernels import CachedKernel, kernel

@kernel(tags=["string", "text", "sequence"], version="1.0.0")
class StringKernel(CachedKernel):
    """String Subsequence Kernel (SSK) for **text and sequence data**.

    The String Kernel measures similarity between strings by counting
    **weighted common subsequences**. Subsequences need not be contiguous ---
    gaps are penalized by a **decay factor** :math:`\\lambda`, allowing
    the kernel to capture long-range dependencies in text.

    Overview
    --------
    The kernel evaluation proceeds as follows:

    1. Enumerate all common **subsequences** of length up to ``subsequence_length``
    2. Weight each subsequence occurrence by :math:`\\lambda^{\\ell}` where :math:`\\ell` accounts for gap penalties
    3. Sum the weighted counts to produce the raw kernel value
    4. Optionally **normalize** by dividing by :math:`\\sqrt{K(s,s) \\cdot K(t,t)}`

    An efficient :math:`O(n \\cdot m \\cdot k)` **dynamic programming** algorithm
    is used, where :math:`n, m` are string lengths and :math:`k` is the
    maximum subsequence length.

    Theory
    ------
    The string subsequence kernel is defined as:

    .. math::
        K(s, t) = \\sum_{u \\in \\Sigma^{\\leq k}} \\sum_{\\mathbf{i}: s[\\mathbf{i}]=u} \\sum_{\\mathbf{j}: t[\\mathbf{j}]=u} \\lambda^{|\\mathbf{i}| + |\\mathbf{j}|}

    where:

    - :math:`\\Sigma^{\\leq k}` --- Set of all subsequences up to length :math:`k`
    - :math:`\\mathbf{i}, \\mathbf{j}` --- Index tuples locating the subsequence in each string
    - :math:`\\lambda \\in (0, 1)` --- Decay factor penalizing gaps between matched characters
    - :math:`|\\mathbf{i}|` --- Span of the index tuple (accounts for non-contiguous matches)

    The normalized version is:

    .. math::
        \\hat{K}(s, t) = \\frac{K(s, t)}{\\sqrt{K(s, s) \\cdot K(t, t)}}

    Parameters
    ----------
    subsequence_length : int, default=3
        Maximum length of subsequences to consider.

    lambda_decay : float, default=0.5
        Decay factor for gaps in subsequences. Must be in ``(0, 1]``.

    normalize : bool, default=True
        Whether to normalize kernel values to :math:`[0, 1]`.

    cache_size : int, default=250007
        Maximum number of cached kernel evaluations.

    Attributes
    ----------
    n_samples\\_ : int
        Number of training strings stored after ``build()``.

    Notes
    -----
    **Complexity:**

    - Single evaluation: :math:`O(n \\cdot m \\cdot k)` where :math:`n, m` are string lengths, :math:`k` = subsequence length
    - Matrix computation: :math:`O(N^2 \\cdot \\bar{n}^2 \\cdot k)` where :math:`N` = number of strings, :math:`\\bar{n}` = average string length

    **When to use StringKernel:**

    - Text classification (spam detection, sentiment analysis)
    - Biological sequence analysis (protein or DNA similarity)
    - When bag-of-words representations lose important sequential information
    - When subsequence-level similarity is more informative than exact matching

    References
    ----------
    .. [Lodhi2002] Lodhi, H., Saunders, C., Shawe-Taylor, J., Cristianini, N. and Watkins, C. (2002).
           **Text Classification Using String Kernels.**
           *Journal of Machine Learning Research*, 2, pp. 419-444.

    .. [Leslie2002] Leslie, C., Eskin, E. and Noble, W.S. (2002).
           **The Spectrum Kernel: A String Kernel for SVM Protein Classification.**
           *Pacific Symposium on Biocomputing*, pp. 564-575.

    See Also
    --------
    :class:`~tuiml.algorithms.svm.kernels.RBFKernel` : Gaussian RBF kernel for numeric data.
    :class:`~tuiml.algorithms.svm.kernels.PrecomputedKernel` : Precomputed kernel matrix for custom kernels.

    Examples
    --------
    Basic usage for text similarity:

    >>> from tuiml.algorithms.svm.kernels import StringKernel
    >>>
    >>> kernel = StringKernel(subsequence_length=3, lambda_decay=0.5)
    >>> kernel.build(["hello world", "hello there", "goodbye world"])
    StringKernel(...)
    >>> value = kernel.compute(0, 1)
    """

    def __init__(self, subsequence_length: int = 3,
                 lambda_decay: float = 0.5,
                 normalize: bool = True,
                 cache_size: int = 250007):
        """Initialize string kernel.

        Parameters
        ----------
        subsequence_length : int, default=3
            Maximum length of subsequences to consider.
        lambda_decay : float, default=0.5
            Decay factor for gaps in subsequences.
        normalize : bool, default=True
            Whether to normalize kernel values.
        cache_size : int, default=250007
            Maximum number of cached kernel evaluations.
        """
        super().__init__(cache_size=cache_size)
        self.subsequence_length = subsequence_length
        self.lambda_decay = lambda_decay
        self.normalize = normalize
        self._strings: List[str] = []
        self._self_similarities: Optional[np.ndarray] = None

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        """Return parameter schema."""
        return {
            "subsequence_length": {
                "type": "integer",
                "default": 3,
                "minimum": 1,
                "description": "Maximum subsequence length"
            },
            "lambda_decay": {
                "type": "number",
                "default": 0.5,
                "minimum": 0,
                "maximum": 1,
                "description": "Decay factor for gaps"
            },
            "normalize": {
                "type": "boolean",
                "default": True,
                "description": "Normalize kernel values"
            }
        }

    def build(self, X) -> "StringKernel":
        """Build kernel with string data.

        Parameters
        ----------
        X : array-like or list of str
            List of strings or array where each row represents a sequence.

        Returns
        -------
        self : StringKernel
            Returns the built instance.
        """
        # Convert to list of strings if needed
        if isinstance(X, np.ndarray):
            if X.dtype.kind in ['U', 'S', 'O']:
                self._strings = [str(s) for s in X]
            else:
                # Assume character indices
                self._strings = [''.join(chr(int(c)) for c in row if c >= 0)
                                for row in X]
        else:
            self._strings = [str(s) for s in X]

        self.n_samples_ = len(self._strings)
        self._is_built = True

        # Precompute self-similarities for normalization
        if self.normalize:
            self._self_similarities = np.array([
                self._compute_ssk(s, s) for s in self._strings
            ])

        return self

    def _compute_ssk(self, s: str, t: str) -> float:
        """Compute string subsequence kernel using dynamic programming.

        Parameters
        ----------
        s : str
            First string.
        t : str
            Second string.

        Returns
        -------
        val : float
            Kernel value (unnormalized).
        """
        n = len(s)
        m = len(t)
        k = self.subsequence_length
        lambda_ = self.lambda_decay

        if n == 0 or m == 0 or k == 0:
            return 0.0

        # Dynamic programming tables
        # K_prime[i][j][l] = sum of weighted matches for substrings s[:i], t[:j]
        # with subsequence length l (but not necessarily ending at i,j)
        # K[l] = sum of weighted matches with subsequence length l

        # Simplified O(n*m*k) algorithm
        K_prime = np.zeros((k, n, m))
        K = np.zeros(k)

        # Base case: length 1 subsequences
        for i in range(n):
            for j in range(m):
                if s[i] == t[j]:
                    K_prime[0, i, j] = lambda_ ** 2
                    K[0] += K_prime[0, i, j]

        # Recursive case for longer subsequences
        for l in range(1, k):
            for i in range(l, n):
                running_sum = 0.0
                for j in range(l, m):
                    running_sum = lambda_ * running_sum
                    if s[i - 1] == t[j - 1]:
                        running_sum += K_prime[l - 1, i - 2, j - 2] if i > 1 and j > 1 else 0

                    if s[i] == t[j]:
                        K_prime[l, i, j] = lambda_ ** 2 * running_sum
                        K[l] += K_prime[l, i, j]

        return float(np.sum(K))

    def evaluate(self, x1, x2) -> float:
        """Evaluate string kernel between two strings.

        Parameters
        ----------
        x1 : str or int
            First string or its index in the built data.
        x2 : str or int
            Second string or its index in the built data.

        Returns
        -------
        val : float
            String kernel value.
        """
        # Handle string inputs directly
        if isinstance(x1, str) and isinstance(x2, str):
            s1, s2 = x1, x2
        elif isinstance(x1, (int, np.integer)) and isinstance(x2, (int, np.integer)):
            s1 = self._strings[int(x1)]
            s2 = self._strings[int(x2)]
        else:
            s1 = str(x1) if not isinstance(x1, str) else x1
            s2 = str(x2) if not isinstance(x2, str) else x2

        k_xy = self._compute_ssk(s1, s2)

        if self.normalize:
            k_xx = self._compute_ssk(s1, s1)
            k_yy = self._compute_ssk(s2, s2)
            normalizer = np.sqrt(k_xx * k_yy)
            if normalizer == 0:
                return 0.0
            return k_xy / normalizer

        return k_xy

    def compute(self, i: int, j: int) -> float:
        """Compute kernel between training strings i and j.

        Parameters
        ----------
        i : int
            Index of first string.
        j : int
            Index of second string.

        Returns
        -------
        val : float
            Kernel value for strings at indices i and j.
        """
        self._check_is_built()

        if self.cache_size != -1:
            key = (min(i, j), max(i, j))
            if key in self._cache:
                self._cache_hits += 1
                return self._cache[key]
            self._cache_misses += 1

        k_xy = self._compute_ssk(self._strings[i], self._strings[j])

        if self.normalize and self._self_similarities is not None:
            normalizer = np.sqrt(self._self_similarities[i] *
                                self._self_similarities[j])
            value = k_xy / normalizer if normalizer > 0 else 0.0
        else:
            value = k_xy

        if self.cache_size != -1:
            if self.cache_size == 0 or len(self._cache) < self.cache_size:
                self._cache[key] = value

        return value

    def __repr__(self) -> str:
        """String representation."""
        return (f"StringKernel(subsequence_length={self.subsequence_length}, "
               f"lambda_decay={self.lambda_decay})")
