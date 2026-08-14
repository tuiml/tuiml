"""BOSS - Bag-of-SFA-Symbols dictionary classification."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from tuiml._cpp_ext import timeseries as _cpp_ts
from tuiml.algorithms.timeseries.classification._base import TimeSeriesClassifier
from tuiml.base.algorithms import classifier


def _boss_distance(query: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Compute the asymmetric BOSS distance from one histogram to many.

    Only words the **query** actually contains contribute. A word absent from
    the query is ignored even if the reference has it in abundance, which is
    what makes the measure robust to the long tail of noise words a reference
    series happens to carry.

    Parameters
    ----------
    query : np.ndarray of shape (n_words,)
        Word-count histogram of the query series.
    reference : np.ndarray of shape (n_references, n_words)
        Word-count histograms to compare against.

    Returns
    -------
    distances : np.ndarray of shape (n_references,)
        Asymmetric squared distance to each reference.
    """
    present = query > 0
    if not present.any():
        return np.zeros(len(reference))
    difference = query[present] - reference[:, present]
    return (difference ** 2).sum(axis=1)


@classifier(
    tags=["timeseries", "classification", "dictionary", "symbolic"],
    version="1.0.0",
)
class BOSSClassifier(TimeSeriesClassifier):
    """BOSS classifies by **which symbolic patterns a series contains**.

    BOSS turns each series into a *bag of words*. Every sliding window is
    reduced to its lowest Fourier coefficients, those are quantised into
    letters, and the resulting words are counted. Two series are similar when
    they contain the same patterns in similar proportions — regardless of
    *where* those patterns occur.

    The low-pass step is what distinguishes it: keeping only low-frequency
    coefficients discards high-frequency detail, which is where much of the
    noise lives.

    Overview
    --------
    1. Slide a window over each series and take the DFT of each window.
    2. Keep the lowest ``word_length`` coefficients, dropping the DC term and
       scaling by the window's standard deviation.
    3. Quantise each coefficient into one of ``alphabet_size`` letters, using
       breakpoints fitted from the training data (Multiple Coefficient
       Binning).
    4. Drop a word identical to its predecessor, so a slowly varying stretch
       contributes one word rather than hundreds.
    5. Count the surviving words per series and classify by nearest neighbour
       under the asymmetric BOSS distance.

    Theory
    ------
    For histograms :math:`B_a, B_b` the BOSS distance is

    .. math::
        d(a, b) = \\sum_{s \\ \\in \\ B_a,\\ B_a(s) > 0}
        \\left( B_a(s) - B_b(s) \\right)^2

    The restriction to words present in :math:`a` makes it **asymmetric**:
    :math:`d(a, b) \\neq d(b, a)` in general. That is deliberate, not an
    oversight — a reference series carrying extra noise words should not be
    penalised for them when the query does not contain them.

    Numerosity reduction — collapsing runs of identical consecutive words —
    matters more than it looks. Without it a long flat stretch dominates the
    histogram purely because it is long, and the representation stops
    describing structure and starts describing duration.

    Parameters
    ----------
    window_size : int or float, default=0.25
        Sliding window length. A float in ``(0, 1]`` is a fraction of the
        series length. Sets the time scale of the patterns BOSS can see.
    word_length : int, default=8
        Number of Fourier coefficients retained per window. Longer words
        capture more shape detail and tolerate less noise.
    alphabet_size : int, default=4
        Letters per coefficient. Four is the near-universal choice; the
        method is famously insensitive to this.
    norm_mean : bool, default=True
        Whether to drop each window's mean, making the representation
        invariant to local offset.
    n_neighbors : int, default=1
        Neighbours to vote under the BOSS distance.

    Attributes
    ----------
    breakpoints_ : np.ndarray of shape (word_length, alphabet_size - 1)
        Fitted quantisation breakpoints per coefficient.
    histograms_ : np.ndarray of shape (n_samples, n_words)
        Training word-count histograms.
    vocabulary_ : np.ndarray of shape (n_words,)
        The word codes actually observed during :meth:`fit`.
    classes_ : np.ndarray of shape (n_classes,)
        Class labels seen during :meth:`fit`.

    Notes
    -----
    **Complexity.** Fitting is :math:`O(n L \\ell)` for ``n`` series of length
    ``L`` and word length :math:`\\ell` — the sliding DFT is advanced by the
    momentary Fourier transform in the shared C++ kernel
    ``tuiml._cpp_ext.timeseries.sfa_transform``, so each window costs
    :math:`O(\\ell)` rather than :math:`O(w \\ell)`. Prediction is
    :math:`O(m n V)` for a vocabulary of size ``V``, since it is a nearest
    neighbour search over histograms.

    **When to use — and where it does not win.** BOSS represents a series by
    *what patterns it contains and how often*, discarding where they occur.
    That is a genuinely different view from every other member of this family,
    and it shows up where position-invariant content is the signal. Measured
    on two synthetic problems, 160 train / 160 test:

    =================================  ======  ==========  =========  =====  =========
    problem                            BOSS    MiniRocket  Shapelet   DTW    Euclidean
    =================================  ======  ==========  =========  =====  =========
    motif **count** (2 vs 6 repeats    0.844   **1.000**   0.956      0.950  0.588
    at random positions)
    frequency under heavy noise        0.775   **0.944**   0.831      0.656  0.869
    (sd 3.0)
    =================================  ======  ==========  =========  =====  =========

    Read that honestly: BOSS beats a Euclidean neighbour decisively when
    position varies (0.844 against 0.588) and beats DTW under noise, but
    :class:`~tuiml.algorithms.timeseries.classification.MiniRocketClassifier`
    beat it on both. If you want one classifier, use MINIROCKET. BOSS earns
    its place as a **diverse component** — a symbolic, frequency-domain view
    that fails differently from convolutional and elastic methods, which is
    precisely why every strong meta-ensemble in the literature includes a
    dictionary member. Combine it through
    :class:`~tuiml.algorithms.ensemble.VotingClassifier`.

    Its cost also grows with the training set, like
    :class:`~tuiml.algorithms.timeseries.classification.DTWNeighborsClassifier`
    and unlike MINIROCKET, so it suits small to moderate datasets.

    The published method ensembles many ``(window_size, word_length,
    norm_mean)`` settings and keeps those within 92% of the best
    cross-validated accuracy. This class implements a **single** parameter
    set, which is weaker; on the noise problem above, sweeping window sizes
    from 25 to 150 and word lengths from 4 to 10 moved accuracy only from
    0.775 to 0.781, so the gap to MINIROCKET there is not a tuning artefact.

    References
    ----------
    .. [Schafer2015] Schäfer, P. (2015). The BOSS is Concerned with Time
       Series Classification in the Presence of Noise. *Data Mining and
       Knowledge Discovery*, 29(6), 1505-1530.
       :doi:`10.1007/s10618-014-0377-7`
    .. [Schafer2012] Schäfer, P., & Högqvist, M. (2012). SFA: A Symbolic
       Fourier Approximation and Index for Similarity Search in High
       Dimensional Datasets. *EDBT*, 516-527.
       :doi:`10.1145/2247596.2247656`

    See Also
    --------
    :class:`~tuiml.algorithms.timeseries.classification.MiniRocketClassifier` : Faster, and its cost does not grow with the training set.
    :class:`~tuiml.algorithms.timeseries.classification.ShapeletTransformClassifier` : Also pattern-based, but interpretable and not noise-tolerant.

    Examples
    --------
    >>> import numpy as np
    >>> from tuiml.algorithms.timeseries.classification import BOSSClassifier
    >>> rng = np.random.default_rng(0)
    >>> t = np.linspace(0, 8 * np.pi, 160)
    >>> # Two classes distinguished by frequency, buried in heavy noise.
    >>> slow = np.sin(t) + rng.normal(0, 0.8, (40, 160))
    >>> fast = np.sin(3 * t) + rng.normal(0, 0.8, (40, 160))
    >>> X = np.vstack([slow, fast])
    >>> y = np.array([0] * 40 + [1] * 40)
    >>> model = BOSSClassifier(window_size=40, word_length=6).fit(X, y)
    >>> bool((model.predict(X) == y).mean() > 0.9)
    True
    """

    def __init__(
        self,
        window_size: float = 0.25,
        word_length: int = 8,
        alphabet_size: int = 4,
        norm_mean: bool = True,
        n_neighbors: int = 1,
    ):
        """Initialize the BOSS classifier.

        Parameters
        ----------
        window_size : int or float, default=0.25
            Sliding window length, absolute or as a fraction of series length.
        word_length : int, default=8
            Fourier coefficients retained per window.
        alphabet_size : int, default=4
            Letters per coefficient.
        norm_mean : bool, default=True
            Drop each window's mean.
        n_neighbors : int, default=1
            Neighbours to vote.
        """
        super().__init__()
        if word_length < 1:
            raise ValueError(f"word_length must be at least 1, got {word_length}")
        if alphabet_size < 2:
            raise ValueError(
                f"alphabet_size must be at least 2, got {alphabet_size}"
            )
        if n_neighbors < 1:
            raise ValueError(f"n_neighbors must be at least 1, got {n_neighbors}")
        self.window_size = window_size
        self.word_length = word_length
        self.alphabet_size = alphabet_size
        self.norm_mean = norm_mean
        self.n_neighbors = n_neighbors

        # Fitted attributes
        self.breakpoints_ = None
        self.histograms_ = None
        self.vocabulary_ = None
        self.y_train_ = None
        self.window_size_ = None

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        """Return JSON Schema for algorithm parameters."""
        return {
            "window_size": {
                "type": "number",
                "default": 0.25,
                "description": "Window length, absolute or fraction of series length"
            },
            "word_length": {
                "type": "integer",
                "default": 8,
                "minimum": 1,
                "description": "Fourier coefficients retained per window"
            },
            "alphabet_size": {
                "type": "integer",
                "default": 4,
                "minimum": 2,
                "description": "Letters each coefficient is quantised into"
            },
            "norm_mean": {
                "type": "boolean",
                "default": True,
                "description": "Drop each window's mean"
            },
            "n_neighbors": {
                "type": "integer",
                "default": 1,
                "minimum": 1,
                "description": "Neighbors to vote under the BOSS distance"
            }
        }

    @classmethod
    def get_capabilities(cls) -> List[str]:
        """Return supported capabilities."""
        return [
            "numeric",
            "multiclass",
            "timeseries",
            "multivariate_timeseries",
            "noise_tolerant",
        ]

    @classmethod
    def get_complexity(cls) -> str:
        """Return complexity analysis."""
        return "Fit: O(n*L*word_length), Predict: O(m*n*V) for vocabulary size V"

    @classmethod
    def get_references(cls) -> List[str]:
        """Return academic citations."""
        return [
            "Schafer, P., 2015. The BOSS is concerned with time series "
            "classification in the presence of noise. DMKD.",
            "Schafer, P. and Hogqvist, M., 2012. SFA: a symbolic Fourier "
            "approximation and index for similarity search in high dimensional "
            "datasets. EDBT."
        ]

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BOSSClassifier":
        """Fit the quantisation breakpoints and build training histograms.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_timepoints) or (n_samples, n_channels, n_timepoints)
            Training series.
        y : np.ndarray of shape (n_samples,)
            Training labels.

        Returns
        -------
        self : BOSSClassifier
            The fitted classifier.
        """
        panel, y = self._validate_fit(X, y)
        self.window_size_ = self._resolve_window(panel.shape[2])

        coefficients = self._sfa(panel)
        self.breakpoints_ = self._fit_breakpoints(coefficients)

        words = self._to_words(coefficients)
        self.vocabulary_ = np.unique(np.concatenate(words))
        self.histograms_ = self._to_histograms(words)
        self.y_train_ = y
        self._is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Return the word-count histogram of each series.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_timepoints) or (n_samples, n_channels, n_timepoints)
            Series to transform.

        Returns
        -------
        histograms : np.ndarray of shape (n_samples, n_words)
            Counts over the fitted vocabulary.
        """
        panel = self._validate_predict(X)
        return self._to_histograms(self._to_words(self._sfa(panel)))

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Classify each series by nearest neighbour under the BOSS distance.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_timepoints) or (n_samples, n_channels, n_timepoints)
            Series to classify.

        Returns
        -------
        y_pred : np.ndarray of shape (n_samples,)
            Predicted labels.
        """
        proba = self.predict_proba(X)
        return self.classes_[proba.argmax(axis=1)]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return the neighbour vote share for each class.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_timepoints) or (n_samples, n_channels, n_timepoints)
            Series to classify.

        Returns
        -------
        proba : np.ndarray of shape (n_samples, n_classes)
            Vote shares, rows summing to one.
        """
        histograms = self.transform(X)
        n_classes = len(self.classes_)
        proba = np.zeros((len(histograms), n_classes))

        k = min(self.n_neighbors, len(self.histograms_))
        for i, histogram in enumerate(histograms):
            distances = _boss_distance(histogram, self.histograms_)
            nearest = np.argpartition(distances, k - 1)[:k]
            for label in self.y_train_[nearest]:
                proba[i, np.searchsorted(self.classes_, label)] += 1.0

        return proba / proba.sum(axis=1, keepdims=True)

    def _resolve_window(self, n_timepoints: int) -> int:
        """Resolve the window length to an absolute number of time steps.

        Parameters
        ----------
        n_timepoints : int
            Series length.

        Returns
        -------
        window : int
            Window length, at least as long as the word it must produce.
        """
        if isinstance(self.window_size, float) and 0.0 < self.window_size <= 1.0:
            window = int(round(self.window_size * n_timepoints))
        else:
            window = int(self.window_size)

        # The window must hold enough samples for the coefficients requested.
        minimum = max(self.word_length + 2, 4)
        window = max(minimum, min(window, n_timepoints))
        if window > n_timepoints:
            raise ValueError(
                f"window_size {window} exceeds the series length {n_timepoints}"
            )
        return window

    def _sfa(self, panel: np.ndarray) -> np.ndarray:
        """Compute SFA coefficients for every window of every series.

        Channels are stacked along the window axis, so a word carries no
        channel identity — a multivariate series is treated as one pooled bag
        of patterns.

        Parameters
        ----------
        panel : np.ndarray of shape (n_samples, n_channels, n_timepoints)
            Series to transform.

        Returns
        -------
        coefficients : np.ndarray of shape (n_samples, n_windows_total, word_length)
            Retained Fourier coefficients per window.
        """
        blocks = [
            np.asarray(
                _cpp_ts.sfa_transform(
                    np.ascontiguousarray(panel[:, channel, :]),
                    self.window_size_,
                    self.word_length,
                    self.norm_mean,
                )
            )
            for channel in range(panel.shape[1])
        ]
        return blocks[0] if len(blocks) == 1 else np.concatenate(blocks, axis=1)

    def _fit_breakpoints(self, coefficients: np.ndarray) -> np.ndarray:
        """Fit equal-depth quantisation breakpoints per coefficient.

        Multiple Coefficient Binning gives each coefficient its own
        breakpoints, drawn from the training distribution, so every letter is
        used about equally often. Fixed Gaussian breakpoints would waste most
        of the alphabet on coefficients whose spread differs from the assumed
        normal.

        Parameters
        ----------
        coefficients : np.ndarray of shape (n_samples, n_windows, word_length)
            Training coefficients.

        Returns
        -------
        breakpoints : np.ndarray of shape (word_length, alphabet_size - 1)
            Ascending breakpoints per coefficient.
        """
        flat = coefficients.reshape(-1, self.word_length)
        quantiles = np.linspace(0, 1, self.alphabet_size + 1)[1:-1]
        breakpoints = np.quantile(flat, quantiles, axis=0).T

        # Ties collapse breakpoints onto each other and silently shrink the
        # alphabet; nudging them apart keeps every letter reachable.
        for row in breakpoints:
            for i in range(1, len(row)):
                if row[i] <= row[i - 1]:
                    row[i] = np.nextafter(row[i - 1], np.inf)
        return breakpoints

    def _to_words(self, coefficients: np.ndarray) -> List[np.ndarray]:
        """Quantise coefficients into word codes, with numerosity reduction.

        Parameters
        ----------
        coefficients : np.ndarray of shape (n_samples, n_windows, word_length)
            Coefficients to quantise.

        Returns
        -------
        words : list of np.ndarray
            One array of integer word codes per series, consecutive duplicates
            removed.
        """
        letters = np.empty(coefficients.shape, dtype=np.int64)
        for f in range(self.word_length):
            letters[:, :, f] = np.searchsorted(
                self.breakpoints_[f], coefficients[:, :, f], side="right"
            )

        # Pack the letters into one integer per window, base alphabet_size.
        powers = self.alphabet_size ** np.arange(self.word_length, dtype=np.int64)
        codes = (letters * powers).sum(axis=2)

        words = []
        for row in codes:
            if len(row) == 0:
                words.append(row)
                continue
            # Numerosity reduction: a run of identical words counts once, so
            # a long unchanging stretch cannot swamp the histogram.
            keep = np.ones(len(row), dtype=bool)
            keep[1:] = row[1:] != row[:-1]
            words.append(row[keep])
        return words

    def _to_histograms(self, words: List[np.ndarray]) -> np.ndarray:
        """Count word occurrences over the fitted vocabulary.

        Parameters
        ----------
        words : list of np.ndarray
            Word codes per series.

        Returns
        -------
        histograms : np.ndarray of shape (n_samples, n_words)
            Counts aligned to ``vocabulary_``. Words unseen during fitting are
            dropped, since they cannot contribute to any comparison.
        """
        histograms = np.zeros((len(words), len(self.vocabulary_)))
        for i, row in enumerate(words):
            if len(row) == 0:
                continue
            index = np.searchsorted(self.vocabulary_, row)
            known = (index < len(self.vocabulary_)) & (
                self.vocabulary_[np.minimum(index, len(self.vocabulary_) - 1)] == row
            )
            np.add.at(histograms[i], index[known], 1.0)
        return histograms

    def __repr__(self) -> str:
        """Return a readable representation of the classifier."""
        return (
            f"BOSSClassifier(window_size={self.window_size}, "
            f"word_length={self.word_length}, "
            f"alphabet_size={self.alphabet_size})"
        )
