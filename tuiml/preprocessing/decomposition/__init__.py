"""
Decomposition transformers for dimensionality reduction and feature extraction.

Available:
    - SklearnPCA: Scikit-Learn Principal Component Analysis
    - SklearnIncrementalPCA: Scikit-Learn Incremental PCA
    - SklearnKernelPCA: Scikit-Learn Kernel PCA
    - SklearnSparsePCA: Scikit-Learn Sparse PCA
    - SklearnMiniBatchSparsePCA: Scikit-Learn Mini-batch Sparse PCA
    - SklearnFactorAnalysis: Scikit-Learn Factor Analysis
    - SklearnFastICA: Scikit-Learn Fast Independent Component Analysis
    - SklearnDictionaryLearning: Scikit-Learn Dictionary Learning
    - SklearnMiniBatchDictionaryLearning: Scikit-Learn Mini-batch Dictionary Learning
    - SklearnNMF: Scikit-Learn Non-Negative Matrix Factorization
    - SklearnMiniBatchNMF: Scikit-Learn Mini-Batch NMF
    - SklearnLatentDirichletAllocation: Scikit-Learn Latent Dirichlet Allocation
    - SklearnSparseCoder: Scikit-Learn Sparse Coder
    - SklearnTruncatedSVD: Scikit-Learn Truncated SVD
"""

try:
    import sklearn
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

if SKLEARN_AVAILABLE:
    from tuiml.preprocessing.decomposition.sklearn_pca import SklearnPCA
    from tuiml.preprocessing.decomposition.sklearn_incremental_pca import SklearnIncrementalPCA
    from tuiml.preprocessing.decomposition.sklearn_kernel_pca import SklearnKernelPCA
    from tuiml.preprocessing.decomposition.sklearn_sparse_pca import SklearnSparsePCA
    from tuiml.preprocessing.decomposition.sklearn_mini_batch_sparse_pca import SklearnMiniBatchSparsePCA
    from tuiml.preprocessing.decomposition.sklearn_factor_analysis import SklearnFactorAnalysis
    from tuiml.preprocessing.decomposition.sklearn_fast_ica import SklearnFastICA
    from tuiml.preprocessing.decomposition.sklearn_dictionary_learning import SklearnDictionaryLearning
    from tuiml.preprocessing.decomposition.sklearn_mini_batch_dictionary_learning import SklearnMiniBatchDictionaryLearning
    from tuiml.preprocessing.decomposition.sklearn_nmf import SklearnNMF
    from tuiml.preprocessing.decomposition.sklearn_mini_batch_nmf import SklearnMiniBatchNMF
    from tuiml.preprocessing.decomposition.sklearn_lda import SklearnLatentDirichletAllocation
    from tuiml.preprocessing.decomposition.sklearn_sparse_coder import SklearnSparseCoder
    from tuiml.preprocessing.decomposition.sklearn_truncated_svd import SklearnTruncatedSVD

__all__ = [
]

if SKLEARN_AVAILABLE:
    __all__.extend([
        "SklearnPCA",
        "SklearnIncrementalPCA",
        "SklearnKernelPCA",
        "SklearnSparsePCA",
        "SklearnMiniBatchSparsePCA",
        "SklearnFactorAnalysis",
        "SklearnFastICA",
        "SklearnDictionaryLearning",
        "SklearnMiniBatchDictionaryLearning",
        "SklearnNMF",
        "SklearnMiniBatchNMF",
        "SklearnLatentDirichletAllocation",
        "SklearnSparseCoder",
        "SklearnTruncatedSVD",
    ])

