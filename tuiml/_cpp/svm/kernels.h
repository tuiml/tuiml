#pragma once

#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <cstddef>
#include <list>
#include <unordered_map>
#include <pybind11/numpy.h>

namespace py = pybind11;

namespace tuiml {
namespace svm {

// ── Kernel types ──────────────────────────────────────────────────────
enum class KernelType { LINEAR = 0, POLY = 1, RBF = 2, SIGMOID = 3, PRECOMPUTED = 4 };

// ── LRU kernel column cache ──────────────────────────────────────────
class KernelCache {
public:
    explicit KernelCache(size_t max_bytes, int n);

    // Return cached column for index i, or nullptr if not cached.
    const std::vector<double>* get(int i);

    // Insert column for index i (takes ownership via move).
    void put(int i, std::vector<double>&& col);

    void clear();

private:
    size_t max_bytes_;
    size_t current_bytes_;
    int n_;

    // LRU list: front = most recently used
    std::list<std::pair<int, std::vector<double>>> lru_;
    std::unordered_map<int, decltype(lru_)::iterator> map_;
};

// ── Kernel evaluator ─────────────────────────────────────────────────
// Holds training data + kernel params. Computes K(i,j) on demand, with
// a configurable LRU column cache.
class KernelEvaluator {
public:
    KernelEvaluator(const double* X, int n, int p,
                    KernelType type, double gamma, int degree,
                    double coef0, size_t cache_mb);

    // Precomputed-kernel constructor: X is the n x n kernel matrix.
    KernelEvaluator(const double* K_precomputed, int n, size_t cache_mb);

    // Single element
    double eval(int i, int j) const;

    // Fill a full column Q_i[0..n-1] = y[i]*y[j]*K(i,j)  (without y — that is
    // the caller's job). Returns a pointer that is valid at least until the
    // next call.
    const std::vector<double>& get_column(int i);

    // Diagonal K(i,i) — precomputed at construction.
    double diag(int i) const { return diag_[i]; }

    int n() const { return n_; }
    int p() const { return p_; }

    // Evaluate kernel between training row i and an external point x.
    double eval_external(int i, const double* x) const;

private:
    void compute_column(int i, std::vector<double>& col) const;

    const double* X_;         // n x p row-major
    int n_, p_;
    KernelType type_;
    double gamma_, coef0_;
    int degree_;
    std::vector<double> diag_;    // K(i,i)
    std::vector<double> sq_norms_; // ||x_i||^2 for RBF
    KernelCache cache_;

    // Temp buffer for get_column when cache is disabled / missed
    mutable std::vector<double> tmp_col_;
};

}  // namespace svm
}  // namespace tuiml
