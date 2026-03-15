#include "svm/kernels.h"
#include <stdexcept>
#include <cstring>

namespace tuiml {
namespace svm {

// ══════════════════════════════════════════════════════════════════════
//  KernelCache
// ══════════════════════════════════════════════════════════════════════

KernelCache::KernelCache(size_t max_bytes, int n)
    : max_bytes_(max_bytes), current_bytes_(0), n_(n) {}

const std::vector<double>* KernelCache::get(int i) {
    auto it = map_.find(i);
    if (it == map_.end()) return nullptr;
    // Move to front (most recently used)
    lru_.splice(lru_.begin(), lru_, it->second);
    return &(it->second->second);
}

void KernelCache::put(int i, std::vector<double>&& col) {
    size_t col_bytes = col.size() * sizeof(double);
    if (col_bytes > max_bytes_) return;  // single column exceeds budget

    // Evict until space available
    while (current_bytes_ + col_bytes > max_bytes_ && !lru_.empty()) {
        auto& back = lru_.back();
        current_bytes_ -= back.second.size() * sizeof(double);
        map_.erase(back.first);
        lru_.pop_back();
    }

    lru_.emplace_front(i, std::move(col));
    map_[i] = lru_.begin();
    current_bytes_ += col_bytes;
}

void KernelCache::clear() {
    lru_.clear();
    map_.clear();
    current_bytes_ = 0;
}

// ══════════════════════════════════════════════════════════════════════
//  KernelEvaluator  —  standard kernels
// ══════════════════════════════════════════════════════════════════════

KernelEvaluator::KernelEvaluator(const double* X, int n, int p,
                                 KernelType type, double gamma, int degree,
                                 double coef0, size_t cache_mb)
    : X_(X), n_(n), p_(p), type_(type),
      gamma_(gamma), coef0_(coef0), degree_(degree),
      cache_(cache_mb * 1024ULL * 1024ULL, n)
{
    // Precompute squared norms for RBF
    if (type_ == KernelType::RBF) {
        sq_norms_.resize(n_);
        for (int i = 0; i < n_; ++i) {
            double s = 0;
            const double* xi = X_ + (size_t)i * p_;
            for (int f = 0; f < p_; ++f) s += xi[f] * xi[f];
            sq_norms_[i] = s;
        }
    }

    // Precompute diagonal K(i,i)
    diag_.resize(n_);
    for (int i = 0; i < n_; ++i)
        diag_[i] = eval(i, i);

    tmp_col_.resize(n_);
}

// Precomputed-kernel constructor
KernelEvaluator::KernelEvaluator(const double* K_precomputed, int n,
                                 size_t cache_mb)
    : X_(K_precomputed), n_(n), p_(n), type_(KernelType::PRECOMPUTED),
      gamma_(0), coef0_(0), degree_(0),
      cache_(cache_mb * 1024ULL * 1024ULL, n)
{
    diag_.resize(n_);
    for (int i = 0; i < n_; ++i)
        diag_[i] = X_[(size_t)i * n_ + i];
    tmp_col_.resize(n_);
}

double KernelEvaluator::eval(int i, int j) const {
    const double* xi = X_ + (size_t)i * p_;
    const double* xj = X_ + (size_t)j * p_;

    switch (type_) {
    case KernelType::LINEAR: {
        double d = 0;
        for (int f = 0; f < p_; ++f) d += xi[f] * xj[f];
        return d;
    }
    case KernelType::POLY: {
        double d = 0;
        for (int f = 0; f < p_; ++f) d += xi[f] * xj[f];
        double base = gamma_ * d + coef0_;
        double result = 1.0;
        for (int e = 0; e < degree_; ++e) result *= base;
        return result;
    }
    case KernelType::RBF: {
        double sq_dist = sq_norms_[i] + sq_norms_[j];
        double dot = 0;
        for (int f = 0; f < p_; ++f) dot += xi[f] * xj[f];
        sq_dist -= 2.0 * dot;
        if (sq_dist < 0) sq_dist = 0;
        return std::exp(-gamma_ * sq_dist);
    }
    case KernelType::SIGMOID: {
        double d = 0;
        for (int f = 0; f < p_; ++f) d += xi[f] * xj[f];
        return std::tanh(gamma_ * d + coef0_);
    }
    case KernelType::PRECOMPUTED:
        return X_[(size_t)i * n_ + j];
    }
    return 0;
}

void KernelEvaluator::compute_column(int i, std::vector<double>& col) const {
    col.resize(n_);
    for (int j = 0; j < n_; ++j)
        col[j] = eval(i, j);
}

const std::vector<double>& KernelEvaluator::get_column(int i) {
    const auto* cached = cache_.get(i);
    if (cached) return *cached;

    std::vector<double> col(n_);
    compute_column(i, col);
    cache_.put(i, std::move(col));

    // After put, the pointer is valid from the cache
    const auto* ptr = cache_.get(i);
    if (ptr) return *ptr;

    // Fallback: cache was too small, use tmp buffer
    compute_column(i, tmp_col_);
    return tmp_col_;
}

double KernelEvaluator::eval_external(int i, const double* x) const {
    const double* xi = X_ + (size_t)i * p_;

    switch (type_) {
    case KernelType::LINEAR: {
        double d = 0;
        for (int f = 0; f < p_; ++f) d += xi[f] * x[f];
        return d;
    }
    case KernelType::POLY: {
        double d = 0;
        for (int f = 0; f < p_; ++f) d += xi[f] * x[f];
        double base = gamma_ * d + coef0_;
        double result = 1.0;
        for (int e = 0; e < degree_; ++e) result *= base;
        return result;
    }
    case KernelType::RBF: {
        double sq_dist = 0;
        for (int f = 0; f < p_; ++f) {
            double d = xi[f] - x[f];
            sq_dist += d * d;
        }
        return std::exp(-gamma_ * sq_dist);
    }
    case KernelType::SIGMOID: {
        double d = 0;
        for (int f = 0; f < p_; ++f) d += xi[f] * x[f];
        return std::tanh(gamma_ * d + coef0_);
    }
    case KernelType::PRECOMPUTED:
        // For precomputed, x[i] already holds the kernel value for row i
        return x[i];
    }
    return 0;
}

}  // namespace svm
}  // namespace tuiml
