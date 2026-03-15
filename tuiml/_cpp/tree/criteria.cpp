#include "criteria.h"
#include <cmath>
#include <numeric>

namespace tuiml {
namespace tree {

double gini_impurity(const std::vector<int>& counts, int n_samples) {
    if (n_samples == 0) return 0.0;
    double sum_sq = 0.0;
    double n = static_cast<double>(n_samples);
    for (int c : counts) {
        double p = static_cast<double>(c) / n;
        sum_sq += p * p;
    }
    return 1.0 - sum_sq;
}

double entropy_impurity(const std::vector<int>& counts, int n_samples) {
    if (n_samples == 0) return 0.0;
    double ent = 0.0;
    double n = static_cast<double>(n_samples);
    for (int c : counts) {
        if (c > 0) {
            double p = static_cast<double>(c) / n;
            ent -= p * std::log2(p);
        }
    }
    return ent;
}

double compute_impurity(const std::vector<int>& counts, int n_samples,
                        const std::string& criterion) {
    if (criterion == "gini") {
        return gini_impurity(counts, n_samples);
    } else {
        // entropy, log_loss, gain_ratio all use entropy for node impurity
        return entropy_impurity(counts, n_samples);
    }
}

double compute_impurity_complement(const std::vector<int>& total_counts,
                                   const std::vector<int>& left_counts,
                                   int n_right,
                                   const std::string& criterion) {
    // right_counts = total_counts - left_counts
    std::vector<int> right_counts(total_counts.size());
    for (size_t i = 0; i < total_counts.size(); i++) {
        right_counts[i] = total_counts[i] - left_counts[i];
    }
    return compute_impurity(right_counts, n_right, criterion);
}

double mse_impurity(const double* y, int n_samples) {
    if (n_samples == 0) return 0.0;
    double sum = 0.0;
    double sq_sum = 0.0;
    for (int i = 0; i < n_samples; i++) {
        sum += y[i];
        sq_sum += y[i] * y[i];
    }
    double mean = sum / n_samples;
    return sq_sum / n_samples - mean * mean;
}

double mse_from_stats(double sum, double sq_sum, int n) {
    if (n == 0) return 0.0;
    double mean = sum / n;
    return sq_sum / n - mean * mean;
}

}  // namespace tree
}  // namespace tuiml
