#include "splitters.h"
#include "criteria.h"
#include "../common/parallel.h"

#include <algorithm>
#include <vector>
#include <numeric>
#include <cmath>
#include <random>
#include <limits>

namespace tuiml {
namespace tree {

std::tuple<int, double, double> best_split_classifier(
    py::array_t<double> X,
    py::array_t<int> y,
    const std::string& criterion,
    int n_classes,
    int min_samples_leaf,
    int random_seed,
    int max_features
) {
    auto X_buf = X.unchecked<2>();
    auto y_buf = y.unchecked<1>();
    int n_samples = static_cast<int>(X_buf.shape(0));
    int n_features = static_cast<int>(X_buf.shape(1));

    // Edge case: not enough samples to split
    if (n_samples < 2 * min_samples_leaf) {
        return {-1, 0.0, std::numeric_limits<double>::lowest()};
    }

    // Permute features and select max_features
    std::mt19937 rng(random_seed);
    std::vector<int> feature_order(n_features);
    std::iota(feature_order.begin(), feature_order.end(), 0);
    std::shuffle(feature_order.begin(), feature_order.end(), rng);
    if (max_features > 0 && max_features < n_features) {
        feature_order.resize(max_features);
    }

    // Pre-compute total class counts for parent impurity
    std::vector<int> total_counts(n_classes, 0);
    for (int i = 0; i < n_samples; i++) {
        total_counts[y_buf(i)]++;
    }
    double parent_imp = compute_impurity(total_counts, n_samples, criterion);

    double best_gain = std::numeric_limits<double>::lowest();
    int best_feature = -1;
    double best_threshold = 0.0;

    // Working buffers (per feature)
    std::vector<int> indices(n_samples);
    std::vector<int> left_counts(n_classes);

    for (int feat_idx : feature_order) {
        // Sort sample indices by feature value
        std::iota(indices.begin(), indices.end(), 0);
        std::sort(indices.begin(), indices.end(),
            [&](int a, int b) {
                return X_buf(a, feat_idx) < X_buf(b, feat_idx);
            });

        // Single-pass cumulative scan
        std::fill(left_counts.begin(), left_counts.end(), 0);

        for (int i = 0; i < n_samples - 1; i++) {
            left_counts[y_buf(indices[i])]++;
            int n_left = i + 1;
            int n_right = n_samples - n_left;

            // Check min_samples_leaf
            if (n_left < min_samples_leaf || n_right < min_samples_leaf) {
                continue;
            }

            // Skip duplicate feature values (no valid split point)
            if (X_buf(indices[i], feat_idx) == X_buf(indices[i + 1], feat_idx)) {
                continue;
            }

            // Compute impurity gain
            double imp_left = compute_impurity(left_counts, n_left, criterion);
            double imp_right = compute_impurity_complement(
                total_counts, left_counts, n_right, criterion);

            double gain = parent_imp
                - (static_cast<double>(n_left) / n_samples) * imp_left
                - (static_cast<double>(n_right) / n_samples) * imp_right;

            // Apply gain_ratio normalization
            if (criterion == "gain_ratio") {
                double p_left = static_cast<double>(n_left) / n_samples;
                double p_right = static_cast<double>(n_right) / n_samples;
                double split_info = -(p_left * std::log2(p_left)
                                    + p_right * std::log2(p_right));
                if (split_info > 0.0) {
                    gain = gain / split_info;
                } else {
                    gain = std::numeric_limits<double>::lowest();
                }
            }

            if (gain > best_gain) {
                best_gain = gain;
                best_feature = feat_idx;
                best_threshold = (X_buf(indices[i], feat_idx)
                                + X_buf(indices[i + 1], feat_idx)) / 2.0;
            }
        }
    }

    return {best_feature, best_threshold, best_gain};
}

std::tuple<int, double, double> best_split_regressor(
    py::array_t<double> X,
    py::array_t<double> y,
    const std::string& criterion,
    int min_samples_leaf,
    int random_seed,
    int max_features
) {
    auto X_buf = X.unchecked<2>();
    auto y_buf = y.unchecked<1>();
    int n_samples = static_cast<int>(X_buf.shape(0));
    int n_features = static_cast<int>(X_buf.shape(1));

    if (n_samples < 2 * min_samples_leaf) {
        return {-1, 0.0, std::numeric_limits<double>::lowest()};
    }

    // Permute features
    std::mt19937 rng(random_seed);
    std::vector<int> feature_order(n_features);
    std::iota(feature_order.begin(), feature_order.end(), 0);
    std::shuffle(feature_order.begin(), feature_order.end(), rng);
    if (max_features > 0 && max_features < n_features) {
        feature_order.resize(max_features);
    }

    // Pre-compute parent impurity (MSE = variance)
    double total_sum = 0.0;
    double total_sq_sum = 0.0;
    for (int i = 0; i < n_samples; i++) {
        total_sum += y_buf(i);
        total_sq_sum += y_buf(i) * y_buf(i);
    }
    double parent_mean = total_sum / n_samples;
    double parent_imp = total_sq_sum / n_samples - parent_mean * parent_mean;

    double best_gain = std::numeric_limits<double>::lowest();
    int best_feature = -1;
    double best_threshold = 0.0;

    std::vector<int> indices(n_samples);

    for (int feat_idx : feature_order) {
        // Sort by feature value
        std::iota(indices.begin(), indices.end(), 0);
        std::sort(indices.begin(), indices.end(),
            [&](int a, int b) {
                return X_buf(a, feat_idx) < X_buf(b, feat_idx);
            });

        // Cumulative scan for sum and sq_sum
        double left_sum = 0.0;
        double left_sq_sum = 0.0;

        for (int i = 0; i < n_samples - 1; i++) {
            double val = y_buf(indices[i]);
            left_sum += val;
            left_sq_sum += val * val;

            int n_left = i + 1;
            int n_right = n_samples - n_left;

            if (n_left < min_samples_leaf || n_right < min_samples_leaf) {
                continue;
            }

            if (X_buf(indices[i], feat_idx) == X_buf(indices[i + 1], feat_idx)) {
                continue;
            }

            double right_sum = total_sum - left_sum;
            double right_sq_sum = total_sq_sum - left_sq_sum;

            double left_mean = left_sum / n_left;
            double right_mean = right_sum / n_right;

            double gain;
            if (criterion == "friedman_mse") {
                gain = (static_cast<double>(n_left) * n_right
                       / (static_cast<double>(n_samples) * n_samples))
                       * (left_mean - right_mean) * (left_mean - right_mean);
            } else {
                // squared_error: parent_mse - weighted child mse
                double left_mse = left_sq_sum / n_left - left_mean * left_mean;
                double right_mse = right_sq_sum / n_right - right_mean * right_mean;
                gain = parent_imp
                    - (static_cast<double>(n_left) / n_samples) * left_mse
                    - (static_cast<double>(n_right) / n_samples) * right_mse;
            }

            if (gain > best_gain) {
                best_gain = gain;
                best_feature = feat_idx;
                best_threshold = (X_buf(indices[i], feat_idx)
                                + X_buf(indices[i + 1], feat_idx)) / 2.0;
            }
        }
    }

    return {best_feature, best_threshold, best_gain};
}

}  // namespace tree
}  // namespace tuiml
