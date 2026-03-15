#include "builder.h"
#include "criteria.h"

#include <algorithm>
#include <vector>
#include <numeric>
#include <cmath>
#include <random>
#include <limits>
#include <stack>

namespace tuiml {
namespace tree {

namespace {

// Internal: find best split for classifier using contiguous local arrays.
// Extracts the subset of data into contiguous arrays and uses the same
// proven logic as the standalone splitter in splitters.cpp.
std::tuple<int, double, double> find_best_split_classifier(
    const py::detail::unchecked_reference<double, 2>& X_buf,
    const py::detail::unchecked_reference<int, 1>& y_buf,
    const std::vector<int>& samples,
    const std::string& criterion,
    int n_classes,
    int min_samples_leaf,
    std::mt19937& rng,
    int max_features,
    int n_features
) {
    int n = static_cast<int>(samples.size());
    if (n < 2 * min_samples_leaf) {
        return {-1, 0.0, std::numeric_limits<double>::lowest()};
    }

    // Extract contiguous sub-arrays for this node's samples.
    // This matches the standalone splitter which receives contiguous
    // numpy arrays (X[mask], y[mask]).
    std::vector<double> X_local(static_cast<size_t>(n) * n_features);
    std::vector<int> y_local(n);
    for (int i = 0; i < n; i++) {
        int orig = samples[i];
        y_local[i] = y_buf(orig);
        for (int f = 0; f < n_features; f++) {
            X_local[static_cast<size_t>(i) * n_features + f] = X_buf(orig, f);
        }
    }

    // Total class counts
    std::vector<int> total_counts(n_classes, 0);
    for (int i = 0; i < n; i++) {
        total_counts[y_local[i]]++;
    }
    double parent_imp = compute_impurity(total_counts, n, criterion);

    // Feature permutation
    std::vector<int> feature_order(n_features);
    std::iota(feature_order.begin(), feature_order.end(), 0);
    std::shuffle(feature_order.begin(), feature_order.end(), rng);
    if (max_features > 0 && max_features < n_features) {
        feature_order.resize(max_features);
    }

    double best_gain = std::numeric_limits<double>::lowest();
    int best_feature = -1;
    double best_threshold = 0.0;

    std::vector<int> indices(n);
    std::vector<int> left_counts(n_classes);

    for (int feat_idx : feature_order) {
        // Fresh 0..n-1 indices sorted by feature value (same as standalone)
        std::iota(indices.begin(), indices.end(), 0);
        std::sort(indices.begin(), indices.end(),
            [&](int a, int b) {
                return X_local[static_cast<size_t>(a) * n_features + feat_idx]
                     < X_local[static_cast<size_t>(b) * n_features + feat_idx];
            });

        std::fill(left_counts.begin(), left_counts.end(), 0);

        for (int i = 0; i < n - 1; i++) {
            left_counts[y_local[indices[i]]]++;
            int n_left = i + 1;
            int n_right = n - n_left;

            if (n_left < min_samples_leaf || n_right < min_samples_leaf) continue;

            double val_i = X_local[static_cast<size_t>(indices[i]) * n_features + feat_idx];
            double val_next = X_local[static_cast<size_t>(indices[i + 1]) * n_features + feat_idx];
            if (val_i == val_next) continue;

            double imp_left = compute_impurity(left_counts, n_left, criterion);
            double imp_right = compute_impurity_complement(
                total_counts, left_counts, n_right, criterion);

            double gain = parent_imp
                - (static_cast<double>(n_left) / n) * imp_left
                - (static_cast<double>(n_right) / n) * imp_right;

            if (criterion == "gain_ratio") {
                double p_left = static_cast<double>(n_left) / n;
                double p_right = static_cast<double>(n_right) / n;
                double split_info = -(p_left * std::log2(p_left)
                                    + p_right * std::log2(p_right));
                gain = (split_info > 0.0) ? gain / split_info
                                          : std::numeric_limits<double>::lowest();
            }

            if (gain > best_gain) {
                best_gain = gain;
                best_feature = feat_idx;
                best_threshold = (val_i + val_next) / 2.0;
            }
        }
    }

    return {best_feature, best_threshold, best_gain};
}

// Internal: find best split for regressor using contiguous local arrays.
std::tuple<int, double, double> find_best_split_regressor(
    const py::detail::unchecked_reference<double, 2>& X_buf,
    const py::detail::unchecked_reference<double, 1>& y_buf,
    const std::vector<int>& samples,
    const std::string& criterion,
    int min_samples_leaf,
    std::mt19937& rng,
    int max_features,
    int n_features
) {
    int n = static_cast<int>(samples.size());
    if (n < 2 * min_samples_leaf) {
        return {-1, 0.0, std::numeric_limits<double>::lowest()};
    }

    // Extract contiguous sub-arrays
    std::vector<double> X_local(static_cast<size_t>(n) * n_features);
    std::vector<double> y_local(n);
    for (int i = 0; i < n; i++) {
        int orig = samples[i];
        y_local[i] = y_buf(orig);
        for (int f = 0; f < n_features; f++) {
            X_local[static_cast<size_t>(i) * n_features + f] = X_buf(orig, f);
        }
    }

    double total_sum = 0.0, total_sq_sum = 0.0;
    for (int i = 0; i < n; i++) {
        total_sum += y_local[i];
        total_sq_sum += y_local[i] * y_local[i];
    }
    double parent_mean = total_sum / n;
    double parent_imp = total_sq_sum / n - parent_mean * parent_mean;

    std::vector<int> feature_order(n_features);
    std::iota(feature_order.begin(), feature_order.end(), 0);
    std::shuffle(feature_order.begin(), feature_order.end(), rng);
    if (max_features > 0 && max_features < n_features) {
        feature_order.resize(max_features);
    }

    double best_gain = std::numeric_limits<double>::lowest();
    int best_feature = -1;
    double best_threshold = 0.0;

    std::vector<int> indices(n);

    for (int feat_idx : feature_order) {
        std::iota(indices.begin(), indices.end(), 0);
        std::sort(indices.begin(), indices.end(),
            [&](int a, int b) {
                return X_local[static_cast<size_t>(a) * n_features + feat_idx]
                     < X_local[static_cast<size_t>(b) * n_features + feat_idx];
            });

        double left_sum = 0.0, left_sq_sum = 0.0;

        for (int i = 0; i < n - 1; i++) {
            double val = y_local[indices[i]];
            left_sum += val;
            left_sq_sum += val * val;

            int n_left = i + 1;
            int n_right = n - n_left;

            if (n_left < min_samples_leaf || n_right < min_samples_leaf) continue;

            double fval_i = X_local[static_cast<size_t>(indices[i]) * n_features + feat_idx];
            double fval_next = X_local[static_cast<size_t>(indices[i + 1]) * n_features + feat_idx];
            if (fval_i == fval_next) continue;

            double right_sum = total_sum - left_sum;
            double right_sq_sum = total_sq_sum - left_sq_sum;
            double left_mean = left_sum / n_left;
            double right_mean = right_sum / n_right;

            double gain;
            if (criterion == "friedman_mse") {
                gain = (static_cast<double>(n_left) * n_right
                       / (static_cast<double>(n) * n))
                       * (left_mean - right_mean) * (left_mean - right_mean);
            } else {
                double left_mse = left_sq_sum / n_left - left_mean * left_mean;
                double right_mse = right_sq_sum / n_right - right_mean * right_mean;
                gain = parent_imp
                    - (static_cast<double>(n_left) / n) * left_mse
                    - (static_cast<double>(n_right) / n) * right_mse;
            }

            if (gain > best_gain) {
                best_gain = gain;
                best_feature = feat_idx;
                best_threshold = (fval_i + fval_next) / 2.0;
            }
        }
    }

    return {best_feature, best_threshold, best_gain};
}

}  // anonymous namespace


std::tuple<
    py::array_t<int>, py::array_t<double>,
    py::array_t<int>, py::array_t<int>,
    py::array_t<double>, int
> build_classifier_tree(
    py::array_t<double> X,
    py::array_t<int> y,
    const std::string& criterion,
    int n_classes,
    int max_depth,
    int min_samples_split,
    int min_samples_leaf,
    double min_impurity_decrease,
    int random_seed,
    int max_features
) {
    auto X_buf = X.unchecked<2>();
    auto y_buf = y.unchecked<1>();
    int n_samples = static_cast<int>(X_buf.shape(0));
    int n_features = static_cast<int>(X_buf.shape(1));

    std::mt19937 rng(random_seed);

    // Dynamic arrays for tree nodes
    std::vector<int> features;
    std::vector<double> thresholds;
    std::vector<int> left_children;
    std::vector<int> right_children;
    std::vector<std::vector<double>> values;

    auto add_node = [&](int feat, double thresh, const std::vector<double>& val) -> int {
        int idx = static_cast<int>(features.size());
        features.push_back(feat);
        thresholds.push_back(thresh);
        left_children.push_back(-1);
        right_children.push_back(-1);
        values.push_back(val);
        return idx;
    };

    // Initial samples
    std::vector<int> all_samples(n_samples);
    std::iota(all_samples.begin(), all_samples.end(), 0);

    auto compute_distribution = [&](const std::vector<int>& samples) {
        std::vector<double> dist(n_classes, 0.0);
        for (int idx : samples) {
            dist[y_buf(idx)] += 1.0;
        }
        double n = static_cast<double>(samples.size());
        for (double& d : dist) d /= n;
        return dist;
    };

    auto compute_impurity_for = [&](const std::vector<int>& samples) {
        std::vector<int> counts(n_classes, 0);
        for (int idx : samples) counts[y_buf(idx)]++;
        return compute_impurity(counts, static_cast<int>(samples.size()), criterion);
    };

    struct StackEntry {
        std::vector<int> samples;
        int depth;
        int node_idx;
    };

    // Add root node
    int root_idx = add_node(-1, 0.0, compute_distribution(all_samples));
    std::stack<StackEntry> stack;
    stack.push({all_samples, 0, root_idx});

    while (!stack.empty()) {
        auto entry = std::move(stack.top());
        stack.pop();

        int n = static_cast<int>(entry.samples.size());
        double impurity = compute_impurity_for(entry.samples);

        // Check leaf conditions
        bool is_leaf = false;
        if (max_depth >= 0 && entry.depth >= max_depth) is_leaf = true;
        if (n < min_samples_split) is_leaf = true;
        if (impurity == 0.0) is_leaf = true;

        if (is_leaf) {
            features[entry.node_idx] = -1;
            continue;
        }

        // Find best split
        auto [best_feat, best_thresh, best_gain] = find_best_split_classifier(
            X_buf, y_buf, entry.samples, criterion, n_classes,
            min_samples_leaf, rng, max_features, n_features);

        if (best_feat == -1 || best_gain < min_impurity_decrease) {
            features[entry.node_idx] = -1;
            continue;
        }

        // Split samples
        std::vector<int> left_samples, right_samples;
        left_samples.reserve(n);
        right_samples.reserve(n);
        for (int idx : entry.samples) {
            if (X_buf(idx, best_feat) <= best_thresh) {
                left_samples.push_back(idx);
            } else {
                right_samples.push_back(idx);
            }
        }

        // Update current node
        features[entry.node_idx] = best_feat;
        thresholds[entry.node_idx] = best_thresh;

        // Create child nodes
        int left_idx = add_node(-1, 0.0, compute_distribution(left_samples));
        int right_idx = add_node(-1, 0.0, compute_distribution(right_samples));
        left_children[entry.node_idx] = left_idx;
        right_children[entry.node_idx] = right_idx;

        // Push children for processing
        stack.push({std::move(right_samples), entry.depth + 1, right_idx});
        stack.push({std::move(left_samples), entry.depth + 1, left_idx});
    }

    // Convert to numpy arrays
    int n_nodes = static_cast<int>(features.size());

    py::array_t<int> feat_arr(n_nodes);
    py::array_t<double> thresh_arr(n_nodes);
    py::array_t<int> left_arr(n_nodes);
    py::array_t<int> right_arr(n_nodes);
    py::array_t<double> val_arr({n_nodes, n_classes});

    auto feat_out = feat_arr.mutable_unchecked<1>();
    auto thresh_out = thresh_arr.mutable_unchecked<1>();
    auto left_out = left_arr.mutable_unchecked<1>();
    auto right_out = right_arr.mutable_unchecked<1>();
    auto val_out = val_arr.mutable_unchecked<2>();

    for (int i = 0; i < n_nodes; i++) {
        feat_out(i) = features[i];
        thresh_out(i) = thresholds[i];
        left_out(i) = left_children[i];
        right_out(i) = right_children[i];
        for (int j = 0; j < n_classes; j++) {
            val_out(i, j) = values[i][j];
        }
    }

    return {feat_arr, thresh_arr, left_arr, right_arr, val_arr, n_nodes};
}

std::tuple<
    py::array_t<int>, py::array_t<double>,
    py::array_t<int>, py::array_t<int>,
    py::array_t<double>, int
> build_regressor_tree(
    py::array_t<double> X,
    py::array_t<double> y,
    const std::string& criterion,
    int max_depth,
    int min_samples_split,
    int min_samples_leaf,
    double min_impurity_decrease,
    int random_seed,
    int max_features
) {
    auto X_buf = X.unchecked<2>();
    auto y_buf = y.unchecked<1>();
    int n_samples = static_cast<int>(X_buf.shape(0));
    int n_features = static_cast<int>(X_buf.shape(1));

    std::mt19937 rng(random_seed);

    std::vector<int> feat_vec;
    std::vector<double> thresh_vec;
    std::vector<int> left_vec;
    std::vector<int> right_vec;
    std::vector<double> val_vec;  // single value per node

    auto add_node = [&](int feat, double thresh, double val) -> int {
        int idx = static_cast<int>(feat_vec.size());
        feat_vec.push_back(feat);
        thresh_vec.push_back(thresh);
        left_vec.push_back(-1);
        right_vec.push_back(-1);
        val_vec.push_back(val);
        return idx;
    };

    auto compute_mean = [&](const std::vector<int>& samples) {
        double s = 0.0;
        for (int idx : samples) s += y_buf(idx);
        return s / samples.size();
    };

    auto compute_mse = [&](const std::vector<int>& samples) {
        double s = 0.0, sq = 0.0;
        int n = static_cast<int>(samples.size());
        for (int idx : samples) {
            s += y_buf(idx);
            sq += y_buf(idx) * y_buf(idx);
        }
        double m = s / n;
        return sq / n - m * m;
    };

    std::vector<int> all_samples(n_samples);
    std::iota(all_samples.begin(), all_samples.end(), 0);

    int root_idx = add_node(-1, 0.0, compute_mean(all_samples));

    struct StackEntry {
        std::vector<int> samples;
        int depth;
        int node_idx;
    };

    std::stack<StackEntry> stack;
    stack.push({all_samples, 0, root_idx});

    while (!stack.empty()) {
        auto entry = std::move(stack.top());
        stack.pop();

        int n = static_cast<int>(entry.samples.size());
        double impurity = compute_mse(entry.samples);

        bool is_leaf = false;
        if (max_depth >= 0 && entry.depth >= max_depth) is_leaf = true;
        if (n < min_samples_split) is_leaf = true;
        if (impurity == 0.0) is_leaf = true;

        if (is_leaf) {
            feat_vec[entry.node_idx] = -1;
            continue;
        }

        auto [best_feat, best_thresh, best_gain] = find_best_split_regressor(
            X_buf, y_buf, entry.samples, criterion,
            min_samples_leaf, rng, max_features, n_features);

        if (best_feat == -1 || best_gain < min_impurity_decrease) {
            feat_vec[entry.node_idx] = -1;
            continue;
        }

        std::vector<int> left_samples, right_samples;
        left_samples.reserve(n);
        right_samples.reserve(n);
        for (int idx : entry.samples) {
            if (X_buf(idx, best_feat) <= best_thresh) {
                left_samples.push_back(idx);
            } else {
                right_samples.push_back(idx);
            }
        }

        feat_vec[entry.node_idx] = best_feat;
        thresh_vec[entry.node_idx] = best_thresh;

        int left_idx = add_node(-1, 0.0, compute_mean(left_samples));
        int right_idx = add_node(-1, 0.0, compute_mean(right_samples));
        left_vec[entry.node_idx] = left_idx;
        right_vec[entry.node_idx] = right_idx;

        stack.push({std::move(right_samples), entry.depth + 1, right_idx});
        stack.push({std::move(left_samples), entry.depth + 1, left_idx});
    }

    int n_nodes = static_cast<int>(feat_vec.size());

    py::array_t<int> feat_arr(n_nodes);
    py::array_t<double> thresh_arr(n_nodes);
    py::array_t<int> left_arr(n_nodes);
    py::array_t<int> right_arr(n_nodes);
    py::array_t<double> val_arr({n_nodes, 1});

    auto feat_out = feat_arr.mutable_unchecked<1>();
    auto thresh_out = thresh_arr.mutable_unchecked<1>();
    auto left_out = left_arr.mutable_unchecked<1>();
    auto right_out = right_arr.mutable_unchecked<1>();
    auto val_out = val_arr.mutable_unchecked<2>();

    for (int i = 0; i < n_nodes; i++) {
        feat_out(i) = feat_vec[i];
        thresh_out(i) = thresh_vec[i];
        left_out(i) = left_vec[i];
        right_out(i) = right_vec[i];
        val_out(i, 0) = val_vec[i];
    }

    return {feat_arr, thresh_arr, left_arr, right_arr, val_arr, n_nodes};
}

}  // namespace tree
}  // namespace tuiml
