#include "treeshap.h"
#include "../common/parallel.h"

#include <cmath>
#include <vector>

namespace tuiml {
namespace shapley {

namespace {

// One entry of the "unique path" carried down the tree: which feature was
// split on, the fraction of background samples that follow this branch
// (zero_fraction), whether this sample follows it (one_fraction), and the
// Shapley weight accumulated so far.
struct PathElement {
    int feature_index = -1;
    double zero_fraction = 0.0;
    double one_fraction = 0.0;
    double weight = 0.0;
};

// Grow the path by one split, updating every element's Shapley weight.
//
// The weights encode, for each subset size, the proportion of feature subsets
// of that size — which is what turns the exponential sum into a polynomial
// recurrence.
void extend_path(std::vector<PathElement>& path, int depth, double zero_fraction,
                 double one_fraction, int feature_index) {
    path[depth].feature_index = feature_index;
    path[depth].zero_fraction = zero_fraction;
    path[depth].one_fraction = one_fraction;
    path[depth].weight = (depth == 0) ? 1.0 : 0.0;

    for (int i = depth - 1; i >= 0; i--) {
        path[i + 1].weight +=
            one_fraction * path[i].weight * (i + 1) / static_cast<double>(depth + 1);
        path[i].weight =
            zero_fraction * path[i].weight * (depth - i) / static_cast<double>(depth + 1);
    }
}

// Remove a feature from the path, reversing extend_path exactly.
//
// Needed when a feature is split on twice down one path: the second split
// must replace the first rather than double-count it.
void unwind_path(std::vector<PathElement>& path, int depth, int path_index) {
    const double one_fraction = path[path_index].one_fraction;
    const double zero_fraction = path[path_index].zero_fraction;
    double next_one_portion = path[depth].weight;

    for (int i = depth - 1; i >= 0; i--) {
        if (one_fraction != 0.0) {
            const double tmp = path[i].weight;
            path[i].weight = next_one_portion * (depth + 1) /
                             static_cast<double>((i + 1) * one_fraction);
            next_one_portion =
                tmp - path[i].weight * zero_fraction * (depth - i) /
                          static_cast<double>(depth + 1);
        } else if (zero_fraction != 0.0) {
            path[i].weight = path[i].weight * (depth + 1) /
                             static_cast<double>(zero_fraction * (depth - i));
        } else {
            path[i].weight = 0.0;
        }
    }

    for (int i = path_index; i < depth; i++) {
        path[i].feature_index = path[i + 1].feature_index;
        path[i].zero_fraction = path[i + 1].zero_fraction;
        path[i].one_fraction = path[i + 1].one_fraction;
    }
}

// Total weight the path would have if the given element were unwound, without
// actually modifying it.
double unwound_path_sum(const std::vector<PathElement>& path, int depth,
                        int path_index) {
    const double one_fraction = path[path_index].one_fraction;
    const double zero_fraction = path[path_index].zero_fraction;
    double next_one_portion = path[depth].weight;
    double total = 0.0;

    for (int i = depth - 1; i >= 0; i--) {
        if (one_fraction != 0.0) {
            const double tmp = next_one_portion * (depth + 1) /
                               static_cast<double>((i + 1) * one_fraction);
            total += tmp;
            next_one_portion = path[i].weight - tmp * zero_fraction *
                                                    (depth - i) /
                                                    static_cast<double>(depth + 1);
        } else if (zero_fraction != 0.0) {
            total += (path[i].weight / zero_fraction) /
                     ((depth - i) / static_cast<double>(depth + 1));
        }
    }
    return total;
}

struct TreeView {
    const int* feature;
    const double* threshold;
    const int* children_left;
    const int* children_right;
    const double* value;
    const double* node_weight;
    int output_dim;
};

void recurse(const TreeView& tree, const double* x, double* out, int node,
             std::vector<PathElement>& path, int depth, double zero_fraction,
             double one_fraction, int feature_index) {
    extend_path(path, depth, zero_fraction, one_fraction, feature_index);

    if (tree.feature[node] < 0) {
        // Leaf: every feature on the path receives its marginal contribution,
        // weighted by the proportion of subsets in which it matters.
        for (int i = 1; i <= depth; i++) {
            const double weight = unwound_path_sum(path, depth, i);
            const PathElement& element = path[i];
            const double factor =
                weight * (element.one_fraction - element.zero_fraction);
            for (int k = 0; k < tree.output_dim; k++) {
                out[element.feature_index * tree.output_dim + k] +=
                    factor * tree.value[node * tree.output_dim + k];
            }
        }
        return;
    }

    const int split_feature = tree.feature[node];
    const int hot = (x[split_feature] <= tree.threshold[node])
                        ? tree.children_left[node]
                        : tree.children_right[node];
    const int cold = (hot == tree.children_left[node]) ? tree.children_right[node]
                                                       : tree.children_left[node];

    double incoming_zero = 1.0;
    double incoming_one = 1.0;

    // If this feature already appears on the path, remove the earlier entry
    // and carry its fractions forward, so one feature is counted once.
    int path_index = 1;
    for (; path_index <= depth; path_index++) {
        if (path[path_index].feature_index == split_feature) break;
    }
    if (path_index <= depth) {
        incoming_zero = path[path_index].zero_fraction;
        incoming_one = path[path_index].one_fraction;
        unwind_path(path, depth, path_index);
        depth -= 1;
    }

    const double parent_weight = tree.node_weight[node];
    const double hot_weight =
        parent_weight > 0.0 ? tree.node_weight[hot] / parent_weight : 0.0;
    const double cold_weight =
        parent_weight > 0.0 ? tree.node_weight[cold] / parent_weight : 0.0;

    std::vector<PathElement> hot_path(path.begin(), path.end());
    recurse(tree, x, out, hot, hot_path, depth + 1, incoming_zero * hot_weight,
            incoming_one, split_feature);

    std::vector<PathElement> cold_path(path.begin(), path.end());
    recurse(tree, x, out, cold, cold_path, depth + 1, incoming_zero * cold_weight,
            0.0, split_feature);
}

}  // namespace

py::array_t<double> tree_shap(py::array_t<int> feature,
                              py::array_t<double> threshold,
                              py::array_t<int> children_left,
                              py::array_t<int> children_right,
                              py::array_t<double> value,
                              py::array_t<double> node_weight,
                              py::array_t<double> X) {
    auto x_buf = X.unchecked<2>();
    auto value_buf = value.unchecked<2>();

    const int n_samples = static_cast<int>(x_buf.shape(0));
    const int n_features = static_cast<int>(x_buf.shape(1));
    const int output_dim = static_cast<int>(value_buf.shape(1));
    const int n_nodes = static_cast<int>(value_buf.shape(0));

    TreeView tree{
        feature.unchecked<1>().data(0),
        threshold.unchecked<1>().data(0),
        children_left.unchecked<1>().data(0),
        children_right.unchecked<1>().data(0),
        value_buf.data(0, 0),
        node_weight.unchecked<1>().data(0),
        output_dim,
    };

    py::array_t<double> result({n_samples, n_features, output_dim});
    auto out = result.mutable_unchecked<3>();
    for (int i = 0; i < n_samples; i++)
        for (int j = 0; j < n_features; j++)
            for (int k = 0; k < output_dim; k++) out(i, j, k) = 0.0;

    // Depth bound: a path can hold at most one entry per node.
    int max_depth = 0;
    {
        std::vector<int> depth_of(n_nodes, 0);
        for (int node = 0; node < n_nodes; node++) {
            const int left = tree.children_left[node];
            const int right = tree.children_right[node];
            if (left >= 0) depth_of[left] = depth_of[node] + 1;
            if (right >= 0) depth_of[right] = depth_of[node] + 1;
            if (depth_of[node] > max_depth) max_depth = depth_of[node];
        }
    }
    const int path_capacity = max_depth + 2;

    WK_PARALLEL_FOR
    for (int i = 0; i < n_samples; i++) {
        std::vector<double> row(static_cast<size_t>(n_features) * output_dim, 0.0);
        std::vector<PathElement> path(path_capacity);
        std::vector<double> sample(n_features);
        for (int j = 0; j < n_features; j++) sample[j] = x_buf(i, j);

        recurse(tree, sample.data(), row.data(), 0, path, 0, 1.0, 1.0, -1);

        for (int j = 0; j < n_features; j++)
            for (int k = 0; k < output_dim; k++)
                out(i, j, k) = row[static_cast<size_t>(j) * output_dim + k];
    }

    return result;
}

}  // namespace shapley
}  // namespace tuiml
