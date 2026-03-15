#include "kd_tree.h"
#include "../common/parallel.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <limits>
#include <vector>

namespace tuiml {
namespace neighbors {

CppKDTree::CppKDTree(int leaf_size)
    : leaf_size_(leaf_size), n_points_(0), n_features_(0) {}

void CppKDTree::build(py::array_t<double> X) {
    auto buf = X.unchecked<2>();
    n_points_  = static_cast<int>(buf.shape(0));
    n_features_ = static_cast<int>(buf.shape(1));

    // Copy data to internal flat storage
    data_.resize(n_points_ * n_features_);
    for (int i = 0; i < n_points_; i++)
        for (int d = 0; d < n_features_; d++)
            data_[i * n_features_ + d] = buf(i, d);

    // Initialise identity permutation
    point_indices_.resize(n_points_);
    std::iota(point_indices_.begin(), point_indices_.end(), 0);

    nodes_.clear();
    nodes_.reserve(2 * n_points_ / std::max(leaf_size_, 1) + 1);

    build_recursive(0, n_points_, 0);
}

int CppKDTree::build_recursive(int start, int end, int depth) {
    int count    = end - start;
    int node_idx = static_cast<int>(nodes_.size());
    nodes_.push_back(Node{});
    nodes_[node_idx].idx_start = start;
    nodes_[node_idx].idx_count = count;

    if (count <= leaf_size_) {
        nodes_[node_idx].split_dim = -1;
        nodes_[node_idx].split_val = 0.0;
        nodes_[node_idx].left  = -1;
        nodes_[node_idx].right = -1;
        return node_idx;
    }

    // Dimension with largest spread
    int    best_dim    = 0;
    double best_spread = -1.0;
    for (int d = 0; d < n_features_; d++) {
        double lo = std::numeric_limits<double>::max();
        double hi = std::numeric_limits<double>::lowest();
        for (int i = start; i < end; i++) {
            double v = point_value(point_indices_[i], d);
            lo = std::min(lo, v);
            hi = std::max(hi, v);
        }
        double spread = hi - lo;
        if (spread > best_spread) {
            best_spread = spread;
            best_dim    = d;
        }
    }

    nodes_[node_idx].split_dim = best_dim;

    // Partition around median via nth_element
    int mid = start + count / 2;
    int dim = best_dim;
    int nf  = n_features_;
    const auto& data_ref = data_;
    std::nth_element(
        point_indices_.begin() + start,
        point_indices_.begin() + mid,
        point_indices_.begin() + end,
        [dim, nf, &data_ref](int a, int b) {
            return data_ref[a * nf + dim] < data_ref[b * nf + dim];
        });

    nodes_[node_idx].split_val = point_value(point_indices_[mid], best_dim);

    // Build children (nodes_ may re-allocate, so save/restore by index)
    int left_idx  = build_recursive(start, mid, depth + 1);
    int right_idx = build_recursive(mid, end, depth + 1);

    nodes_[node_idx].left  = left_idx;
    nodes_[node_idx].right = right_idx;
    return node_idx;
}

void CppKDTree::query_recursive(
        int node_idx, const double* q, int k,
        std::priority_queue<std::pair<double,int>>& heap) const {

    const Node& node = nodes_[node_idx];

    if (node.split_dim == -1) {
        // Leaf – check all points
        for (int i = node.idx_start; i < node.idx_start + node.idx_count; i++) {
            int pt = point_indices_[i];
            double sq = sq_distance(q, pt);
            if (static_cast<int>(heap.size()) < k) {
                heap.push({sq, pt});
            } else if (sq < heap.top().first) {
                heap.pop();
                heap.push({sq, pt});
            }
        }
        return;
    }

    // Decide which child is closer
    double diff  = q[node.split_dim] - node.split_val;
    int first  = (diff <= 0.0) ? node.left  : node.right;
    int second = (diff <= 0.0) ? node.right : node.left;

    if (first >= 0)
        query_recursive(first, q, k, heap);

    // Prune the far child
    double sq_diff = diff * diff;
    if (second >= 0 &&
        (static_cast<int>(heap.size()) < k || sq_diff < heap.top().first)) {
        query_recursive(second, q, k, heap);
    }
}

std::tuple<py::array_t<double>, py::array_t<int>>
CppKDTree::query(py::array_t<double> X_query, int k) {
    auto buf = X_query.unchecked<2>();
    int n_query = static_cast<int>(buf.shape(0));
    k = std::min(k, n_points_);

    py::array_t<double> distances({n_query, k});
    py::array_t<int>    indices({n_query, k});
    auto dist_out = distances.mutable_unchecked<2>();
    auto idx_out  = indices.mutable_unchecked<2>();

    WK_PARALLEL_FOR
    for (int q = 0; q < n_query; q++) {
        std::vector<double> qpt(n_features_);
        for (int d = 0; d < n_features_; d++)
            qpt[d] = buf(q, d);

        std::priority_queue<std::pair<double,int>> heap;
        query_recursive(0, qpt.data(), k, heap);

        int count = std::min(k, static_cast<int>(heap.size()));
        for (int i = count - 1; i >= 0; i--) {
            dist_out(q, i) = std::sqrt(heap.top().first);
            idx_out(q, i)  = heap.top().second;
            heap.pop();
        }
    }

    return std::make_tuple(distances, indices);
}

}  // namespace neighbors
}  // namespace tuiml
