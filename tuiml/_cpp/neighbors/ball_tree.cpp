#include "ball_tree.h"
#include "../common/parallel.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <limits>
#include <vector>

namespace tuiml {
namespace neighbors {

CppBallTree::CppBallTree(int leaf_size)
    : leaf_size_(leaf_size), n_points_(0), n_features_(0) {}

void CppBallTree::build(py::array_t<double> X) {
    auto buf = X.unchecked<2>();
    n_points_   = static_cast<int>(buf.shape(0));
    n_features_ = static_cast<int>(buf.shape(1));

    data_.resize(n_points_ * n_features_);
    for (int i = 0; i < n_points_; i++)
        for (int d = 0; d < n_features_; d++)
            data_[i * n_features_ + d] = buf(i, d);

    point_indices_.resize(n_points_);
    std::iota(point_indices_.begin(), point_indices_.end(), 0);

    nodes_.clear();
    centroids_.clear();

    build_recursive(0, n_points_);
}

int CppBallTree::build_recursive(int start, int end) {
    int count    = end - start;
    int node_idx = static_cast<int>(nodes_.size());
    nodes_.push_back(Node{});

    // ---- centroid ----
    int c_off = static_cast<int>(centroids_.size());
    centroids_.resize(centroids_.size() + n_features_, 0.0);
    for (int i = start; i < end; i++) {
        int pt = point_indices_[i];
        for (int d = 0; d < n_features_; d++)
            centroids_[c_off + d] += data_[pt * n_features_ + d];
    }
    for (int d = 0; d < n_features_; d++)
        centroids_[c_off + d] /= count;

    // ---- radius ----
    double max_sq = 0.0;
    for (int i = start; i < end; i++) {
        int pt = point_indices_[i];
        double sq = 0.0;
        for (int d = 0; d < n_features_; d++) {
            double diff = data_[pt * n_features_ + d] - centroids_[c_off + d];
            sq += diff * diff;
        }
        max_sq = std::max(max_sq, sq);
    }

    nodes_[node_idx].centroid_offset = c_off;
    nodes_[node_idx].radius    = std::sqrt(max_sq);
    nodes_[node_idx].idx_start = start;
    nodes_[node_idx].idx_count = count;

    if (count <= leaf_size_) {
        nodes_[node_idx].left  = -1;
        nodes_[node_idx].right = -1;
        return node_idx;
    }

    // ---- split along dimension with largest spread ----
    int    best_dim    = 0;
    double best_spread = -1.0;
    for (int d = 0; d < n_features_; d++) {
        double lo = std::numeric_limits<double>::max();
        double hi = std::numeric_limits<double>::lowest();
        for (int i = start; i < end; i++) {
            double v = data_[point_indices_[i] * n_features_ + d];
            lo = std::min(lo, v);
            hi = std::max(hi, v);
        }
        double spread = hi - lo;
        if (spread > best_spread) {
            best_spread = spread;
            best_dim    = d;
        }
    }

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

    int left_idx  = build_recursive(start, mid);
    int right_idx = build_recursive(mid, end);

    nodes_[node_idx].left  = left_idx;
    nodes_[node_idx].right = right_idx;
    return node_idx;
}

void CppBallTree::query_recursive(
        int node_idx, const double* q, int k,
        std::priority_queue<std::pair<double,int>>& heap) const {

    const Node& node = nodes_[node_idx];

    // Prune via bounding-ball
    double sq_dc  = sq_distance_centroid(q, node_idx);
    double dc     = std::sqrt(sq_dc);
    double min_d  = dc - node.radius;
    if (min_d < 0.0) min_d = 0.0;

    if (static_cast<int>(heap.size()) >= k &&
        min_d * min_d >= heap.top().first) {
        return;
    }

    if (node.left == -1) {
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

    // Visit the closer child first
    double dl = sq_distance_centroid(q, node.left);
    double dr = sq_distance_centroid(q, node.right);
    int first  = (dl <= dr) ? node.left  : node.right;
    int second = (dl <= dr) ? node.right : node.left;

    query_recursive(first,  q, k, heap);
    query_recursive(second, q, k, heap);
}

std::tuple<py::array_t<double>, py::array_t<int>>
CppBallTree::query(py::array_t<double> X_query, int k) {
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
