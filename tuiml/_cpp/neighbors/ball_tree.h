#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <tuple>
#include <vector>
#include <queue>

namespace py = pybind11;

namespace tuiml {
namespace neighbors {

class CppBallTree {
public:
    explicit CppBallTree(int leaf_size = 30);

    void build(py::array_t<double> X);

    /// Batch query: returns (distances[n_query, k], indices[n_query, k]).
    std::tuple<py::array_t<double>, py::array_t<int>>
    query(py::array_t<double> X_query, int k);

    int num_nodes() const { return static_cast<int>(nodes_.size()); }

private:
    struct Node {
        int left;
        int right;
        int idx_start;
        int idx_count;
        double radius;
        int centroid_offset;   // offset into centroids_ (stride = n_features_)
    };

    int leaf_size_;
    int n_points_;
    int n_features_;
    std::vector<double> data_;            // flat: n_points * n_features
    std::vector<int>    point_indices_;
    std::vector<Node>   nodes_;
    std::vector<double> centroids_;       // flat: n_nodes * n_features

    int build_recursive(int start, int end);

    void query_recursive(int node_idx, const double* q, int k,
                         std::priority_queue<std::pair<double,int>>& heap) const;

    inline double sq_distance(const double* a, int pt_idx) const {
        double sum = 0.0;
        const double* b = data_.data() + pt_idx * n_features_;
        for (int d = 0; d < n_features_; d++) {
            double diff = a[d] - b[d];
            sum += diff * diff;
        }
        return sum;
    }

    inline double sq_distance_centroid(const double* a, int node_idx) const {
        int off = nodes_[node_idx].centroid_offset;
        double sum = 0.0;
        for (int d = 0; d < n_features_; d++) {
            double diff = a[d] - centroids_[off + d];
            sum += diff * diff;
        }
        return sum;
    }
};

}  // namespace neighbors
}  // namespace tuiml
