#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <tuple>
#include <vector>
#include <queue>

namespace py = pybind11;

namespace tuiml {
namespace neighbors {

class CppKDTree {
public:
    explicit CppKDTree(int leaf_size = 30);

    void build(py::array_t<double> X);

    /// Batch query: returns (distances[n_query, k], indices[n_query, k]).
    std::tuple<py::array_t<double>, py::array_t<int>>
    query(py::array_t<double> X_query, int k);

    int num_nodes() const { return static_cast<int>(nodes_.size()); }

private:
    struct Node {
        int split_dim;      // -1 for leaf
        double split_val;
        int left;           // child index, -1 if absent
        int right;
        int idx_start;      // start in point_indices_
        int idx_count;      // number of points (>0 for leaf)
    };

    int leaf_size_;
    int n_points_;
    int n_features_;
    std::vector<double> data_;          // flat: n_points * n_features
    std::vector<int>    point_indices_; // permuted original indices
    std::vector<Node>   nodes_;

    int build_recursive(int start, int end, int depth);

    void query_recursive(int node_idx, const double* q, int k,
                         std::priority_queue<std::pair<double,int>>& heap) const;

    inline double point_value(int pt_idx, int dim) const {
        return data_[pt_idx * n_features_ + dim];
    }

    inline double sq_distance(const double* a, int pt_idx) const {
        double sum = 0.0;
        const double* b = data_.data() + pt_idx * n_features_;
        for (int d = 0; d < n_features_; d++) {
            double diff = a[d] - b[d];
            sum += diff * diff;
        }
        return sum;
    }
};

}  // namespace neighbors
}  // namespace tuiml
