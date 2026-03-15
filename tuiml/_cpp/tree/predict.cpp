#include "predict.h"
#include "../common/parallel.h"

#include <vector>

namespace tuiml {
namespace tree {

py::array_t<double> predict_batch(
    py::array_t<int> feature,
    py::array_t<double> threshold,
    py::array_t<int> children_left,
    py::array_t<int> children_right,
    py::array_t<double> value,
    py::array_t<double> X
) {
    auto feat_buf = feature.unchecked<1>();
    auto thresh_buf = threshold.unchecked<1>();
    auto left_buf = children_left.unchecked<1>();
    auto right_buf = children_right.unchecked<1>();
    auto val_buf = value.unchecked<2>();
    auto X_buf = X.unchecked<2>();

    int n_samples = static_cast<int>(X_buf.shape(0));
    int value_width = static_cast<int>(val_buf.shape(1));

    // Allocate output: (n_samples, value_width)
    py::array_t<double> result({n_samples, value_width});
    auto res_buf = result.mutable_unchecked<2>();

    WK_PARALLEL_FOR
    for (int i = 0; i < n_samples; i++) {
        // Traverse tree for sample i
        int node_idx = 0;
        while (left_buf(node_idx) != -1) {
            int f = feat_buf(node_idx);
            if (X_buf(i, f) <= thresh_buf(node_idx)) {
                node_idx = left_buf(node_idx);
            } else {
                node_idx = right_buf(node_idx);
            }
        }
        // Copy leaf value
        for (int j = 0; j < value_width; j++) {
            res_buf(i, j) = val_buf(node_idx, j);
        }
    }

    return result;
}

}  // namespace tree
}  // namespace tuiml
