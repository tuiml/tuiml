#include "knn.h"
#include "../common/parallel.h"

#include <cmath>
#include <algorithm>
#include <vector>
#include <queue>

namespace tuiml {
namespace neighbors {

std::tuple<py::array_t<double>, py::array_t<int>>
brute_knn_query(py::array_t<double> X_train,
                py::array_t<double> X_query, int k) {
    auto train = X_train.unchecked<2>();
    auto query = X_query.unchecked<2>();

    int n_train = static_cast<int>(train.shape(0));
    int n_query = static_cast<int>(query.shape(0));
    int n_feat  = static_cast<int>(train.shape(1));
    k = std::min(k, n_train);

    py::array_t<double> distances({n_query, k});
    py::array_t<int>    indices({n_query, k});
    auto dist_buf = distances.mutable_unchecked<2>();
    auto idx_buf  = indices.mutable_unchecked<2>();

    WK_PARALLEL_FOR
    for (int q = 0; q < n_query; q++) {
        // max-heap: largest distance on top, so we can cheaply evict it
        std::priority_queue<std::pair<double, int>> heap;

        for (int t = 0; t < n_train; t++) {
            double sum = 0.0;
            for (int f = 0; f < n_feat; f++) {
                double diff = query(q, f) - train(t, f);
                sum += diff * diff;
            }
            double dist = std::sqrt(sum);

            if (static_cast<int>(heap.size()) < k) {
                heap.push({dist, t});
            } else if (dist < heap.top().first) {
                heap.pop();
                heap.push({dist, t});
            }
        }

        // extract sorted (smallest distance first)
        int count = static_cast<int>(heap.size());
        for (int i = count - 1; i >= 0; i--) {
            dist_buf(q, i) = heap.top().first;
            idx_buf(q, i)  = heap.top().second;
            heap.pop();
        }
    }

    return std::make_tuple(distances, indices);
}

py::array_t<int>
brute_knn_classify(py::array_t<double> X_train,
                   py::array_t<int> y_train,
                   py::array_t<double> X_query,
                   int k, int n_classes,
                   const std::string& weight_mode) {

    auto [dists, idxs] = brute_knn_query(X_train, X_query, k);
    auto d_buf = dists.unchecked<2>();
    auto i_buf = idxs.unchecked<2>();
    auto y_buf = y_train.unchecked<1>();

    int n_query  = static_cast<int>(d_buf.shape(0));
    int actual_k = static_cast<int>(d_buf.shape(1));

    py::array_t<int> predictions(n_query);
    auto pred_buf = predictions.mutable_unchecked<1>();

    WK_PARALLEL_FOR
    for (int q = 0; q < n_query; q++) {
        std::vector<double> class_weights(n_classes, 0.0);

        for (int j = 0; j < actual_k; j++) {
            int label = y_buf(i_buf(q, j));
            double w = 1.0;

            if (weight_mode == "distance") {
                double d = d_buf(q, j);
                w = (d > 0.0) ? 1.0 / d : 1e10;
            } else if (weight_mode == "similarity") {
                w = 1.0 / (1.0 + d_buf(q, j));
            }

            if (label >= 0 && label < n_classes) {
                class_weights[label] += w;
            }
        }

        int best_class = 0;
        double best_weight = class_weights[0];
        for (int c = 1; c < n_classes; c++) {
            if (class_weights[c] > best_weight) {
                best_weight = class_weights[c];
                best_class  = c;
            }
        }
        pred_buf(q) = best_class;
    }

    return predictions;
}

py::array_t<double>
brute_knn_regress(py::array_t<double> X_train,
                  py::array_t<double> y_train,
                  py::array_t<double> X_query,
                  int k,
                  const std::string& weight_mode) {

    auto [dists, idxs] = brute_knn_query(X_train, X_query, k);
    auto d_buf = dists.unchecked<2>();
    auto i_buf = idxs.unchecked<2>();
    auto y_buf = y_train.unchecked<1>();

    int n_query  = static_cast<int>(d_buf.shape(0));
    int actual_k = static_cast<int>(d_buf.shape(1));

    py::array_t<double> predictions(n_query);
    auto pred_buf = predictions.mutable_unchecked<1>();

    WK_PARALLEL_FOR
    for (int q = 0; q < n_query; q++) {
        double weighted_sum = 0.0;
        double weight_sum   = 0.0;

        for (int j = 0; j < actual_k; j++) {
            double w = 1.0;
            if (weight_mode == "distance") {
                double d = d_buf(q, j);
                w = (d > 0.0) ? 1.0 / d : 1e10;
            } else if (weight_mode == "similarity") {
                w = 1.0 / (1.0 + d_buf(q, j));
            }

            weighted_sum += w * y_buf(i_buf(q, j));
            weight_sum   += w;
        }

        pred_buf(q) = (weight_sum > 0.0) ? weighted_sum / weight_sum : 0.0;
    }

    return predictions;
}

}  // namespace neighbors
}  // namespace tuiml
