#include "metrics.h"
#include "../common/parallel.h"

#include <cmath>
#include <algorithm>

namespace tuiml {
namespace distance {

py::array_t<double> euclidean_distance(py::array_t<double> X, py::array_t<double> Y) {
    auto X_buf = X.unchecked<2>();
    auto Y_buf = Y.unchecked<2>();

    int n = static_cast<int>(X_buf.shape(0));
    int m = static_cast<int>(Y_buf.shape(0));
    int d = static_cast<int>(X_buf.shape(1));

    py::array_t<double> result({n, m});
    auto res = result.mutable_unchecked<2>();

    WK_PARALLEL_FOR
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            double sum = 0.0;
            for (int k = 0; k < d; k++) {
                double diff = X_buf(i, k) - Y_buf(j, k);
                sum += diff * diff;
            }
            res(i, j) = std::sqrt(sum);
        }
    }

    return result;
}

py::array_t<double> manhattan_distance(py::array_t<double> X, py::array_t<double> Y) {
    auto X_buf = X.unchecked<2>();
    auto Y_buf = Y.unchecked<2>();

    int n = static_cast<int>(X_buf.shape(0));
    int m = static_cast<int>(Y_buf.shape(0));
    int d = static_cast<int>(X_buf.shape(1));

    py::array_t<double> result({n, m});
    auto res = result.mutable_unchecked<2>();

    WK_PARALLEL_FOR
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            double sum = 0.0;
            for (int k = 0; k < d; k++) {
                sum += std::abs(X_buf(i, k) - Y_buf(j, k));
            }
            res(i, j) = sum;
        }
    }

    return result;
}

py::array_t<double> cosine_distance(py::array_t<double> X, py::array_t<double> Y) {
    auto X_buf = X.unchecked<2>();
    auto Y_buf = Y.unchecked<2>();

    int n = static_cast<int>(X_buf.shape(0));
    int m = static_cast<int>(Y_buf.shape(0));
    int d = static_cast<int>(X_buf.shape(1));

    py::array_t<double> result({n, m});
    auto res = result.mutable_unchecked<2>();

    // Pre-compute norms for Y
    std::vector<double> Y_norms(m);
    for (int j = 0; j < m; j++) {
        double norm = 0.0;
        for (int k = 0; k < d; k++) {
            norm += Y_buf(j, k) * Y_buf(j, k);
        }
        Y_norms[j] = std::sqrt(norm);
    }

    WK_PARALLEL_FOR
    for (int i = 0; i < n; i++) {
        double x_norm = 0.0;
        for (int k = 0; k < d; k++) {
            x_norm += X_buf(i, k) * X_buf(i, k);
        }
        x_norm = std::sqrt(x_norm);

        for (int j = 0; j < m; j++) {
            double dot = 0.0;
            for (int k = 0; k < d; k++) {
                dot += X_buf(i, k) * Y_buf(j, k);
            }
            double denom = x_norm * Y_norms[j];
            if (denom > 0.0) {
                res(i, j) = 1.0 - dot / denom;
            } else {
                res(i, j) = 0.0;
            }
        }
    }

    return result;
}

}  // namespace distance
}  // namespace tuiml
