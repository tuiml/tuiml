#pragma once

#include <vector>
#include <pybind11/numpy.h>
#include "svm/kernels.h"

namespace py = pybind11;

namespace tuiml {
namespace svm {

// ── SVRModel: trained SVR result ─────────────────────────────────────
struct SVRModel {
    std::vector<int> sv_indices;       // indices of support vectors in training data
    std::vector<double> dual_coef;     // alpha_i - alpha_i* for SVs
    double rho;                        // bias = -b
    int n_iter;
    int n_features;
};

// ── Train / predict functions ────────────────────────────────────────

SVRModel svr_train(py::array_t<double> X, py::array_t<double> y,
                   int kernel_type, double C, double epsilon,
                   double gamma, int degree, double coef0,
                   double tol, int max_iter, size_t cache_mb,
                   bool shrinking);

SVRModel svr_train_precomputed(py::array_t<double> K, py::array_t<double> y,
                               double C, double epsilon,
                               double tol, int max_iter,
                               size_t cache_mb, bool shrinking);

py::array_t<double> svr_predict(const SVRModel& model,
                                py::array_t<double> X_train,
                                py::array_t<double> X_test,
                                int kernel_type, double gamma,
                                int degree, double coef0);

py::array_t<double> svr_predict_precomputed(const SVRModel& model,
                                            py::array_t<double> K_test);

}  // namespace svm
}  // namespace tuiml
