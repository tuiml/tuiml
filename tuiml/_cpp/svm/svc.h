#pragma once

#include <vector>
#include <pybind11/numpy.h>
#include "svm/kernels.h"

namespace py = pybind11;

namespace tuiml {
namespace svm {

// ── SVCModel: trained SVC result ─────────────────────────────────────
struct SVCModel {
    // Per binary sub-model (OvO)
    struct BinaryModel {
        int class_a, class_b;                  // original class labels (indices)
        std::vector<int> sv_indices;           // indices into training data
        std::vector<double> dual_coef;         // alpha_i * y_i for support vectors
        double rho;                            // bias = -b
        int n_iter;
    };

    int n_classes;
    std::vector<int> classes;                  // unique class labels (sorted)
    std::vector<BinaryModel> models;           // n_classes*(n_classes-1)/2
    int n_features;

    // Flattened support vector data for Python exposure
    std::vector<int> all_sv_indices;           // unique sorted SV indices
    std::vector<double> all_sv_data;           // SV feature data (n_sv x n_features)
    std::vector<double> all_dual_coef;         // (n_classes-1) x n_sv dual coefficients
    std::vector<double> intercept;             // rho per binary model
    std::vector<int> n_support;                // SVs per class
};

// ── Train / predict functions ────────────────────────────────────────

SVCModel svc_train(py::array_t<double> X, py::array_t<int> y,
                   int kernel_type, double C,
                   double gamma, int degree, double coef0,
                   double tol, int max_iter, size_t cache_mb,
                   bool shrinking);

SVCModel svc_train_precomputed(py::array_t<double> K, py::array_t<int> y,
                               double C, double tol, int max_iter,
                               size_t cache_mb, bool shrinking);

py::array_t<int> svc_predict(const SVCModel& model,
                             py::array_t<double> X_train,
                             py::array_t<double> X_test,
                             int kernel_type, double gamma,
                             int degree, double coef0);

py::array_t<int> svc_predict_precomputed(const SVCModel& model,
                                         py::array_t<double> K_test);

py::array_t<double> svc_decision_function(const SVCModel& model,
                                          py::array_t<double> X_train,
                                          py::array_t<double> X_test,
                                          int kernel_type, double gamma,
                                          int degree, double coef0);

py::array_t<double> svc_decision_function_precomputed(
    const SVCModel& model, py::array_t<double> K_test);

}  // namespace svm
}  // namespace tuiml
