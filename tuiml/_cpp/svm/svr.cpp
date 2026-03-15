#include "svm/svr.h"
#include "svm/smo_solver.h"
#include "common/parallel.h"

#include <cmath>
#include <algorithm>
#include <cstdio>

namespace tuiml {
namespace svm {

// ── svr_train ────────────────────────────────────────────────────────

SVRModel svr_train(py::array_t<double> X_arr, py::array_t<double> y_arr,
                   int kernel_type, double C, double epsilon,
                   double gamma, int degree, double coef0,
                   double tol, int max_iter, size_t cache_mb,
                   bool shrinking) {

    auto X = X_arr.unchecked<2>();
    auto y = y_arr.unchecked<1>();
    int n = (int)X.shape(0);
    int p = (int)X.shape(1);

    KernelEvaluator kern(X_arr.data(), n, p,
                         static_cast<KernelType>(kernel_type),
                         gamma, degree, coef0, cache_mb);

    SMOSolverSVR solver(kern, y_arr.data(), n, C, epsilon, tol, max_iter,
                        shrinking);
    SMOResult result = solver.solve();

    SVRModel model;
    model.rho = result.rho;
    model.n_iter = result.n_iter;
    model.n_features = p;

    for (int idx : result.sv_indices) {
        model.sv_indices.push_back(idx);
        model.dual_coef.push_back(result.alpha[idx]);
    }

    return model;
}

SVRModel svr_train_precomputed(py::array_t<double> K_arr, py::array_t<double> y_arr,
                               double C, double epsilon,
                               double tol, int max_iter,
                               size_t cache_mb, bool shrinking) {

    auto K = K_arr.unchecked<2>();
    auto y = y_arr.unchecked<1>();
    int n = (int)K.shape(0);

    KernelEvaluator kern(K_arr.data(), n, cache_mb);

    SMOSolverSVR solver(kern, y_arr.data(), n, C, epsilon, tol, max_iter,
                        shrinking);
    SMOResult result = solver.solve();

    SVRModel model;
    model.rho = result.rho;
    model.n_iter = result.n_iter;
    model.n_features = 0;

    for (int idx : result.sv_indices) {
        model.sv_indices.push_back(idx);
        model.dual_coef.push_back(result.alpha[idx]);
    }

    return model;
}

// ── svr_predict ──────────────────────────────────────────────────────

py::array_t<double> svr_predict(const SVRModel& model,
                                py::array_t<double> X_train_arr,
                                py::array_t<double> X_test_arr,
                                int kernel_type, double gamma,
                                int degree, double coef0) {
    auto X_train = X_train_arr.unchecked<2>();
    auto X_test = X_test_arr.unchecked<2>();
    int n_test = (int)X_test.shape(0);
    int p = (int)X_train.shape(1);

    const double* X_train_ptr = X_train_arr.data();
    const double* X_test_ptr = X_test_arr.data();

    py::array_t<double> result(n_test);
    auto r = result.mutable_unchecked<1>();

    for (int t = 0; t < n_test; ++t) {
        const double* xt = X_test_ptr + (size_t)t * p;
        double pred = -model.rho;  // f(x) = sum(coef * K) - rho

        for (size_t s = 0; s < model.sv_indices.size(); ++s) {
            int sv_idx = model.sv_indices[s];
            const double* sv = X_train_ptr + (size_t)sv_idx * p;

            double kval = 0;
            switch (static_cast<KernelType>(kernel_type)) {
            case KernelType::LINEAR: {
                for (int f = 0; f < p; ++f) kval += sv[f] * xt[f];
                break;
            }
            case KernelType::POLY: {
                double d = 0;
                for (int f = 0; f < p; ++f) d += sv[f] * xt[f];
                double base = gamma * d + coef0;
                kval = 1.0;
                for (int e = 0; e < degree; ++e) kval *= base;
                break;
            }
            case KernelType::RBF: {
                double sq = 0;
                for (int f = 0; f < p; ++f) {
                    double diff = sv[f] - xt[f];
                    sq += diff * diff;
                }
                kval = std::exp(-gamma * sq);
                break;
            }
            case KernelType::SIGMOID: {
                double d = 0;
                for (int f = 0; f < p; ++f) d += sv[f] * xt[f];
                kval = std::tanh(gamma * d + coef0);
                break;
            }
            default:
                kval = 0;
            }

            pred += model.dual_coef[s] * kval;
        }
        r(t) = pred;
    }

    return result;
}

py::array_t<double> svr_predict_precomputed(const SVRModel& model,
                                            py::array_t<double> K_test_arr) {
    auto K_test = K_test_arr.unchecked<2>();
    int n_test = (int)K_test.shape(0);
    int n_train = (int)K_test.shape(1);

    const double* K_test_ptr = K_test_arr.data();

    py::array_t<double> result(n_test);
    auto r = result.mutable_unchecked<1>();

    for (int t = 0; t < n_test; ++t) {
        const double* kt = K_test_ptr + (size_t)t * n_train;
        double pred = -model.rho;

        for (size_t s = 0; s < model.sv_indices.size(); ++s) {
            int sv_idx = model.sv_indices[s];
            pred += model.dual_coef[s] * kt[sv_idx];
        }
        r(t) = pred;
    }

    return result;
}

}  // namespace svm
}  // namespace tuiml
