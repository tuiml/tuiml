#include "svm/svc.h"
#include "svm/smo_solver.h"
#include "common/parallel.h"

#include <algorithm>
#include <set>
#include <map>
#include <cmath>
#include <numeric>

namespace tuiml {
namespace svm {

// ── svc_train ────────────────────────────────────────────────────────

SVCModel svc_train(py::array_t<double> X_arr, py::array_t<int> y_arr,
                   int kernel_type, double C,
                   double gamma, int degree, double coef0,
                   double tol, int max_iter, size_t cache_mb,
                   bool shrinking) {

    auto X = X_arr.unchecked<2>();
    auto y = y_arr.unchecked<1>();
    int n = (int)X.shape(0);
    int p = (int)X.shape(1);

    // Discover classes
    std::set<int> cls_set;
    for (int i = 0; i < n; ++i) cls_set.insert(y(i));
    std::vector<int> classes(cls_set.begin(), cls_set.end());
    int n_classes = (int)classes.size();

    // Build class->index map
    std::map<int, int> cls_idx;
    for (int c = 0; c < n_classes; ++c) cls_idx[classes[c]] = c;

    // Group sample indices by class
    std::vector<std::vector<int>> class_samples(n_classes);
    for (int i = 0; i < n; ++i)
        class_samples[cls_idx[y(i)]].push_back(i);

    SVCModel model;
    model.n_classes = n_classes;
    model.classes = classes;
    model.n_features = p;

    const double* X_ptr = X_arr.data();

    // One-vs-One decomposition
    for (int a = 0; a < n_classes; ++a) {
        for (int b = a + 1; b < n_classes; ++b) {
            // Collect samples for this pair
            const auto& sa = class_samples[a];
            const auto& sb = class_samples[b];
            int n_ab = (int)(sa.size() + sb.size());

            // Build sub-problem data
            std::vector<double> X_sub(n_ab * p);
            std::vector<double> y_sub(n_ab);
            std::vector<int> orig_indices(n_ab);

            int idx = 0;
            for (int i : sa) {
                std::copy(X_ptr + (size_t)i * p, X_ptr + (size_t)i * p + p,
                          X_sub.data() + (size_t)idx * p);
                y_sub[idx] = 1.0;
                orig_indices[idx] = i;
                ++idx;
            }
            for (int i : sb) {
                std::copy(X_ptr + (size_t)i * p, X_ptr + (size_t)i * p + p,
                          X_sub.data() + (size_t)idx * p);
                y_sub[idx] = -1.0;
                orig_indices[idx] = i;
                ++idx;
            }

            KernelEvaluator kern(X_sub.data(), n_ab, p,
                                 static_cast<KernelType>(kernel_type),
                                 gamma, degree, coef0, cache_mb);
            SMOSolver solver(kern, y_sub.data(), n_ab, C, tol, max_iter,
                             shrinking);
            SMOResult result = solver.solve();

            // Build binary model
            SVCModel::BinaryModel bm;
            bm.class_a = a;
            bm.class_b = b;
            bm.rho = result.rho;
            bm.n_iter = result.n_iter;

            for (int sv : result.sv_indices) {
                bm.sv_indices.push_back(orig_indices[sv]);
                bm.dual_coef.push_back(result.alpha[sv] * y_sub[sv]);
            }

            model.models.push_back(std::move(bm));
        }
    }

    // Build flattened attributes
    std::set<int> all_sv_set;
    for (const auto& bm : model.models)
        for (int idx : bm.sv_indices)
            all_sv_set.insert(idx);

    model.all_sv_indices.assign(all_sv_set.begin(), all_sv_set.end());
    int n_sv = (int)model.all_sv_indices.size();

    model.all_sv_data.resize((size_t)n_sv * p);
    for (int s = 0; s < n_sv; ++s) {
        int orig = model.all_sv_indices[s];
        std::copy(X_ptr + (size_t)orig * p, X_ptr + (size_t)orig * p + p,
                  model.all_sv_data.data() + (size_t)s * p);
    }

    // Build index map from original index -> position in all_sv_indices
    std::map<int, int> sv_pos;
    for (int s = 0; s < n_sv; ++s)
        sv_pos[model.all_sv_indices[s]] = s;

    // Dual coefficients: (n_classes-1) x n_sv, following sklearn/libsvm convention
    model.all_dual_coef.assign((size_t)(n_classes - 1) * n_sv, 0.0);

    int model_idx = 0;
    for (int a = 0; a < n_classes; ++a) {
        for (int b = a + 1; b < n_classes; ++b) {
            const auto& bm = model.models[model_idx++];
            for (size_t s = 0; s < bm.sv_indices.size(); ++s) {
                int pos = sv_pos[bm.sv_indices[s]];
                int orig = bm.sv_indices[s];
                int cls = cls_idx[y(orig)];
                // Row index in dual_coef: for SVs of class a in model (a,b), row = b-1
                // for SVs of class b in model (a,b), row = a
                int row;
                if (cls == a) row = b - 1;
                else          row = a;
                // Accumulate (there's only one model per pair, so just set)
                model.all_dual_coef[(size_t)row * n_sv + pos] = bm.dual_coef[s];
            }
        }
    }

    // Intercept (rho, negated to match sklearn convention of -rho = b)
    model.intercept.resize(model.models.size());
    for (size_t m = 0; m < model.models.size(); ++m)
        model.intercept[m] = -model.models[m].rho;

    // n_support per class
    model.n_support.resize(n_classes, 0);
    for (int sv_orig : model.all_sv_indices)
        model.n_support[cls_idx[y(sv_orig)]]++;

    return model;
}

// ── svc_train_precomputed ────────────────────────────────────────────

SVCModel svc_train_precomputed(py::array_t<double> K_arr, py::array_t<int> y_arr,
                               double C, double tol, int max_iter,
                               size_t cache_mb, bool shrinking) {

    auto K = K_arr.unchecked<2>();
    auto y = y_arr.unchecked<1>();
    int n = (int)K.shape(0);

    std::set<int> cls_set;
    for (int i = 0; i < n; ++i) cls_set.insert(y(i));
    std::vector<int> classes(cls_set.begin(), cls_set.end());
    int n_classes = (int)classes.size();

    std::map<int, int> cls_idx;
    for (int c = 0; c < n_classes; ++c) cls_idx[classes[c]] = c;

    std::vector<std::vector<int>> class_samples(n_classes);
    for (int i = 0; i < n; ++i)
        class_samples[cls_idx[y(i)]].push_back(i);

    const double* K_ptr = K_arr.data();

    SVCModel model;
    model.n_classes = n_classes;
    model.classes = classes;
    model.n_features = 0;  // not applicable for precomputed

    for (int a = 0; a < n_classes; ++a) {
        for (int b = a + 1; b < n_classes; ++b) {
            const auto& sa = class_samples[a];
            const auto& sb = class_samples[b];
            int n_ab = (int)(sa.size() + sb.size());

            // Extract sub-kernel-matrix
            std::vector<int> sub_idx;
            sub_idx.reserve(n_ab);
            for (int i : sa) sub_idx.push_back(i);
            for (int i : sb) sub_idx.push_back(i);

            std::vector<double> K_sub(n_ab * n_ab);
            for (int r = 0; r < n_ab; ++r)
                for (int c = 0; c < n_ab; ++c)
                    K_sub[r * n_ab + c] = K_ptr[(size_t)sub_idx[r] * n + sub_idx[c]];

            std::vector<double> y_sub(n_ab);
            for (int i = 0; i < (int)sa.size(); ++i) y_sub[i] = 1.0;
            for (int i = 0; i < (int)sb.size(); ++i) y_sub[sa.size() + i] = -1.0;

            KernelEvaluator kern(K_sub.data(), n_ab, cache_mb);
            SMOSolver solver(kern, y_sub.data(), n_ab, C, tol, max_iter,
                             shrinking);
            SMOResult result = solver.solve();

            SVCModel::BinaryModel bm;
            bm.class_a = a;
            bm.class_b = b;
            bm.rho = result.rho;
            bm.n_iter = result.n_iter;

            for (int sv : result.sv_indices) {
                bm.sv_indices.push_back(sub_idx[sv]);
                bm.dual_coef.push_back(result.alpha[sv] * y_sub[sv]);
            }

            model.models.push_back(std::move(bm));
        }
    }

    // Build flattened attributes
    std::set<int> all_sv_set;
    for (const auto& bm : model.models)
        for (int idx : bm.sv_indices)
            all_sv_set.insert(idx);

    model.all_sv_indices.assign(all_sv_set.begin(), all_sv_set.end());
    int n_sv = (int)model.all_sv_indices.size();

    std::map<int, int> sv_pos;
    for (int s = 0; s < n_sv; ++s)
        sv_pos[model.all_sv_indices[s]] = s;

    model.all_dual_coef.assign((size_t)(n_classes - 1) * n_sv, 0.0);
    int model_idx = 0;
    for (int a = 0; a < n_classes; ++a) {
        for (int b = a + 1; b < n_classes; ++b) {
            const auto& bm = model.models[model_idx++];
            for (size_t s = 0; s < bm.sv_indices.size(); ++s) {
                int pos = sv_pos[bm.sv_indices[s]];
                int orig = bm.sv_indices[s];
                int cls = cls_idx[y(orig)];
                int row = (cls == a) ? b - 1 : a;
                model.all_dual_coef[(size_t)row * n_sv + pos] = bm.dual_coef[s];
            }
        }
    }

    model.intercept.resize(model.models.size());
    for (size_t m = 0; m < model.models.size(); ++m)
        model.intercept[m] = -model.models[m].rho;

    model.n_support.resize(n_classes, 0);
    for (int sv_orig : model.all_sv_indices)
        model.n_support[cls_idx[y(sv_orig)]]++;

    return model;
}

// ── Helper: compute decision values for OvO ──────────────────────────

static void compute_decision_values(
    const SVCModel& model,
    const double* X_train, int n_train, int p,
    const double* X_test, int n_test,
    int kernel_type, double gamma, int degree, double coef0,
    std::vector<double>& dec_values)
{
    int n_models = (int)model.models.size();
    dec_values.resize((size_t)n_test * n_models);

    for (int t = 0; t < n_test; ++t) {
        const double* xt = X_test + (size_t)t * p;

        for (int m = 0; m < n_models; ++m) {
            const auto& bm = model.models[m];
            double decision = 0;

            for (size_t s = 0; s < bm.sv_indices.size(); ++s) {
                int sv_idx = bm.sv_indices[s];
                const double* sv = X_train + (size_t)sv_idx * p;

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

                decision += bm.dual_coef[s] * kval;
            }

            decision -= bm.rho;  // f(x) = sum(alpha_i*y_i*K) - rho
            dec_values[(size_t)t * n_models + m] = decision;
        }
    }
}

static void compute_decision_values_precomputed(
    const SVCModel& model,
    const double* K_test, int n_test, int n_train,
    std::vector<double>& dec_values)
{
    int n_models = (int)model.models.size();
    dec_values.resize((size_t)n_test * n_models);

    for (int t = 0; t < n_test; ++t) {
        const double* kt = K_test + (size_t)t * n_train;

        for (int m = 0; m < n_models; ++m) {
            const auto& bm = model.models[m];
            double decision = 0;

            for (size_t s = 0; s < bm.sv_indices.size(); ++s) {
                int sv_idx = bm.sv_indices[s];
                decision += bm.dual_coef[s] * kt[sv_idx];
            }

            decision -= bm.rho;
            dec_values[(size_t)t * n_models + m] = decision;
        }
    }
}

// ── svc_predict ──────────────────────────────────────────────────────

py::array_t<int> svc_predict(const SVCModel& model,
                             py::array_t<double> X_train_arr,
                             py::array_t<double> X_test_arr,
                             int kernel_type, double gamma,
                             int degree, double coef0) {
    auto X_train = X_train_arr.unchecked<2>();
    auto X_test = X_test_arr.unchecked<2>();
    int n_train = (int)X_train.shape(0);
    int n_test = (int)X_test.shape(0);
    int p = (int)X_train.shape(1);

    std::vector<double> dec_values;
    compute_decision_values(model, X_train_arr.data(), n_train, p,
                            X_test_arr.data(), n_test,
                            kernel_type, gamma, degree, coef0,
                            dec_values);

    int n_classes = model.n_classes;
    int n_models = (int)model.models.size();

    py::array_t<int> result(n_test);
    auto r = result.mutable_unchecked<1>();

    for (int t = 0; t < n_test; ++t) {
        // OvO voting
        std::vector<int> votes(n_classes, 0);
        int m = 0;
        for (int a = 0; a < n_classes; ++a) {
            for (int b = a + 1; b < n_classes; ++b) {
                if (dec_values[(size_t)t * n_models + m] > 0)
                    votes[a]++;
                else
                    votes[b]++;
                ++m;
            }
        }
        int best = (int)(std::max_element(votes.begin(), votes.end()) - votes.begin());
        r(t) = model.classes[best];
    }

    return result;
}

py::array_t<int> svc_predict_precomputed(const SVCModel& model,
                                         py::array_t<double> K_test_arr) {
    auto K_test = K_test_arr.unchecked<2>();
    int n_test = (int)K_test.shape(0);
    int n_train = (int)K_test.shape(1);

    std::vector<double> dec_values;
    compute_decision_values_precomputed(model, K_test_arr.data(), n_test, n_train,
                                       dec_values);

    int n_classes = model.n_classes;
    int n_models = (int)model.models.size();

    py::array_t<int> result(n_test);
    auto r = result.mutable_unchecked<1>();

    for (int t = 0; t < n_test; ++t) {
        std::vector<int> votes(n_classes, 0);
        int m = 0;
        for (int a = 0; a < n_classes; ++a) {
            for (int b = a + 1; b < n_classes; ++b) {
                if (dec_values[(size_t)t * n_models + m] > 0)
                    votes[a]++;
                else
                    votes[b]++;
                ++m;
            }
        }
        int best = (int)(std::max_element(votes.begin(), votes.end()) - votes.begin());
        r(t) = model.classes[best];
    }

    return result;
}

// ── svc_decision_function ────────────────────────────────────────────

py::array_t<double> svc_decision_function(const SVCModel& model,
                                          py::array_t<double> X_train_arr,
                                          py::array_t<double> X_test_arr,
                                          int kernel_type, double gamma,
                                          int degree, double coef0) {
    auto X_train = X_train_arr.unchecked<2>();
    auto X_test = X_test_arr.unchecked<2>();
    int n_train = (int)X_train.shape(0);
    int n_test = (int)X_test.shape(0);
    int p = (int)X_train.shape(1);

    std::vector<double> dec_values;
    compute_decision_values(model, X_train_arr.data(), n_train, p,
                            X_test_arr.data(), n_test,
                            kernel_type, gamma, degree, coef0,
                            dec_values);

    int n_models = (int)model.models.size();

    if (model.n_classes == 2) {
        py::array_t<double> result(n_test);
        auto r = result.mutable_unchecked<1>();
        for (int t = 0; t < n_test; ++t)
            r(t) = dec_values[t];
        return result;
    } else {
        py::array_t<double> result({n_test, n_models});
        auto r = result.mutable_unchecked<2>();
        for (int t = 0; t < n_test; ++t)
            for (int m = 0; m < n_models; ++m)
                r(t, m) = dec_values[(size_t)t * n_models + m];
        return result;
    }
}

py::array_t<double> svc_decision_function_precomputed(
    const SVCModel& model, py::array_t<double> K_test_arr) {
    auto K_test = K_test_arr.unchecked<2>();
    int n_test = (int)K_test.shape(0);
    int n_train = (int)K_test.shape(1);

    std::vector<double> dec_values;
    compute_decision_values_precomputed(model, K_test_arr.data(), n_test, n_train,
                                       dec_values);

    int n_models = (int)model.models.size();

    if (model.n_classes == 2) {
        py::array_t<double> result(n_test);
        auto r = result.mutable_unchecked<1>();
        for (int t = 0; t < n_test; ++t)
            r(t) = dec_values[t];
        return result;
    } else {
        py::array_t<double> result({n_test, n_models});
        auto r = result.mutable_unchecked<2>();
        for (int t = 0; t < n_test; ++t)
            for (int m = 0; m < n_models; ++m)
                r(t, m) = dec_values[(size_t)t * n_models + m];
        return result;
    }
}

}  // namespace svm
}  // namespace tuiml
