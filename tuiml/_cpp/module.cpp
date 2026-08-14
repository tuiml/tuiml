#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "tree/splitters.h"
#include "tree/criteria.h"
#include "tree/predict.h"
#include "tree/builder.h"
#include "distance/metrics.h"
#include "neighbors/knn.h"
#include "neighbors/kd_tree.h"
#include "neighbors/ball_tree.h"
#include "common/parallel.h"
#include "svm/svc.h"
#include "svm/svr.h"
#include "linear/sgd.h"
#include "clustering/hierarchical.h"
#include "clustering/em.h"
#include "stats/pava.h"
#include "stats/ecdf.h"
#include "stats/histogram.h"
#include "timeseries/dtw.h"
#include "timeseries/convolution.h"
#include "timeseries/shapelet.h"
#include "timeseries/sfa.h"

namespace py = pybind11;

PYBIND11_MODULE(_cpp_ext, m) {
    m.doc() = "TuiML C++ acceleration backend";

    // ── tree submodule ─────────────────────────────────────────────────
    auto tree = m.def_submodule("tree", "Tree algorithm kernels");

    tree.def("best_split_classifier",
             &tuiml::tree::best_split_classifier,
             py::arg("X"), py::arg("y"),
             py::arg("criterion"), py::arg("n_classes"),
             py::arg("min_samples_leaf"), py::arg("random_seed"),
             py::arg("max_features"),
             "Find the best binary split for classification.");

    tree.def("best_split_regressor",
             &tuiml::tree::best_split_regressor,
             py::arg("X"), py::arg("y"),
             py::arg("criterion"), py::arg("min_samples_leaf"),
             py::arg("random_seed"), py::arg("max_features"),
             "Find the best binary split for regression.");

    tree.def("predict_batch",
             &tuiml::tree::predict_batch,
             py::arg("feature"), py::arg("threshold"),
             py::arg("children_left"), py::arg("children_right"),
             py::arg("value"), py::arg("X"),
             "Batch prediction over a flattened tree.");

    tree.def("build_classifier_tree",
             &tuiml::tree::build_classifier_tree,
             py::arg("X"), py::arg("y"),
             py::arg("criterion"), py::arg("n_classes"),
             py::arg("max_depth"), py::arg("min_samples_split"),
             py::arg("min_samples_leaf"), py::arg("min_impurity_decrease"),
             py::arg("random_seed"), py::arg("max_features"),
             "Build a complete classification tree, returning flat arrays.");

    tree.def("build_regressor_tree",
             &tuiml::tree::build_regressor_tree,
             py::arg("X"), py::arg("y"),
             py::arg("criterion"), py::arg("max_depth"),
             py::arg("min_samples_split"), py::arg("min_samples_leaf"),
             py::arg("min_impurity_decrease"), py::arg("random_seed"),
             py::arg("max_features"),
             "Build a complete regression tree, returning flat arrays.");

    // ── distance submodule ─────────────────────────────────────────────
    auto dist = m.def_submodule("distance", "Distance computation kernels");

    dist.def("euclidean", &tuiml::distance::euclidean_distance,
             py::arg("X"), py::arg("Y"),
             "Pairwise Euclidean distance matrix.");

    dist.def("manhattan", &tuiml::distance::manhattan_distance,
             py::arg("X"), py::arg("Y"),
             "Pairwise Manhattan distance matrix.");

    dist.def("cosine", &tuiml::distance::cosine_distance,
             py::arg("X"), py::arg("Y"),
             "Pairwise cosine distance matrix.");

    // ── neighbors submodule ────────────────────────────────────────────
    auto nn = m.def_submodule("neighbors", "Neighbor search kernels");

    nn.def("brute_knn_query",
           &tuiml::neighbors::brute_knn_query,
           py::arg("X_train"), py::arg("X_query"), py::arg("k"),
           "Batch brute-force k-nearest-neighbor query.");

    nn.def("brute_knn_classify",
           &tuiml::neighbors::brute_knn_classify,
           py::arg("X_train"), py::arg("y_train"), py::arg("X_query"),
           py::arg("k"), py::arg("n_classes"), py::arg("weight_mode"),
           "Brute-force KNN classification (returns int labels).");

    nn.def("brute_knn_regress",
           &tuiml::neighbors::brute_knn_regress,
           py::arg("X_train"), py::arg("y_train"), py::arg("X_query"),
           py::arg("k"), py::arg("weight_mode"),
           "Brute-force KNN regression (returns predicted values).");

    py::class_<tuiml::neighbors::CppKDTree>(nn, "KDTree")
        .def(py::init<int>(), py::arg("leaf_size") = 30)
        .def("build", &tuiml::neighbors::CppKDTree::build,
             py::arg("X"), "Build the KD-tree from training data.")
        .def("query", &tuiml::neighbors::CppKDTree::query,
             py::arg("X_query"), py::arg("k"),
             "Batch k-nearest-neighbor query.")
        .def("num_nodes", &tuiml::neighbors::CppKDTree::num_nodes);

    py::class_<tuiml::neighbors::CppBallTree>(nn, "BallTree")
        .def(py::init<int>(), py::arg("leaf_size") = 30)
        .def("build", &tuiml::neighbors::CppBallTree::build,
             py::arg("X"), "Build the Ball-tree from training data.")
        .def("query", &tuiml::neighbors::CppBallTree::query,
             py::arg("X_query"), py::arg("k"),
             "Batch k-nearest-neighbor query.")
        .def("num_nodes", &tuiml::neighbors::CppBallTree::num_nodes);

    // ── svm submodule ─────────────────────────────────────────────────
    auto svm = m.def_submodule("svm", "SVM solver kernels");

    py::class_<tuiml::svm::SVCModel>(svm, "SVCModel")
        .def_readonly("n_classes", &tuiml::svm::SVCModel::n_classes)
        .def_readonly("classes", &tuiml::svm::SVCModel::classes)
        .def_readonly("n_features", &tuiml::svm::SVCModel::n_features)
        .def_readonly("all_sv_indices", &tuiml::svm::SVCModel::all_sv_indices)
        .def_readonly("all_sv_data", &tuiml::svm::SVCModel::all_sv_data)
        .def_readonly("all_dual_coef", &tuiml::svm::SVCModel::all_dual_coef)
        .def_readonly("intercept", &tuiml::svm::SVCModel::intercept)
        .def_readonly("n_support", &tuiml::svm::SVCModel::n_support);

    py::class_<tuiml::svm::SVRModel>(svm, "SVRModel")
        .def_readonly("sv_indices", &tuiml::svm::SVRModel::sv_indices)
        .def_readonly("dual_coef", &tuiml::svm::SVRModel::dual_coef)
        .def_readonly("rho", &tuiml::svm::SVRModel::rho)
        .def_readonly("n_iter", &tuiml::svm::SVRModel::n_iter)
        .def_readonly("n_features", &tuiml::svm::SVRModel::n_features);

    svm.def("svc_train", &tuiml::svm::svc_train,
            py::arg("X"), py::arg("y"),
            py::arg("kernel_type"), py::arg("C"),
            py::arg("gamma"), py::arg("degree"), py::arg("coef0"),
            py::arg("tol"), py::arg("max_iter"),
            py::arg("cache_mb") = 200,
            py::arg("shrinking") = true,
            "Train a C-SVC using native SMO.");

    svm.def("svc_train_precomputed", &tuiml::svm::svc_train_precomputed,
            py::arg("K"), py::arg("y"),
            py::arg("C"), py::arg("tol"), py::arg("max_iter"),
            py::arg("cache_mb") = 200,
            py::arg("shrinking") = true,
            "Train a C-SVC using a precomputed kernel matrix.");

    svm.def("svc_predict", &tuiml::svm::svc_predict,
            py::arg("model"), py::arg("X_train"), py::arg("X_test"),
            py::arg("kernel_type"), py::arg("gamma"),
            py::arg("degree"), py::arg("coef0"),
            "Predict labels using a trained SVC model.");

    svm.def("svc_predict_precomputed", &tuiml::svm::svc_predict_precomputed,
            py::arg("model"), py::arg("K_test"),
            "Predict labels using a precomputed kernel test matrix.");

    svm.def("svc_decision_function", &tuiml::svm::svc_decision_function,
            py::arg("model"), py::arg("X_train"), py::arg("X_test"),
            py::arg("kernel_type"), py::arg("gamma"),
            py::arg("degree"), py::arg("coef0"),
            "Compute OvO decision function values.");

    svm.def("svc_decision_function_precomputed",
            &tuiml::svm::svc_decision_function_precomputed,
            py::arg("model"), py::arg("K_test"),
            "Decision function for precomputed kernel.");

    svm.def("svr_train", &tuiml::svm::svr_train,
            py::arg("X"), py::arg("y"),
            py::arg("kernel_type"), py::arg("C"), py::arg("epsilon"),
            py::arg("gamma"), py::arg("degree"), py::arg("coef0"),
            py::arg("tol"), py::arg("max_iter"),
            py::arg("cache_mb") = 200,
            py::arg("shrinking") = true,
            "Train an epsilon-SVR using native SMO.");

    svm.def("svr_train_precomputed", &tuiml::svm::svr_train_precomputed,
            py::arg("K"), py::arg("y"),
            py::arg("C"), py::arg("epsilon"),
            py::arg("tol"), py::arg("max_iter"),
            py::arg("cache_mb") = 200,
            py::arg("shrinking") = true,
            "Train an epsilon-SVR using a precomputed kernel matrix.");

    svm.def("svr_predict", &tuiml::svm::svr_predict,
            py::arg("model"), py::arg("X_train"), py::arg("X_test"),
            py::arg("kernel_type"), py::arg("gamma"),
            py::arg("degree"), py::arg("coef0"),
            "Predict values using a trained SVR model.");

    svm.def("svr_predict_precomputed", &tuiml::svm::svr_predict_precomputed,
            py::arg("model"), py::arg("K_test"),
            "Predict values using a precomputed kernel test matrix.");

    // ── linear submodule ──────────────────────────────────────────────
    auto lin = m.def_submodule("linear", "Linear model kernels (SGD)");

    py::class_<tuiml::linear::SGDResult>(lin, "SGDResult")
        .def_readonly("weights", &tuiml::linear::SGDResult::weights)
        .def_readonly("bias",    &tuiml::linear::SGDResult::bias)
        .def_readonly("n_iter",  &tuiml::linear::SGDResult::n_iter);

    lin.def("sgd_fit_classifier", &tuiml::linear::sgd_fit_classifier,
            py::arg("X"), py::arg("y"),
            py::arg("n_classes"),
            py::arg("loss_type"),
            py::arg("penalty_type"),
            py::arg("alpha"),
            py::arg("l1_ratio"),
            py::arg("eta0"),
            py::arg("lr_schedule"),
            py::arg("power_t"),
            py::arg("n_epochs"),
            py::arg("batch_size"),
            py::arg("tol"),
            py::arg("patience"),
            py::arg("shuffle"),
            py::arg("random_seed"),
            py::arg("weights_init"),
            py::arg("bias_init"),
            "Train an SGD classifier (OvR for multiclass).");

    lin.def("sgd_fit_regressor", &tuiml::linear::sgd_fit_regressor,
            py::arg("X"), py::arg("y"),
            py::arg("loss_type"),
            py::arg("penalty_type"),
            py::arg("alpha"),
            py::arg("l1_ratio"),
            py::arg("eta0"),
            py::arg("lr_schedule"),
            py::arg("power_t"),
            py::arg("epsilon"),
            py::arg("n_epochs"),
            py::arg("batch_size"),
            py::arg("tol"),
            py::arg("patience"),
            py::arg("shuffle"),
            py::arg("random_seed"),
            py::arg("weights_init"),
            py::arg("bias_init"),
            "Train an SGD regressor.");

    lin.def("sgd_decision_function", &tuiml::linear::sgd_decision_function,
            py::arg("X"), py::arg("weights"), py::arg("bias"),
            "Compute linear decision function scores.");

    // ── clustering submodule ──────────────────────────────────────────
    auto clu = m.def_submodule("clustering", "Clustering algorithm kernels");

    clu.def("hierarchical_fit", &tuiml::clustering::hierarchical_fit,
            py::arg("X"), py::arg("n_clusters"), py::arg("linkage"),
            "Agglomerative hierarchical clustering (Ward/complete/average/single).");

    py::class_<tuiml::clustering::EMResult>(clu, "EMResult")
        .def_readonly("weights",       &tuiml::clustering::EMResult::weights)
        .def_readonly("means",         &tuiml::clustering::EMResult::means)
        .def_readonly("covariances",   &tuiml::clustering::EMResult::covariances)
        .def_readonly("n_iter",        &tuiml::clustering::EMResult::n_iter)
        .def_readonly("log_likelihood",&tuiml::clustering::EMResult::log_likelihood)
        .def_readonly("cov_type",      &tuiml::clustering::EMResult::cov_type);

    clu.def("em_fit", &tuiml::clustering::em_fit,
            py::arg("X"), py::arg("n_components"),
            py::arg("covariance_type"), py::arg("max_iter"),
            py::arg("tol"), py::arg("reg_covar"),
            py::arg("n_init"), py::arg("random_seed"),
            "Fit a Gaussian Mixture Model via EM.");

    clu.def("em_predict", &tuiml::clustering::em_predict,
            py::arg("X"), py::arg("weights"), py::arg("means"),
            py::arg("covariances"), py::arg("covariance_type"),
            "Predict cluster labels from fitted GMM parameters.");

    clu.def("em_log_resp", &tuiml::clustering::em_log_resp,
            py::arg("X"), py::arg("weights"), py::arg("means"),
            py::arg("covariances"), py::arg("covariance_type"),
            "Compute log responsibilities (n_samples, n_components).");

    // ── utility functions ──────────────────────────────────────────────
    m.def("get_num_threads", &tuiml::get_num_threads,
          "Get current number of OpenMP threads.");
    m.def("set_num_threads", &tuiml::set_num_threads,
          py::arg("n"), "Set number of OpenMP threads.");

    m.attr("__version__") = "0.1.9";

#ifdef TUIML_USE_OPENMP
    m.attr("has_openmp") = true;
#else
    m.attr("has_openmp") = false;
#endif

    // ── stats submodule ────────────────────────────────────────────────
    auto stats = m.def_submodule("stats", "Shared statistical kernels");

    stats.def("pool_adjacent_violators",
              &tuiml::stats::pool_adjacent_violators,
              py::arg("y"), py::arg("w"), py::arg("increasing") = true,
              "Isotonic least-squares fit of a pre-sorted sequence (PAVA).");

    stats.def("isotonic_fit",
              &tuiml::stats::isotonic_fit,
              py::arg("x"), py::arg("y"), py::arg("w"),
              py::arg("increasing") = true,
              "Sort by x, then PAVA; returns (x_sorted, fitted).");

    stats.def("tail_probabilities",
              &tuiml::stats::tail_probabilities,
              py::arg("X_train"), py::arg("X_query"),
              "Per-dimension empirical left/right tail probabilities.");

    stats.def("skewness",
              &tuiml::stats::skewness,
              py::arg("X"),
              "Adjusted Fisher-Pearson skewness of each column.");

    stats.def("equal_width_histogram",
              &tuiml::stats::equal_width_histogram,
              py::arg("X"), py::arg("n_bins"),
              "Per-dimension equal-width histogram; returns (edges, density).");

    stats.def("equal_frequency_histogram",
              &tuiml::stats::equal_frequency_histogram,
              py::arg("X"), py::arg("n_bins"),
              "Per-dimension quantile histogram; returns (edges, density).");

    stats.def("histogram_density",
              &tuiml::stats::histogram_density,
              py::arg("edges"), py::arg("density"), py::arg("X_query"),
              "Density of each query value under a fitted histogram.");

    // ── timeseries submodule ───────────────────────────────────────────
    auto ts = m.def_submodule("timeseries", "Time-series kernels");

    ts.def("dtw_distance",
           &tuiml::timeseries::dtw_distance,
           py::arg("a"), py::arg("b"), py::arg("window") = -1,
           py::arg("cutoff") = 0.0,
           "DTW distance between two univariate series.");

    ts.def("lb_keogh_envelope",
           &tuiml::timeseries::lb_keogh_envelope,
           py::arg("series"), py::arg("window"),
           "Running min/max envelope under a Sakoe-Chiba band.");

    ts.def("lb_keogh",
           &tuiml::timeseries::lb_keogh,
           py::arg("query"), py::arg("lower"), py::arg("upper"),
           "LB_Keogh lower bound of the DTW distance.");

    ts.def("dtw_pairwise",
           &tuiml::timeseries::dtw_pairwise,
           py::arg("A"), py::arg("B"), py::arg("window") = -1,
           "Full pairwise DTW matrix over (n, channels, length) inputs.");

    ts.def("dtw_knn",
           &tuiml::timeseries::dtw_knn,
           py::arg("A"), py::arg("B"), py::arg("k"), py::arg("window") = -1,
           "k nearest neighbours under DTW, with LB_Keogh pruning.");

    ts.def("minirocket_kernel_indices",
           &tuiml::timeseries::minirocket_kernel_indices,
           "The 84 fixed MINIROCKET kernels as (84, 3) gamma positions.");

    ts.def("minirocket_biases",
           &tuiml::timeseries::minirocket_biases,
           py::arg("X"), py::arg("dilations"),
           py::arg("features_per_dilation"), py::arg("quantiles"),
           py::arg("seed"),
           "Fit MINIROCKET bias quantiles from convolution output.");

    ts.def("minirocket_transform",
           &tuiml::timeseries::minirocket_transform,
           py::arg("X"), py::arg("dilations"),
           py::arg("features_per_dilation"), py::arg("biases"),
           "Apply a fitted MINIROCKET transform, returning PPV features.");

    ts.def("shapelet_distances",
           &tuiml::timeseries::shapelet_distances,
           py::arg("X"), py::arg("shapelets"), py::arg("offsets"),
           py::arg("lengths"),
           "Minimum z-normalised distance from each series to each shapelet.");

    ts.def("sfa_transform",
           &tuiml::timeseries::sfa_transform,
           py::arg("X"), py::arg("window_size"), py::arg("word_length"),
           py::arg("norm_mean") = true,
           "Low-frequency DFT coefficients of every sliding window (SFA).");
}
