#pragma once

#include <vector>
#include <cstddef>
#include "svm/kernels.h"

namespace tuiml {
namespace svm {

// ── Result returned by SMO solvers ───────────────────────────────────
struct SMOResult {
    std::vector<double> alpha;     // Lagrange multipliers (size n)
    double rho;                    // Bias = -b
    int n_iter;                    // Iterations used
    std::vector<int> sv_indices;   // Indices where |alpha| > 0
};

// ── C-SVC SMO Solver (binary, labels +1/-1) ─────────────────────────
// Uses WSS3 (Fan et al. 2005) working-set selection with shrinking.
class SMOSolver {
public:
    SMOSolver(KernelEvaluator& kern, const double* y, int n,
              double C, double tol, int max_iter, bool shrinking = true);

    SMOResult solve();

private:
    // WSS3: select working pair (i, j)
    // Returns false if optimality reached.
    bool select_working_set(int& out_i, int& out_j);

    void update_gradient_after_step(int i, int j,
                                    double delta_i, double delta_j);

    // Shrinking: deactivate bounded variables
    void do_shrinking();
    void reconstruct_gradient();

    KernelEvaluator& kern_;
    int n_;                   // total training size
    const double* y_;         // +1 / -1 labels
    double C_;
    double tol_;
    int max_iter_;
    bool shrinking_;

    std::vector<double> alpha_;
    std::vector<double> G_;   // gradient = sum_j alpha_j y_j K(i,j) - 1
    std::vector<double> G_bar_; // gradient of fully-bounded variables

    // Active set
    std::vector<int> active_set_;
    int active_size_;
    bool unshrink_;
};

// ── Epsilon-SVR SMO Solver ──────────────────────────────────────────
// Reformulated as 2n problem with alpha and alpha* variables.
class SMOSolverSVR {
public:
    SMOSolverSVR(KernelEvaluator& kern, const double* y, int n,
                 double C, double epsilon, double tol, int max_iter,
                 bool shrinking = true);

    SMOResult solve();

private:
    bool select_working_set(int& out_i, int& out_j);
    void do_shrinking();
    void reconstruct_gradient();

    KernelEvaluator& kern_;
    int n_;
    const double* y_;
    double C_, epsilon_, tol_;
    int max_iter_;
    bool shrinking_;

    // 2n variables: index k < n  => alpha_k, index k >= n => alpha*_{k-n}
    std::vector<double> alpha_;   // size 2n
    std::vector<double> G_;       // gradient, size 2n
    std::vector<double> G_bar_;

    std::vector<int> active_set_;
    int active_size_;
    bool unshrink_;

    // Sign helper: +1 for alpha, -1 for alpha*
    double sign(int k) const { return (k < n_) ? 1.0 : -1.0; }
    int original_index(int k) const { return (k < n_) ? k : k - n_; }
    double get_C(int k) const { return C_; }
};

}  // namespace svm
}  // namespace tuiml
