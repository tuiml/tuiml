#include "svm/smo_solver.h"
#include "common/parallel.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <cstdio>

namespace tuiml {
namespace svm {

static constexpr double INF = std::numeric_limits<double>::infinity();
static constexpr double TAU = 1e-12;  // min second-order denominator

// ══════════════════════════════════════════════════════════════════════
//  SMOSolver  (C-SVC,  binary +1/-1)
// ══════════════════════════════════════════════════════════════════════

SMOSolver::SMOSolver(KernelEvaluator& kern, const double* y, int n,
                     double C, double tol, int max_iter, bool shrinking)
    : kern_(kern), n_(n), y_(y), C_(C), tol_(tol),
      max_iter_(max_iter), shrinking_(shrinking),
      alpha_(n, 0.0), G_(n, -1.0),  // G_i = -1 initially (grad of dual obj)
      G_bar_(n, 0.0), active_size_(n), unshrink_(false)
{
    active_set_.resize(n);
    std::iota(active_set_.begin(), active_set_.end(), 0);
}

bool SMOSolver::select_working_set(int& out_i, int& out_j) {
    // WSS3 from Fan, Chen, Lin (2005)
    // Select i = argmax { -y_i * G_i : i in I_up }
    // Select j = argmin { -t^2 / (Q_ii + Q_jj - 2*Q_ij) } over I_low

    double Gmax = -INF;
    double Gmax2 = -INF;
    int Gmax_idx = -1;
    int Gmin_idx = -1;
    double obj_diff_min = INF;

    for (int t = 0; t < active_size_; ++t) {
        int i = active_set_[t];
        if (y_[i] == 1) {
            if (alpha_[i] < C_) {
                if (-G_[i] >= Gmax) {
                    Gmax = -G_[i];
                    Gmax_idx = i;
                }
            }
        } else {
            if (alpha_[i] > 0) {
                if (G_[i] >= Gmax) {
                    Gmax = G_[i];
                    Gmax_idx = i;
                }
            }
        }
    }

    int i = Gmax_idx;
    if (i == -1) return false;

    const auto& Q_i = kern_.get_column(i);

    for (int t = 0; t < active_size_; ++t) {
        int j = active_set_[t];
        if (y_[j] == 1) {
            if (alpha_[j] > 0) {
                double grad_diff = Gmax + G_[j];
                if (grad_diff > 0) {
                    if (G_[j] >= Gmax2) Gmax2 = G_[j];
                    // a_ij = K(i,i) + K(j,j) - 2*K(i,j)
                    double quad_coef = kern_.diag(i) + kern_.diag(j) - 2.0 * Q_i[j];
                    if (quad_coef <= 0) quad_coef = TAU;
                    double obj_diff = -(grad_diff * grad_diff) / quad_coef;
                    if (obj_diff <= obj_diff_min) {
                        Gmin_idx = j;
                        obj_diff_min = obj_diff;
                    }
                }
            }
        } else {
            if (alpha_[j] < C_) {
                double grad_diff = Gmax - G_[j];
                if (grad_diff > 0) {
                    if (-G_[j] >= Gmax2) Gmax2 = -G_[j];
                    double quad_coef = kern_.diag(i) + kern_.diag(j) - 2.0 * Q_i[j];
                    if (quad_coef <= 0) quad_coef = TAU;
                    double obj_diff = -(grad_diff * grad_diff) / quad_coef;
                    if (obj_diff <= obj_diff_min) {
                        Gmin_idx = j;
                        obj_diff_min = obj_diff;
                    }
                }
            }
        }
    }

    // Check stopping criterion
    if (Gmax + Gmax2 < tol_ || Gmin_idx == -1)
        return false;

    out_i = Gmax_idx;
    out_j = Gmin_idx;
    return true;
}

void SMOSolver::update_gradient_after_step(int i, int j,
                                           double delta_i, double delta_j) {
    const auto& Q_i = kern_.get_column(i);
    const auto& Q_j = kern_.get_column(j);

    for (int t = 0; t < active_size_; ++t) {
        int k = active_set_[t];
        G_[k] += Q_i[k] * delta_i + Q_j[k] * delta_j;
    }

    // Update G_bar for bounded variables tracking
    bool i_is_upper = (alpha_[i] >= C_);
    bool j_is_upper = (alpha_[j] >= C_);

    // If i becomes upper bounded
    if (!i_is_upper && alpha_[i] + delta_i * y_[i] >= C_) {
        // noop for simplicity — reconstruct handles this
    }
}

void SMOSolver::do_shrinking() {
    // Simple shrinking: remove variables at bounds whose gradient indicates
    // they will stay at bounds.
    double Gmax1 = -INF;  // max { -y_i G_i : i in I_up }
    double Gmax2 = -INF;  // max {  y_j G_j : j in I_low }

    for (int t = 0; t < active_size_; ++t) {
        int i = active_set_[t];
        if (y_[i] == 1) {
            if (alpha_[i] < C_ && -G_[i] > Gmax1) Gmax1 = -G_[i];
            if (alpha_[i] > 0  &&  G_[i] > Gmax2) Gmax2 =  G_[i];
        } else {
            if (alpha_[i] > 0  &&  G_[i] > Gmax1) Gmax1 =  G_[i];
            if (alpha_[i] < C_ && -G_[i] > Gmax2) Gmax2 = -G_[i];
        }
    }

    if (!unshrink_ && Gmax1 + Gmax2 <= tol_ * 10) {
        unshrink_ = true;
        reconstruct_gradient();
        active_size_ = n_;
        // Reset active set
        active_set_.resize(n_);
        std::iota(active_set_.begin(), active_set_.end(), 0);
    }

    for (int t = 0; t < active_size_; ++t) {
        int i = active_set_[t];
        bool shrink = false;
        if (y_[i] == 1) {
            if (alpha_[i] >= C_ && G_[i] > Gmax2) shrink = true;
            if (alpha_[i] <= 0  && -G_[i] > Gmax1) shrink = true;
        } else {
            if (alpha_[i] >= C_ && -G_[i] > Gmax1) shrink = true;
            if (alpha_[i] <= 0  &&  G_[i] > Gmax2) shrink = true;
        }
        if (shrink) {
            --active_size_;
            std::swap(active_set_[t], active_set_[active_size_]);
            --t;
        }
    }
}

void SMOSolver::reconstruct_gradient() {
    // Rebuild gradient from scratch for inactive variables
    for (int t = active_size_; t < n_; ++t) {
        int i = active_set_[t];
        G_[i] = -1.0;
    }

    for (int j = 0; j < n_; ++j) {
        if (alpha_[j] == 0) continue;
        const auto& Q_j = kern_.get_column(j);
        double aj_yj = alpha_[j] * y_[j];
        for (int t = active_size_; t < n_; ++t) {
            int i = active_set_[t];
            G_[i] += aj_yj * y_[i] * Q_j[i];
        }
    }
}

SMOResult SMOSolver::solve() {
    int iter = 0;
    int counter = std::min(n_, 1000) + 1;

    while (iter < max_iter_) {
        // Periodically shrink
        if (shrinking_ && --counter == 0) {
            counter = std::min(n_, 1000);
            do_shrinking();
        }

        int i, j;
        if (!select_working_set(i, j)) {
            // Reconstruct full gradient and check again
            reconstruct_gradient();
            active_size_ = n_;
            active_set_.resize(n_);
            std::iota(active_set_.begin(), active_set_.end(), 0);
            if (!select_working_set(i, j))
                break;  // Truly optimal
        }

        ++iter;

        // Get kernel values
        const auto& Q_i = kern_.get_column(i);
        const auto& Q_j = kern_.get_column(j);

        double C_i = C_;
        double C_j = C_;

        double old_alpha_i = alpha_[i];
        double old_alpha_j = alpha_[j];

        if (y_[i] != y_[j]) {
            // Different labels — Hessian is always K_ii + K_jj - 2*K_ij
            double quad_coef = kern_.diag(i) + kern_.diag(j) - 2.0 * Q_i[j];
            if (quad_coef <= 0) quad_coef = TAU;
            double delta = (-G_[i] - G_[j]) / quad_coef;
            double diff = alpha_[i] - alpha_[j];
            alpha_[i] += delta;
            alpha_[j] += delta;

            if (diff > 0) {
                if (alpha_[j] < 0) { alpha_[j] = 0; alpha_[i] = diff; }
            } else {
                if (alpha_[i] < 0) { alpha_[i] = 0; alpha_[j] = -diff; }
            }
            if (diff > C_i - C_j) {
                if (alpha_[i] > C_i) { alpha_[i] = C_i; alpha_[j] = C_i - diff; }
            } else {
                if (alpha_[j] > C_j) { alpha_[j] = C_j; alpha_[i] = C_j + diff; }
            }
        } else {
            // Same labels
            double quad_coef = kern_.diag(i) + kern_.diag(j) - 2.0 * Q_i[j];
            if (quad_coef <= 0) quad_coef = TAU;
            double delta = (G_[i] - G_[j]) / quad_coef;
            double sum = alpha_[i] + alpha_[j];
            alpha_[i] -= delta;
            alpha_[j] += delta;

            if (sum > C_i) {
                if (alpha_[i] > C_i) { alpha_[i] = C_i; alpha_[j] = sum - C_i; }
            } else {
                if (alpha_[j] < 0) { alpha_[j] = 0; alpha_[i] = sum; }
            }
            if (sum > C_j) {
                if (alpha_[j] > C_j) { alpha_[j] = C_j; alpha_[i] = sum - C_j; }
            } else {
                if (alpha_[i] < 0) { alpha_[i] = 0; alpha_[j] = sum; }
            }
        }

        // Update gradient
        double delta_i = (alpha_[i] - old_alpha_i) * y_[i];
        double delta_j = (alpha_[j] - old_alpha_j) * y_[j];

        for (int t = 0; t < active_size_; ++t) {
            int k = active_set_[t];
            G_[k] += Q_i[k] * y_[k] * delta_i + Q_j[k] * y_[k] * delta_j;
        }
    }

    // Compute rho (bias)
    double rho = 0;
    int nr_free = 0;
    double ub = INF, lb = -INF, sum_free = 0;

    for (int i = 0; i < n_; ++i) {
        double yG = y_[i] * G_[i];
        if (alpha_[i] >= C_) {
            if (y_[i] == -1) ub = std::min(ub, yG);
            else             lb = std::max(lb, yG);
        } else if (alpha_[i] <= 0) {
            if (y_[i] == +1) ub = std::min(ub, yG);
            else             lb = std::max(lb, yG);
        } else {
            ++nr_free;
            sum_free += yG;
        }
    }

    if (nr_free > 0)
        rho = sum_free / nr_free;
    else
        rho = (ub + lb) / 2.0;

    // Build result
    SMOResult result;
    result.alpha = alpha_;
    result.rho = rho;
    result.n_iter = iter;

    for (int i = 0; i < n_; ++i) {
        if (std::abs(alpha_[i]) > 1e-8)
            result.sv_indices.push_back(i);
    }

    return result;
}

// ══════════════════════════════════════════════════════════════════════
//  SMOSolverSVR  (epsilon-SVR)
// ══════════════════════════════════════════════════════════════════════

SMOSolverSVR::SMOSolverSVR(KernelEvaluator& kern, const double* y, int n,
                           double C, double epsilon, double tol, int max_iter,
                           bool shrinking)
    : kern_(kern), n_(n), y_(y), C_(C), epsilon_(epsilon),
      tol_(tol), max_iter_(max_iter), shrinking_(shrinking),
      alpha_(2 * n, 0.0), G_(2 * n), G_bar_(2 * n, 0.0),
      active_size_(2 * n), unshrink_(false)
{
    // Initialize gradient:
    // For k < n  (alpha_k):     G_k = epsilon - y_k
    // For k >= n (alpha*_{k-n}): G_k = epsilon + y_{k-n}
    for (int i = 0; i < n_; ++i) {
        G_[i]      = epsilon_ - y_[i];
        G_[i + n_] = epsilon_ + y_[i];
    }
    active_set_.resize(2 * n);
    std::iota(active_set_.begin(), active_set_.end(), 0);
}

bool SMOSolverSVR::select_working_set(int& out_i, int& out_j) {
    // WSS3 adapted for SVR (matches libsvm's Solver with sign-aware branches)
    double Gmax = -INF;
    double Gmax2 = -INF;
    int Gmax_idx = -1;
    int Gmin_idx = -1;
    double obj_diff_min = INF;

    // Select i from I_up (sign-aware violation measure)
    for (int t = 0; t < active_size_; ++t) {
        int k = active_set_[t];
        double Ck = get_C(k);

        if (sign(k) == 1) {
            // alpha var: I_up = alpha < C, violation = -G
            if (alpha_[k] < Ck) {
                if (-G_[k] >= Gmax) {
                    Gmax = -G_[k];
                    Gmax_idx = k;
                }
            }
        } else {
            // alpha* var: I_up = alpha > 0, violation = G
            if (alpha_[k] > 0) {
                if (G_[k] >= Gmax) {
                    Gmax = G_[k];
                    Gmax_idx = k;
                }
            }
        }
    }

    int i = Gmax_idx;
    if (i == -1) return false;

    int oi = original_index(i);
    const auto& Q_oi = kern_.get_column(oi);

    // Select j from I_low (sign-aware grad_diff)
    for (int t = 0; t < active_size_; ++t) {
        int k = active_set_[t];
        double Ck = get_C(k);

        if (sign(k) == 1) {
            // alpha var: I_low = alpha > 0
            if (alpha_[k] > 0) {
                double grad_diff = Gmax + G_[k];
                if (G_[k] >= Gmax2) Gmax2 = G_[k];
                if (grad_diff > 0) {
                    int ok = original_index(k);
                    double quad_coef = kern_.diag(oi) + kern_.diag(ok)
                                       - 2.0 * Q_oi[ok];
                    if (quad_coef <= 0) quad_coef = TAU;
                    double obj_diff = -(grad_diff * grad_diff) / quad_coef;
                    if (obj_diff <= obj_diff_min) {
                        Gmin_idx = k;
                        obj_diff_min = obj_diff;
                    }
                }
            }
        } else {
            // alpha* var: I_low = alpha < C
            if (alpha_[k] < Ck) {
                double grad_diff = Gmax - G_[k];
                if (-G_[k] >= Gmax2) Gmax2 = -G_[k];
                if (grad_diff > 0) {
                    int ok = original_index(k);
                    double quad_coef = kern_.diag(oi) + kern_.diag(ok)
                                       - 2.0 * Q_oi[ok];
                    if (quad_coef <= 0) quad_coef = TAU;
                    double obj_diff = -(grad_diff * grad_diff) / quad_coef;
                    if (obj_diff <= obj_diff_min) {
                        Gmin_idx = k;
                        obj_diff_min = obj_diff;
                    }
                }
            }
        }
    }

    if (Gmax + Gmax2 < tol_ || Gmin_idx == -1)
        return false;

    out_i = Gmax_idx;
    out_j = Gmin_idx;
    return true;
}

void SMOSolverSVR::do_shrinking() {
    double Gmax1 = -INF;  // max violation over I_up
    double Gmax2 = -INF;  // max violation over I_low

    for (int t = 0; t < active_size_; ++t) {
        int k = active_set_[t];
        double Ck = get_C(k);

        if (sign(k) == 1) {
            if (alpha_[k] < Ck && -G_[k] > Gmax1) Gmax1 = -G_[k];
            if (alpha_[k] > 0  &&  G_[k] > Gmax2) Gmax2 =  G_[k];
        } else {
            if (alpha_[k] > 0  &&  G_[k] > Gmax1) Gmax1 =  G_[k];
            if (alpha_[k] < Ck && -G_[k] > Gmax2) Gmax2 = -G_[k];
        }
    }

    if (!unshrink_ && Gmax1 + Gmax2 <= tol_ * 10) {
        unshrink_ = true;
        reconstruct_gradient();
        active_size_ = 2 * n_;
        active_set_.resize(2 * n_);
        std::iota(active_set_.begin(), active_set_.end(), 0);
    }

    for (int t = 0; t < active_size_; ++t) {
        int k = active_set_[t];
        double Ck = get_C(k);
        bool shrink = false;

        if (sign(k) == 1) {
            if (alpha_[k] >= Ck &&  G_[k] > Gmax2) shrink = true;
            if (alpha_[k] <= 0  && -G_[k] > Gmax1) shrink = true;
        } else {
            if (alpha_[k] >= Ck && -G_[k] > Gmax2) shrink = true;
            if (alpha_[k] <= 0  &&  G_[k] > Gmax1) shrink = true;
        }

        if (shrink) {
            --active_size_;
            std::swap(active_set_[t], active_set_[active_size_]);
            --t;
        }
    }
}

void SMOSolverSVR::reconstruct_gradient() {
    for (int t = active_size_; t < 2 * n_; ++t) {
        int k = active_set_[t];
        int ok = original_index(k);
        G_[k] = (sign(k) == 1) ? (epsilon_ - y_[ok]) : (epsilon_ + y_[ok]);
    }

    for (int j = 0; j < 2 * n_; ++j) {
        if (alpha_[j] == 0) continue;
        int oj = original_index(j);
        const auto& Q_oj = kern_.get_column(oj);
        double sj = sign(j);
        for (int t = active_size_; t < 2 * n_; ++t) {
            int k = active_set_[t];
            int ok = original_index(k);
            double sk = sign(k);
            G_[k] += alpha_[j] * sj * sk * Q_oj[ok];
        }
    }
}

SMOResult SMOSolverSVR::solve() {
    int iter = 0;
    int counter = std::min(2 * n_, 1000) + 1;

    while (iter < max_iter_) {
        if (shrinking_ && --counter == 0) {
            counter = std::min(2 * n_, 1000);
            do_shrinking();
        }

        int i, j;
        if (!select_working_set(i, j)) {
            reconstruct_gradient();
            active_size_ = 2 * n_;
            active_set_.resize(2 * n_);
            std::iota(active_set_.begin(), active_set_.end(), 0);
            if (!select_working_set(i, j))
                break;
        }

        ++iter;

        int oi = original_index(i), oj = original_index(j);
        double si = sign(i), sj = sign(j);
        const auto& Q_oi = kern_.get_column(oi);
        const auto& Q_oj = kern_.get_column(oj);

        double Ci = get_C(i);
        double Cj = get_C(j);

        double old_alpha_i = alpha_[i];
        double old_alpha_j = alpha_[j];

        // Hessian is always K(oi,oi) + K(oj,oj) - 2*K(oi,oj)
        double quad_coef = kern_.diag(oi) + kern_.diag(oj)
                          - 2.0 * Q_oi[oj];
        if (quad_coef <= 0) quad_coef = TAU;

        if (si != sj) {
            double delta = (-G_[i] - G_[j]) / quad_coef;
            double diff = alpha_[i] - alpha_[j];
            alpha_[i] += delta;
            alpha_[j] += delta;

            if (diff > 0) {
                if (alpha_[j] < 0) { alpha_[j] = 0; alpha_[i] = diff; }
            } else {
                if (alpha_[i] < 0) { alpha_[i] = 0; alpha_[j] = -diff; }
            }
            if (diff > Ci - Cj) {
                if (alpha_[i] > Ci) { alpha_[i] = Ci; alpha_[j] = Ci - diff; }
            } else {
                if (alpha_[j] > Cj) { alpha_[j] = Cj; alpha_[i] = Cj + diff; }
            }
        } else {
            double delta = (G_[i] - G_[j]) / quad_coef;
            double sum = alpha_[i] + alpha_[j];
            alpha_[i] -= delta;
            alpha_[j] += delta;

            if (sum > Ci) {
                if (alpha_[i] > Ci) { alpha_[i] = Ci; alpha_[j] = sum - Ci; }
            } else {
                if (alpha_[j] < 0) { alpha_[j] = 0; alpha_[i] = sum; }
            }
            if (sum > Cj) {
                if (alpha_[j] > Cj) { alpha_[j] = Cj; alpha_[i] = sum - Cj; }
            } else {
                if (alpha_[i] < 0) { alpha_[i] = 0; alpha_[j] = sum; }
            }
        }

        // Update gradient for all active variables
        double delta_i = alpha_[i] - old_alpha_i;
        double delta_j = alpha_[j] - old_alpha_j;

        for (int t = 0; t < active_size_; ++t) {
            int k = active_set_[t];
            int ok = original_index(k);
            double sk = sign(k);
            G_[k] += Q_oi[ok] * si * sk * delta_i + Q_oj[ok] * sj * sk * delta_j;
        }
    }

    // Compute rho (use sign(k)*G[k] to match libsvm convention)
    double rho = 0;
    int nr_free = 0;
    double ub = INF, lb = -INF, sum_free = 0;

    for (int k = 0; k < 2 * n_; ++k) {
        double Ck = get_C(k);
        double yG = sign(k) * G_[k];
        if (alpha_[k] >= Ck) {
            if (sign(k) == -1) ub = std::min(ub, yG);
            else                lb = std::max(lb, yG);
        } else if (alpha_[k] <= 0) {
            if (sign(k) == +1) ub = std::min(ub, yG);
            else                lb = std::max(lb, yG);
        } else {
            ++nr_free;
            sum_free += yG;
        }
    }

    if (nr_free > 0)
        rho = sum_free / nr_free;
    else
        rho = (ub + lb) / 2.0;

    // Combine alpha and alpha* into net coefficients
    SMOResult result;
    result.rho = rho;
    result.n_iter = iter;
    result.alpha.resize(n_);
    for (int i = 0; i < n_; ++i) {
        result.alpha[i] = alpha_[i] - alpha_[i + n_];
    }
    for (int i = 0; i < n_; ++i) {
        if (std::abs(result.alpha[i]) > 1e-8)
            result.sv_indices.push_back(i);
    }

    return result;
}

}  // namespace svm
}  // namespace tuiml
