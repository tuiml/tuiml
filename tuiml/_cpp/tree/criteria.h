#pragma once

#include <vector>
#include <string>

namespace tuiml {
namespace tree {

// Compute impurity from class counts
double compute_impurity(const std::vector<int>& counts, int n_samples,
                        const std::string& criterion);

// Compute impurity from complement counts (total - left = right)
double compute_impurity_complement(const std::vector<int>& total_counts,
                                   const std::vector<int>& left_counts,
                                   int n_right,
                                   const std::string& criterion);

// Individual criterion functions
double gini_impurity(const std::vector<int>& counts, int n_samples);
double entropy_impurity(const std::vector<int>& counts, int n_samples);

// Regression criteria
double mse_impurity(const double* y, int n_samples);
double mse_from_stats(double sum, double sq_sum, int n);

}  // namespace tree
}  // namespace tuiml
