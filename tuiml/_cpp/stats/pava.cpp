#include "pava.h"

#include <algorithm>
#include <numeric>
#include <vector>

namespace tuiml {
namespace stats {

namespace {

// In-place PAVA over contiguous blocks. ``value`` holds the weighted block
// means, ``weight`` the block masses and ``size`` the sample counts. Returns
// the number of surviving blocks.
int pava_blocks(std::vector<double>& value, std::vector<double>& weight,
                std::vector<int>& size, int n) {
    int blocks = 0;
    for (int i = 0; i < n; i++) {
        double v = value[i];
        double w = weight[i];
        int s = size[i];

        // Merge backwards while the new block violates monotonicity.
        while (blocks > 0 && value[blocks - 1] > v) {
            blocks--;
            double total = weight[blocks] + w;
            v = (weight[blocks] * value[blocks] + w * v) / total;
            w = total;
            s += size[blocks];
        }

        value[blocks] = v;
        weight[blocks] = w;
        size[blocks] = s;
        blocks++;
    }
    return blocks;
}

}  // namespace

py::array_t<double> pool_adjacent_violators(py::array_t<double> y,
                                            py::array_t<double> w,
                                            bool increasing) {
    auto y_buf = y.unchecked<1>();
    auto w_buf = w.unchecked<1>();
    int n = static_cast<int>(y_buf.shape(0));

    py::array_t<double> result(n);
    auto res = result.mutable_unchecked<1>();
    if (n == 0) return result;

    // A decreasing fit is the increasing fit of the reversed sequence.
    std::vector<double> value(n), weight(n);
    std::vector<int> size(n, 1);
    for (int i = 0; i < n; i++) {
        int src = increasing ? i : (n - 1 - i);
        value[i] = y_buf(src);
        weight[i] = w_buf(src);
    }

    int blocks = pava_blocks(value, weight, size, n);

    // Expand block means back to one value per sample.
    int pos = 0;
    for (int b = 0; b < blocks; b++) {
        for (int k = 0; k < size[b]; k++) {
            int dst = increasing ? pos : (n - 1 - pos);
            res(dst) = value[b];
            pos++;
        }
    }

    return result;
}

std::pair<py::array_t<double>, py::array_t<double>> isotonic_fit(
    py::array_t<double> x, py::array_t<double> y, py::array_t<double> w,
    bool increasing) {
    auto x_buf = x.unchecked<1>();
    auto y_buf = y.unchecked<1>();
    auto w_buf = w.unchecked<1>();
    int n = static_cast<int>(x_buf.shape(0));

    std::vector<int> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(),
                     [&](int a, int b) { return x_buf(a) < x_buf(b); });

    py::array_t<double> xs(n), ys(n), ws(n);
    auto xs_buf = xs.mutable_unchecked<1>();
    auto ys_buf = ys.mutable_unchecked<1>();
    auto ws_buf = ws.mutable_unchecked<1>();
    for (int i = 0; i < n; i++) {
        xs_buf(i) = x_buf(order[i]);
        ys_buf(i) = y_buf(order[i]);
        ws_buf(i) = w_buf(order[i]);
    }

    py::array_t<double> fitted = pool_adjacent_violators(ys, ws, increasing);
    return {xs, fitted};
}

}  // namespace stats
}  // namespace tuiml
