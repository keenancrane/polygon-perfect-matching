#include "mwpm/algorithms.hpp"

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace mwpm {

namespace {

// Standard interval DP for non-crossing matchings on points lying in convex
// position. Let cost[i][j] be the minimum total weight of a perfect matching
// of points[i..j] (inclusive). The recurrence is
//
//   cost[i][j] = min over k in {i+1, i+3, ..., j} of
//                  d(points[i], points[k]) + cost[i+1..k-1] + cost[k+1..j]
//
// (with k - i odd, so that both sub-ranges have even length). For convenience
// we treat empty ranges as having cost 0 and an empty matching.

struct DpScratch {
    std::vector<double> dist;
    std::vector<double> cost;
    std::vector<double> cost_t;
    std::vector<int> partner;
    std::vector<std::pair<int, int>> stack;
};

}  // namespace

MatchingResult dp_matching(const std::vector<Point>& points) {
    const int n = static_cast<int>(points.size());
    if (n % 2 != 0) {
        throw std::invalid_argument("dp_matching requires an even number of points");
    }
    MatchingResult result;
    if (n == 0) return result;

    // Distances are reused by O(N^3) transitions; precomputing removes the
    // sqrt from the hot loop. cost_t mirrors cost transposed, so both
    // subproblem lookups in the recurrence are contiguous as k increases.
    static thread_local DpScratch scratch;
    const auto n_size = static_cast<std::size_t>(n);
    const auto nn = n_size * n_size;
    auto& dist = scratch.dist;
    dist.resize(nn);
    for (int i = 0; i < n; ++i) {
        const double xi = points[static_cast<std::size_t>(i)].x;
        const double yi = points[static_cast<std::size_t>(i)].y;
        const auto base_i = static_cast<std::size_t>(i) * n_size;
        for (int j = i + 1; j < n; j += 2) {
            const double dx = points[static_cast<std::size_t>(j)].x - xi;
            const double dy = points[static_cast<std::size_t>(j)].y - yi;
            dist[base_i + static_cast<std::size_t>(j)] = std::sqrt(dx * dx + dy * dy);
        }
    }

    const int stride = n + 1;
    const auto stride_size = static_cast<std::size_t>(stride);
    const auto table_size = stride_size * stride_size;
    auto& cost = scratch.cost;
    auto& cost_t = scratch.cost_t;
    cost.resize(table_size);
    cost_t.resize(table_size);
    for (int i = 0; i < n; ++i) {
        cost[(static_cast<std::size_t>(i) + 1) * stride_size +
             static_cast<std::size_t>(i)] = 0.0;
        cost_t[static_cast<std::size_t>(i) * stride_size +
               static_cast<std::size_t>(i + 1)] = 0.0;
    }
    // partner[i][j] = the index k that point i is paired with for the optimal
    // matching of [i..j]. Used to reconstruct the matching pairs.
    auto& partner = scratch.partner;
    partner.resize(table_size);

    // Fill by interval length. Lengths 2, 4, 6, ... up to n.
    for (int len = 2; len <= n; len += 2) {
        for (int i = 0; i + len - 1 < n; ++i) {
            const int j = i + len - 1;
            const auto dist_base = static_cast<std::size_t>(i) * n_size;
            const auto out = static_cast<std::size_t>(i) * stride + j;
            const auto left_base = static_cast<std::size_t>(i + 1) * stride;
            const auto right_base = static_cast<std::size_t>(j) * stride;
            double best = std::numeric_limits<double>::infinity();
            int best_k = -1;
            for (int k = i + 1; k <= j; k += 2) {
                // Pair i with k, leaving sub-intervals [i+1..k-1] and [k+1..j].
                const double total = dist[dist_base + k] +
                                     cost[left_base + k - 1] +
                                     cost_t[right_base + k + 1];
                if (total < best) {
                    best = total;
                    best_k = k;
                }
            }
            cost[out] = best;
            cost_t[right_base + i] = best;
            partner[out] = best_k;
        }
    }

    result.cost = cost[static_cast<std::size_t>(n - 1)];

    // Reconstruct pairs by walking the partner table.
    std::vector<Pair> pairs;
    pairs.reserve(static_cast<std::size_t>(n / 2));

    // Stack-based traversal over intervals.
    auto& stack = scratch.stack;
    stack.clear();
    stack.reserve(static_cast<std::size_t>(n / 2));
    stack.emplace_back(0, n - 1);
    while (!stack.empty()) {
        auto [i, j] = stack.back();
        stack.pop_back();
        if (i > j) continue;
        const int k = partner[static_cast<std::size_t>(i) * stride + j];
        pairs.emplace_back(static_cast<std::size_t>(i), static_cast<std::size_t>(k));
        if (i + 1 <= k - 1) stack.emplace_back(i + 1, k - 1);
        if (k + 1 <= j) stack.emplace_back(k + 1, j);
    }
    result.pairs = std::move(pairs);
    return result;
}

}  // namespace mwpm
