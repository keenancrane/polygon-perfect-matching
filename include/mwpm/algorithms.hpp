#pragma once

#include <vector>

#include "mwpm/point.hpp"

namespace mwpm {

// Brute force: enumerate every perfect matching (including crossing ones) by
// always pairing the lowest-index unmatched point with each of its options, and
// keep the minimum cost found. Crossing matchings strictly dominate
// non-crossing ones in Euclidean cost, so the minimum is non-crossing. Runs in
// O(N!!) time and is intended for N <= 20.
MatchingResult brute_force_matching(const std::vector<Point>& points);

// O(N^3) interval dynamic programming: C(i, j) = minimum cost matching of the
// (cyclically contiguous) sub-range points[i..j], assuming an even number of
// points. Recurrence pairs i with some k in (i, j], splitting into independent
// sub-ranges (i+1..k-1) and (k+1..j).
MatchingResult dp_matching(const std::vector<Point>& points);

// Marcotte & Suri divide-and-conquer (Find_Matching, Section 4 of the paper).
// `simple_scan` selects how G_1 / G_2 are built during the conquer phase:
//   - false (default): use the SMAWK matrix-searching algorithm of Aggarwal,
//     Klawe, Moran, Shor & Wilber (1987) on the totally monotone weighted-
//     distance matrix. Conquer is O(N), so the recurrence becomes
//     T(n) = 2 T(n/2) + O(n) and the algorithm runs in O(N log N) time.
//   - true: use direct O(|R| * |L|) weighted-distance comparisons. Conquer
//     is O(N^2) and the recurrence yields O(N^2) overall. Useful as a
//     reference implementation and as a debugging fallback.
// Either variant returns the same matching cost; the SMAWK and scan variants
// are cross-checked inside validation mode of the CLI driver.
MatchingResult marcotte_suri_matching(const std::vector<Point>& points,
                                      bool simple_scan = false);

// Same algorithm and cost as marcotte_suri_matching, but skips constructing
// the output pair list. Intended for benchmark/challenge modes that only need
// the optimum objective value.
double marcotte_suri_cost(const std::vector<Point>& points,
                          bool simple_scan = false);

// Exact hybrid used for performance sweeps: optimized interval DP for tiny
// inputs, tuned Marcotte-Suri/SMAWK for larger inputs.
MatchingResult optimized_matching(const std::vector<Point>& points);
double optimized_cost(const std::vector<Point>& points);

}  // namespace mwpm
