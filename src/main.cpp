#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <string_view>
#include <vector>

#include "mwpm/algorithms.hpp"
#include "mwpm/generator.hpp"
#include "mwpm/geometry.hpp"
#include "mwpm/io.hpp"
#include "mwpm/point.hpp"

namespace {

constexpr int kBruteForceCap = 20;
constexpr int kDpCap = 4000;
constexpr double kCostTolerance = 1e-7;

struct BenchmarkOptions {
    int trials = 100;
    int max_points = 16;
    int exact_points = 0;
    bool validate = false;
    bool simple_scan = false;
    bool optimized = false;
    bool cost_only = false;
    bool skip_dp = false;
    bool skip_brute = false;
    std::uint64_t seed = 0;
    bool seed_specified = false;
};

struct MatchOptions {
    std::string input_path;
    std::string output_path;  // empty => stdout
    bool write_svg = false;
    bool simple_scan = false;
    bool validate = false;
};

enum class Mode { Match, Benchmark };

struct CliOptions {
    Mode mode = Mode::Match;
    MatchOptions match;
    BenchmarkOptions benchmark;
};

void print_usage(std::ostream& os, const char* prog) {
    os << "Usage:\n"
       << "  " << prog << " [options] <input.json>     compute a matching\n"
       << "  " << prog << " --benchmark [options]      benchmark / validate\n"
       << "\n"
       << "Matching mode options:\n"
       << "  <input.json>             Input polygon and boundary points (JSON)\n"
       << "  -o, --output <file>      Write result JSON (default: stdout)\n"
       << "      --svg                Write SVG visualization to <input.json>.svg\n"
       << "      --simple-scan        Use O(N^2) scan instead of SMAWK\n"
       << "  -v, --validate           Cross-check BF / DP / Paper on this input\n"
       << "\n"
       << "Benchmark mode options (--benchmark):\n"
       << "  -t, --trials <int>       Number of random geometries (default 100)\n"
       << "  -m, --max-points <int>   Maximum total points per triangle (default 16)\n"
       << "  -e, --exact-points <int> Force every trial to have exactly this many\n"
       << "                           points (even, >= 2). Overrides --max-points.\n"
       << "  -v, --validate           Cross-check BF / DP / Paper on every trial\n"
       << "      --simple-scan        Benchmark only the O(N^2) scan variant\n"
       << "      --optimized          Benchmark the exact hybrid optimized solver\n"
       << "      --cost-only          Benchmark objective value only, not pair output\n"
       << "      --skip-dp            Skip DP regardless of N\n"
       << "      --skip-brute         Skip brute force regardless of N\n"
       << "      --seed <uint64>      Deterministic RNG seed\n"
       << "\n"
       << "  -h, --help               Show this help text\n";
}

bool parse_int(const std::string& s, int& out) {
    try {
        std::size_t pos = 0;
        int v = std::stoi(s, &pos);
        if (pos != s.size()) return false;
        out = v;
        return true;
    } catch (...) {
        return false;
    }
}

bool parse_uint64(const std::string& s, std::uint64_t& out) {
    try {
        std::size_t pos = 0;
        std::uint64_t v = std::stoull(s, &pos);
        if (pos != s.size()) return false;
        out = v;
        return true;
    } catch (...) {
        return false;
    }
}

bool parse_cli(int argc, char** argv, CliOptions& opts) {
    for (int i = 1; i < argc; ++i) {
        const std::string_view a(argv[i]);
        auto next_arg = [&](const char* flag) -> const char* {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for " << flag << "\n";
                return nullptr;
            }
            return argv[++i];
        };

        if (a == "--benchmark") {
            opts.mode = Mode::Benchmark;
        } else if (a == "-o" || a == "--output") {
            if (opts.mode == Mode::Benchmark) {
                std::cerr << "--output is not valid in benchmark mode\n";
                return false;
            }
            const char* v = next_arg("--output");
            if (!v) return false;
            opts.match.output_path = v;
        } else if (a == "--svg") {
            if (opts.mode == Mode::Benchmark) {
                std::cerr << "--svg is not valid in benchmark mode\n";
                return false;
            }
            opts.match.write_svg = true;
        } else if (a == "--simple-scan") {
            if (opts.mode == Mode::Benchmark) {
                opts.benchmark.simple_scan = true;
            } else {
                opts.match.simple_scan = true;
            }
        } else if (a == "--cost-only") {
            if (opts.mode != Mode::Benchmark) {
                std::cerr << "--cost-only requires --benchmark\n";
                return false;
            }
            opts.benchmark.cost_only = true;
        } else if (a == "--optimized") {
            if (opts.mode != Mode::Benchmark) {
                std::cerr << "--optimized requires --benchmark\n";
                return false;
            }
            opts.benchmark.optimized = true;
        } else if (a == "-t" || a == "--trials") {
            if (opts.mode != Mode::Benchmark) {
                std::cerr << "-t/--trials requires --benchmark\n";
                return false;
            }
            const char* v = next_arg("--trials");
            if (!v || !parse_int(v, opts.benchmark.trials) || opts.benchmark.trials <= 0) {
                return false;
            }
        } else if (a == "-m" || a == "--max-points") {
            if (opts.mode != Mode::Benchmark) {
                std::cerr << "-m/--max-points requires --benchmark\n";
                return false;
            }
            const char* v = next_arg("--max-points");
            if (!v || !parse_int(v, opts.benchmark.max_points) ||
                opts.benchmark.max_points < 2) {
                return false;
            }
        } else if (a == "-e" || a == "--exact-points") {
            if (opts.mode != Mode::Benchmark) {
                std::cerr << "-e/--exact-points requires --benchmark\n";
                return false;
            }
            const char* v = next_arg("--exact-points");
            if (!v || !parse_int(v, opts.benchmark.exact_points) ||
                opts.benchmark.exact_points < 2 ||
                opts.benchmark.exact_points % 2 != 0) {
                std::cerr << "--exact-points must be an even integer >= 2\n";
                return false;
            }
        } else if (a == "-v" || a == "--validate") {
            if (opts.mode == Mode::Benchmark) {
                opts.benchmark.validate = true;
            } else {
                opts.match.validate = true;
            }
        } else if (a == "--skip-dp") {
            if (opts.mode != Mode::Benchmark) {
                std::cerr << "--skip-dp requires --benchmark\n";
                return false;
            }
            opts.benchmark.skip_dp = true;
        } else if (a == "--skip-brute") {
            if (opts.mode != Mode::Benchmark) {
                std::cerr << "--skip-brute requires --benchmark\n";
                return false;
            }
            opts.benchmark.skip_brute = true;
        } else if (a == "--seed") {
            if (opts.mode != Mode::Benchmark) {
                std::cerr << "--seed requires --benchmark\n";
                return false;
            }
            const char* v = next_arg("--seed");
            if (!v || !parse_uint64(v, opts.benchmark.seed)) return false;
            opts.benchmark.seed_specified = true;
        } else if (a == "-h" || a == "--help") {
            print_usage(std::cout, argv[0]);
            std::exit(0);
        } else if (!a.empty() && a[0] == '-') {
            std::cerr << "Unknown argument: " << a << "\n";
            return false;
        } else {
            if (!opts.match.input_path.empty()) {
                std::cerr << "Unexpected extra argument: " << a << "\n";
                return false;
            }
            opts.match.input_path = std::string(a);
        }
    }

    if (opts.mode == Mode::Match && opts.match.input_path.empty()) {
        std::cerr << "Matching mode requires an input JSON file.\n";
        return false;
    }
    if (opts.match.validate && opts.match.input_path.empty()) {
        opts.mode = Mode::Benchmark;
        opts.benchmark.validate = true;
        opts.match.validate = false;
    }
    if (opts.mode == Mode::Benchmark && !opts.match.input_path.empty()) {
        std::cerr << "Benchmark mode does not take an input JSON file.\n";
        return false;
    }
    return true;
}

double sum_pair_costs(const std::vector<mwpm::Point>& points,
                      const std::vector<mwpm::Pair>& pairs) {
    double s = 0.0;
    for (const auto& [i, j] : pairs) s += mwpm::distance(points[i], points[j]);
    return s;
}

bool matching_is_perfect(std::size_t n, const std::vector<mwpm::Pair>& pairs) {
    if (pairs.size() != n / 2) return false;
    std::vector<int> count(n, 0);
    for (const auto& [i, j] : pairs) {
        if (i >= n || j >= n || i == j) return false;
        ++count[i];
        ++count[j];
    }
    for (int c : count) {
        if (c != 1) return false;
    }
    return true;
}

void dump_failure(std::ostream& os,
                  const std::vector<mwpm::Point>& points,
                  const mwpm::MatchingResult& dp,
                  const mwpm::MatchingResult& paper_smawk,
                  const mwpm::MatchingResult& paper_scan,
                  const mwpm::MatchingResult* brute) {
    os << std::setprecision(17);
    os << "FAILURE: matchings disagree.\n";
    os << "Points (" << points.size() << ", CCW):\n";
    for (std::size_t i = 0; i < points.size(); ++i) {
        os << "  [" << i << "] (" << points[i].x << ", " << points[i].y << ")\n";
    }
    auto dump = [&](const char* name, const mwpm::MatchingResult& r) {
        os << name << " cost=" << r.cost << " pairs:";
        for (const auto& [i, j] : r.pairs) os << " (" << i << "," << j << ")";
        os << "\n";
    };
    dump("DP        ", dp);
    dump("PaperSMAWK", paper_smawk);
    dump("PaperScan ", paper_scan);
    if (brute) dump("Brute     ", *brute);
}

void write_output_json_stream(std::ostream& os,
                              const mwpm::MatchingResult& result,
                              const mwpm::BuiltGeometry& geometry) {
    os << std::setprecision(17);
    os << "{\n  \"cost\": " << result.cost << ",\n  \"segments\": [\n";
    for (std::size_t k = 0; k < result.pairs.size(); ++k) {
        const auto& [i, j] = result.pairs[k];
        const auto& a = geometry.refs[i];
        const auto& b = geometry.refs[j];
        os << "    [[" << a.edge << ", " << a.index << "], [" << b.edge << ", "
           << b.index << "]]";
        if (k + 1 < result.pairs.size()) os << ',';
        os << '\n';
    }
    os << "  ]\n}\n";
}

bool validate_all_algorithms(const std::vector<mwpm::Point>& points) {
    const std::size_t n = points.size();
    const mwpm::MatchingResult dp = mwpm::dp_matching(points);
    const mwpm::MatchingResult paper_smawk =
        mwpm::marcotte_suri_matching(points, /*simple_scan=*/false);
    const mwpm::MatchingResult paper_scan =
        mwpm::marcotte_suri_matching(points, /*simple_scan=*/true);

    bool ran_brute = n <= static_cast<std::size_t>(kBruteForceCap);
    mwpm::MatchingResult brute;
    if (ran_brute) {
        brute = mwpm::brute_force_matching(points);
    }

    const double dp_recomputed = sum_pair_costs(points, dp.pairs);
    const double smawk_recomputed = sum_pair_costs(points, paper_smawk.pairs);
    const double scan_recomputed = sum_pair_costs(points, paper_scan.pairs);

    bool ok = matching_is_perfect(n, dp.pairs) &&
              matching_is_perfect(n, paper_smawk.pairs) &&
              matching_is_perfect(n, paper_scan.pairs) &&
              std::abs(dp.cost - dp_recomputed) < kCostTolerance &&
              std::abs(paper_smawk.cost - smawk_recomputed) < kCostTolerance &&
              std::abs(paper_scan.cost - scan_recomputed) < kCostTolerance &&
              std::abs(dp.cost - paper_smawk.cost) < kCostTolerance &&
              std::abs(dp.cost - paper_scan.cost) < kCostTolerance;

    if (ran_brute) {
        ok = ok && matching_is_perfect(n, brute.pairs) &&
             std::abs(brute.cost - sum_pair_costs(points, brute.pairs)) < kCostTolerance &&
             std::abs(brute.cost - dp.cost) < kCostTolerance;
    }

    if (!ok) {
        dump_failure(std::cerr, points, dp, paper_smawk, paper_scan,
                     ran_brute ? &brute : nullptr);
    }
    return ok;
}

int run_matching(const MatchOptions& opts) {
    const mwpm::PolygonInput input = mwpm::read_input_json(opts.input_path);
    const mwpm::BuiltGeometry geometry = mwpm::build_geometry(input);

    if (opts.validate && !validate_all_algorithms(geometry.points)) {
        return 1;
    }

    const mwpm::MatchingResult result =
        mwpm::marcotte_suri_matching(geometry.points, opts.simple_scan);

    if (opts.output_path.empty()) {
        write_output_json_stream(std::cout, result, geometry);
    } else {
        mwpm::write_output_json(opts.output_path, result, geometry);
    }

    if (opts.write_svg) {
        const std::string svg_path = opts.input_path + ".svg";
        mwpm::write_solution_svg(svg_path, input, geometry, result);
        std::cerr << "Wrote " << svg_path << '\n';
    }

    if (opts.validate) {
        std::cerr << "VALIDATION OK: all applicable algorithms agreed within tolerance "
                  << kCostTolerance << ".\n";
    }

    return 0;
}

int run_benchmark(const BenchmarkOptions& opts) {
    std::uint64_t seed = opts.seed;
    if (!opts.seed_specified) {
        std::random_device rd;
        seed = (static_cast<std::uint64_t>(rd()) << 32) ^ rd();
    }
    std::mt19937_64 rng(seed);

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Mode      : " << (opts.validate ? "validation" : "benchmark") << "\n";
    std::cout << "Trials    : " << opts.trials << "\n";
    if (opts.exact_points > 0) {
        std::cout << "Points/Tri: exactly " << opts.exact_points << "\n";
    } else {
        std::cout << "MaxPoints : " << opts.max_points << "\n";
    }
    std::cout << "Seed      : " << seed << "\n";
    if (!opts.validate) {
        std::cout << "Conquer   : ";
        if (opts.optimized) {
            std::cout << "optimized exact hybrid";
        } else {
            std::cout << (opts.simple_scan ? "simple-scan (O(N^2))" : "SMAWK (O(N))");
        }
        std::cout << "\n";
        std::cout << "Output    : " << (opts.cost_only ? "cost only" : "full matching")
                  << "\n";
    }

    using Clock = std::chrono::steady_clock;
    using std::chrono::duration_cast;
    using std::chrono::nanoseconds;

    std::int64_t total_ns_brute = 0;
    std::int64_t total_ns_dp = 0;
    std::int64_t total_ns_paper_smawk = 0;
    std::int64_t total_ns_paper_scan = 0;
    std::int64_t total_ns_optimized = 0;
    int trials_brute = 0;
    int trials_dp = 0;
    int trials_paper_smawk = 0;
    int trials_paper_scan = 0;
    int trials_optimized = 0;

    const bool run_smawk = opts.validate || opts.optimized || !opts.simple_scan;
    const bool run_scan = opts.validate || opts.simple_scan;

    for (int t = 0; t < opts.trials; ++t) {
        const auto tri = (opts.exact_points > 0)
                             ? mwpm::generate_random_triangle_exact(rng, opts.exact_points)
                             : mwpm::generate_random_triangle(rng, opts.max_points);
        const auto points = mwpm::generate_points_on_triangle(rng, tri);
        const std::size_t n = points.size();
        if (n < 2 || n % 2 != 0) {
            continue;
        }

        const bool run_dp =
            opts.validate || (!opts.skip_dp && static_cast<int>(n) <= kDpCap);
        mwpm::MatchingResult dp;
        auto t0 = Clock::now();
        auto t1 = t0;
        if (run_dp) {
            t0 = Clock::now();
            dp = mwpm::dp_matching(points);
            t1 = Clock::now();
            total_ns_dp += duration_cast<nanoseconds>(t1 - t0).count();
            ++trials_dp;
        }

        mwpm::MatchingResult paper_smawk;
        double paper_smawk_cost = 0.0;
        if (run_smawk) {
            t0 = Clock::now();
            if (opts.validate) {
                paper_smawk = mwpm::marcotte_suri_matching(points, /*simple_scan=*/false);
                paper_smawk_cost = paper_smawk.cost;
            } else if (opts.optimized && opts.cost_only) {
                paper_smawk_cost = mwpm::optimized_cost(points);
            } else if (opts.optimized) {
                paper_smawk = mwpm::optimized_matching(points);
                paper_smawk_cost = paper_smawk.cost;
            } else if (opts.cost_only) {
                paper_smawk_cost = mwpm::marcotte_suri_cost(points, /*simple_scan=*/false);
            } else {
                paper_smawk = mwpm::marcotte_suri_matching(points, /*simple_scan=*/false);
                paper_smawk_cost = paper_smawk.cost;
            }
            t1 = Clock::now();
            total_ns_paper_smawk += duration_cast<nanoseconds>(t1 - t0).count();
            ++trials_paper_smawk;
        }
        mwpm::MatchingResult paper_scan;
        double paper_scan_cost = 0.0;
        if (run_scan) {
            t0 = Clock::now();
            if (opts.validate) {
                paper_scan = mwpm::marcotte_suri_matching(points, /*simple_scan=*/true);
                paper_scan_cost = paper_scan.cost;
            } else if (opts.cost_only) {
                paper_scan_cost = mwpm::marcotte_suri_cost(points, /*simple_scan=*/true);
            } else {
                paper_scan = mwpm::marcotte_suri_matching(points, /*simple_scan=*/true);
                paper_scan_cost = paper_scan.cost;
            }
            t1 = Clock::now();
            total_ns_paper_scan += duration_cast<nanoseconds>(t1 - t0).count();
            ++trials_paper_scan;
        }

        mwpm::MatchingResult optimized;
        if (opts.validate) {
            t0 = Clock::now();
            optimized = mwpm::optimized_matching(points);
            t1 = Clock::now();
            total_ns_optimized += duration_cast<nanoseconds>(t1 - t0).count();
            ++trials_optimized;
        }

        bool ran_brute = false;
        mwpm::MatchingResult brute;
        if (!opts.skip_brute && static_cast<int>(n) <= kBruteForceCap) {
            t0 = Clock::now();
            brute = mwpm::brute_force_matching(points);
            t1 = Clock::now();
            total_ns_brute += duration_cast<nanoseconds>(t1 - t0).count();
            ++trials_brute;
            ran_brute = true;
        }

        if (opts.validate) {
            const double dp_recomputed = sum_pair_costs(points, dp.pairs);
            const double smawk_recomputed = sum_pair_costs(points, paper_smawk.pairs);
            const double scan_recomputed = sum_pair_costs(points, paper_scan.pairs);
            const double optimized_recomputed = sum_pair_costs(points, optimized.pairs);

            bool ok = matching_is_perfect(n, dp.pairs) &&
                      matching_is_perfect(n, paper_smawk.pairs) &&
                      matching_is_perfect(n, paper_scan.pairs) &&
                      matching_is_perfect(n, optimized.pairs) &&
                      std::abs(dp.cost - dp_recomputed) < kCostTolerance &&
                      std::abs(paper_smawk.cost - smawk_recomputed) < kCostTolerance &&
                      std::abs(paper_scan.cost - scan_recomputed) < kCostTolerance &&
                      std::abs(optimized.cost - optimized_recomputed) < kCostTolerance &&
                      std::abs(dp.cost - paper_smawk.cost) < kCostTolerance &&
                      std::abs(dp.cost - paper_scan.cost) < kCostTolerance &&
                      std::abs(dp.cost - optimized.cost) < kCostTolerance;

            if (ran_brute) {
                ok = ok && matching_is_perfect(n, brute.pairs) &&
                     std::abs(brute.cost - sum_pair_costs(points, brute.pairs)) <
                         kCostTolerance &&
                     std::abs(brute.cost - dp.cost) < kCostTolerance;
            }

            if (!ok) {
                dump_failure(std::cerr, points, dp, paper_smawk, paper_scan,
                             ran_brute ? &brute : nullptr);
                return 1;
            }
        }

        if ((t + 1) % 100 == 0 || t + 1 == opts.trials) {
            const double sample_cost = run_dp     ? dp.cost
                                       : run_smawk ? paper_smawk_cost
                                                   : paper_scan_cost;
            std::cout << "  trial " << (t + 1) << "/" << opts.trials
                      << " n=" << n << " cost=" << sample_cost << "\n";
        }
    }

    auto fmt_ms = [](std::int64_t ns) {
        return static_cast<double>(ns) / 1e6;
    };

    std::cout << "\n--- Aggregated timings ---\n";
    std::cout << "Brute force (n<=" << kBruteForceCap << "): "
              << trials_brute << " trials, total "
              << fmt_ms(total_ns_brute) << " ms\n";
    std::cout << "DP                       : " << trials_dp << " trials, total "
              << fmt_ms(total_ns_dp) << " ms\n";
    if (run_smawk) {
        std::cout << (opts.optimized && !opts.validate
                          ? "Optimized exact         : "
                          : "Marcotte & Suri (SMAWK)  : ")
                  << trials_paper_smawk
                  << " trials, total " << fmt_ms(total_ns_paper_smawk) << " ms\n";
    }
    if (run_scan) {
        std::cout << "Marcotte & Suri (scan)   : " << trials_paper_scan
                  << " trials, total " << fmt_ms(total_ns_paper_scan) << " ms\n";
    }
    if (opts.validate) {
        std::cout << "Optimized exact          : " << trials_optimized
                  << " trials, total " << fmt_ms(total_ns_optimized) << " ms\n";
    }

    if (opts.validate) {
        std::cout << "\nVALIDATION OK: BF / DP / Paper (SMAWK) / Paper (scan) / Optimized "
                     "all agreed within tolerance " << kCostTolerance << ".\n";
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    CliOptions opts;
    if (!parse_cli(argc, argv, opts)) {
        print_usage(std::cerr, argv[0]);
        return 2;
    }

    try {
        if (opts.mode == Mode::Benchmark) {
            return run_benchmark(opts.benchmark);
        }
        return run_matching(opts.match);
    } catch (const mwpm::GeometryError& e) {
        std::cerr << "Geometry error: " << e.what() << '\n';
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }
}
