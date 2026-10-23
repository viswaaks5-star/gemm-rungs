// sgemm · the ladder's driver: checks every rung against cuBLAS, then times it.
//
//   sgemm                            check, then time, every rung at 128..4096 (square)
//   sgemm --kernels 0,1              rung ids from sgemm/registry.cuh; 0 is cuBLAS
//   sgemm --sizes 1024,4096          square sizes
//   sgemm --shape 4096x11008x4096    any M x N x K (repeatable; replaces --sizes)
//   sgemm --check-only               correctness only
//   sgemm --graph 100                time CUDA-graph replays of 100 launches each
//   sgemm --profile                  one launch per (rung, shape), no checks: for ncu and nsys
//   sgemm --skip-slow 1000           skip a shape whose launch is estimated above 1000 ms (default;
//                                    0 disables; estimates need a timed smaller shape first)
//   sgemm --allow-power-cap          keep reps that saw SwPowerCap (diagnostic only)
//   sgemm --reps 7 --rep-ms 300 --poll-ms 5
//
// Per shape: exact test, numerics test, timing. A rung that fails a test is not run again.
// The full method is in docs/methodology.md; scripts/ladder.sh is the standard way to run this.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iterator>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>

#include "benchmark.cuh"
#include "check.cuh"
#include "cuda_utils.cuh"
#include "problem.cuh"
#include "registry.cuh"
#include "report.cuh"
#include "timing.cuh"

namespace ladder {
namespace {

constexpr int kDefaultSizes[] = {128, 256, 512, 1024, 2048, 4096};
// Correctness-only shapes. M, N and K all differ and none is a multiple of 32, so a swapped
// index or a missing edge guard can't hide behind symmetry or a whole number of tiles.
constexpr int kOddShapes[][3] = {{257, 129, 67}, {31, 1000, 77}, {1000, 31, 513}};

struct Options {
    std::vector<Shape> shapes;
    std::vector<int> kernel_ids;   // empty: every registered rung
    bool check_only = false;
    bool profile = false;
    int graph = 0;                 // launches per CUDA graph; 0 = plain back-to-back launches
    double skip_slow_ms = 1000.0;
    TimingOptions timing;
};

void usage() {
    std::fprintf(stderr,
                 "usage: sgemm [--kernels 0,1,..] [--sizes 128,256,.. | --shape MxNxK ...]\n"
                 "             [--check-only | --profile] [--graph G] [--skip-slow MS]\n"
                 "             [--allow-power-cap] [--reps N] [--rep-ms MS] [--poll-ms MS]\n");
}

std::vector<int> parse_int_list(const std::string& s) {
    std::vector<int> out;
    size_t pos = 0;
    while (pos <= s.size()) {
        size_t end = s.find(',', pos);
        if (end == std::string::npos) end = s.size();
        const std::string tok = s.substr(pos, end - pos);
        char* rest = nullptr;
        const long v = std::strtol(tok.c_str(), &rest, 10);
        if (tok.empty() || *rest != '\0' || v < 0) {
            std::fprintf(stderr, "bad number '%s' in list '%s'\n", tok.c_str(), s.c_str());
            std::exit(EXIT_FAILURE);
        }
        out.push_back(static_cast<int>(v));
        pos = end + 1;
    }
    return out;
}

Shape parse_shape(const std::string& s) {
    Shape sh{};
    char tail = 0;
    if (std::sscanf(s.c_str(), "%dx%dx%d%c", &sh.M, &sh.N, &sh.K, &tail) != 3 || sh.M <= 0 ||
        sh.N <= 0 || sh.K <= 0) {
        std::fprintf(stderr, "bad shape '%s', expected MxNxK\n", s.c_str());
        std::exit(EXIT_FAILURE);
    }
    return sh;
}

Options parse_args(int argc, char** argv) {
    Options o;
    std::vector<int> sizes;
    for (int k = 1; k < argc; ++k) {
        const std::string arg = argv[k];
        const bool value = k + 1 < argc;
        if (arg == "--kernels" && value) o.kernel_ids = parse_int_list(argv[++k]);
        else if (arg == "--sizes" && value) sizes = parse_int_list(argv[++k]);
        else if (arg == "--shape" && value) o.shapes.push_back(parse_shape(argv[++k]));
        else if (arg == "--graph" && value) o.graph = std::atoi(argv[++k]);
        else if (arg == "--skip-slow" && value) o.skip_slow_ms = std::atof(argv[++k]);
        else if (arg == "--reps" && value) o.timing.reps = std::atoi(argv[++k]);
        else if (arg == "--rep-ms" && value) o.timing.rep_ms = std::atof(argv[++k]);
        else if (arg == "--poll-ms" && value) o.timing.poll_ms = std::atoi(argv[++k]);
        else if (arg == "--check-only") o.check_only = true;
        else if (arg == "--profile") o.profile = true;
        else if (arg == "--allow-power-cap") o.timing.allow_power_cap = true;
        else {
            usage();
            std::exit(EXIT_FAILURE);
        }
    }
    if (!o.shapes.empty() && !sizes.empty()) {
        std::fprintf(stderr, "--sizes and --shape are mutually exclusive\n");
        std::exit(EXIT_FAILURE);
    }
    if (o.shapes.empty()) {
        if (sizes.empty()) sizes.assign(std::begin(kDefaultSizes), std::end(kDefaultSizes));
        for (int n : sizes) {
            if (n <= 0) {
                std::fprintf(stderr, "sizes must be positive\n");
                std::exit(EXIT_FAILURE);
            }
            o.shapes.push_back({n, n, n});
        }
    }
    if (o.graph < 0 || o.timing.reps < 1 || !(o.timing.rep_ms > 0.0) || o.skip_slow_ms < 0.0) {
        usage();
        std::exit(EXIT_FAILURE);
    }
    return o;
}

// One launch per (rung, shape), in a fixed order and with no tests, so a profiler sees only
// these launches: `ncu --set full ./bin/sgemm --profile --kernels 1 --sizes 1024` profiles
// exactly one kernel (scripts/profile.sh wraps this). Each launch also sits in an NVTX range
// "<rung> MxNxK" inside a range "profile", which labels it in an nsys timeline. cuBLAS may
// launch two kernels for one call (split-K plus a reduction); nsys shows which.
void profile(const std::vector<SgemmKernel>& kernels, const std::vector<Shape>& shapes) {
    nvtxRangePushA("profile");
    int n = 0;
    for (const Shape& s : shapes) {
        Problem P(s);
        fill_uniform(P.hA, 21);
        fill_uniform(P.hB, 22);
        P.upload_inputs();
        for (const SgemmKernel& k : kernels) {
            if (!k.supports(s.M, s.N, s.K)) continue;
            const std::string name = std::string(k.name) + " " + shape_str(s);
            nvtxRangePushA(name.c_str());
            P.launch(k, 1.0f, 0.0f, nullptr);
            CUDA_CHECK_LAST();
            nvtxRangePop();
            std::printf("profile launch %d: %s\n", ++n, name.c_str());
        }
    }
    nvtxRangePop();
}

int run(int argc, char** argv) {
    const Options o = parse_args(argc, argv);

    const gpu::Nvml nvml;
    const gpu::DeviceInfo dev = gpu::query_device();
    gpu::print_device(dev);
    gpu::print_environment(nvml.dev);

    cublas_init();
    const Peak peak{dev.sm_count, fp32_lanes_per_sm(dev.major, dev.minor)};
    print_context(dev, peak, o.timing, o.graph, o.check_only ? 0.0 : o.skip_slow_ms);

    std::vector<SgemmKernel> kernels;
    for (const SgemmKernel& k : sgemm_rungs()) {
        if (o.kernel_ids.empty() ||
            std::find(o.kernel_ids.begin(), o.kernel_ids.end(), k.id) != o.kernel_ids.end())
            kernels.push_back(k);
    }
    if (kernels.empty()) {
        std::fprintf(stderr, "no registered rung matches --kernels\n");
        return EXIT_FAILURE;
    }
    print_resident(kernels, dev);

    if (o.profile) {
        profile(kernels, o.shapes);
        cublas_shutdown();
        return EXIT_SUCCESS;
    }

    const size_t n = kernels.size();
    int failures = 0, skipped = 0;
    std::vector<bool> broken(n, false);
    // The most recent timed shape per rung: --skip-slow scales its time by FLOPs.
    std::vector<double> last_ms(n, 0.0), last_flops(n, 0.0);
    auto fail = [&](size_t i, const char* what, const Shape& s, const std::string& why) {
        std::printf("  %-16s %-16s %s FAIL: %s\n", kernels[i].name, shape_str(s).c_str(), what,
                    why.c_str());
        broken[i] = true;
        ++failures;
    };

    std::printf("\nexact test on odd shapes: integers in [-2, 2], two passes, every element vs cuBLAS\n");
    for (const auto& os : kOddShapes) {
        Problem P(Shape{os[0], os[1], os[2]});
        fill_small_ints(P.hA, 1);
        fill_small_ints(P.hB, 2);
        fill_small_ints(P.hC0, 3);
        P.upload_inputs();
        for (size_t i = 0; i < n; ++i) {
            const SgemmKernel& k = kernels[i];
            if (k.id == 0 || broken[i]) continue;
            if (!k.supports(P.s.M, P.s.N, P.s.K)) {
                std::printf("  %-16s %-16s skipped: shape not supported\n", k.name, shape_str(P.s).c_str());
                continue;
            }
            const std::string e = exact_test(k, P);
            if (e.empty()) std::printf("  %-16s %-16s exact pass\n", k.name, shape_str(P.s).c_str());
            else fail(i, "exact", P.s, e);
        }
    }

    if (!o.check_only) {
        std::printf("\n");
        print_csv_header();
    }

    for (const Shape& s : o.shapes) {
        print_shape_model(s, dev);
        Problem P(s);

        // 1. Exact test, for every rung that can run at this shape.
        fill_small_ints(P.hA, 11);
        fill_small_ints(P.hB, 12);
        fill_small_ints(P.hC0, 13);
        P.upload_inputs();
        std::vector<bool> ok(n, false);
        for (size_t i = 0; i < n; ++i) {
            const SgemmKernel& k = kernels[i];
            if (broken[i]) {
                std::printf("  %-16s skipped: failed an earlier test\n", k.name);
                continue;
            }
            if (!k.supports(s.M, s.N, s.K)) {
                std::printf("  %-16s skipped: shape not supported\n", k.name);
                continue;
            }
            if (!o.check_only && o.skip_slow_ms > 0.0 && last_ms[i] > 0.0) {
                const double estimate = last_ms[i] * flops(s) / last_flops[i];
                if (estimate > o.skip_slow_ms) {
                    std::printf("  %-16s skipped: estimated %.0f ms per launch (%.2f ms at the last "
                                "shape, scaled by FLOPs) is above --skip-slow %.0f\n",
                                k.name, estimate, last_ms[i], o.skip_slow_ms);
                    ++skipped;
                    continue;
                }
            }
            const std::string e = exact_test(k, P);
            if (!e.empty()) fail(i, "exact", s, e);
            else ok[i] = true;
        }

        // 2. Numerics test, on the data that will also be timed.
        fill_uniform(P.hA, 21);
        fill_uniform(P.hB, 22);
        P.upload_inputs();
        const Reference ref = make_reference(P);
        const double bound = gamma_k(s.K);
        // cuBLAS's own error, always, so every row has its baseline even when cuBLAS isn't
        // selected. fp32 SGEMM lands near 1e-7 here at any K; TF32 would land near 1e-5.
        const double cublas_err = numerics_error(rung_00_cublas(), P, ref);
        if (!(cublas_err <= 2e-6))
            std::printf("  WARNING: cuBLAS error %.2e x sum|ab| is far above fp32's ~1e-7; reduced "
                        "precision (TF32?) would make every %%cuBLAS figure unfair\n", cublas_err);
        for (size_t i = 0; i < n; ++i) {
            if (!ok[i]) continue;
            const double err = kernels[i].id == 0 ? cublas_err : numerics_error(kernels[i], P, ref);
            if (!(err <= bound)) {
                char why[160];
                std::snprintf(why, sizeof(why), "max error %.3g x sum|ab| exceeds gamma_K = %.3g",
                              err, bound);
                fail(i, "numerics", s, why);
                ok[i] = false;
                continue;
            }
            std::printf("  %-16s exact pass; numerics: max error %.2e x sum|ab| (cuBLAS %.2e, "
                        "ratio %.2f; bound %.2e)\n",
                        kernels[i].name, err, cublas_err, err / cublas_err, bound);
        }

        // 3. Timing.
        if (o.check_only) continue;
        print_timing_header();
        double cublas_ms = -1.0;
        for (size_t i = 0; i < n; ++i) {
            if (!ok[i]) continue;
            const SgemmKernel& k = kernels[i];
            const std::string range = std::string(k.name) + " " + shape_str(s);
            nvtxRangePushA(range.c_str());
            gpu::Result r;
            int per_launch = 1;
            if (o.graph > 0) {
                const Graph graph(k, P, o.graph);
                auto launch = [&] { CUDA_CHECK(cudaGraphLaunch(graph.exec, nullptr)); };
                r = bench(launch, o.timing, nvml.dev);
                per_launch = o.graph;
            } else {
                auto launch = [&] { P.launch(k, 1.0f, 0.0f, nullptr); };
                r = bench(launch, o.timing, nvml.dev);
            }
            nvtxRangePop();
            if (r.accepted == 0) {
                std::printf("%-16s %16s no accepted reps (reasons above)\n", k.name, shape_str(s).c_str());
                continue;
            }
            last_ms[i] = r.median_ms / per_launch;
            last_flops[i] = flops(s);
            if (k.id == 0) cublas_ms = last_ms[i];
            print_timing_row(k.name, s, r, per_launch, peak, cublas_ms, o.graph);
        }
    }

    std::printf("\n");
    if (failures == 0)
        std::printf(o.check_only ? "all configurations verified\n"
                                 : "all configurations verified before timing\n");
    else
        std::printf("%d check(s) FAILED; a failing rung is not run after its failure (FAIL lines above)\n",
                    failures);
    if (skipped > 0)
        std::printf("%d configuration(s) skipped by --skip-slow\n", skipped);
    cublas_shutdown();
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}

}  // namespace
}  // namespace ladder

int main(int argc, char** argv) {
    // Piped into tee, stdout is fully buffered and stderr is not, so the two would interleave
    // out of order in the saved log. Line buffering keeps the log in the order things happened.
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    return ladder::run(argc, argv);
}
