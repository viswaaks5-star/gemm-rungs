#pragma once
// Everything the driver prints. Lines that start with "CSV," are for tools/ladder_table.py;
// the rest is for people.

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "00_cublas/kernel.cuh"
#include "benchmark.cuh"
#include "cuda_utils.cuh"
#include "problem.cuh"
#include "timing.cuh"

namespace ladder {

struct Peak {
    int sm_count = 0;
    int lanes = 0;   // fp32 FMA lanes per SM; 0 when unknown
};

inline int fp32_lanes_per_sm(int major, int minor) {
    if (major == 8 && minor == 0) return 64;   // GA100
    if (major == 7) return 64;                 // Volta, Turing
    if (major >= 8) return 128;                // GA10x, Ada, Hopper, Blackwell
    return 0;
}

inline void print_context(const gpu::DeviceInfo& dev, const Peak& peak, const TimingOptions& t,
                          int graph, double skip_slow_ms) {
    int version = 0;
    CUBLAS_CHECK(cublasGetVersion(cublas_handle(), &version));
    cublasMath_t mode{};
    CUBLAS_CHECK(cublasGetMathMode(cublas_handle(), &mode));
    if (peak.lanes) {
        const double flop_per_clk = 2.0 * peak.lanes * peak.sm_count;
        std::printf("fp32     : %d FMA lanes/SM x %d SMs: peak %.0f FLOP/clk = %.3f TFLOP/s per GHz of SM clock\n",
                    peak.lanes, peak.sm_count, flop_per_clk, flop_per_clk / 1000.0);
    } else {
        std::printf("fp32     : FMA lanes per SM unknown for sm_%d%d; %%peak not reported\n",
                    dev.major, dev.minor);
    }
    std::printf("cuBLAS   : version %d, math mode %d (0 = CUBLAS_DEFAULT_MATH)\n", version,
                static_cast<int>(mode));
    const std::string how = graph ? "CUDA-graph replays of " + std::to_string(graph) + " launches"
                                  : std::string("back-to-back launches");
    std::printf("timing   : %s, reps of ~%.0f ms, %s\n", how.c_str(), t.rep_ms,
                t.allow_power_cap ? "reps with SwPowerCap ACCEPTED (diagnostic), SM spread tolerance 10%"
                                  : "reps with SwPowerCap rejected");
    if (skip_slow_ms > 0)
        std::printf("guard    : a rung whose launch is estimated above %.0f ms is skipped at that shape "
                    "(--skip-slow; Windows' GPU watchdog kills kernels near 2 s)\n", skip_slow_ms);
    std::printf("data     : timed and numerics on uniform [-1, 1); exact test on integers in [-2, 2]; "
                "timed with alpha = 1, beta = 0\n");
}

inline void print_resident(const std::vector<SgemmKernel>& kernels, const gpu::DeviceInfo& dev) {
    for (const SgemmKernel& k : kernels) {
        if (!k.func) continue;
        cudaFuncAttributes a{};
        CUDA_CHECK(cudaFuncGetAttributes(&a, k.func));
        int blocks = 0;
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, k.func, k.block_threads, 0));
        const int warps = blocks * k.block_threads / 32;
        std::printf("resident : %-16s %3d regs/thread, %6zu B smem, %zu B local%s; %4d threads/block -> "
                    "%d blocks/SM = %d warps/SM (%.0f%%), %d blocks on the GPU\n",
                    k.name, a.numRegs, a.sharedSizeBytes, a.localSizeBytes,
                    a.localSizeBytes ? " (SPILLS)" : "", k.block_threads, blocks, warps,
                    100.0 * blocks * k.block_threads / dev.max_threads_per_sm, blocks * dev.sm_count);
    }
}

inline void print_shape_model(const Shape& s, const gpu::DeviceInfo& dev) {
    const double bytes = 4.0 * (static_cast<double>(s.M) * s.K + static_cast<double>(s.K) * s.N +
                                static_cast<double>(s.M) * s.N);
    const double mib = bytes / (1024.0 * 1024.0);
    const double l2_mib = dev.l2_bytes / (1024.0 * 1024.0);
    std::printf("\n== %s: %.4g GFLOP; compulsory traffic %.2f MB (read A and B, write C), %.1f FLOP/B; "
                "working set %.1f MiB vs %.0f MiB L2: %s\n",
                shape_str(s).c_str(), flops(s) / 1e9, bytes / 1e6, flops(s) / bytes, mib, l2_mib,
                mib <= l2_mib ? "L2-resident across back-to-back launches" : "streams from DRAM");
}

inline void print_timing_header() {
    std::printf("\n%-16s %16s %10s %9s %6s %8s %8s %11s %5s %7s\n", "rung", "M x N x K", "ms",
                "GFLOP/s", "%peak", "%cuBLAS", "FLOP/clk", "SM MHz", "reps", "spread");
    std::printf("%s\n", std::string(105, '-').c_str());
}

inline void print_csv_header() {
    std::printf("CSV,rung,M,N,K,ms,min_ms,max_ms,gflops,pct_peak,pct_cublas,flop_per_clk_sm,"
                "sm_lo,sm_hi,mem_lo,mem_hi,reps_accepted,reps_total,graph\n");
}

// %peak divides by the highest SM clock seen in any accepted rep: the least flattering
// denominator. FLOP/clk is per SM and per SM clock, so it doesn't move with the clock; it is
// the number the model in docs/model.md predicts, and the fp32 peak on sm_86/89 is 256.
inline void print_timing_row(const char* name, const Shape& s, const gpu::Result& r, int per_launch,
                             const Peak& peak, double cublas_ms, int graph) {
    const double ms = r.median_ms / per_launch;
    const double gflops = flops(s) / (ms * 1.0e6);
    const double flop_per_clk = flops(s) / (ms * 1.0e-3 * r.sm_hi * 1.0e6 * peak.sm_count);
    const double pct_peak = peak.lanes ? 100.0 * flop_per_clk / (2.0 * peak.lanes) : NAN;
    const double pct_cublas = cublas_ms > 0.0 ? 100.0 * cublas_ms / ms : NAN;
    char sm[24], reps[16], pc[16];
    std::snprintf(sm, sizeof(sm), "%u-%u", r.sm_lo, r.sm_hi);
    std::snprintf(reps, sizeof(reps), "%d/%d", r.accepted, r.total);
    if (std::isnan(pct_cublas)) std::snprintf(pc, sizeof(pc), "n/a");
    else std::snprintf(pc, sizeof(pc), "%.1f%%", pct_cublas);
    std::printf("%-16s %16s %10.4f %9.1f %5.1f%% %8s %8.1f %11s %5s %6.2f%%\n", name,
                shape_str(s).c_str(), ms, gflops, pct_peak, pc, flop_per_clk, sm, reps,
                100.0 * (r.max_ms - r.min_ms) / r.median_ms);
    const std::string g = graph ? ", graph of " + std::to_string(graph) : std::string();
    std::printf("%-16s ramp %.0f ms to SM %u MHz%s, mem %u-%u MHz, %.0f clock samples/rep, reasons: %s%s\n",
                "", r.ramp.ms, r.ramp.sm_mhz, r.ramp.converged ? "" : " (NOT converged)", r.mem_lo,
                r.mem_hi, r.samples_per_rep, gpu::clock_reasons_str(r.reasons, r.reasons_known).c_str(),
                g.c_str());
    for (const std::string& note : r.notes) std::printf("%-16s %s\n", "", note.c_str());
    std::printf("CSV,%s,%d,%d,%d,%.6f,%.6f,%.6f,%.2f,%.2f,%.2f,%.3f,%u,%u,%u,%u,%d,%d,%d\n", name,
                s.M, s.N, s.K, ms, r.min_ms / per_launch, r.max_ms / per_launch, gflops, pct_peak,
                pct_cublas, flop_per_clk, r.sm_lo, r.sm_hi, r.mem_lo, r.mem_hi, r.accepted, r.total,
                graph);
}

}  // namespace ladder
