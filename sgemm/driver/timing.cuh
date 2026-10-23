#pragma once
// Timing: the harness's gated reps, with knobs sized to each launch, and CUDA-graph replays.

#include <algorithm>
#include <cmath>

#include <cuda_runtime.h>

#include "00_cublas/kernel.cuh"
#include "benchmark.cuh"
#include "cuda_utils.cuh"
#include "problem.cuh"

namespace ladder {

struct TimingOptions {
    int reps = 7;
    double rep_ms = 300.0;         // target length of one timed rep
    int poll_ms = 5;
    bool allow_power_cap = false;  // diagnostic only: keep reps that saw SwPowerCap
};

inline int clamp_int(double v, int lo, int hi) {
    if (!(v > lo)) return lo;   // also catches NaN
    if (v > hi) return hi;
    return static_cast<int>(v);
}

// Mean time per launch over n back-to-back launches, by CUDA events.
template <typename L>
double time_launches_ms(L& launch, int n) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < n; ++i) launch();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return static_cast<double>(ms) / n;
}

// One launch can take 5 us (cuBLAS at 128^3) or a second (naive at 4096^3), and no fixed
// iteration count serves both. So the harness's knobs come from a pilot timing: reps of
// ~rep_ms; warm-up batches of >= 10 ms of GPU work between clock samples, so a tiny kernel
// doesn't leave the GPU idle (and the clock low) while the host synchronizes; and a ramp
// window spanning >= 200 ms of load that never needs more than 10 samples.
template <typename L>
gpu::Result bench(L& launch, const TimingOptions& o, nvmlDevice_t nv) {
    for (int i = 0; i < 2; ++i) launch();   // first use: lazy module load, cuBLAS heuristics
    CUDA_CHECK_LAST();
    double t1 = time_launches_ms(launch, 1);
    if (t1 < 1.0)
        t1 = time_launches_ms(launch, clamp_int(std::ceil(5.0 / std::max(t1, 1e-3)), 2, 5000));

    gpu::BenchConfig cfg;
    cfg.reps = o.reps;
    cfg.poll_ms = o.poll_ms;
    cfg.exit_if_all_rejected = false;
    if (o.allow_power_cap) {
        cfg.reject_reasons &= ~static_cast<unsigned long long>(nvmlClocksThrottleReasonSwPowerCap);
        cfg.sm_spread_tol = 0.10;   // a power-capped clock dithers between bins
    }
    cfg.iters = clamp_int(std::ceil(o.rep_ms / t1), 1, 1 << 20);
    cfg.rep_warmup = clamp_int(cfg.iters / 8.0, 1, 1 << 16);
    cfg.ramp_batch = clamp_int(std::ceil(10.0 / t1), 1, 1 << 20);
    const double step_ms = std::max(cfg.ramp_sample_ms, cfg.ramp_batch * t1);
    cfg.ramp_window = clamp_int(std::ceil(200.0 / step_ms), 3, 10);
    cfg.ramp_timeout_ms = std::max(5000.0, cfg.ramp_min_ms + 4.0 * cfg.ramp_window * step_ms);
    return gpu::benchmark(nv, launch, cfg);
}

// A CUDA graph of g launches of one rung. Replaying it costs the host one launch instead
// of g, which separates kernel time from launch overhead at small sizes.
struct Graph {
    cudaGraphExec_t exec = nullptr;
    Graph(const SgemmKernel& k, Problem& P, int g) {
        cudaStream_t s;
        CUDA_CHECK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
        set_cublas_stream(s);   // before capture, not during it
        cudaGraph_t graph;
        CUDA_CHECK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
        for (int i = 0; i < g; ++i) P.launch(k, 1.0f, 0.0f, s);
        CUDA_CHECK(cudaStreamEndCapture(s, &graph));
        CUDA_CHECK(cudaGraphInstantiate(&exec, graph, 0));
        CUDA_CHECK(cudaGraphDestroy(graph));
        set_cublas_stream(nullptr);
        CUDA_CHECK(cudaStreamDestroy(s));
    }
    ~Graph() {
        if (exec) cudaGraphExecDestroy(exec);
    }
    Graph(const Graph&) = delete;
    Graph& operator=(const Graph&) = delete;
};

}  // namespace ladder
