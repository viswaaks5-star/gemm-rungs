#pragma once
// The correctness contract. A rung that fails either test is never timed.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <utility>
#include <vector>

#include "00_cublas/kernel.cuh"
#include "cuda_utils.cuh"
#include "problem.cuh"

namespace ladder {

// Exact test. With integer inputs in [-2, 2], every product and every partial sum is an
// integer of magnitude at most 4K, which fp32 holds exactly while 4K < 2^24. So every correct
// summation order gives the same values, and the rung must equal cuBLAS in every element.
// cuBLAS itself is checked against int64 arithmetic on sampled entries.
//   pass 1: alpha = 1, beta = 0, C pre-filled with NaN   (BLAS: C is not read when beta = 0)
//   pass 2: alpha = 2, beta = 1, C = integers in [-2, 2] (alpha and beta paths, still exact)
// Returns an empty string on success, else what went wrong first.
inline std::string exact_pass(const SgemmKernel& k, Problem& P, float alpha, float beta) {
    const Shape s = P.s;
    for (DeviceBuffer* c : {&P.C, &P.Cref}) {
        if (beta == 0.0f) c->fill_nan();
        else c->upload(P.hC0);
    }
    launch_cublas(s.M, s.N, s.K, alpha, P.A.p, P.B.p, beta, P.Cref.p, nullptr);
    P.launch(k, alpha, beta, nullptr);
    CUDA_CHECK_LAST();
    P.Cref.download(P.hRef);
    P.C.download(P.hOut);

    char buf[256];
    Rng r(0xC0FFEEull + static_cast<uint64_t>(s.M) * 7919 + static_cast<uint64_t>(s.N));
    for (int t = 0; t < 256; ++t) {
        const int i = r.below(s.M);
        const int j = r.below(s.N);
        int64_t dot = 0;
        for (int kk = 0; kk < s.K; ++kk)
            dot += static_cast<int64_t>(P.a(i, kk)) * static_cast<int64_t>(P.b(kk, j));
        const double want = alpha * static_cast<double>(dot) +
                            (beta != 0.0f ? beta * static_cast<double>(P.hC0[P.idx(i, j)]) : 0.0);
        if (!(static_cast<double>(P.hRef[P.idx(i, j)]) == want)) {
            std::snprintf(buf, sizeof(buf), "cuBLAS itself is wrong at (%d, %d): %g, want %g", i,
                          j, P.hRef[P.idx(i, j)], want);
            return buf;
        }
    }

    size_t bad = 0, first = 0;
    for (size_t e = 0; e < P.hOut.size(); ++e) {
        if (!(P.hOut[e] == P.hRef[e])) {   // catches NaN; +0 and -0 compare equal, as they should
            if (bad == 0) first = e;
            ++bad;
        }
    }
    if (bad == 0) return "";
    std::snprintf(buf, sizeof(buf),
                  "alpha %g beta %g: %zu of %zu elements differ, first at (%zu, %zu): got %g, cuBLAS %g",
                  alpha, beta, bad, P.hOut.size(), first / static_cast<size_t>(s.N),
                  first % static_cast<size_t>(s.N), P.hOut[first], P.hRef[first]);
    return buf;
}

inline std::string exact_test(const SgemmKernel& k, Problem& P) {
    std::string e = exact_pass(k, P, 1.0f, 0.0f);
    if (e.empty()) e = exact_pass(k, P, 2.0f, 1.0f);
    return e;
}

// Numerics test, on uniform [-1, 1) inputs, with fp64 as truth (a product of two floats is
// exact in fp64). Errors are relative to sum_k |a_ik b_kj|, the scale of the classical bound
// that holds for every summation order:
//     |fl(c) - c| <= gamma_K * sum_k |a_ik b_kj|,   gamma_K = K u / (1 - K u),   u = 2^-24
// A correct fp32 rung can never exceed gamma_K, and in practice sits far below it. The
// informative number is the ratio to cuBLAS: near 1x is normal, 100x is a precision bug.
struct Reference {
    std::vector<std::pair<int, int>> at;
    std::vector<double> value, scale;
};

constexpr int kSamples = 512;

inline Reference make_reference(const Problem& P) {
    Reference ref;
    Rng r(0x5EEDull + static_cast<uint64_t>(P.s.K));
    for (int t = 0; t < kSamples; ++t) {
        const int i = r.below(P.s.M);
        const int j = r.below(P.s.N);
        double dot = 0.0, scale = 0.0;
        for (int kk = 0; kk < P.s.K; ++kk) {
            const double p = static_cast<double>(P.a(i, kk)) * static_cast<double>(P.b(kk, j));
            dot += p;
            scale += std::fabs(p);
        }
        ref.at.emplace_back(i, j);
        ref.value.push_back(dot);
        ref.scale.push_back(scale);
    }
    return ref;
}

inline double gamma_k(int K) {
    const double ku = K * std::ldexp(1.0, -24);
    return ku / (1.0 - ku);
}

// Launch once with alpha = 1, beta = 0 and return the largest scaled error over the samples.
inline double numerics_error(const SgemmKernel& k, Problem& P, const Reference& ref) {
    P.C.fill_nan();
    P.launch(k, 1.0f, 0.0f, nullptr);
    CUDA_CHECK_LAST();
    P.C.download(P.hOut);
    double worst = 0.0;
    for (size_t t = 0; t < ref.at.size(); ++t) {
        const float got = P.hOut[P.idx(ref.at[t].first, ref.at[t].second)];
        if (!std::isfinite(got)) return INFINITY;
        const double err =
            std::fabs(static_cast<double>(got) - ref.value[t]) / std::max(ref.scale[t], 1e-300);
        worst = std::max(worst, err);
    }
    return worst;
}

}  // namespace ladder
