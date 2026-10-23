#pragma once
// TEMPLATE: copied by `cp -r sgemm/_template sgemm/NN_name`, never compiled from here.
// Replace NN, name and NAME, then include the copy in sgemm/registry.cuh.
//
// Rung NN · name. What changes from rung NN-1, in one sentence. The reasoning lives in
// README.md next to this file; comments here say what the code does, not why it's fast.

#include "rung.cuh"

__global__ void sgemm_NAME(int M, int N, int K, float alpha, const float* __restrict__ A,
                           const float* __restrict__ B, float beta, float* __restrict__ C) {
}

inline void launch_NAME(int M, int N, int K, float alpha, const float* A, const float* B,
                        float beta, float* C, cudaStream_t stream) {
    const dim3 block(/* threads per block */);
    const dim3 grid(/* tiles covering C */);
    sgemm_NAME<<<grid, block, 0, stream>>>(M, N, K, alpha, A, B, beta, C);
}

// Shapes this rung is correct for. If it only handles tile multiples, say so here and the
// driver skips other shapes instead of reporting a failure.
inline bool supports_NAME(int M, int N, int K) { return true; }

inline SgemmKernel rung_NN_NAME() {
    return {NN, "NN_name", launch_NAME, supports_NAME, reinterpret_cast<const void*>(sgemm_NAME),
            /* threads per block */ 0};
}
