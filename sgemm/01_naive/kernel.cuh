#pragma once
// Rung 1 · naive. One thread computes one element of C: a full dot product over K, then
// one store. The reasoning, prediction and results are in README.md next to this file.
//
// Use exactly the worklog's mapping, even though it's the slow one. Rung 2 is defined
// against it, so the comparison only means something if this stays as written:
//
//     row = blockIdx.x * 32 + threadIdx.x        // threadIdx.x walks DOWN the rows of C
//     col = blockIdx.y * 32 + threadIdx.y
//
// blockDim = (32, 32) and gridDim = (CEIL_DIV(M, 32), CEIL_DIV(N, 32)), as the launcher has.
// Guard row < M && col < N. Accumulate in a register. When beta == 0, store alpha * acc
// without reading C (C may hold NaN); otherwise store alpha * acc + beta * C[row][col].

#include "rung.cuh"

__global__ void sgemm_naive(int M, int N, int K, float alpha, const float* __restrict__ A,
                            const float* __restrict__ B, float beta, float* __restrict__ C) {
    // TODO(you). Until this is written, the exact test reports FAIL and the rung is never timed.
}

inline void launch_naive(int M, int N, int K, float alpha, const float* A, const float* B,
                         float beta, float* C, cudaStream_t stream) {
    const dim3 block(32, 32);
    const dim3 grid(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
    sgemm_naive<<<grid, block, 0, stream>>>(M, N, K, alpha, A, B, beta, C);
}

inline SgemmKernel rung_01_naive() {
    return {1, "01_naive", launch_naive, any_shape, reinterpret_cast<const void*>(sgemm_naive),
            32 * 32};
}
