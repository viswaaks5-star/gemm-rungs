#pragma once
// Test-only ladder, found before sgemm/registry.cuh on the mock build's include path: the
// real cuBLAS rung plus one fake rung with an empty kernel. mock.cpp "runs" every kernel
// launch as a CPU GEMM, correct or deliberately broken (MOCK_KERNEL), so the driver's checks
// are tested without a GPU and without depending on the real rungs' device code.

#include <vector>

#include "00_cublas/kernel.cuh"

__global__ void sgemm_fake(int, int, int, float, const float*, const float*, float, float*) {}

inline void launch_fake(int M, int N, int K, float alpha, const float* A, const float* B,
                        float beta, float* C, cudaStream_t stream) {
    sgemm_fake<<<dim3(CEIL_DIV(M, 32), CEIL_DIV(N, 32)), dim3(32, 32), 0, stream>>>(M, N, K, alpha,
                                                                                   A, B, beta, C);
}

inline std::vector<SgemmKernel> sgemm_rungs() {
    return {
        rung_00_cublas(),
        {1, "01_fake", launch_fake, any_shape, reinterpret_cast<const void*>(sgemm_fake), 32 * 32},
    };
}
