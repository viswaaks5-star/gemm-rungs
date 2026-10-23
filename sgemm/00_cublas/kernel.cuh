#pragma once
// Rung 0 · cuBLAS: the reference every rung is checked and measured against.
// How it's called, and why that's a fair baseline, is in README.md next to this file.

#include <cstdio>
#include <cstdlib>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "rung.cuh"

#define CUBLAS_CHECK(call)                                                          \
    do {                                                                            \
        const cublasStatus_t st_ = (call);                                          \
        if (st_ != CUBLAS_STATUS_SUCCESS) {                                         \
            std::fprintf(stderr, "cuBLAS error at %s:%d\n  call:   %s\n  status: %d\n", \
                         __FILE__, __LINE__, #call, static_cast<int>(st_));          \
            std::exit(EXIT_FAILURE);                                                \
        }                                                                           \
    } while (0)

inline cublasHandle_t& cublas_handle() {
    static cublasHandle_t h = nullptr;
    return h;
}

// The stream last given to cublasSetStream, so steady-state launches skip the call.
inline cudaStream_t& cublas_current_stream() {
    static cudaStream_t s = nullptr;
    return s;
}

// Created once, outside every timed region. CUBLAS_DEFAULT_MATH keeps SGEMM in fp32: TF32
// needs an explicit opt-in. The kernel names in an nsys trace are the check that it stayed
// there, and the numerics test warns if cuBLAS's error ever looks like reduced precision.
inline void cublas_init() {
    CUBLAS_CHECK(cublasCreate(&cublas_handle()));
    CUBLAS_CHECK(cublasSetMathMode(cublas_handle(), CUBLAS_DEFAULT_MATH));
}

inline void cublas_shutdown() {
    CUBLAS_CHECK(cublasDestroy(cublas_handle()));
    cublas_handle() = nullptr;
}

inline void set_cublas_stream(cudaStream_t s) {
    if (s != cublas_current_stream()) {
        CUBLAS_CHECK(cublasSetStream(cublas_handle(), s));
        cublas_current_stream() = s;
    }
}

// cuBLAS is column-major. A row-major X is, byte for byte, the column-major X^T, so the
// row-major C = A B is the column-major C^T = B^T A^T: pass B first, with m = N and n = M.
inline void launch_cublas(int M, int N, int K, float alpha, const float* A, const float* B,
                          float beta, float* C, cudaStream_t stream) {
    set_cublas_stream(stream);
    CUBLAS_CHECK(cublasSgemm(cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, N,
                             A, K, &beta, C, N));
}

inline SgemmKernel rung_00_cublas() {
    return {0, "00_cublas", launch_cublas, any_shape, nullptr, 0};
}
