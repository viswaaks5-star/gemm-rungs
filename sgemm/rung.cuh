#pragma once
// The interface every rung implements: one kernel, its launcher, and a descriptor the
// registry lists. Row-major fp32 throughout:
//
//     C[M x N] = alpha * A[M x K] * B[K x N] + beta * C
//
// The contract, enforced by driver/check.cuh before anything is timed:
//   - correct for every shape `supports` accepts, including ones that aren't tile multiples
//     (a rung that only handles multiples says so in `supports`, and the driver skips the rest);
//   - beta == 0 means C is not read (the driver pre-fills C with NaN, as BLAS allows);
//   - on integer-valued inputs, equal to cuBLAS in every element.
//
// A launcher only enqueues work on `stream` and returns.

#include <cuda_runtime.h>

#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))

using SgemmLaunch = void (*)(int M, int N, int K, float alpha, const float* A, const float* B,
                             float beta, float* C, cudaStream_t stream);
using SgemmSupports = bool (*)(int M, int N, int K);

struct SgemmKernel {
    int id;                 // rung number: the NN in sgemm/NN_name/
    const char* name;       // "NN_name", matching the directory
    SgemmLaunch launch;
    SgemmSupports supports;
    const void* func;       // the __global__ function, for register and occupancy reports;
                            // nullptr for cuBLAS
    int block_threads;      // threads per block it launches with
};

inline bool any_shape(int, int, int) { return true; }
