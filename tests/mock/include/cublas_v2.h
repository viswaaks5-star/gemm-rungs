// Stand-in for cuBLAS, for tests/mock only (see include/cuda_runtime.h).
#pragma once
#include "cuda_runtime.h"

typedef struct cublasContext* cublasHandle_t;
enum cublasStatus_t { CUBLAS_STATUS_SUCCESS = 0 };
enum cublasOperation_t { CUBLAS_OP_N = 0, CUBLAS_OP_T = 1 };
enum cublasMath_t { CUBLAS_DEFAULT_MATH = 0, CUBLAS_PEDANTIC_MATH = 2, CUBLAS_TF32_TENSOR_OP_MATH = 3 };

extern "C" {
cublasStatus_t cublasCreate(cublasHandle_t*);
cublasStatus_t cublasDestroy(cublasHandle_t);
cublasStatus_t cublasSetStream(cublasHandle_t, cudaStream_t);
cublasStatus_t cublasSetMathMode(cublasHandle_t, cublasMath_t);
cublasStatus_t cublasGetMathMode(cublasHandle_t, cublasMath_t*);
cublasStatus_t cublasGetVersion(cublasHandle_t, int*);
cublasStatus_t cublasSgemm(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int,
                           const float*, const float*, int, const float*, int, const float*,
                           float*, int);
}
