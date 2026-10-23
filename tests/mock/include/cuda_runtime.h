// Stand-in for the CUDA runtime header, for tests/mock only: the declarations the driver
// uses, with the CUDA 12/13 signatures. clang compiles the driver's host code against this
// (--cuda-host-only), and mock.cpp implements it on the CPU. Never used for a real build.
#pragma once
#include <cstddef>
#include <stdlib.h>

#ifdef __CUDA__
#define __host__ __attribute__((host))
#define __device__ __attribute__((device))
#define __global__ __attribute__((global))
#define __shared__ __attribute__((shared))
#define __constant__ __attribute__((constant))
#include <__clang_cuda_builtin_vars.h>
#else
#define __host__
#define __device__
#define __global__
#define __shared__
#define __constant__
#endif
#define __forceinline__ __inline__ __attribute__((always_inline))
#define __restrict__ __restrict

struct dim3 {
    unsigned x, y, z;
    __host__ __device__ constexpr dim3(unsigned a = 1, unsigned b = 1, unsigned c = 1)
        : x(a), y(b), z(c) {}
};

enum cudaError_t { cudaSuccess = 0, cudaErrorNotReady = 600 };
enum cudaMemcpyKind { cudaMemcpyHostToDevice = 1, cudaMemcpyDeviceToHost = 2 };
enum cudaStreamCaptureMode {
    cudaStreamCaptureModeGlobal = 0,
    cudaStreamCaptureModeThreadLocal = 1,
    cudaStreamCaptureModeRelaxed = 2
};
#define cudaStreamNonBlocking 0x01

typedef struct CUstream_st* cudaStream_t;
typedef struct CUevent_st* cudaEvent_t;
typedef struct CUgraph_st* cudaGraph_t;
typedef struct CUgraphExec_st* cudaGraphExec_t;

struct cudaDeviceProp {
    char name[256];
    int multiProcessorCount, major, minor, memoryBusWidth, l2CacheSize,
        maxThreadsPerMultiProcessor, pciBusID, pciDeviceID, pciDomainID;
};
struct cudaFuncAttributes {
    size_t sharedSizeBytes, constSizeBytes, localSizeBytes;
    int maxThreadsPerBlock, numRegs, ptxVersion, binaryVersion;
};

extern "C" {
const char* cudaGetErrorString(cudaError_t);
cudaError_t cudaGetLastError(void);
cudaError_t cudaDeviceSynchronize(void);
cudaError_t cudaGetDeviceProperties(cudaDeviceProp*, int);
cudaError_t cudaMalloc(void**, size_t);
cudaError_t cudaFree(void*);
cudaError_t cudaMemcpy(void*, const void*, size_t, cudaMemcpyKind);
cudaError_t cudaMemset(void*, int, size_t);
cudaError_t cudaEventCreate(cudaEvent_t*);
cudaError_t cudaEventRecord(cudaEvent_t, cudaStream_t = 0);
cudaError_t cudaEventSynchronize(cudaEvent_t);
cudaError_t cudaEventQuery(cudaEvent_t);
cudaError_t cudaEventElapsedTime(float*, cudaEvent_t, cudaEvent_t);
cudaError_t cudaEventDestroy(cudaEvent_t);
cudaError_t cudaStreamCreateWithFlags(cudaStream_t*, unsigned int);
cudaError_t cudaStreamDestroy(cudaStream_t);
cudaError_t cudaStreamBeginCapture(cudaStream_t, cudaStreamCaptureMode);
cudaError_t cudaStreamEndCapture(cudaStream_t, cudaGraph_t*);
cudaError_t cudaGraphInstantiate(cudaGraphExec_t*, cudaGraph_t, unsigned long long = 0);
cudaError_t cudaGraphDestroy(cudaGraph_t);
cudaError_t cudaGraphExecDestroy(cudaGraphExec_t);
cudaError_t cudaGraphLaunch(cudaGraphExec_t, cudaStream_t);
cudaError_t cudaFuncGetAttributes(cudaFuncAttributes*, const void*);
cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(int*, const void*, int, size_t);
cudaError_t cudaRuntimeGetVersion(int*);
cudaError_t cudaDriverGetVersion(int*);
// Kernel-launch entry points clang emits for <<<...>>> (both the current and legacy ABI).
unsigned __cudaPushCallConfiguration(dim3, dim3, size_t = 0, cudaStream_t = 0);
cudaError_t cudaConfigureCall(dim3, dim3, size_t = 0, cudaStream_t = 0);
}

template <class T>
inline cudaError_t cudaMalloc(T** p, size_t n) {
    return cudaMalloc(reinterpret_cast<void**>(p), n);
}
