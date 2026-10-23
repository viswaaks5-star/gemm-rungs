// CPU stand-ins for the CUDA runtime, cuBLAS and NVML: enough to run the driver's host code
// end to end with no GPU. Device memory is host memory, a kernel launch runs a CPU GEMM, and
// "GPU time" advances in proportion to FLOPs, so timing, rejection and reporting all execute.
//
//   MOCK_KERNEL=good     a launched rung computes C correctly (default)
//   MOCK_KERNEL=bug      ... but drops the last k-term
//   MOCK_KERNEL=nanread  ... but reads C when beta == 0
//   MOCK_KERNEL=none     ... does nothing, like an unwritten rung
//   MOCK_POWERCAP=1      NVML reports SwPowerCap on every sample

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "cublas_v2.h"
#include "cuda_runtime.h"
#include "nvml.h"

namespace {
using Clock = std::chrono::steady_clock;
dim3 g_grid, g_block;
bool g_capturing = false;
double g_fake_us = 0.0;   // simulated GPU time, advanced by every launch

const char* kernel_mode() {
    const char* m = std::getenv("MOCK_KERNEL");
    return m ? m : "good";
}

// What a launched rung does, on the CPU. Row-major, BLAS beta semantics unless broken on purpose.
void emulate_rung(int M, int N, int K, float alpha, const float* A, const float* B, float beta,
                  float* C) {
    const char* mode = kernel_mode();
    if (!std::strcmp(mode, "none")) return;
    const int kmax = !std::strcmp(mode, "bug") ? K - 1 : K;
    const bool read_c_always = !std::strcmp(mode, "nanread");
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j) {
            float acc = 0.0f;
            for (int k = 0; k < kmax; ++k)
                acc = std::fmaf(A[static_cast<size_t>(i) * K + k], B[static_cast<size_t>(k) * N + j], acc);
            float& c = C[static_cast<size_t>(i) * N + j];
            c = (beta == 0.0f && !read_c_always) ? alpha * acc : alpha * acc + beta * c;
        }
}

unsigned char g_args[256];
size_t g_arg_offsets[16];
int g_nargs = 0;
}  // namespace

struct CUevent_st { Clock::time_point t; };
struct CUstream_st { int unused = 0; };
struct CUgraph_st { int unused = 0; };
struct CUgraphExec_st { int unused = 0; };

extern "C" {

// ------------------------------------------------------------------------ runtime
const char* cudaGetErrorString(cudaError_t) { return "mock error"; }
cudaError_t cudaGetLastError(void) { return cudaSuccess; }
cudaError_t cudaDeviceSynchronize(void) { return cudaSuccess; }
cudaError_t cudaGetDeviceProperties(cudaDeviceProp* p, int) {
    std::memset(p, 0, sizeof(*p));
    std::strcpy(p->name, "MOCK GPU (CPU stand-in, sm_89 shape)");
    p->multiProcessorCount = 24;
    p->major = 8;
    p->minor = 9;
    p->memoryBusWidth = 128;
    p->l2CacheSize = 32 << 20;
    p->maxThreadsPerMultiProcessor = 1536;
    return cudaSuccess;
}
cudaError_t cudaMalloc(void** p, size_t n) {
    *p = std::malloc(n ? n : 1);
    return cudaSuccess;
}
cudaError_t cudaFree(void* p) {
    std::free(p);
    return cudaSuccess;
}
cudaError_t cudaMemcpy(void* d, const void* s, size_t n, cudaMemcpyKind) {
    std::memcpy(d, s, n);
    return cudaSuccess;
}
cudaError_t cudaMemset(void* d, int v, size_t n) {
    std::memset(d, v, n);
    return cudaSuccess;
}
cudaError_t cudaEventCreate(cudaEvent_t* e) {
    *e = new CUevent_st;
    return cudaSuccess;
}
cudaError_t cudaEventRecord(cudaEvent_t e, cudaStream_t) {
    e->t = Clock::now() + std::chrono::nanoseconds(static_cast<long long>(g_fake_us * 1000.0));
    return cudaSuccess;
}
cudaError_t cudaEventSynchronize(cudaEvent_t) { return cudaSuccess; }
cudaError_t cudaEventQuery(cudaEvent_t) { return cudaSuccess; }
cudaError_t cudaEventElapsedTime(float* ms, cudaEvent_t a, cudaEvent_t b) {
    *ms = std::chrono::duration<float, std::milli>(b->t - a->t).count();
    return cudaSuccess;
}
cudaError_t cudaEventDestroy(cudaEvent_t e) {
    delete e;
    return cudaSuccess;
}
cudaError_t cudaStreamCreateWithFlags(cudaStream_t* s, unsigned) {
    *s = new CUstream_st;
    return cudaSuccess;
}
cudaError_t cudaStreamDestroy(cudaStream_t s) {
    delete s;
    return cudaSuccess;
}
cudaError_t cudaStreamBeginCapture(cudaStream_t, cudaStreamCaptureMode) {
    g_capturing = true;
    return cudaSuccess;
}
cudaError_t cudaStreamEndCapture(cudaStream_t, cudaGraph_t* g) {
    g_capturing = false;
    *g = new CUgraph_st;
    return cudaSuccess;
}
cudaError_t cudaGraphInstantiate(cudaGraphExec_t* e, cudaGraph_t, unsigned long long) {
    *e = new CUgraphExec_st;
    return cudaSuccess;
}
cudaError_t cudaGraphDestroy(cudaGraph_t g) {
    delete g;
    return cudaSuccess;
}
cudaError_t cudaGraphExecDestroy(cudaGraphExec_t e) {
    delete e;
    return cudaSuccess;
}
cudaError_t cudaGraphLaunch(cudaGraphExec_t, cudaStream_t) {
    g_fake_us += 50.0;
    return cudaSuccess;
}
cudaError_t cudaFuncGetAttributes(cudaFuncAttributes* a, const void*) {
    std::memset(a, 0, sizeof(*a));
    a->numRegs = 30;
    return cudaSuccess;
}
cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(int* n, const void*, int, size_t) {
    *n = 1;
    return cudaSuccess;
}
cudaError_t cudaRuntimeGetVersion(int* v) {
    *v = 13010;
    return cudaSuccess;
}
cudaError_t cudaDriverGetVersion(int* v) {
    *v = 13030;
    return cudaSuccess;
}

// ---------------------------------------------------- kernel launch (both clang ABIs)
cudaError_t cudaLaunchKernel(const void*, dim3 grid, dim3 block, void** args, size_t, cudaStream_t) {
    if (g_capturing) return cudaSuccess;
    const int M = *static_cast<int*>(args[0]);
    const int N = *static_cast<int*>(args[1]);
    const int K = *static_cast<int*>(args[2]);
    if (static_cast<int>(grid.x * block.x) < M || static_cast<int>(grid.y * block.y) < N) {
        std::fprintf(stderr, "mock: launch grid does not cover C\n");
        std::exit(3);
    }
    emulate_rung(M, N, K, *static_cast<float*>(args[3]), *static_cast<const float**>(args[4]),
                 *static_cast<const float**>(args[5]), *static_cast<float*>(args[6]),
                 *static_cast<float**>(args[7]));
    g_fake_us += 2.0 * M * N * K / 1e5;   // as if 100 GFLOP/s
    return cudaSuccess;
}
unsigned __cudaPushCallConfiguration(dim3 g, dim3 b, size_t, cudaStream_t) {
    g_grid = g;
    g_block = b;
    return 0;
}
cudaError_t __cudaPopCallConfiguration(dim3* g, dim3* b, size_t* shared, void* stream) {
    *g = g_grid;
    *b = g_block;
    *shared = 0;
    *static_cast<void**>(stream) = nullptr;
    return cudaSuccess;
}
cudaError_t cudaConfigureCall(dim3 g, dim3 b, size_t, cudaStream_t) {
    g_grid = g;
    g_block = b;
    g_nargs = 0;
    return cudaSuccess;
}
cudaError_t cudaSetupArgument(const void* arg, size_t size, size_t offset) {
    std::memcpy(g_args + offset, arg, size);
    g_arg_offsets[g_nargs++] = offset;
    return cudaSuccess;
}
cudaError_t cudaLaunch(const void* func) {
    void* args[16];
    for (int i = 0; i < g_nargs; ++i) args[i] = g_args + g_arg_offsets[i];
    g_nargs = 0;
    return cudaLaunchKernel(func, g_grid, g_block, args, 0, nullptr);
}
void** __cudaRegisterFatBinary(void*) {
    static void* handle = nullptr;
    return &handle;
}
void __cudaRegisterFatBinaryEnd(void**) {}
void __cudaUnregisterFatBinary(void**) {}
void __cudaRegisterFunction(void**, const void*, char*, const char*, int, void*, void*, void*,
                            void*, int*) {}

// --------------------------------------------------------------------------- cuBLAS
// Column-major NN, accumulating even and odd k separately so its rounding differs from the
// rung's: the numerics test then has two different fp32 orders to compare, as on a real GPU.
cublasStatus_t cublasCreate(cublasHandle_t* h) {
    *h = reinterpret_cast<cublasHandle_t>(0x1);
    return CUBLAS_STATUS_SUCCESS;
}
cublasStatus_t cublasDestroy(cublasHandle_t) { return CUBLAS_STATUS_SUCCESS; }
cublasStatus_t cublasSetStream(cublasHandle_t, cudaStream_t) { return CUBLAS_STATUS_SUCCESS; }
cublasStatus_t cublasSetMathMode(cublasHandle_t, cublasMath_t) { return CUBLAS_STATUS_SUCCESS; }
cublasStatus_t cublasGetMathMode(cublasHandle_t, cublasMath_t* m) {
    *m = CUBLAS_DEFAULT_MATH;
    return CUBLAS_STATUS_SUCCESS;
}
cublasStatus_t cublasGetVersion(cublasHandle_t, int* v) {
    *v = 130100;
    return CUBLAS_STATUS_SUCCESS;
}
cublasStatus_t cublasSgemm(cublasHandle_t, cublasOperation_t ta, cublasOperation_t tb, int m, int n,
                           int k, const float* alpha, const float* A, int lda, const float* B,
                           int ldb, const float* beta, float* C, int ldc) {
    if (ta != CUBLAS_OP_N || tb != CUBLAS_OP_N) std::exit(4);
    if (g_capturing) return CUBLAS_STATUS_SUCCESS;
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < m; ++i) {
            float even = 0.0f, odd = 0.0f;
            for (int l = 0; l < k; ++l) {
                const float t = A[i + static_cast<size_t>(l) * lda] * B[l + static_cast<size_t>(j) * ldb];
                if (l & 1) odd += t;
                else even += t;
            }
            float& c = C[i + static_cast<size_t>(j) * ldc];
            c = (*beta == 0.0f) ? *alpha * (even + odd) : *alpha * (even + odd) + *beta * c;
        }
    g_fake_us += 2.0 * m * n * k / 2e5;   // as if 200 GFLOP/s
    return CUBLAS_STATUS_SUCCESS;
}

// ----------------------------------------------------------------------------- NVML
const char* nvmlErrorString(nvmlReturn_t) { return "mock"; }
nvmlReturn_t nvmlInit_v2(void) { return NVML_SUCCESS; }
nvmlReturn_t nvmlShutdown(void) { return NVML_SUCCESS; }
nvmlReturn_t nvmlDeviceGetCount_v2(unsigned* c) {
    *c = 1;
    return NVML_SUCCESS;
}
nvmlReturn_t nvmlDeviceGetHandleByIndex_v2(unsigned, nvmlDevice_t* d) {
    *d = reinterpret_cast<nvmlDevice_t>(0x1);
    return NVML_SUCCESS;
}
nvmlReturn_t nvmlDeviceGetPciInfo_v3(nvmlDevice_t, nvmlPciInfo_t* p) {
    p->domain = p->bus = p->device = 0;
    return NVML_SUCCESS;
}
nvmlReturn_t nvmlDeviceGetClockInfo(nvmlDevice_t, nvmlClockType_t t, unsigned* v) {
    *v = t == NVML_CLOCK_MEM ? 8201 : 2400;
    return NVML_SUCCESS;
}
nvmlReturn_t nvmlDeviceGetMaxClockInfo(nvmlDevice_t, nvmlClockType_t t, unsigned* v) {
    *v = t == NVML_CLOCK_MEM ? 8001 : 3105;
    return NVML_SUCCESS;
}
nvmlReturn_t nvmlDeviceGetCurrentClocksEventReasons(nvmlDevice_t, unsigned long long* r) {
    *r = std::getenv("MOCK_POWERCAP") ? 0x4ull : 0ull;
    return NVML_SUCCESS;
}
nvmlReturn_t nvmlSystemGetDriverVersion(char* b, unsigned n) {
    std::snprintf(b, n, "mock");
    return NVML_SUCCESS;
}

// ----------------------------------------------------------------------------- NVTX
int nvtxRangePushA(const char*) { return 0; }
int nvtxRangePop(void) { return 0; }

}  // extern "C"
