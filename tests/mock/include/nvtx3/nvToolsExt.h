// Stand-in for NVTX, for tests/mock only (see include/cuda_runtime.h).
#pragma once
extern "C" {
int nvtxRangePushA(const char*);
int nvtxRangePop(void);
}
