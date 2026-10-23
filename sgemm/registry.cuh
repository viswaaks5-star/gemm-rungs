#pragma once
// The ladder. Everything runs in this order: cuBLAS first, so every other row can report
// its percentage of cuBLAS.
//
// Adding a rung:
//   1. cp -r sgemm/_template sgemm/NN_name, and write the kernel in its kernel.cuh
//   2. include it below, and add rung_NN_name() to the list

#include <vector>

#include "00_cublas/kernel.cuh"
#include "01_naive/kernel.cuh"

inline std::vector<SgemmKernel> sgemm_rungs() {
    return {
        rung_00_cublas(),
        rung_01_naive(),
    };
}
