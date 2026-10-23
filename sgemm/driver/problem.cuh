#pragma once
// Shapes, test data and device buffers.

#include <cstdint>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "cuda_utils.cuh"
#include "rung.cuh"

namespace ladder {

struct Shape {
    int M, N, K;
};

inline std::string shape_str(const Shape& s) {
    return std::to_string(s.M) + "x" + std::to_string(s.N) + "x" + std::to_string(s.K);
}

inline double flops(const Shape& s) { return 2.0 * s.M * s.N * static_cast<double>(s.K); }

// splitmix64 with fixed seeds: every run sees the same matrices.
struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed) {}
    uint64_t next() {
        uint64_t z = (s += 0x9E3779B97F4A7C15ull);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    }
    // Uniform in [-1, 1) from 24 random bits: exact in fp32.
    float uniform() { return static_cast<float>(next() >> 40) * (2.0f / 16777216.0f) - 1.0f; }
    int below(int n) { return static_cast<int>(next() % static_cast<uint64_t>(n)); }
};

// What gets timed. On a power-limited GPU the data matters: low-entropy matrices toggle
// fewer bits, draw less power and run at a higher clock, so timing uses random reals.
inline void fill_uniform(std::vector<float>& v, uint64_t seed) {
    Rng r(seed);
    for (float& x : v) x = r.uniform();
}

// What the exact test uses: -2, -1, 0, 1, 2.
inline void fill_small_ints(std::vector<float>& v, uint64_t seed) {
    Rng r(seed);
    for (float& x : v) x = static_cast<float>(r.below(5) - 2);
}

struct DeviceBuffer {
    float* p = nullptr;
    size_t n = 0;
    explicit DeviceBuffer(size_t count) : n(count) { CUDA_CHECK(cudaMalloc(&p, bytes())); }
    ~DeviceBuffer() { cudaFree(p); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    size_t bytes() const { return n * sizeof(float); }
    void upload(const std::vector<float>& h) {
        CUDA_CHECK(cudaMemcpy(p, h.data(), bytes(), cudaMemcpyHostToDevice));
    }
    void download(std::vector<float>& h) const {
        h.resize(n);
        CUDA_CHECK(cudaMemcpy(h.data(), p, bytes(), cudaMemcpyDeviceToHost));
    }
    void fill_nan() { CUDA_CHECK(cudaMemset(p, 0xFF, bytes())); }   // 0xFFFFFFFF is a NaN
};

// One shape's matrices, on the host and the device. C is the rung's output; Cref is cuBLAS's.
struct Problem {
    Shape s;
    std::vector<float> hA, hB, hC0, hRef, hOut;
    DeviceBuffer A, B, C, Cref;
    explicit Problem(const Shape& sh)
        : s(sh),
          hA(static_cast<size_t>(sh.M) * sh.K),
          hB(static_cast<size_t>(sh.K) * sh.N),
          hC0(static_cast<size_t>(sh.M) * sh.N),
          A(hA.size()),
          B(hB.size()),
          C(hC0.size()),
          Cref(hC0.size()) {}
    float a(int i, int k) const { return hA[static_cast<size_t>(i) * s.K + k]; }
    float b(int k, int j) const { return hB[static_cast<size_t>(k) * s.N + j]; }
    size_t idx(int i, int j) const { return static_cast<size_t>(i) * s.N + j; }
    void upload_inputs() {
        A.upload(hA);
        B.upload(hB);
    }
    void launch(const SgemmKernel& k, float alpha, float beta, cudaStream_t stream) {
        k.launch(s.M, s.N, s.K, alpha, A.p, B.p, beta, C.p, stream);
    }
};

}  // namespace ladder
