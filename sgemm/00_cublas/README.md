# 00 · cuBLAS

> The reference. Every rung is checked against it element by element, and its time is the 100% in every "% of cuBLAS".

## How it's called

[kernel.cuh](kernel.cuh) calls `cublasSgemm` with:

- **fp32 arithmetic.** `CUBLAS_DEFAULT_MATH`. TF32 needs an explicit opt-in and is never enabled. A TF32 baseline would trade precision for speed, and no fp32 rung could fairly be compared to it.
- **Row-major data, via the transpose identity.** cuBLAS is column-major, and a row-major matrix is, byte for byte, the column-major layout of its transpose. So row-major C = A·B is computed as column-major Cᵀ = Bᵀ·Aᵀ: B goes first, with m = N and n = M. Nothing is copied or transposed.
- **One handle**, created outside every timed region, and warmed up like every rung.

## What it runs

cuBLAS picks a different kernel for each shape, and that choice is part of what every rung competes with. From nsys ([machine.md, M3](../../docs/machine.md#m3-which-kernels-does-cublas-run-and-are-they-fp32)):

| N | kernel(s) | GPU time |
|---|---|---|
| 128 | | |
| 256 | | |
| 512 | | |
| 1024 | | |
| 2048 | | |
| 4096 | | |

For comparison, on Simon Boehm's A6000 it ran a 32×32 sliced kernel at 128³, split-K plus a reduction kernel at 256³, and a 128×64 kernel from 1024³ up.

## Numerics

The driver checks cuBLAS itself on every run, two ways:
- on integer inputs, exactly, against int64 arithmetic;
- on real inputs, against fp64, where the error should sit near 1e-7 × sum|a·b| ([machine.md, M4](../../docs/machine.md#m4-how-large-is-cublass-own-error)).
