# Methodology

How every number in this repository is taken, and what makes it trustworthy. The timing harness and its rules come from inference-kernels ([methodology](https://github.com/viswaaks5-star/inference-kernels/blob/main/docs/methodology.md)). This page covers what changes when the kernel is compute-bound, and the rules specific to GEMM.

## Correctness before timing

Every rung passes two tests at every shape before it is timed. A rung that fails is not run again in that process.

- **Exact test.** Inputs are integers in [-2, 2]. Every product and partial sum is then an integer far below 2^24, which fp32 holds exactly. So every correct summation order gives identical results, and the rung must equal cuBLAS in every element. cuBLAS itself is checked against int64 arithmetic on sampled entries. A second pass uses alpha = 2 and beta = 1. Before the beta = 0 pass, C is filled with NaN: BLAS says C is not read when beta = 0, and a rung that reads it fails.
- **Numerics test.** Inputs are uniform in [-1, 1). 512 sampled entries are computed in fp64. Error is reported relative to sum|a·b|, the scale of the classical bound gamma_K = K·u / (1 − K·u) that holds for any summation order. A rung must stay under that bound. The informative number is its ratio to cuBLAS's error: near 1× is normal, 100× is a precision bug.
- **Odd shapes.** 257×129×67, 31×1000×77 and 1000×31×513. M, N and K differ and none is a multiple of 32, so swapped indices and missing edge guards can't hide.

## What is timed

Only kernel launches. Allocation, copies and checks all sit outside every timed region.

- **Inputs are random reals**, uniform in [-1, 1), never the integer test data. On a power-limited GPU, low-entropy inputs draw less power and run at a higher clock.
- **Launches run back to back with warm caches.** Up to 1024³ the whole problem is L2-resident, and the log says so for each shape.
- **`--graph G` times CUDA-graph replays** of G launches instead, which removes the per-launch host cost at small sizes (machine.md, M2).

## How a rep is timed

This is the inference-kernels harness: the clock ramps to steady state, each rep is gated on NVML clock samples taken every 5 ms, and the result is the median of accepted reps. Two things change for SGEMM:

- **The knobs are sized to the launch.** One launch takes anywhere from 5 µs to 1 s across the ladder, so iterations, warm-up and ramp batches come from a pilot timing (`driver/timing.cuh`), not from constants.
- **The SM clock is the denominator.** A compute-bound kernel's peak scales with the SM clock, the way a bandwidth-bound kernel's scales with the memory clock. Under SGEMM the laptop's power limit lowers the clock (M1). So results are taken with the SM clock locked below the power-cap point, and reps that see SwPowerCap are rejected. `--allow-power-cap` keeps those reps, as a diagnostic only.

## What is reported

- **GFLOP/s:** 2MNK ÷ time.
- **% of peak:** against 2 × 128 lanes × 24 SMs × the highest SM clock in any accepted rep. That's the least flattering denominator.
- **FLOP/clk per SM:** independent of the clock, and exactly the quantity the model predicts ([model.md](model.md)). The fp32 peak is 256.
- **% of cuBLAS:** from the same process, in the same clock state.

Every log starts with the commit (marked `-dirty` if built from uncommitted code), the CUDA and driver versions, the OS, the clocks, and the registers and resident blocks for each rung. Lines starting with `CSV,` feed `tools/ladder_table.py`.

## cuBLAS, fairly

- **Created once**, outside timing.
- **fp32 math.** `CUBLAS_DEFAULT_MATH`, so no TF32.
- **Row-major** through the transpose identity, so no copies.
- **Warmed up** like every rung.

The kernel names in an nsys trace confirm what it ran (M3). The numerics test warns if cuBLAS's error ever looks like reduced precision.

## Profiling

Counts come from Nsight Compute and times from the harness, and the two are never mixed. ncu replays each kernel, flushes the caches and serializes execution, so its timings describe a different run.

- **`--profile`** launches each (rung, shape) once, with no checks, so a profiler sees only those launches.
- **`scripts/profile.sh`** exports the readable report into the rung's `results/` and keeps the binary `.ncu-rep` out of git.
- **`scripts/sass.sh`** records the compiled SASS and its instruction mix.

## Reproducibility

- **Every binary records its commit.** The Makefile bakes `git describe --dirty` into the binary and rebuilds whenever that string changes.
- **Ladder runs come from committed code.** `scripts/ladder.sh` refuses to run on uncommitted changes, saves the raw log in `sgemm/runs/`, and regenerates the README table from that log. Published numbers are never typed by hand.
- **Predictions come first.** Each rung's prediction is committed before its first run, and the git history shows the order.
- **Slow launches are skipped.** `--skip-slow` (default 1000 ms) skips a shape whose launch is estimated to take more than a second, scaling the last timed shape's time by FLOPs. Windows' GPU watchdog resets the GPU when a kernel runs for about 2 s (machine.md).

## Reporting checklist

Every published number states: GPU, CUDA and driver versions, OS, power mode, SM clock (and whether it was locked), shape, dtype, timing mode, reps accepted out of reps taken, and the baseline it is compared against.
