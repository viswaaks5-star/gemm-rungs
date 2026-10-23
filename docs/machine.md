# The machine

Facts about the test machine that every rung inherits, each one established by an experiment (M1, M2, …, in the [experiment format](https://github.com/viswaaks5-star/inference-kernels/blob/main/docs/experiment-format.md)). The bandwidth-side facts (memory clock, DRAM ceiling, buffer placement) were established in inference-kernels' [01 vector add](https://github.com/viswaaks5-star/inference-kernels/tree/main/kernels/01_vector_add) and are only summarized here.

## Facts

| fact | value | from |
|---|---|---|
| GPU | RTX 4060 Laptop GPU: AD107, sm_89, 24 SMs, 32 MiB L2 | every log header |
| DRAM bandwidth, sustained | about 247 GB/s (262.4 GB/s theoretical, in turbo mode) | inference-kernels 01 |
| SM clock under SGEMM load | | M1 |
| Launch overhead under WSL2 | | M2 |
| Kernels cuBLAS runs, per size | | M3 |
| cuBLAS's own error | | M4 |

## M1. What SM clock can this laptop sustain under SGEMM?

**Hypothesis.** SGEMM keeps every FP32 lane busy, so the laptop's power limit, not the boost table, sets the clock.

**Prediction.** `./bin/sgemm --kernels 0 --sizes 4096` rejects every rep for SwPowerCap, at an SM clock below the ~2730 MHz seen under vector add. If no rep sees SwPowerCap, the hypothesis is false and the natural clock is usable as is.

**Setup.** From an administrator PowerShell on Windows, run `nvidia-smi -lgc 2400,2400`. Then step the clock down about 150 MHz at a time until 7 of 7 reps pass at 4096³. Record that clock and the power mode; every ladder run uses it. `--allow-power-cap` shows the same run without the gate. `nvidia-smi -rgc` unlocks.

**Result.**

**Verdict.**

## M2. Does event timing measure the kernel at small sizes under WSL2?

**Hypothesis.** Under WSL2, a kernel launch costs the host several microseconds. So back-to-back launches of a 128³ SGEMM measure the launcher, not the kernel.

**Prediction.** At 128³ and 256³, the default timing reads higher than both `--graph 100` and nsys's median kernel duration. From 1024³ up, all three agree.

**Setup.**
```bash
./bin/sgemm --kernels 0 --sizes 128,256,512,1024 --reps 3
./bin/sgemm --kernels 0 --sizes 128,256,512,1024 --reps 3 --graph 100
mkdir -p profiles && nsys profile --stats=true -o profiles/m2 ./bin/sgemm --kernels 0 --sizes 128,256,512,1024 --reps 3
```
Compare against the medians in nsys's `cuda_gpu_kern_sum` table.

**Result.**

**Verdict.**

## M3. Which kernels does cuBLAS run, and are they fp32?

**Hypothesis.** cuBLAS picks a different SIMT kernel for each size, and all of them are fp32 (no TF32).

**Prediction.** No kernel name in M2's nsys output contains `tf32`. Small sizes use small tiles or split-K. On Simon Boehm's A6000 it ran a 32×32 sliced kernel at 128³, split-K plus a reduction kernel at 256³, and a 128×64 kernel from 1024³ up.

**Result.** The table goes in [00_cublas](../sgemm/00_cublas/README.md#what-it-runs). Every difference from the A6000 is a data point about how cuBLAS dispatches on sm_89.

**Verdict.**

## M4. How large is cuBLAS's own error?

**Hypothesis.** cuBLAS's error is plain fp32 rounding.

**Prediction.** About 1e-7 × sum|a·b| at every size (u = 2^-24 ≈ 6e-8). The driver warns above 2e-6, which is where TF32 would show up.

**Result.**

**Verdict.**

## Windows' GPU watchdog

WSL2 runs CUDA through Windows' display driver model (WDDM). Windows resets the GPU when a single kernel runs past its timeout, which is 2 s by default. The naive rung at 4096³ takes about 1.1 s at 2.73 GHz and longer at a locked lower clock. So the driver's `--skip-slow` (default 1000 ms) skips any shape it estimates would run longer than that, and clock experiments on slow rungs use 2048³. An error reading "the launch timed out and was terminated" is the watchdog, not the kernel.
