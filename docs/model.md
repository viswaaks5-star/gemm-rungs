# The model

Every rung predicts its number from this page before it is measured. Figures are for the test machine; [machine.md](machine.md) holds the measured facts they rest on.

## 1. Work and data

C = A·B with A of size M×K and B of size K×N. One element of C is a dot product: K multiply-adds. A GPU does a multiply-add in one instruction, an FMA, counted as 2 FLOPs.

- **Work:** 2MNK FLOPs. Square N: 2N³, which is 137.4 GFLOP at N = 4096.
- **Compulsory traffic:** read A and B once, write C once (beta = 0): 4(MK + KN + MN) bytes. Square N: 12N², which is 201 MB at N = 4096.
- **Intensity at compulsory traffic:** N/6 FLOPs per byte for square N. Vector add sits at 1/12.

Each element of A is needed by every output in its row, which is N outputs, and each element of B by every output in its column. Loaded once, a number can be used N times. Capturing that reuse is the whole game. Each rung captures more of it, one memory level closer to the math units: DRAM, then L2, then L1 and shared memory, then registers.

## 2. The two limits

- **Math:** 24 SMs × 128 FP32 lanes × 2 FLOPs = 6,144 FLOPs per clock, or 6.144 TFLOP/s per GHz of SM clock. That's 16.8 TFLOP/s at the 2.73 GHz seen under vector add. SGEMM's sustained clock is lower (machine.md, M1).
- **DRAM:** about 247 GB/s sustained (262.4 GB/s theoretical).
- **Ridge point:** about 68 FLOPs per byte against 247 GB/s (64 against the theoretical bandwidth). Above it, math is the limit; below it, memory is.

| N | GFLOP | compulsory MB | FLOP/B | math floor, 2.73 GHz | DRAM floor, 247 GB/s | fits the 32 MiB L2? |
|---|---|---|---|---|---|---|
| 128 | 0.0042 | 0.20 | 21 | 0.25 µs | 0.80 µs | yes |
| 256 | 0.034 | 0.79 | 43 | 2.0 µs | 3.2 µs | yes |
| 512 | 0.27 | 3.1 | 85 | 16 µs | 13 µs | yes |
| 1024 | 2.15 | 12.6 | 171 | 128 µs | 51 µs | yes |
| 2048 | 17.2 | 50.3 | 341 | 1.02 ms | 0.20 ms | no |
| 4096 | 137.4 | 201 | 683 | 8.19 ms | 0.82 ms | no |

Read the table as three regimes, not one roofline:

- **N ≤ 256:** both floors sit below the cost of one kernel launch and one trip to memory. The kernel is latency-bound, and neither limit applies.
- **512 to 1024:** the whole problem stays in L2 across back-to-back launches. The L2's bandwidth matters, not DRAM's.
- **2048 to 4096:** compute-bound, as long as the kernel's actual DRAM traffic stays under about 10× compulsory. Tile size and block order decide that (section 5).

With no reuse at all, every output fetches its own row and column: 8N³ bytes, which is 550 GB and about 2.2 s at N = 4096. Caches recover part of that even for the naive rung, and the rungs recover the rest.

## 3. The SM, per clock (sm_89)

- **Warps.** Threads run in warps of 32 consecutive threads, with threadIdx.x varying fastest. One instruction issued to a warp executes for all 32.
- **Schedulers.** Each SM has 4 warp schedulers. Every clock, each one issues one instruction from a warp that is ready.
- **FP32.** 128 FMA lanes, which is one FFMA warp-instruction per scheduler per clock. **Peak needs every issue slot to be an FFMA.** So the ceiling is at most peak × (FFMAs ÷ all instructions issued), and every load, address computation, barrier and branch is paid out of the FMA budget. `scripts/sass.sh` reports that share for a rung's hot loop.
- **Shared memory.** 32 banks of 4 bytes, serving one wavefront (up to 128 bytes) per clock. A conflict-free 32-bit load, or a broadcast, is 1 wavefront; an n-way bank conflict is n.
- **L1.** One wavefront per clock. An access it can't serve in one wavefront (many cache lines, or bank conflicts) splits into several, processed one after another.
- **Limits per SM:** 48 warps (1,536 threads), 24 blocks, 100 KB of shared memory (1 KB reserved per block), 64K registers (at most 255 per thread). [Ada tuning guide](https://docs.nvidia.com/cuda/archive/13.3.0/ada-tuning-guide/index.html)

## 4. How to reason about a rung

1. **Find the shared resource** every warp on the SM competes for: the L1, shared memory, issue slots, or DRAM.
2. **Count what one unit of work costs it, and what that work produces** in FMAs. That ratio is the ceiling in FMA per clock per SM. It doesn't depend on how many warps there are, because they all share the one resource.
3. **Check that enough work is in flight** to keep the resource busy every clock: enough warps, or enough independent instructions per warp. Too little, and the rung lands below the ceiling. *The shared resource sets the ceiling; the number of warps decides whether you reach it.*
4. **Count waves.** Only blocks-per-SM × 24 blocks run at once, so a partial last wave leaves SMs idle.
5. **Convert to the harness's unit:** FLOP/clk per SM is 2 × FMA/clk/SM. It doesn't depend on the clock, so it compares directly with the measurement, and 256 is the fp32 peak.

## 5. Traffic depends on block order

A prediction for rung 5, to test there. At 4096³ with 128×128 tiles and 2 blocks per SM:

- A wave of 48 blocks covers 1.5 rows of tiles.
- Each wave reads all of B, which is 64 MiB, twice the L2, so nothing survives from one wave to the next.
- That's about 1.5 GB of DRAM traffic against 0.2 GB compulsory. At cuBLAS-like speed, it's about half the DRAM bandwidth.

Simon Boehm found that reordering blocks made no difference on his A6000, where the L2 hit rate was already about 80%. This model predicts it matters here, at the top of the ladder. The test is `dram__bytes_read.sum` at rung 5, with and without the reordering.
