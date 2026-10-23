# 01 · naive

> One thread per element of C, each computing a full dot product. The baseline every later rung is measured against.

## Idea

Thread (x, y) of the block computes one element of C: it walks k from 0 to K−1 accumulating A[row][k]·B[k][col], then stores the result. The mapping is the worklog's, kept deliberately: `row` comes from threadIdx.x, so threadIdx.x walks down the rows of C. Rung 2 is defined against this choice.

## Model

Written before measuring, from [docs/model.md](../../docs/model.md).

**What a warp touches.** threadIdx.x varies fastest, so a warp is 32 consecutive rows of a single column of C. Each k-step, the warp:
- loads `A[row][k]`: 32 rows that are K·4 bytes apart. That's 32 different cache lines, and when K is a multiple of 32 they all fall in the same bank. About 32 L1 wavefronts.
- loads `B[k][col]`: the same address for all 32 threads, a broadcast. 1 wavefront.
- does 32 FMAs.

**The shared resource is the L1.** There is one per SM, shared by every warp on it, serving about one wavefront per clock. However the warps interleave, each warp-step costs the L1 33 clocks and yields 32 FMAs. That's **about 0.97 FMA/clk/SM, 0.76% of peak**, or 1.94 FLOP/clk in the harness's units.

**Reaching the ceiling.** A block is 32 warps (1,024 threads), and the 1,536-thread limit fits one block per SM. 32 warps is plenty to keep the L1 busy every clock.

**Why nearly everything hits in L1.**
- All 32 warps of a block work on the same 32 rows of A, so the first warp to fetch a line pays for it and the others hit.
- A 32-byte sector holds 8 consecutive k-values, so each thread reuses it for 8 steps.
- The block's 32 columns of B sit in one cache line, shared the same way.

**Waves.** 24 blocks run at once, one per SM. At N = 128 there are only 16 blocks, so one partial wave leaves 8 SMs idle.

**On someone else's GPU.** Simon Boehm's A6000 has 84 SMs: 84 × 0.97 × 2 × ~1.8 GHz ≈ 293 GFLOP/s, against 309 measured. That's within about 5%, but his clock isn't reported, so the match could be luck.

**What would falsify this.**
- E1: locking the SM clock lower doesn't slow the kernel proportionally.
- E2: the counters show far more or fewer than 33 L1 wavefronts per warp-step, or a low L1 hit rate.

## Prediction

> Commit this section before the first run, and put the commit's link here.

| N | blocks | waves | predicted FLOP/clk/SM | predicted GFLOP/s at ___ MHz |
|---|---|---|---|---|
| 128 | | | | |
| 256 | | | | |
| 512 | | | 1.94 | |
| 1024 | | | 1.94 | |
| 2048 | | | 1.94 | |

4096³ takes about a second per launch, so `--skip-slow` should skip it (see [machine.md](../../docs/machine.md#windows-gpu-watchdog)).

## Result

## Experiments

### E1. Is rung 1 limited inside the SM or by memory?

**Hypothesis.** The L1 sets the speed, and the L1 runs on the SM clock.

**Prediction.** At 2048³, locking the SM clock at 1545 MHz instead of 2400 MHz makes each launch 2400/1545 ≈ 1.55× slower. If it barely slows down, the model is wrong and memory is the limit.

**Setup.** Lock the clock with `nvidia-smi -lgc` from an administrator PowerShell, then run `./bin/sgemm --kernels 1 --sizes 2048`. Use 2048³, not 4096³: at 1545 MHz one 4096³ launch gets close to Windows' 2-second watchdog.

**Result.**

**Verdict.**

### E2. Do the hardware counters agree with the wavefront count?

**Hypothesis.** The access pattern is exactly the one in the model, and the L1 serves it from cache.

**Prediction.** `scripts/profile.sh 1 1024` shows:
- about 16.5 sectors per global load request (32 for A, 1 for B), and 32 per store request;
- an L1 hit rate above 95%;
- DRAM traffic near zero;
- about 33 L1 wavefronts per warp per k-step.

**Result.**

**Verdict.**

### E3. Does the wave count explain the small sizes?

**Hypothesis.** Below 512³, a partial last wave is the only reason a rung falls short of its ceiling.

**Prediction.** The measured FLOP/clk at 128³ and 256³ matches the prediction table above.

**Result.**

**Verdict.**

## Verdict

## What rung 2 must fix
