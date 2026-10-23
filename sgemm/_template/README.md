# NN · name

> One sentence: what this rung changes, and what that buys.

**Diff from the previous rung:** `diff sgemm/PP_prev/kernel.cuh sgemm/NN_name/kernel.cuh`

## Idea

What changes, in plain words, and which bottleneck of the previous rung it attacks (that rung's "What the next rung must fix").

## Model

Written before measuring, from [docs/model.md](../../docs/model.md):
- **The shared resource** every warp competes for, and why it's the one that binds.
- **Its cost per unit of work**, and from that the ceiling, in FMA/clk/SM and FLOP/clk/SM.
- **Reaching the ceiling:** registers, shared memory and threads per block give blocks and warps per SM. Is that enough to keep the resource busy?
- **Waves:** blocks per wave, and the tail at each size.
- **What would falsify it.**

## Prediction

> Committed before the first run: (link to the commit)

| N | predicted FLOP/clk/SM | predicted GFLOP/s at ___ MHz |
|---|---|---|

## Result

From [`sgemm/runs/<log>`](../runs/): measured against predicted, at every size.

| N | predicted | measured | error |
|---|---|---|---|

## Experiments

### E1. A question the result raises

**Hypothesis.** One claim that could turn out to be wrong.

**Prediction.** What the numbers look like if it holds, and if it doesn't. Written before running.

**Setup.** What changed and what was held fixed.

**Result.** The numbers, with a link to the raw output in results/.

**Verdict.** Supported, falsified or inconclusive, judged against the prediction.

**What went wrong.** Leave this out when nothing did.

## Verdict

Did the model hold? Where does the explanation stop?

## What the next rung must fix

The new bottleneck, with the evidence that names it: stall reasons, counters, SASS.
