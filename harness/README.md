# harness

Gated timing and environment reporting, vendored from [inference-kernels](https://github.com/viswaaks5-star/inference-kernels) at commit `73614a7` (its `common/` directory). The rules it implements are written up in that repository's [methodology](https://github.com/viswaaks5-star/inference-kernels/blob/main/docs/methodology.md). [docs/methodology.md](../docs/methodology.md) covers what changes for compute-bound kernels.

This copy adds three fields to `BenchConfig`, each defaulting to the original behaviour:

| field | default | why |
|---|---|---|
| `reject_reasons` | the original set | `--allow-power-cap` drops SwPowerCap, as a diagnostic on a power-limited laptop |
| `ramp_batch` | 4 | a 5 µs launch needs about 2,000 launches between clock checks; a 1 s launch needs one |
| `exit_if_all_rejected` | `true` | a full ladder run reports a rejected configuration and moves on |

It also fixes one comment: the deviation flag is the measured 0.2% (3σ), not the 0.5% guess.

The two copies are kept in sync by hand. After changing either one, compare them:

```bash
diff -r harness ../inference-kernels/common
```
