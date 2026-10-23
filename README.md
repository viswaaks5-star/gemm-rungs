# gemm-rungs

[![ci](https://github.com/viswaaks5-star/gemm-rungs/actions/workflows/ci.yml/badge.svg)](https://github.com/viswaaks5-star/gemm-rungs/actions/workflows/ci.yml)

Matrix multiplication on a laptop GPU, from a naive kernel up to cuBLAS, one rung at a time. Every rung is predicted from first principles before it is measured, and checked element by element against cuBLAS before it is timed. Every published number links to the raw log it came from.

**Status:** harness ready; rung 1 in progress.

## Results

<!-- ladder:begin -->
No ladder run yet. `scripts/ladder.sh` writes this table from a committed log in [sgemm/runs/](sgemm/runs/).
<!-- ladder:end -->

## The ladder

FP32 SGEMM on CUDA cores. Rung numbers follow [Simon Boehm's worklog](https://siboehm.com/articles/22/CUDA-MMM), the ladder this one climbs.

| rung | idea | page |
|---|---|---|
| 00 | cuBLAS: the reference every rung is checked and measured against | [00_cublas](sgemm/00_cublas/) |
| 01 | naive: one thread per output | [01_naive](sgemm/01_naive/) |
| 02 | global memory coalescing | |
| 03 | shared-memory tiling | |
| 04 | 1D register tiling | |
| 05 | 2D register tiling | |
| 06–10 | vectorized access, bank conflicts, autotuning, warp tiling | |

Tensor-core ladders come later, beside `sgemm/`.

## Hardware

NVIDIA GeForce RTX 4060 Laptop GPU (AD107, sm_89): 24 SMs, 32 MiB L2, 128-bit GDDR6 · CUDA 13.1 · driver 610.74 · Ubuntu 26.04 under WSL2. FP32 peak is 6.144 TFLOP/s per GHz of SM clock; DRAM sustains about 247 GB/s. Every log begins with the environment it ran in, so this line can be checked against any of them.

## Reproduce

```bash
make                         # bin/sgemm, for the GPU in this machine
./bin/sgemm --check-only     # correctness of every rung, no timing
scripts/ladder.sh            # the full ladder: a log in sgemm/runs/, the table above regenerated
scripts/profile.sh 1 1024    # Nsight Compute on rung 1 at 1024³, into sgemm/01_naive/results/
scripts/sass.sh 1            # rung 1's SASS and instruction mix, into sgemm/01_naive/results/
```

Needs the CUDA toolkit (12.x or newer), an NVIDIA driver, and `python3` for the tools.

## How to read this repository

- **Two minutes:** this page.
- **Ten minutes per rung:** its page, always in the same order: the idea, the model, the prediction (committed before the first run), the result, the verdict, and what the next rung must fix.
- **Everything:** [the model](docs/model.md) every prediction starts from, [the methodology](docs/methodology.md) behind every number, and [the machine](docs/machine.md) they were measured on.

## Layout

```
sgemm/
  NN_name/       one rung: kernel.cuh, its page (README.md), its evidence (results/)
  registry.cuh   the ladder, one line per rung
  rung.cuh       the interface every rung implements
  driver/        checks every rung against cuBLAS, then times it
  runs/          full-ladder logs: the source of the table above
  _template/     copied to start a rung
harness/         gated timing, vendored from inference-kernels
docs/            the model, the methodology, the machine
scripts/         the standard runs: ladder, profile, SASS
tools/           log to table, SASS to instruction mix
tests/           the driver, tested on a CPU mock with no GPU
```

## Credits

The ladder follows Simon Boehm's [How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance](https://siboehm.com/articles/22/CUDA-MMM). The timing harness comes from [inference-kernels](https://github.com/viswaaks5-star/inference-kernels).

## License

[MIT](LICENSE)
