#!/usr/bin/env bash
# The SASS the compiler produced for one rung, and its instruction mix. The FFMA share of the
# hot loop is the issue-bound ceiling in docs/model.md. Writes sass.txt and sass_mix.txt into
# the rung's results/.
#
#   scripts/sass.sh <rung id>
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ $# -ne 1 ]]; then
    echo "usage: $0 <rung id>" >&2
    exit 2
fi
if [[ "$1" == 0 ]]; then
    echo "rung 0 is cuBLAS: its SASS lives in libcublas, not bin/sgemm." >&2
    echo "Find the kernel it runs with nsys (docs/machine.md, M3), then: cuobjdump -sass -fun <name> <libcublas>" >&2
    exit 2
fi
nn=$(printf '%02d' "$1")
dir=$(compgen -G "sgemm/${nn}_*" | head -n 1 || true)
if [[ -z "$dir" ]]; then
    echo "no rung directory sgemm/${nn}_*" >&2
    exit 2
fi
kernel=$(grep -oE '__global__ void [A-Za-z0-9_]+' "$dir/kernel.cuh" | head -n 1 | awk '{print $3}')
if [[ -z "$kernel" ]]; then
    echo "no __global__ function in $dir/kernel.cuh" >&2
    exit 2
fi

make --no-print-directory
mkdir -p "$dir/results"
cuobjdump -sass bin/sgemm > bin/sgemm.sass
python3 tools/sass_mix.py bin/sgemm.sass --match "$kernel" --out "$dir/results/sass.txt" \
    | tee "$dir/results/sass_mix.txt"
