#!/usr/bin/env bash
# Nsight Compute on exactly one launch: one rung at one size. The readable report goes in the
# rung's results/ (committed); the binary .ncu-rep stays in profiles/ (ignored by git), for
# opening in the Nsight Compute GUI. Counts from here, times from the harness: never mix them.
#
#   scripts/profile.sh <rung id> <N> [extra ncu arguments]
#   scripts/profile.sh 1 1024
#   scripts/profile.sh 1 1024 --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum
#
# Check metric names first (ncu --query-metrics): a misspelled one prints n/a, no error.
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ $# -lt 2 ]]; then
    echo "usage: $0 <rung id> <N> [extra ncu arguments]" >&2
    exit 2
fi
id=$1 n=$2
shift 2
nn=$(printf '%02d' "$id")
dir=$(compgen -G "sgemm/${nn}_*" | head -n 1 || true)
if [[ -z "$dir" ]]; then
    echo "no rung directory sgemm/${nn}_*" >&2
    exit 2
fi

make --no-print-directory
mkdir -p profiles "$dir/results"
name="$(basename "$dir")_${n}"
custom=$#
if [[ $custom -eq 0 ]]; then set -- --set full; fi
ncu "$@" -f -o "profiles/$name" ./bin/sgemm --profile --kernels "$id" --sizes "$n"
ncu --import "profiles/$name.ncu-rep" --page details > "$dir/results/ncu_${n}.txt"
if [[ $custom -gt 0 ]] && grep -q "n/a" "$dir/results/ncu_${n}.txt"; then
    echo "warning: n/a in the report; a metric name may be misspelled" >&2
fi
echo "wrote $dir/results/ncu_${n}.txt (full report: profiles/$name.ncu-rep)"
