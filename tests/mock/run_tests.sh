#!/usr/bin/env bash
# Tests the driver (sgemm/driver/) with no GPU and no CUDA toolkit. clang compiles the
# driver's host code against the stand-in headers in include/, mock.cpp implements them on the
# CPU, and each scenario below checks one promise the driver makes: a correct rung passes and
# is timed; a broken one is caught and never timed; every mode runs.
#
# You only need this when you change the driver. Rung code is not involved: the mock build
# uses tests/mock/registry.cuh, with one fake rung, in place of sgemm/registry.cuh.
#
#   bash tests/mock/run_tests.sh            (CXX=clang++-18 to pick a clang)

set -uo pipefail
cd "$(dirname "$0")/../.."
CXX=${CXX:-clang++}
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
M=tests/mock

echo "building the driver against the CPU mock with $CXX"
"$CXX" -x cuda -nocudainc -nocudalib --cuda-host-only -std=c++17 -O1 -Wall -Wextra \
    -Wno-unused-parameter -include $M/include/cuda_runtime.h -I$M/include -I$M -Iharness -Isgemm \
    -DGIT_SHA='"mock"' -c sgemm/driver/main.cu -o "$work/main.o" || exit 1
"$CXX" -std=c++17 -O1 -Wall -I$M/include -c $M/mock.cpp -o "$work/mock.o" || exit 1
"$CXX" "$work/main.o" "$work/mock.o" -o "$work/sgemm" || exit 1

bin="$work/sgemm"
log="$work/log"
pass=0
fail=0
fast=(--reps 3 --rep-ms 5)

run() { "$@" > "$log" 2>&1; echo $? > "$work/rc"; }
has() { grep -qE -- "$1" "$log"; }
lacks() { ! grep -qE -- "$1" "$log"; }
exit_is() { [[ "$(cat "$work/rc")" == "$1" ]]; }
expect() {
    local what=$1
    shift
    if "$@"; then
        pass=$((pass + 1))
        echo "  ok    $what"
    else
        fail=$((fail + 1))
        echo "  FAIL  $what"
        tail -n 30 "$log" | sed 's/^/        | /'
    fi
}

echo "a correct rung"
run env MOCK_KERNEL=good "$bin" --sizes 64,96 "${fast[@]}"
expect "exits 0" exit_is 0
expect "passes the odd shapes" has "01_fake +257x129x67 +exact pass"
expect "is timed, with cuBLAS first" has "^CSV,00_cublas,96,96,96,"
expect "is timed" has "^CSV,01_fake,96,96,96,"
expect "says everything was verified" has "all configurations verified before timing"

echo "a rung that drops the last k-term"
run env MOCK_KERNEL=bug "$bin" --sizes 64 "${fast[@]}"
expect "exits 1" exit_is 1
expect "fails the exact test" has "exact FAIL: alpha 1 beta 0"
expect "is never timed" lacks "^CSV,01_fake"

echo "a rung that reads C when beta = 0"
run env MOCK_KERNEL=nanread "$bin" --sizes 64 --check-only
expect "is caught by the NaN pre-fill" has "exact FAIL: .*got -?nan"

echo "a rung not written yet"
run env MOCK_KERNEL=none "$bin" --sizes 64 --check-only
expect "fails" has "exact FAIL"

echo "a power-capped GPU"
run env MOCK_POWERCAP=1 "$bin" --sizes 64 --kernels 0 "${fast[@]}"
expect "rejects every rep but finishes the sweep" has "no accepted reps"
expect "names the reason" has "reasons: SwPowerCap"
run env MOCK_POWERCAP=1 "$bin" --sizes 64 --kernels 0 --allow-power-cap "${fast[@]}"
expect "--allow-power-cap keeps the reps" has "^CSV,00_cublas,64,64,64,"

echo "modes"
run "$bin" --sizes 64 --graph 10 "${fast[@]}"
expect "--graph times graph replays" has "graph of 10"
run "$bin" --sizes 64,128 --profile
expect "--profile launches each rung once per shape" has "profile launch 4: 01_fake 128x128x128"
expect "--profile runs no checks" lacks "exact (pass|FAIL)"
run "$bin" --sizes 64,128 --skip-slow 0.001 "${fast[@]}"
expect "--skip-slow skips a shape estimated too slow" has "skipped: estimated"
expect "--skip-slow reports the count" has "configuration\(s\) skipped by --skip-slow"

echo "argument errors"
run "$bin" --sizes 1,,2
expect "empty list item" has "bad number ''"
run "$bin" --shape 3x4
expect "malformed shape" has "bad shape '3x4'"
run "$bin" --kernels 9
expect "unknown rung" has "no registered rung matches --kernels"

echo
echo "$pass passed, $fail failed"
[[ $fail -eq 0 ]]
