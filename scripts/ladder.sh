#!/usr/bin/env bash
# A full ladder run: every rung and cuBLAS at every size, in one process and one clock state,
# so the rows are comparable. Saves the raw log in sgemm/runs/ and regenerates the README's
# results table from it. Arguments go to bin/sgemm, e.g. scripts/ladder.sh --sizes 128,1024
#
# Lock the SM clock first (docs/machine.md, M1), then commit the log and README together.
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ -n "$(git status --porcelain --untracked-files=no 2>/dev/null)" && "${ALLOW_DIRTY:-0}" != 1 ]]; then
    echo "uncommitted changes: commit first, so the log names code that exists." >&2
    echo "(ALLOW_DIRTY=1 runs anyway; the log and the README table will say -dirty)" >&2
    exit 1
fi

make --no-print-directory
sha=$(git describe --always --dirty --abbrev=7 2>/dev/null || echo unknown)
mkdir -p sgemm/runs
log="sgemm/runs/$(date +%Y-%m-%d_%H%M%S)_${sha}.txt"

if ! ./bin/sgemm "$@" 2>&1 | tee "$log"; then
    # Keep the evidence, but under a name no one can mistake for a good run.
    mv "$log" "${log%.txt}_failed.txt"
    echo >&2
    echo "bin/sgemm reported a failure; the README table is unchanged. Log: ${log%.txt}_failed.txt" >&2
    exit 1
fi
python3 tools/ladder_table.py "$log" --readme README.md
echo "saved $log; commit it together with README.md"
