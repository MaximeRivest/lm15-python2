#!/usr/bin/env bash
# Run the loopback TTFR benchmark for every client, N samples each, warm-up first.

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-$REPO_ROOT/.venv/bin/python}"
N="${1:-5}"   # samples per lib; default 5

RESULTS="$REPO_ROOT/benchmarks/results"
mkdir -p "$RESULTS"
ts=$(date -u +%Y%m%dT%H%M%SZ)
out="$RESULTS/local-$ts.jsonl"
: > "$out"

libs=(lm15-sync lm15-async requests urllib3 httpx-sync httpx-async)

echo "Running $N samples per lib against localhost loopback..."
echo "Output: $out"
echo

for lib in "${libs[@]}"; do
    # Warm-up (discarded): importers + TCP stack warm
    PYTHONPATH="$REPO_ROOT" "$PY" "$REPO_ROOT/benchmarks/bench_ttfr_local.py" "$lib" > /dev/null 2>&1 || {
        echo "$lib: not installed, skipping"
        continue
    }
    echo -n "$lib: "
    for i in $(seq 1 "$N"); do
        PYTHONPATH="$REPO_ROOT" "$PY" "$REPO_ROOT/benchmarks/bench_ttfr_local.py" "$lib" >> "$out"
        echo -n "."
    done
    echo " done"
done

# Create / update a stable 'latest' symlink for convenience
ln -sfn "local-$ts.jsonl" "$RESULTS/local-latest.jsonl"

echo
echo "Aggregate: $PY $REPO_ROOT/benchmarks/analyze.py $out"
