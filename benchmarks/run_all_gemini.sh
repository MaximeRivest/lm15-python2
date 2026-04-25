#!/usr/bin/env bash
# Run the Gemini TTFT benchmark for every client, randomized order, N samples each.
#
# Usage: GEMINI_API_KEY=... bash run_all_gemini.sh [N]
#   N: samples per library (default 8)
#
# Randomization avoids penalizing whichever library happens to run first
# (cold TCP path to Google's edge) — each run is a fresh process anyway.

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-$REPO_ROOT/.venv/bin/python}"
N="${1:-8}"

if [ -z "${GEMINI_API_KEY:-}" ]; then
    echo "GEMINI_API_KEY not set" >&2
    exit 2
fi

RESULTS="$REPO_ROOT/benchmarks/results"
mkdir -p "$RESULTS"
ts=$(date -u +%Y%m%dT%H%M%SZ)
out="$RESULTS/gemini-$ts.jsonl"
: > "$out"

libs=(lm15-sync lm15-async httpx-sync httpx-async requests aiohttp genai-sdk litellm)

# Build the full sample list (each lib repeated N times), then shuffle
order=()
for i in $(seq 1 "$N"); do
    for lib in "${libs[@]}"; do
        order+=("$lib")
    done
done
order=($(printf "%s\n" "${order[@]}" | shuf))

total=${#order[@]}
echo "Running $total samples total (N=$N per lib), randomized order..."
echo "Model: ${LM15_BENCH_GEMINI_MODEL:-gemini-3.1-flash-lite-preview}"
echo "Output: $out"
echo

count=0
for lib in "${order[@]}"; do
    count=$((count + 1))
    printf "[%3d/%3d] %-14s " "$count" "$total" "$lib"
    if line=$(PYTHONPATH="$REPO_ROOT" GEMINI_API_KEY="$GEMINI_API_KEY" \
              "$PY" "$REPO_ROOT/benchmarks/bench_gemini_ttft.py" "$lib" 2>/dev/null); then
        echo "$line" >> "$out"
        tok=$(echo "$line" | "$PY" -c "import json,sys; r=json.loads(sys.stdin.read()); print(f'{r[\"request_ms\"]:>5.0f}ms first={r[\"first_token\"]!r}')")
        echo "$tok"
    else
        echo "(failed, skipping)"
    fi
    sleep 0.1  # be gentle
done

ln -sfn "gemini-$ts.jsonl" "$RESULTS/gemini-latest.jsonl"

echo
echo "Aggregate: $PY $REPO_ROOT/benchmarks/analyze.py $out"
