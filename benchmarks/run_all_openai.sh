#!/usr/bin/env bash
# Run the OpenAI TTFT benchmark for every client, randomized order, N samples each.
#
# Usage: OPENAI_API_KEY=sk-... bash run_all_openai.sh [N]
#   N: samples per library (default 8)
#
# Randomization avoids penalizing whichever library happens to run first
# (cold TCP path to OpenAI's edge) — each run is a fresh process anyway.

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-$REPO_ROOT/.venv/bin/python}"
N="${1:-8}"

if [ -z "${OPENAI_API_KEY:-}" ]; then
    echo "OPENAI_API_KEY not set" >&2
    exit 2
fi

RESULTS="$REPO_ROOT/benchmarks/results"
mkdir -p "$RESULTS"
ts=$(date -u +%Y%m%dT%H%M%SZ)
out="$RESULTS/openai-$ts.jsonl"
: > "$out"

libs=(lm15-sync lm15-async httpx-sync httpx-async requests aiohttp openai-sdk litellm)

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
echo "Model: ${LM15_BENCH_MODEL:-gpt-4.1-nano}"
echo "Output: $out"
echo

count=0
for lib in "${order[@]}"; do
    count=$((count + 1))
    printf "[%3d/%3d] %-14s " "$count" "$total" "$lib"
    if line=$(PYTHONPATH="$REPO_ROOT" OPENAI_API_KEY="$OPENAI_API_KEY" \
              "$PY" "$REPO_ROOT/benchmarks/bench_openai_ttft.py" "$lib" 2>/dev/null); then
        echo "$line" >> "$out"
        tok=$(echo "$line" | "$PY" -c "import json,sys; r=json.loads(sys.stdin.read()); print(f'{r[\"request_ms\"]:>5.0f}ms first={r[\"first_token\"]!r}')")
        echo "$tok"
    else
        echo "(failed, skipping)"
    fi
    sleep 0.1  # be gentle
done

ln -sfn "openai-$ts.jsonl" "$RESULTS/openai-latest.jsonl"

echo
echo "Aggregate: $PY $REPO_ROOT/benchmarks/analyze.py $out"
