#!/usr/bin/env bash
# Run the Groq TTFT benchmark for every client, randomized order, N samples each.
#
# Usage: GROQ_API_KEY=gsk-... bash run_all_groq.sh [N]
#   N: samples per library (default 8)

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-$REPO_ROOT/.venv/bin/python}"
N="${1:-8}"

if [ -z "${GROQ_API_KEY:-}" ]; then
    echo "GROQ_API_KEY not set" >&2
    exit 2
fi

RESULTS="$REPO_ROOT/benchmarks/results"
mkdir -p "$RESULTS"
ts=$(date -u +%Y%m%dT%H%M%SZ)
out="$RESULTS/groq-$ts.jsonl"
: > "$out"

libs=(lm15-sync lm15-async httpx-sync httpx-async requests aiohttp groq-sdk litellm)

order=()
for i in $(seq 1 "$N"); do
    for lib in "${libs[@]}"; do
        order+=("$lib")
    done
done
order=($(printf "%s\n" "${order[@]}" | shuf))

total=${#order[@]}
echo "Running $total samples total (N=$N per lib), randomized order..."
echo "Model: ${LM15_BENCH_GROQ_MODEL:-llama-3.1-8b-instant}"
echo "Output: $out"
echo

count=0
for lib in "${order[@]}"; do
    count=$((count + 1))
    printf "[%3d/%3d] %-14s " "$count" "$total" "$lib"
    if line=$(PYTHONPATH="$REPO_ROOT" GROQ_API_KEY="$GROQ_API_KEY" \
              "$PY" "$REPO_ROOT/benchmarks/bench_groq_ttft.py" "$lib" 2>/dev/null); then
        echo "$line" >> "$out"
        tok=$(echo "$line" | "$PY" -c "import json,sys; r=json.loads(sys.stdin.read()); print(f'{r[\"request_ms\"]:>5.0f}ms first={r[\"first_token\"]!r}')")
        echo "$tok"
    else
        echo "(failed, skipping)"
    fi
    sleep 0.1
done

ln -sfn "groq-$ts.jsonl" "$RESULTS/groq-latest.jsonl"

echo
echo "Aggregate: $PY $REPO_ROOT/benchmarks/analyze.py $out"
