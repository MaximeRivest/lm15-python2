#!/usr/bin/env bash
# Isolated cold-import timing per library.
# Each run is a fresh Python process that imports ONLY the library in question.
# Prints best-of-5 in ms.

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-$REPO_ROOT/.venv/bin/python}"

echo "=== Cold import time (fresh process, best of 5, ms) ==="
echo "Python: $($PY --version)"
echo

for lib in lm15-sync lm15-async urllib3 requests httpx-sync httpx-async aiohttp openai-sdk genai-sdk groq-sdk anthropic-sdk litellm; do
    case $lib in
        lm15-sync)     cmd="from lm15.transports import StdlibTransport" ;;
        lm15-async)    cmd="from lm15.transports import StdlibAsyncTransport" ;;
        httpx-sync|httpx-async)  cmd="import httpx" ;;
        requests)      cmd="import requests" ;;
        aiohttp)       cmd="import aiohttp" ;;
        urllib3)       cmd="import urllib3" ;;
        openai-sdk)    cmd="import openai" ;;
        genai-sdk)     cmd="from google import genai" ;;
        groq-sdk)      cmd="import groq" ;;
        anthropic-sdk) cmd="import anthropic" ;;
        litellm)       cmd="import litellm" ;;
    esac

    best=""
    for i in 1 2 3 4 5; do
        t=$(PYTHONPATH="$REPO_ROOT" "$PY" -c "
import time
t0 = time.perf_counter()
$cmd
t1 = time.perf_counter()
print(f'{(t1-t0)*1000:.2f}')
" 2>/dev/null || echo "SKIP")
        if [ "$t" = "SKIP" ]; then
            continue
        fi
        if [ -z "$best" ]; then
            best=$t
        else
            is_less=$(awk -v a="$t" -v b="$best" 'BEGIN{print (a<b) ? 1 : 0}')
            if [ "$is_less" = "1" ]; then best=$t; fi
        fi
    done
    if [ -z "$best" ]; then
        printf "%-14s  (not installed, skipped)\n" "$lib"
    else
        printf "%-14s  %6s ms\n" "$lib" "$best"
    fi
done
