#!/usr/bin/env bash
# Install-time benchmark: fresh venv per library, uv + pip, no cache.
#
# Produces a two-column table: "uv pip install X" wall-clock vs
# "pip install X" wall-clock, for every HTTP client + LLM SDK in the
# comparison set.  No network caching is allowed on either side.

set -euo pipefail

if ! command -v uv >/dev/null 2>&1; then
    echo "uv not found — required for this benchmark" >&2
    exit 2
fi

WORK=$(mktemp -d /tmp/lm15-install-bench-XXXXXX)
trap "rm -rf $WORK" EXIT

# A pip-only base venv we can cheaply clone for each pip test.
uv venv --quiet "$WORK/_pipbase" --python 3.10 >/dev/null
uv pip install --quiet --python "$WORK/_pipbase/bin/python" pip >/dev/null

echo "=== Install time (fresh venv, no cache) ==="
echo
printf "%-14s  %10s  %10s\n" "library" "uv (s)" "pip (s)"
echo "----------------------------------------"

for lib in urllib3 requests httpx aiohttp openai google-genai groq anthropic litellm; do
    # uv path
    uv_venv="$WORK/${lib}_uv"
    uv venv --quiet "$uv_venv" --python 3.10 >/dev/null
    t0=$(date +%s.%N)
    if ! uv pip install --quiet --no-cache --python "$uv_venv/bin/python" "$lib" >/dev/null 2>&1; then
        echo "$lib: uv install FAILED"
        continue
    fi
    t1=$(date +%s.%N)
    uv_s=$(awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.2f", b-a}')

    # pip path — clone the pip-only base venv
    pip_venv="$WORK/${lib}_pip"
    cp -r "$WORK/_pipbase" "$pip_venv"
    t0=$(date +%s.%N)
    if ! "$pip_venv/bin/pip" install --quiet --no-cache-dir --disable-pip-version-check "$lib" >/dev/null 2>&1; then
        echo "$lib: pip install FAILED"
        continue
    fi
    t1=$(date +%s.%N)
    pip_s=$(awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.2f", b-a}')

    printf "%-14s  %10s  %10s\n" "$lib" "${uv_s}s" "${pip_s}s"
done
