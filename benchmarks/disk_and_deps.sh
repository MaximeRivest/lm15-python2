#!/usr/bin/env bash
# Report wheel size, on-disk footprint, and transitive dep count for each library.
# Creates one throwaway venv per library to isolate measurements.

set -euo pipefail

if ! command -v uv >/dev/null 2>&1; then
    echo "uv not found — required for this benchmark" >&2
    exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK=$(mktemp -d /tmp/lm15-disk-bench-XXXXXX)
trap "rm -rf $WORK" EXIT

echo "=== Disk footprint and dependency count ==="
echo "(one throwaway venv per lib; Python 3.10 base)"
echo

printf "%-12s  %10s  %12s  %12s  %6s\n" "library" "wheel MB" "installed MB" "files .py/.so" "deps"
printf '%s\n' '-------------------------------------------------------------------------'

# Shared pip-only venv used to download wheels (never mixed with lib venvs).
PIP_ENV="$WORK/_piponly"
uv venv --quiet "$PIP_ENV" --python 3.10 >/dev/null 2>&1
uv pip install --quiet --python "$PIP_ENV/bin/python" pip >/dev/null 2>&1

for lib in urllib3 requests httpx aiohttp openai google-genai groq anthropic litellm; do
    venv="$WORK/$lib"
    uv venv --quiet "$venv" --python 3.10 >/dev/null 2>&1 || {
        echo "$lib: uv venv failed" >&2
        continue
    }
    # Install the library only (no pip) so site-packages reflects real install
    uv pip install --quiet --no-cache --python "$venv/bin/python" "$lib" >/dev/null 2>&1 || {
        echo "$lib: install failed" >&2
        continue
    }
    # Download wheels to a separate tree using the pip-only venv
    wheels="$WORK/${lib}_wheels"
    mkdir -p "$wheels"
    "$PIP_ENV/bin/python" -m pip download --quiet --no-cache-dir --dest "$wheels" "$lib" >/dev/null 2>&1 || true

    wheel_bytes=$(du -sb "$wheels" 2>/dev/null | awk '{print $1}')
    sp=$("$venv/bin/python" -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
    # Apparent bytes, minus virtualenv shim
    inst_bytes=$(du -sb "$sp" 2>/dev/null | awk '{print $1}')
    shim_bytes=$(du -sb "$sp"/_virtualenv* 2>/dev/null | awk '{s+=$1} END {print s+0}')
    real_bytes=$((inst_bytes - shim_bytes))

    py_count=$(find "$sp" -name '*.py' -not -path '*/__pycache__/*' -not -path '*_virtualenv*' -not -path '*.dist-info/*' | wc -l)
    so_count=$(find "$sp" -name '*.so' -not -path '*/__pycache__/*' | wc -l)

    # Transitive deps: everything uv lists minus the root package
    dep_count=$(uv pip list --python "$venv/bin/python" 2>/dev/null \
                | tail -n +3 \
                | awk '{print $1}' \
                | { grep -iv -E "^(${lib//-/_}|${lib})$" || true; } \
                | wc -l)

    awk_fmt() { awk -v b="$1" 'BEGIN{printf "%.2f", b/1048576}'; }
    printf "%-12s  %10s  %12s  %12s  %6s\n" \
        "$lib" \
        "$(awk_fmt "$wheel_bytes")" \
        "$(awk_fmt "$real_bytes")" \
        "$py_count/$so_count" \
        "$dep_count"
done

# lm15.transports stats (measured from source)
transports_bytes=$(du -sb "$REPO_ROOT/lm15/transports" 2>/dev/null | awk '{print $1}')
transports_py=$(find "$REPO_ROOT/lm15/transports" -name '*.py' -not -path '*/__pycache__/*' | wc -l)
awk_fmt() { awk -v b="$1" 'BEGIN{printf "%.2f", b/1048576}'; }

printf "%-12s  %10s  %12s  %12s  %6s\n" \
    "lm15 (ours)" \
    "0.00" \
    "$(awk_fmt "$transports_bytes")" \
    "$transports_py/0" \
    "0"
