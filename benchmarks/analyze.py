"""Aggregate a JSONL result file into a median / p25 / p75 table.

Usage:
    python analyze.py results/<file>.jsonl
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path


ORDER = [
    "lm15-sync",
    "lm15-async",
    "requests",
    "urllib3",
    "httpx-sync",
    "httpx-async",
    "aiohttp",
    # Provider SDKs (each bench scenario surfaces only its own):
    "openai-sdk",
    "genai-sdk",
    "groq-sdk",
    "anthropic-sdk",
    # Meta-SDK:
    "litellm",
]


def percentile(vs: list[float], q: float) -> float:
    if not vs:
        return float("nan")
    s = sorted(vs)
    k = (len(s) - 1) * q
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


def col(rows: list[dict], key: str) -> list[float]:
    return [r[key] for r in rows if r.get(key) is not None]


def fmt(v: float) -> str:
    if v != v:  # nan
        return "       —  "
    return f"{v:>9.1f}"


def main() -> None:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <results.jsonl>", file=sys.stderr)
        sys.exit(1)

    path = Path(sys.argv[1])
    rows: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or not line.startswith("{"):
                continue
            rows.append(json.loads(line))

    if not rows:
        print("no results found", file=sys.stderr)
        sys.exit(1)

    by_lib: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_lib[r["lib"]].append(r)

    scenario = rows[0].get("scenario", "unknown")
    model = rows[0].get("model")
    print(f"=== {scenario}" + (f" ({model})" if model else "") + " ===")
    print(f"n={sum(len(v) for v in by_lib.values())} samples across {len(by_lib)} libraries")
    print()

    # Decide which columns to show based on the scenario
    if scenario == "local-loopback":
        cols = [("import_ms", "import"),
                ("client_ms", "client"),
                ("request_ms", "request"),
                ("total_ms", "total")]
    else:
        cols = [("import_ms", "import"),
                ("client_ms", "client"),
                ("request_ms", "request"),
                ("first_byte_ms", "first_byte"),
                ("first_token_ms", "first_token"),
                ("complete_ms", "complete"),
                ("total_ms", "total")]

    # Header
    header = f"{'lib':12s} {'n':>3s}  " + "  ".join(
        f"{label:>9s}" for _, label in cols
    )
    print(header)
    print("-" * len(header))

    # Medians
    def _row(lib: str, runs: list[dict]) -> str:
        parts = [f"{lib:12s} {len(runs):>3d}"]
        for key, _label in cols:
            parts.append(fmt(percentile(col(runs, key), 0.5)))
        return "  ".join(parts)

    for lib in ORDER:
        runs = by_lib.get(lib, [])
        if not runs:
            continue
        print(_row(lib, runs))
    for lib in sorted(by_lib):
        if lib in ORDER:
            continue
        print(_row(lib, by_lib[lib]))

    print()
    print("All values are medians (ms).")
    print()

    # IQR for stability assessment
    print("=== IQR (p25 → p75) — smaller = more stable ===")
    for lib in ORDER + [k for k in sorted(by_lib) if k not in ORDER]:
        runs = by_lib.get(lib, [])
        if not runs:
            continue
        overhead_key = "total_ms" if scenario == "local-loopback" else "first_token_ms"
        vs = col(runs, overhead_key)
        if not vs:
            continue
        lo = percentile(vs, 0.25)
        hi = percentile(vs, 0.75)
        print(f"  {lib:12s}  {overhead_key}: {lo:>6.1f}ms → {hi:>6.1f}ms  (spread {hi-lo:>5.1f}ms)")

    # Cold overhead (network-independent) — applies to any streaming LLM scenario
    if scenario != "local-loopback":
        print()
        print("=== Network-independent cold overhead (import + client_ms) ===")
        print("This is what the library itself costs; excludes server TTFB jitter.")
        for lib in ORDER + [k for k in sorted(by_lib) if k not in ORDER]:
            runs = by_lib.get(lib, [])
            if not runs:
                continue
            overhead = [r["import_ms"] + r["client_ms"] for r in runs]
            med = percentile(overhead, 0.5)
            lo = percentile(overhead, 0.25)
            hi = percentile(overhead, 0.75)
            print(f"  {lib:12s}  {med:>6.1f}ms   (p25→p75: {lo:>5.1f} → {hi:>5.1f})")

        # Correctness sanity check — make sure every lib got a visible token
        tokens_by_lib = {lib: {r["first_token"] for r in runs} for lib, runs in by_lib.items()}
        print()
        print("=== Correctness: first_token values seen ===")
        for lib in ORDER + [k for k in sorted(by_lib) if k not in ORDER]:
            tokens = tokens_by_lib.get(lib)
            if not tokens:
                continue
            print(f"  {lib:12s} -> {sorted(t for t in tokens if t)}")


if __name__ == "__main__":
    main()
