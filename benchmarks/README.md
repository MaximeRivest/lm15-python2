# lm15 transport benchmarks

Reproducible benchmarks comparing `lm15.transports` (stdlib-only, zero deps)
against `httpx`, `requests`, `aiohttp`, `urllib3`, the provider SDKs
(`openai`, `google-genai`, `groq`, `anthropic`), and the meta-SDK
`litellm`.

**See [`REPORT.md`](./REPORT.md)** for a consolidated, narrative comparison
that combines disk, dependencies, install time, cold import, and real
time-to-first-token against OpenAI, Gemini, and Groq.

## What's in here

| Script | Purpose |
|---|---|
| `bench_import.sh` | Isolated cold-import time per library (5 runs, best-of) |
| `bench_ttfr_local.py` | Time-to-first-request against a local loopback HTTP server (no network) |
| `bench_openai_ttft.py` | Time-to-first-token against OpenAI `/v1/chat/completions` |
| `bench_gemini_ttft.py` | Time-to-first-token against Gemini `streamGenerateContent` |
| `bench_groq_ttft.py` | Time-to-first-token against Groq (OpenAI-compatible endpoint) |
| `_bench_lib.py` | Shared per-library HTTP streaming runners (imported by the above) |
| `analyze.py` | Aggregate JSONL results → median/p25/p75 table |
| `run_all_local.sh` | Run the loopback benchmark for every client |
| `run_all_openai.sh` | Run the OpenAI benchmark, randomized order, N samples per lib |
| `run_all_gemini.sh` | Run the Gemini benchmark, randomized order, N samples per lib |
| `run_all_groq.sh` | Run the Groq benchmark, randomized order, N samples per lib |
| `disk_and_deps.sh` | Report wheel size, on-disk footprint, transitive deps per library |
| `install_time.sh` | Fresh-venv install time for each library via both `uv` and `pip` |

All scripts write JSONL to `results/` so you can re-aggregate without
re-running.

## Setup

```bash
# From the repo root
uv pip install --python .venv/bin/python pytest pytest-asyncio httpx requests aiohttp urllib3

# Load API keys (OPENAI_API_KEY, GEMINI_API_KEY) — adjust path to your .env
source ../.env
```

## Running

### Cold-import timing

```bash
cd benchmarks
bash bench_import.sh
```

Measures a fresh Python process importing only the library in question,
5 runs best-of.  Numbers include lazy-import chains triggered by the
top-level module.

### Loopback TTFR

```bash
bash run_all_local.sh
python analyze.py results/local-latest.jsonl
```

Starts an in-process HTTP server on a fresh port per run, measures end-to-end
cost of one GET from fresh process start.  Eliminates network variance
entirely — what you measure is pure client-side overhead.

### OpenAI TTFT

```bash
source ../.env   # OPENAI_API_KEY must be exported
bash run_all_openai.sh 8   # 8 samples per library, 48 total
python analyze.py results/openai-latest.jsonl
```

Sends a real `POST /v1/chat/completions` with `stream=true` and
`model=gpt-4.1-nano`, measures when the first `delta.content` token arrives.
Randomizes the order across libraries to avoid penalizing whichever runs
first (cold TCP path to OpenAI's edge).

Cost per run: 1 request × ~20 tokens output on gpt-4.1-nano — pennies.

### Gemini TTFT

```bash
source ../.env   # GEMINI_API_KEY must be exported
bash run_all_gemini.sh 8   # 8 samples per library, 48 total
python analyze.py results/gemini-latest.jsonl
```

Sends a real `POST streamGenerateContent?alt=sse&key=...` to
`gemini-3.1-flash-lite-preview`, measures when the first non-empty
`candidates[0].content.parts[*].text` arrives.  Key protocol
differences from OpenAI handled by `bench_gemini_ttft.py`:

- Auth via `?key=...` query parameter (not `Authorization` header)
- Gemini's first chunk for thinking models may contain a `thoughtSignature`
  with empty text — the parser skips those and waits for real content.
- No `[DONE]` terminator; stream ends at connection close.

### Groq TTFT

```bash
export GROQ_API_KEY=gsk-...
bash run_all_groq.sh 8
python analyze.py results/groq-latest.jsonl
```

Groq serves an OpenAI-compatible API at `api.groq.com/openai/v1/...`, so
the request shape, auth scheme, and SSE format are identical to OpenAI's.
The interesting number here is `request_ms`: Groq's LPU inference regularly
hits server-side TTFB in the ~90 ms range — dramatically faster than
OpenAI or Gemini — which makes client-side overhead (the ~125 ms that
`httpx.Client()` spends on lazy imports + SSL setup) visible as a
*majority* of total time, not a rounding error.

## Reading the numbers

- **import_ms** — process start → `import X` finishes.  This is what hurts
  cold-start scenarios (serverless, CLI tools).
- **client_ms** — import → `Client()` / `Transport()` constructor done.
  For `httpx` this is where the SSL context + lazy imports bite (~65 ms).
- **request_ms** — client ready → first response byte.  Includes DNS, TCP,
  TLS, request send, and server-side time-to-first-byte.  Highly variable
  for real-network benchmarks; near-zero for loopback.
- **first_token_ms** — response headers → first SSE `content` delta parsed.
  Time the client spent framing HTTP + parsing SSE; client-side work only.
- **total_ms** — process start → response fully consumed.  What a user
  waits for in a cold-start scenario.

For benchmarking fairness, **cold overhead** (`import_ms + client_ms`)
is the most stable comparison across libraries — it's purely client-side
and doesn't depend on OpenAI's current load.

## Reference results

Canonical runs archived in `results/` (48 samples per library, randomized
order, Linux, Python 3.13).

Each benchmark compares **7 libraries**:

- 6 generic HTTP clients: `lm15-sync`, `lm15-async`, `requests`, `httpx-sync`, `httpx-async`, `aiohttp`
- 1 provider SDK per scenario: `openai-sdk`, `genai-sdk` (google-genai), `groq-sdk`

### OpenAI TTFT (gpt-4.1-nano), medians in ms

Source: `results/openai-reference-7libs-56samples.jsonl`

```
lib            n     import    client    request   first_tok    total
------------------------------------------------------------------------
lm15-sync      8       45.5       0.0      453.5        0.3      525.0
lm15-async     8       40.3       0.2      410.7        0.1      455.5
requests       8      101.6       0.1      517.5        0.2      624.5
httpx-sync     8       61.8      64.8      481.6        0.3      607.3
httpx-async    8       72.6      64.9      441.3        0.6      581.2
aiohttp        8      178.8       0.4      560.7        0.1      748.9
openai-sdk     8      498.5      43.7      713.2        0.3     1257.7
```

### Gemini TTFT (gemini-3.1-flash-lite-preview), medians in ms

Source: `results/gemini-reference-7libs-56samples.jsonl`

```
lib            n     import    client    request   first_tok    total
------------------------------------------------------------------------
lm15-sync      8       44.4       0.0     1098.4        0.1     1143.7
lm15-async     8       43.5       0.2      928.8        0.1      974.4
requests       8      100.6       0.1     1113.9        0.1     1217.8
httpx-sync     8       64.2      64.6      922.5        0.1     1047.0
httpx-async    8       74.4      65.4     1029.3        0.1     1172.8
aiohttp        8      167.9       0.3      947.7        0.1     1130.5
genai-sdk      8      978.4      88.6      914.9        0.0     1978.6
```

### Groq TTFT (llama-3.1-8b-instant), medians in ms

Source: `results/groq-reference-7libs-56samples.jsonl`

```
lib            n     import    client    request   first_tok    total
------------------------------------------------------------------------
lm15-sync      8       46.5       0.0       87.9        1.6      189.5
lm15-async     8       45.0       0.3      121.5        1.0      188.6
requests       8      102.4       0.1      125.3        1.1      232.9
httpx-sync     8       60.9      67.0       88.0        0.6      233.0
httpx-async    8       77.1      66.9       97.0        0.7      240.1
aiohttp        8      165.3       0.3      125.4        0.2      302.8
groq-sdk       8      210.5      45.5      168.7        0.4      423.7
```

### Network-independent cold overhead (import + client_ms)

This is pure client-side cost, identical regardless of network weather:

```
Scenario       lm15      requests    httpx    aiohttp    SDK
-----------------------------------------------------------------
OpenAI        46 ms     102 ms     125 ms    179 ms    544 ms
Gemini        44 ms     101 ms     128 ms    168 ms   1068 ms
Groq          47 ms     103 ms     126 ms    166 ms    259 ms
```

**The three provider SDKs are the slowest clients to start up by a wide
margin.**  The openai SDK takes ~500 ms of CPU time before it can even
send a request, google-genai takes ~1000 ms (because it drags in pydantic,
websockets, tenacity, parts of grpc, etc.), and groq's 259 ms is only fast
because it's a minimal fork of the openai SDK with fewer endpoints.

### Total time to first token (worst case)

On Groq, where server inference is ~90 ms, total time to first token
ranges over a **~2.2×** spread depending purely on your library choice:

- `lm15-sync`: **190 ms**
- `httpx-sync`: **233 ms**
- `aiohttp`: **303 ms**
- `groq-sdk`: **424 ms**

The SDK spends more time booting up Python than the inference engine
spends generating the token.  On OpenAI gpt-4.1-nano, `openai-sdk` takes
**1258 ms total** vs `lm15-async`'s **456 ms** — the SDK almost triples
the cold-start cost.

### Why the SDKs are so heavy

Rough per-SDK import cost breakdown (from observation — not measured line
by line):

- **openai** (~500 ms): imports httpx + httpcore + h11 + anyio + pydantic
  (v2) + typing_inspection + jiter + distro + sniffio.  The SDK builds a
  full validation layer on top of httpx.
- **google-genai** (~1000 ms): imports httpx + websockets + tenacity + a
  chunk of `google.*`, loads pydantic v2, imports `concurrent.futures`,
  initializes generated API types.
- **groq** (~250 ms): fork of the openai SDK with fewer generated
  endpoints, so it inherits the httpx + pydantic cost minus some surface
  area.

None of these are *doing* anything the raw HTTP clients don't do — they
are pure validation + convenience layers on top of httpx.  When the goal
is "send one request, parse the stream", paying 250–1000 ms for
ergonomics you don't need is expensive.

This is the argument lm15 makes structurally: **if the library is only
ever the transport, then the transport should be 1–2 kilobytes of
stdlib glue, not a dependency tree with a 1-second import time.**

### Isolated cold-import (fresh process, best of 5)

```
lm15-sync      50 ms     httpx-sync    78 ms
lm15-async     52 ms     httpx-async   72 ms
urllib3        78 ms     requests     108 ms
                         aiohttp      141 ms
```

### Network-independent cold overhead (import + client construction)

```
lm15-sync        4.5 ms     (p25→p75:   4.3 →   4.7)
lm15-async       4.9 ms     (p25→p75:   4.7 →   5.4)
requests        75.8 ms     (p25→p75:  75.3 →  75.9)
aiohttp         96.8 ms     (p25→p75:  94.2 → 102.8)
httpx-sync      98.6 ms     (p25→p75:  95.7 → 101.6)
httpx-async    100.5 ms     (p25→p75:  98.0 → 102.7)
```

### Disk / dependency comparison

From `disk_and_deps.sh`:

```
Library         Wheel MB    On-disk MB    .py/.so     Deps
lm15 (ours)        0.00         0.14        8/0         0
urllib3            0.13         0.45       36/0         0
requests           0.61         1.95       77/3         4
httpx              0.57         2.05      125/0         7
aiohttp            2.54         9.20      107/8        10
```

## Results files

| File | What |
|---|---|
| `results/openai-reference-7libs-56samples.jsonl` | 56-sample authoritative OpenAI run, gpt-4.1-nano |
| `results/gemini-reference-7libs-56samples.jsonl` | 56-sample authoritative Gemini run, gemini-3.1-flash-lite-preview |
| `results/groq-reference-7libs-56samples.jsonl` | 56-sample authoritative Groq run, llama-3.1-8b-instant |
| `results/openai-latest.jsonl` | Symlink to most recent OpenAI run |
| `results/gemini-latest.jsonl` | Symlink to most recent Gemini run |
| `results/groq-latest.jsonl` | Symlink to most recent Groq run |
| `results/local-latest.jsonl` | Symlink to most recent loopback run |
| `results/<scenario>-<timestamp>.jsonl` | Timestamped runs as you accumulate them |
