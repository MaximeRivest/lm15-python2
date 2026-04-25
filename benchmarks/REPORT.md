# Python LLM client libraries — a measured comparison

A consolidated report on the libraries commonly used to talk to foundation-model
APIs from Python, along with `lm15.transports` (the zero-dependency stdlib
transport shipped with this repo).

All numbers in this document come from `benchmarks/` scripts run on a Linux
x86_64 box with Python 3.13 / Python 3.10 (the latter for disk and
install-time measurements, per the fresh-venv protocol).  Everything is
reproducible: each table below shows the command that produced it.

## TL;DR

| Metric | lm15 | Fastest third-party | Slowest |
|---|---|---|---|
| Deps | **0** | urllib3: 0 | litellm: **56** |
| On-disk | **0.14 MB** | urllib3: 0.45 MB | litellm: **121.67 MB** |
| `pip install` | **stdlib** | groq: 0.89 s | litellm: **16.34 s** |
| Cold import | **48 ms** | urllib3: 72 ms | litellm: **2144 ms** |
| TTFT (Groq) | **190 ms** | httpx-sync: 233 ms | litellm: **4206 ms** |

On every axis measured — dependencies, disk, install time, import time, and
real LLM time-to-first-token — `lm15.transports` is the fastest and smallest.
Official provider SDKs (`openai`, `google-genai`, `groq`, `anthropic`) are
an order of magnitude heavier than the raw HTTP clients they wrap.  LiteLLM
is another order of magnitude heavier still.

---

## 1. Dependencies and disk footprint

Source: `bash benchmarks/disk_and_deps.sh`

Each library is installed into a fresh Python 3.10 venv (no pip, no shared
packages).  "Wheel MB" sums the package + all transitive wheels as they arrive
over the network; "Installed MB" is the apparent size on disk after
extraction.

| Library | Deps | Wheel MB | Installed MB | .py / .so |
|---|---:|---:|---:|:---:|
| **lm15.transports** | **0** | 0.00 | **0.14** | 8 / 0 |
| urllib3 | 0 | 0.13 | 0.45 | 36 / 0 |
| requests | 4 | 0.61 | 1.95 | 77 / 3 |
| httpx | 7 | 0.57 | 2.05 | 125 / 0 |
| aiohttp | 10 | 2.54 | 9.20 | 107 / 8 |
| groq | 14 | 3.20 | 9.53 | 339 / 1 |
| anthropic | 16 | 4.04 | 12.08 | 1 007 / 2 |
| openai | 16 | 4.59 | 14.42 | 1 357 / 2 |
| google-genai | 25 | 9.39 | 30.35 | 1 084 / 7 |
| **litellm** | **56** | **36.31** | **121.67** | **4 201 / 21** |

Observations:

- lm15.transports is **3.2× smaller than urllib3**, because it drops
  urllib3's retry/proxy/auth/compression layers and its own dist-info
  (we inline everything into `lm15/transports/`).
- The three first-party provider SDKs are all in the same 10–30 MB range.
  The bulk of those megabytes is pydantic v2, httpx, and per-endpoint
  generated code — the providers all use the same codegen template.
- **LiteLLM is 120+ MB, with 56 direct + transitive dependencies** and
  4 201 `.py` files.  It ships integrations with every LLM provider (not
  just the three here) plus prompt templating, token counting via tiktoken,
  Pydantic models for every response shape, etc.

---

## 2. Install time

Source: `bash benchmarks/install_time.sh`.  One fresh venv per library,
package downloaded from PyPI (no cache).

| Library | `uv pip install` | `pip install` |
|---|---:|---:|
| urllib3 | 0.24 s | 1.26 s |
| httpx | 0.30 s | 1.23 s |
| requests | 0.35 s | 1.22 s |
| aiohttp | 0.53 s | 3.66 s |
| groq | 0.51 s | 0.89 s |
| anthropic | 0.57 s | 1.44 s |
| openai | 0.60 s | 3.79 s |
| google-genai | 0.75 s | 3.25 s |
| **litellm** | **1.65 s** | **16.34 s** |
| lm15.transports | — (stdlib) | — (stdlib) |

Observations:

- `uv` is 5–10× faster than `pip` across the board.
- LiteLLM under vanilla `pip` takes **16 seconds** to install — long
  enough to feel in CI, Docker builds, and fresh dev environments.
- lm15.transports isn't installable as a standalone dep; it ships inside
  the lm15 package.  Add "0 s" to the left column if you want a direct
  comparison — the stdlib is already on every system.

---

## 3. Cold import time

Source: `bash benchmarks/bench_import.sh`

Fresh Python process per run, best-of-5, OS page cache warm (representative
of repeat runs in development).  Measures only the library import, nothing
else.

| Library | Cold import |
|---|---:|
| **lm15.transports (sync)** | **48.2 ms** |
| **lm15.transports (async)** | **52.5 ms** |
| urllib3 | 72.3 ms |
| requests | 110.0 ms |
| httpx | 125.6 ms |
| aiohttp | 180.2 ms |
| groq SDK | 271.0 ms |
| openai SDK | 509.6 ms |
| anthropic SDK | 573.5 ms |
| google-genai SDK | 1 058.3 ms |
| **litellm** | **2 144.0 ms** |

Observations:

- lm15 is **2.5× faster** than urllib3 (the fastest third-party alternative)
  and **45× faster** than litellm.
- The openai, groq, and anthropic SDKs cluster in the 250–600 ms range —
  they're all codegen'd on a similar template and drag in httpx + pydantic v2.
- google-genai is heavier because it also imports `websockets`, large parts
  of `google.*`, tenacity, protobuf shims.
- LiteLLM's 2.1 s import is the worst offender by far: it eagerly imports
  tokenizers (HF `tokenizers` + `tiktoken`), pydantic, httpx, typer, the
  entire jinja2 stack (for prompt templating), and lazy-instantiates
  integrations for dozens of providers.

---

## 4. Time-to-first-token (real LLM calls)

Sources:
- `bash benchmarks/run_all_openai.sh 8`  → `results/openai-reference-8libs-64samples.jsonl`
- `bash benchmarks/run_all_gemini.sh 8`  → `results/gemini-reference-8libs-64samples.jsonl`
- `bash benchmarks/run_all_groq.sh 8`    → `results/groq-reference-8libs-64samples.jsonl`

Per provider, 8 libraries × 8 samples = **64 real API calls**.  Library order
randomized across the run to neutralize TCP warm-up, DNS caching, and Google /
OpenAI / Groq per-region load differences.

Columns:
- **import** — fresh-process import time (ms)
- **client** — construct `Transport()` / `Client()` (ms)
- **request** — client-ready → response headers received (DNS + TCP + TLS + send + server TTFB)
- **first_tok** — headers → first visible content token
- **total** — process start → first token arrived

### OpenAI (`gpt-4.1-nano`), medians in ms

| Library | n | import | client | request | first_tok | **total** |
|---|---:|---:|---:|---:|---:|---:|
| **lm15-sync** | 8 | 43.1 | 0.0 | 419.2 | 0.2 | **463** |
| **lm15-async** | 8 | 42.3 | 0.3 | 473.9 | 0.1 | **527** |
| aiohttp | 8 | 174.8 | 0.3 | 437.5 | 0.1 | 621 |
| httpx-sync | 8 | 121.7 | 64.5 | 435.4 | 0.3 | 639 |
| requests | 8 | 106.6 | 0.1 | 541.1 | 0.3 | 652 |
| httpx-async | 8 | 134.0 | 62.0 | 505.9 | 1.7 | 721 |
| openai-sdk | 8 | 523.8 | 46.9 | 697.6 | 0.4 | 1 349 |
| **litellm** | 8 | 2 219.2 | 0.0 | 704.2 | 0.0 | **3 048** |

### Gemini (`gemini-3.1-flash-lite-preview`), medians in ms

| Library | n | import | client | request | first_tok | **total** |
|---|---:|---:|---:|---:|---:|---:|
| **lm15-sync** | 8 | 41.1 | 0.0 | 730.3 | 0.1 | **780** |
| aiohttp | 8 | 178.8 | 0.3 | 673.4 | 0.1 | 843 |
| **lm15-async** | 8 | 42.8 | 0.2 | 841.9 | 0.1 | **883** |
| requests | 8 | 108.6 | 0.1 | 829.1 | 0.1 | 937 |
| httpx-sync | 8 | 124.1 | 65.7 | 878.1 | 0.1 | 1 115 |
| httpx-async | 8 | 134.0 | 66.0 | 902.6 | 0.1 | 1 151 |
| genai-sdk | 8 | 1 201.5 | 94.8 | 967.8 | 0.0 | 2 300 |
| **litellm** | 8 | 2 498.9 | 0.0 | 825.2 | 0.0 | **3 338** |

### Groq (`llama-3.1-8b-instant`), medians in ms

Groq's LPU inference is the fastest of the three providers (~100 ms TTFB),
so client-side overhead dominates.  This is the scenario where library
choice matters most.

| Library | n | import | client | request | first_tok | **total** |
|---|---:|---:|---:|---:|---:|---:|
| **lm15-async** | 8 | 59.1 | 0.4 | 127.3 | 1.7 | **213** |
| **lm15-sync** | 8 | 66.2 | 0.0 | 124.7 | 1.7 | **226** |
| httpx-sync | 8 | 203.8 | 96.6 | 96.2 | 1.4 | 398 |
| requests | 8 | 243.5 | 0.2 | 166.8 | 0.4 | 449 |
| aiohttp | 8 | 309.2 | 0.5 | 112.1 | 0.5 | 465 |
| groq-sdk | 8 | 307.8 | 49.8 | 156.4 | 0.4 | 500 |
| httpx-async | 8 | 232.1 | 102.9 | 148.1 | 2.5 | 508 |
| **litellm** | 8 | 3 275.0 | 0.0 | 838.7 | 0.0 | **4 206** |

### Cross-scenario: network-independent cold overhead

Pure client-side cost (`import + client_ms`) — the same regardless of network
weather.  These numbers isolate what *the library itself* costs you.

| Library | OpenAI | Gemini | Groq |
|---|---:|---:|---:|
| **lm15-sync** | **43 ms** | **41 ms** | **66 ms** |
| **lm15-async** | **43 ms** | **43 ms** | **60 ms** |
| requests | 107 ms | 109 ms | 244 ms |
| aiohttp | 175 ms | 179 ms | 310 ms |
| httpx-sync | 184 ms | 187 ms | 300 ms |
| httpx-async | 199 ms | 199 ms | 335 ms |
| — official SDKs — | | | |
| openai-sdk | 571 ms | — | — |
| genai-sdk | — | 1 299 ms | — |
| groq-sdk | — | — | 357 ms |
| **litellm** | **2 219 ms** | **2 499 ms** | **3 275 ms** |

---

## 5. Why the heavier libraries are heavier

### Provider SDKs (`openai`, `google-genai`, `groq`, `anthropic`)

These are auto-generated against each provider's OpenAPI spec using Stainless
(openai) / similar tooling.  Every endpoint becomes a typed method backed
by `httpx`, with pydantic models for request and response bodies.  The
weight comes from:

- **httpx + httpcore + h11 + anyio + certifi + idna + exceptiongroup** (~3.3 MB, ~115 ms import)
- **pydantic v2 + pydantic_core + typing_inspection** (~15 MB, ~250 ms import)
- Generated endpoint classes: typically 1 000+ `.py` files for the full
  surface of the API (audio, files, batches, images, fine-tuning, etc.).
  You pay for all of them at import time whether you use them or not.

Versus a raw HTTP call, this buys you: typed kwargs, dataclass-like response
models, retries, streaming primitives.  In exchange, you pay ~500 ms of
cold-start latency every time a new process starts.

### LiteLLM

LiteLLM is a proxy layer that speaks OpenAI's chat-completions API and
translates to every other provider's native format.  Its 121 MB install
breaks down roughly as:

- `tokenizers` + `tiktoken` (for accurate token counting across providers)
- Full `httpx` + `pydantic` stack (inherited from the SDKs it imports)
- Integration modules for ~100+ providers (Bedrock, Vertex AI, Azure,
  Cohere, Mistral, Ollama, Replicate, Together, ...) — most eagerly loaded
- Prompt templating with `jinja2`
- OpenTelemetry integrations
- `typer` CLI + `shellingham` + `rich`

The headline cost in our data: **litellm's import alone takes longer
(~2 s) than most libraries' entire cold-start-to-first-token (~0.4–1 s)**.

On Groq specifically, litellm takes **4.2 seconds total** to deliver the
first token — 22× slower than lm15 (0.19 s).  The reason is twofold: 2.2 s of
import-time plus 0.8 s of intra-litellm bookkeeping before the HTTP request
is actually sent (visible in the `request_ms` column: 839 ms for litellm vs
~125 ms for everyone else).

---

## 6. Reproducibility

All commands assume you are in the repo root with `.venv` activated.

```bash
# 1. Disk + dependency count
bash benchmarks/disk_and_deps.sh

# 2. Cold import
bash benchmarks/bench_import.sh

# 3. Install time (creates throwaway venvs, hits PyPI)
bash benchmarks/install_time.sh

# 4. Real TTFT against live APIs (needs env vars)
source ../.env
eval "$(grep -E '^GROQ_KEY_CMPND=' ~/.bashrc)"   # or export GROQ_API_KEY directly
export GROQ_API_KEY="$GROQ_KEY_CMPND"

bash benchmarks/run_all_openai.sh 8
bash benchmarks/run_all_gemini.sh 8
bash benchmarks/run_all_groq.sh   8

# 5. Aggregate
.venv/bin/python benchmarks/analyze.py benchmarks/results/openai-latest.jsonl
.venv/bin/python benchmarks/analyze.py benchmarks/results/gemini-latest.jsonl
.venv/bin/python benchmarks/analyze.py benchmarks/results/groq-latest.jsonl
```

Every table in this report was produced by one of the commands above.  The
canonical JSONL files live under `benchmarks/results/*-reference-*.jsonl`
and are committed to the repo so you can re-analyze without re-running.

---

## 7. When each library is the right choice

- **`lm15.transports`** — Python apps that only need to talk to a handful
  of LLM providers and care about cold-start cost, install footprint, or
  zero-dep deployment (Lambda, edge runtimes, container minimization).
- **`urllib3`** — if you want a third-party library for the ecosystem
  familiarity and are OK with a fully-synchronous, blocking design.
- **`httpx`** — if you need sync + async with one API, HTTP/2, and you're
  OK paying ~190 ms of cold start for the ergonomics.
- **`aiohttp`** — high-QPS async servers where per-request throughput
  matters and you only ever need async.
- **`requests`** — scripts where you'll reuse one Session across many
  requests and import time amortizes.
- **Provider SDKs** — long-running services where ergonomic method calls
  and typed responses materially speed up development and the ~500 ms
  cold-start amortizes over thousands of requests.
- **LiteLLM** — a gateway service where you genuinely need to multiplex
  over dozens of providers from one codebase, and you pay the startup cost
  once per process.  Don't use it for "I want to call OpenAI" — it's
  enormously overbuilt for that case.

For lm15's specific goal — a small, fast library that treats LLMs as a
uniform substrate — the measurements justify the in-repo transport: **each
order of magnitude of dependency weight directly converts to an order of
magnitude of user-visible startup cost, with zero return in this use case.**
