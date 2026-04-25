"""Time-to-first-token against OpenAI's real /v1/chat/completions endpoint.

Measures, per run in a fresh Python process:
    process_start → import         (cold-import time)
    import → client                (client/transport construction)
    client → response-headers      (DNS + TCP + TLS + request send + server TTFB)
    headers → first body byte      (pure body-stream latency)
    headers → first 'content' tok  (body stream + SSE parse + JSON decode)
    headers → stream complete      (all chunks drained)
    total

Supports both raw-HTTP clients (lm15, httpx, requests, aiohttp) and the
official `openai` Python SDK.

Usage:
    OPENAI_API_KEY=sk-... python bench_openai_ttft.py <client>
"""

from __future__ import annotations

import json
import os
import sys

from _bench_lib import (
    RUNNERS,
    litellm_openai_ops,
    openai_chat_completions_parser,
    openai_sdk_ops,
    run_sdk_sync,
)


URL = "https://api.openai.com/v1/chat/completions"
MODEL = os.environ.get("LM15_BENCH_MODEL", "gpt-4.1-nano")
SCENARIO = "openai-chat-completions"
PROMPT = "Say 'hello' and nothing else."
MAX_TOKENS = 20

_PAYLOAD = {
    "model": MODEL,
    "messages": [{"role": "user", "content": PROMPT}],
    "stream": True,
    "max_tokens": MAX_TOKENS,
}

ALL_CLIENTS = list(RUNNERS) + ["openai-sdk", "litellm"]


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in ALL_CLIENTS:
        print(f"usage: {sys.argv[0]} <{ '|'.join(ALL_CLIENTS) }>", file=sys.stderr)
        sys.exit(2)

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY not set", file=sys.stderr)
        sys.exit(2)

    client = sys.argv[1]

    if client == "openai-sdk":
        result = run_sdk_sync(
            ops=openai_sdk_ops(model=MODEL, prompt=PROMPT, max_tokens=MAX_TOKENS),
            api_key=api_key,
            lib=client,
            scenario=SCENARIO,
            model=MODEL,
        )
    elif client == "litellm":
        result = run_sdk_sync(
            ops=litellm_openai_ops(model=MODEL, prompt=PROMPT, max_tokens=MAX_TOKENS),
            api_key=api_key,
            lib=client,
            scenario=SCENARIO,
            model=MODEL,
        )
    else:
        headers = [
            ("Authorization", f"Bearer {api_key}"),
            ("Content-Type", "application/json"),
            ("Accept", "text/event-stream"),
        ]
        body = json.dumps(_PAYLOAD).encode()
        result = RUNNERS[client](
            url=URL,
            headers=headers,
            body=body,
            parser=openai_chat_completions_parser,
            lib=client,
            scenario=SCENARIO,
            model=MODEL,
        )
    print(json.dumps(result))


if __name__ == "__main__":
    main()
