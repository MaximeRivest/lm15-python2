"""Time-to-first-token against Groq's OpenAI-compatible chat completions.

Groq serves an OpenAI-compatible API at https://api.groq.com/openai/v1/...
so the request shape, auth scheme, and SSE format are identical to OpenAI's
/v1/chat/completions.  The only differences are the URL, the model name,
and the fact that Groq's LPU inference is dramatically faster at
time-to-first-token (often < 100 ms server-side, vs 300-500 ms on OpenAI).

Also supports the official `groq` Python SDK via the "groq-sdk" client
choice.  Note the groq SDK is a thin fork of the openai SDK, so its
startup/import cost will be comparable to openai-sdk.

Usage:
    GROQ_API_KEY=gsk-... python bench_groq_ttft.py <client>
"""

from __future__ import annotations

import json
import os
import sys

from _bench_lib import (
    RUNNERS,
    groq_sdk_ops,
    litellm_groq_ops,
    openai_chat_completions_parser,
    run_sdk_sync,
)


URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = os.environ.get("LM15_BENCH_GROQ_MODEL", "llama-3.1-8b-instant")
SCENARIO = "groq-chat-completions"
PROMPT = "Say 'hello' and nothing else."
MAX_TOKENS = 20

_PAYLOAD = {
    "model": MODEL,
    "messages": [{"role": "user", "content": PROMPT}],
    "stream": True,
    "max_completion_tokens": MAX_TOKENS,
}

ALL_CLIENTS = list(RUNNERS) + ["groq-sdk", "litellm"]


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in ALL_CLIENTS:
        print(f"usage: {sys.argv[0]} <{ '|'.join(ALL_CLIENTS) }>", file=sys.stderr)
        sys.exit(2)

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("GROQ_API_KEY not set", file=sys.stderr)
        sys.exit(2)

    client = sys.argv[1]

    if client == "groq-sdk":
        result = run_sdk_sync(
            ops=groq_sdk_ops(model=MODEL, prompt=PROMPT, max_tokens=MAX_TOKENS),
            api_key=api_key,
            lib=client,
            scenario=SCENARIO,
            model=MODEL,
        )
    elif client == "litellm":
        result = run_sdk_sync(
            ops=litellm_groq_ops(model=MODEL, prompt=PROMPT, max_tokens=MAX_TOKENS),
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
