"""Time-to-first-token against Gemini's streamGenerateContent endpoint.

Mirrors bench_openai_ttft.py but for Google's Gemini API.  Key protocol
differences from OpenAI:

    * Authentication: ?key=<API_KEY> query parameter (no Authorization header)
    * URL must include ?alt=sse for Server-Sent Events framing; without it
      Google streams a JSON array (one element per chunk) which is harder
      to parse incrementally.
    * Payload uses `contents` + `generationConfig` schema, not messages.
    * First token lives at candidates[0].content.parts[N].text.  For
      thinking models, the first chunk may contain a thoughtSignature with
      empty text \u2014 we skip those and wait for real content.
    * No explicit [DONE] terminator; stream ends at connection close or
      when a chunk carries `finishReason`.

Also supports the official `google-genai` Python SDK via the "genai-sdk"
client choice.

Usage:
    GEMINI_API_KEY=... python bench_gemini_ttft.py <client>
"""

from __future__ import annotations

import json
import os
import sys

from _bench_lib import (
    RUNNERS,
    gemini_parser,
    genai_sdk_ops,
    litellm_gemini_ops,
    run_sdk_sync,
)


MODEL = os.environ.get("LM15_BENCH_GEMINI_MODEL", "gemini-3.1-flash-lite-preview")
SCENARIO = "gemini-stream-generate-content"
PROMPT = "Say 'hello' and nothing else."

_PAYLOAD = {
    "contents": [{"role": "user", "parts": [{"text": PROMPT}]}],
    "generationConfig": {"thinkingConfig": {"thinkingLevel": "MINIMAL"}},
}

ALL_CLIENTS = list(RUNNERS) + ["genai-sdk", "litellm"]


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in ALL_CLIENTS:
        print(f"usage: {sys.argv[0]} <{ '|'.join(ALL_CLIENTS) }>", file=sys.stderr)
        sys.exit(2)

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("GEMINI_API_KEY not set", file=sys.stderr)
        sys.exit(2)

    client = sys.argv[1]

    if client == "genai-sdk":
        result = run_sdk_sync(
            ops=genai_sdk_ops(model=MODEL, prompt=PROMPT),
            api_key=api_key,
            lib=client,
            scenario=SCENARIO,
            model=MODEL,
        )
    elif client == "litellm":
        result = run_sdk_sync(
            ops=litellm_gemini_ops(model=MODEL, prompt=PROMPT),
            api_key=api_key,
            lib=client,
            scenario=SCENARIO,
            model=MODEL,
        )
    else:
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{MODEL}:streamGenerateContent?alt=sse&key={api_key}"
        )
        headers = [
            ("Content-Type", "application/json"),
            ("Accept", "text/event-stream"),
        ]
        body = json.dumps(_PAYLOAD).encode()
        result = RUNNERS[client](
            url=url,
            headers=headers,
            body=body,
            parser=gemini_parser,
            lib=client,
            scenario=SCENARIO,
            model=MODEL,
        )
    print(json.dumps(result))


if __name__ == "__main__":
    main()
