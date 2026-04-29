#!/usr/bin/env python3
"""Validate canonical JSON serde fixtures and protobuf roundtrips."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lm15 import protobuf as pbconv  # noqa: E402
from lm15 import serde  # noqa: E402
from lm15.types import (  # noqa: E402
    AudioGenerationRequest,
    AudioGenerationResponse,
    AudioPart,
    BatchRequest,
    BatchResponse,
    EmbeddingRequest,
    EmbeddingResponse,
    FileUploadRequest,
    FileUploadResponse,
    ImageGenerationRequest,
    ImageGenerationResponse,
    ImagePart,
    Message,
    Request,
    Response,
    TextPart,
    Usage,
)

FIXTURE_PATH = ROOT / "serde" / "canonical.json"
REPORT_DIR = ROOT / "reports"
JsonObject = dict[str, Any]


@dataclass(frozen=True)
class SerdeResult:
    case_id: str
    kind: str
    check: str
    status: str
    reason: str | None = None


JsonToObj = Callable[[JsonObject], Any]
ObjToJson = Callable[[Any], JsonObject]

KIND_SERDE: dict[str, tuple[JsonToObj, ObjToJson]] = {
    "part": (serde.part_from_dict, serde.part_to_dict),
    "message": (serde.message_from_dict, serde.message_to_dict),
    "tool": (serde.tool_from_dict, serde.tool_to_dict),
    "tool_choice": (serde.tool_choice_from_dict, serde.tool_choice_to_dict),
    "reasoning": (serde.reasoning_from_dict, serde.reasoning_to_dict),
    "config": (serde.config_from_dict, serde.config_to_dict),
    "error_detail": (serde.error_detail_from_dict, serde.error_detail_to_dict),
    "delta": (serde.delta_from_dict, serde.delta_to_dict),
    "usage": (serde.usage_from_dict, serde.usage_to_dict),
    "stream_event": (serde.stream_event_from_dict, serde.stream_event_to_dict),
    "request": (serde.request_from_dict, serde.request_to_dict),
    "response": (serde.response_from_dict, serde.response_to_dict),
    "audio_format": (serde.audio_format_from_dict, serde.audio_format_to_dict),
    "live_config": (serde.live_config_from_dict, serde.live_config_to_dict),
    "live_client_event": (serde.live_client_event_from_dict, serde.live_client_event_to_dict),
    "live_server_event": (serde.live_server_event_from_dict, serde.live_server_event_to_dict),
}

PROTO_KINDS = set(KIND_SERDE)


def load_cases() -> list[JsonObject]:
    return list(json.loads(FIXTURE_PATH.read_text()).get("cases", []))


def clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        out = {k: clean_json(v) for k, v in value.items()}
        return {k: v for k, v in out.items() if v not in (None, "", [], {})}
    if isinstance(value, list):
        return [clean_json(v) for v in value]
    return value


def check_json_case(case: JsonObject) -> tuple[SerdeResult, Any | None]:
    case_id = str(case["id"])
    kind = str(case["kind"])
    value = case["value"]
    if kind not in KIND_SERDE:
        return SerdeResult(case_id, kind, "json", "error", f"unknown kind: {kind}"), None
    from_dict, to_dict = KIND_SERDE[kind]
    try:
        obj = from_dict(value)
        got = clean_json(to_dict(obj))
        expected = clean_json(value)
    except Exception as exc:
        return SerdeResult(case_id, kind, "json", "error", str(exc)), None
    if got != expected:
        return SerdeResult(case_id, kind, "json", "fail", f"expected {expected!r}, got {got!r}"), obj
    return SerdeResult(case_id, kind, "json", "pass"), obj


def check_proto_case(case: JsonObject, obj: Any) -> SerdeResult:
    case_id = str(case["id"])
    kind = str(case["kind"])
    if kind not in PROTO_KINDS:
        return SerdeResult(case_id, kind, "protobuf", "skip", "kind has no protobuf mapping")
    try:
        message = pbconv.to_proto(obj)
        out = pbconv.from_proto(message)
    except Exception as exc:
        return SerdeResult(case_id, kind, "protobuf", "error", str(exc))
    if out != obj:
        return SerdeResult(case_id, kind, "protobuf", "fail", f"expected {obj!r}, got {out!r}")
    return SerdeResult(case_id, kind, "protobuf", "pass")


def endpoint_samples() -> list[tuple[str, Any]]:
    req = Request(model="m", messages=(Message.user("hello"),))
    resp = Response(
        id="resp_1",
        model="m",
        message=Message.assistant("hello"),
        finish_reason="stop",
        usage=Usage(input_tokens=1, output_tokens=1, total_tokens=2),
    )
    return [
        ("endpoint.embedding_request", EmbeddingRequest(model="embed", inputs=("hello", "world"), extensions={"task": "search"})),
        ("endpoint.embedding_response", EmbeddingResponse(model="embed", vectors=((0.1, 0.2),), usage=Usage(input_tokens=1, total_tokens=1))),
        ("endpoint.file_upload_request", FileUploadRequest(filename="hello.txt", bytes_data=b"hello", media_type="text/plain", model="m")),
        ("endpoint.file_upload_response", FileUploadResponse(id="file_1", provider_data={"purpose": "test"})),
        ("endpoint.batch_request", BatchRequest(model="m", requests=(req,), extensions={"completion_window": "24h"})),
        ("endpoint.batch_response", BatchResponse(id="batch_1", status="completed", provider_data={"n": 1})),
        ("endpoint.image_generation_request", ImageGenerationRequest(model="img", prompt="a cat", size="1024x1024")),
        ("endpoint.image_generation_response", ImageGenerationResponse(images=(ImagePart(media_type="image/png", data="aGk="),), id="img_1", model="img")),
        ("endpoint.audio_generation_request", AudioGenerationRequest(model="tts", prompt="hello", voice="alloy", format="wav")),
        ("endpoint.audio_generation_response", AudioGenerationResponse(audio=AudioPart(media_type="audio/wav", data="aGk="), id="aud_1", model="tts")),
        ("endpoint.request_wrapper", req),
        ("endpoint.response_wrapper", resp),
        ("endpoint.text_part", TextPart("hello")),
    ]


def check_endpoint_proto_samples() -> list[SerdeResult]:
    results: list[SerdeResult] = []
    for case_id, obj in endpoint_samples():
        try:
            message = pbconv.to_proto(obj)
            out = pbconv.from_proto(message)
        except Exception as exc:
            results.append(SerdeResult(case_id, "endpoint", "protobuf", "error", str(exc)))
            continue
        if out != obj:
            results.append(SerdeResult(case_id, "endpoint", "protobuf", "fail", f"expected {obj!r}, got {out!r}"))
        else:
            results.append(SerdeResult(case_id, "endpoint", "protobuf", "pass"))
    return results


def result_to_dict(result: SerdeResult) -> JsonObject:
    return {
        "id": result.case_id,
        "kind": result.kind,
        "check": result.check,
        "status": result.status,
        "reason": result.reason,
    }


def write_markdown(results: list[SerdeResult], path: Path) -> None:
    counts: JsonObject = {}
    for result in results:
        key = f"{result.check}:{result.status}"
        counts[key] = counts.get(key, 0) + 1
    lines = [
        "# Serde fixture conformance",
        "",
        "Generated by `conformance/check_serde_fixtures.py`.",
        "",
        "## Summary",
        "",
        "| Check/status | Count |",
        "|---|---:|",
    ]
    for key in sorted(counts):
        lines.append(f"| {key} | {counts[key]} |")
    lines.extend(["", "## Cases", "", "| Case | Kind | Check | Status | Reason |", "|---|---|---|---|---|"])
    for result in results:
        reason = (result.reason or "").replace("|", "\\|")
        lines.append(f"| `{result.case_id}` | {result.kind} | {result.check} | {result.status} | {reason} |")
    path.write_text("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--json", type=Path)
    parser.add_argument("--markdown", type=Path)
    args = parser.parse_args(argv)

    results: list[SerdeResult] = []
    for case in load_cases():
        json_result, obj = check_json_case(case)
        results.append(json_result)
        if json_result.status == "pass" and obj is not None:
            results.append(check_proto_case(case, obj))
    results.extend(check_endpoint_proto_samples())

    counts: JsonObject = {}
    for result in results:
        counts[result.status] = counts.get(result.status, 0) + 1

    REPORT_DIR.mkdir(exist_ok=True)
    json_path = args.json or REPORT_DIR / "serde-fixtures.json"
    md_path = args.markdown or REPORT_DIR / "serde-fixtures.md"
    json_path.write_text(json.dumps({"summary": counts, "total": len(results), "results": [result_to_dict(r) for r in results]}, indent=2, sort_keys=True) + "\n")
    write_markdown(results, md_path)

    print(f"serde fixture conformance: {counts} / total={len(results)}")
    print(f"json: {json_path}")
    print(f"markdown: {md_path}")

    if args.strict and any(result.status not in {"pass", "skip"} for result in results):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
