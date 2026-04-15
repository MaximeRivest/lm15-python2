"""
lm15.providers.anthropic — Anthropic Messages API adapter.

Translates between universal Request/Response and Anthropic's wire format.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from typing import Any, ClassVar

from ..errors import (
    AuthError,
    BillingError,
    ContextLengthError,
    InvalidRequestError,
    ProviderError,
    RateLimitError,
    ServerError,
    TimeoutError,
    canonical_error_code,
    map_http_error,
)
from ..features import EndpointSupport, ProviderManifest
from ..protocols import Capabilities
from ..sse import SSEEvent
from ..transports.base import HttpRequest, HttpResponse, Transport
from ..types import (
    BatchRequest,
    BatchResponse,
    Delta,
    FileUploadRequest,
    FileUploadResponse,
    Message,
    Part,
    Request,
    Response,
    StreamEvent,
    ErrorDetail,
    TextPart,
    ThinkingPart,
    ImagePart,
    AudioPart,
    DocumentPart,
    ToolCallPart,
    ToolResultPart,
    Usage,
    text,
    tool_call,
)
from .base import BaseProviderAdapter
from .common import parts_to_text, source_to_anthropic

# Canonical builtin tool name → Anthropic wire type
_ANTHROPIC_BUILTIN_MAP: dict[str, str] = {
    "web_search": "web_search_20250305",
    "code_execution": "code_execution_20250522",
}


def _builtin_to_anthropic(tool: Any) -> dict[str, Any]:
    wire_type = _ANTHROPIC_BUILTIN_MAP.get(tool.name)
    if wire_type:
        out: dict[str, Any] = {"type": wire_type, "name": tool.name}
        if tool.config:
            out.update(tool.config)
        return out
    out = {"type": tool.name, "name": tool.name}
    if tool.config:
        out.update(tool.config)
    return out


@dataclass(slots=True)
class AnthropicAdapter(BaseProviderAdapter):
    api_key: str
    transport: Transport
    base_url: str = "https://api.anthropic.com/v1"
    api_version: str = "2023-06-01"

    provider: str = "anthropic"
    capabilities: Capabilities = Capabilities(
        input_modalities=frozenset({"text", "image", "document"}),
        output_modalities=frozenset({"text"}),
        features=frozenset({"streaming", "tools", "reasoning", "files", "batch"}),
    )
    supports: ClassVar[EndpointSupport] = EndpointSupport(complete=True, stream=True, files=True, batches=True)
    manifest: ClassVar[ProviderManifest] = ProviderManifest(
        provider="anthropic", supports=supports,
        auth_modes=("x-api-key",), env_keys=("ANTHROPIC_API_KEY",),
    )

    _error_type_map: ClassVar[dict[str, type[ProviderError]]] = {
        "authentication_error": AuthError,
        "permission_error": AuthError,
        "billing_error": BillingError,
        "rate_limit_error": RateLimitError,
        "request_too_large": InvalidRequestError,
        "not_found_error": InvalidRequestError,
        "invalid_request_error": InvalidRequestError,
        "api_error": ServerError,
        "overloaded_error": ServerError,
        "timeout_error": TimeoutError,
    }

    @staticmethod
    def _is_context_length_message(msg: str) -> bool:
        m = msg.lower()
        return (
            "prompt is too long" in m
            or "too many tokens" in m
            or "context window" in m
            or "context length" in m
            or ("token" in m and ("limit" in m or "exceed" in m))
        )

    def _make_error_detail(self, provider_code: str, message: str) -> ErrorDetail:
        cls = self._error_type_map.get(provider_code, ProviderError)
        if provider_code == "invalid_request_error" and self._is_context_length_message(message):
            cls = ContextLengthError
        return ErrorDetail(
            code=canonical_error_code(cls),
            message=message,
            provider_code=provider_code or "provider",
        )

    def normalize_error(self, status: int, body: str) -> ProviderError:
        try:
            data = json.loads(body)
            err = data.get("error", {})
            msg = err.get("message", "") if isinstance(err, dict) else str(err)
            err_type = err.get("type", "") if isinstance(err, dict) else ""
            request_id = data.get("request_id", "") if isinstance(data, dict) else ""

            if err_type == "invalid_request_error" and self._is_context_length_message(msg):
                if request_id and request_id not in msg:
                    msg = f"{msg} (request_id={request_id})"
                return ContextLengthError(msg)

            cls = self._error_type_map.get(err_type)
            if cls:
                if request_id and request_id not in msg:
                    msg = f"{msg} (request_id={request_id})"
                return cls(msg)

            if err_type and err_type not in msg:
                msg = f"{msg} ({err_type})"
            if request_id and request_id not in msg:
                msg = f"{msg} (request_id={request_id})"
        except Exception:
            msg = body.strip()[:200] or f"HTTP {status}"
        return map_http_error(status, msg)

    def _headers(self) -> dict[str, str]:
        return {
            "x-api-key": self.api_key,
            "anthropic-version": self.api_version,
            "content-type": "application/json",
        }

    def _part_to_wire(self, p: Part) -> dict[str, Any]:
        """Convert a Part to Anthropic content block format."""
        if isinstance(p, TextPart):
            out: dict[str, Any] = {"type": "text", "text": p.text}
        elif isinstance(p, ImagePart):
            out = {"type": "image", "source": source_to_anthropic(p.source)}
        elif isinstance(p, DocumentPart):
            out = {"type": "document", "source": source_to_anthropic(p.source)}
        elif isinstance(p, ToolCallPart):
            out = {"type": "tool_use", "id": p.id, "name": p.name, "input": p.input}
        elif isinstance(p, ToolResultPart):
            content_text = parts_to_text(p.content) if p.content else ""
            out = {"type": "tool_result", "tool_use_id": p.id}
            if content_text:
                out["content"] = content_text
            if p.is_error:
                out["is_error"] = True
        else:
            # Fallback: extract text if possible
            out = {"type": "text", "text": getattr(p, "text", "")}

        # Cache control from metadata
        metadata = getattr(p, "metadata", None)
        cache_meta = (metadata or {}).get("cache") if metadata else None
        if cache_meta is True:
            out["cache_control"] = {"type": "ephemeral"}
        elif isinstance(cache_meta, dict):
            out["cache_control"] = cache_meta

        return out

    def _payload(self, request: Request, stream: bool) -> dict[str, Any]:
        ext = request.config.extensions or {}
        prompt_caching = bool(ext.get("prompt_caching"))

        messages = [
            {
                "role": "user" if m.role == "tool" else m.role,
                "content": [self._part_to_wire(p) for p in m.parts],
            }
            for m in request.messages
        ]

        if prompt_caching and len(messages) >= 2 and messages[-2].get("content"):
            messages[-2]["content"][-1].setdefault("cache_control", {"type": "ephemeral"})

        payload: dict[str, Any] = {
            "model": request.model,
            "messages": messages,
            "stream": stream,
            "max_tokens": request.config.max_tokens or 1024,
        }

        if request.system:
            if isinstance(request.system, str):
                if prompt_caching:
                    payload["system"] = [{"type": "text", "text": request.system, "cache_control": {"type": "ephemeral"}}]
                else:
                    payload["system"] = request.system
            else:
                payload["system"] = parts_to_text(tuple(request.system))

        if request.config.temperature is not None:
            payload["temperature"] = request.config.temperature

        if request.tools:
            tools_wire: list[dict[str, Any]] = []
            for t in request.tools:
                if t.type == "function":
                    tools_wire.append({
                        "name": t.name,
                        "description": t.description,
                        "input_schema": t.parameters or {"type": "object", "properties": {}},
                    })
                elif t.type == "builtin":
                    tools_wire.append(_builtin_to_anthropic(t))
            payload["tools"] = tools_wire

        if request.config.reasoning and request.config.reasoning.enabled:
            payload["thinking"] = {
                "type": "enabled",
                "budget_tokens": request.config.reasoning.budget or 1024,
            }

        # Provider-specific passthrough
        if ext:
            passthrough = {k: v for k, v in ext.items() if k != "prompt_caching"}
            payload.update(passthrough)

        return payload

    def build_request(self, request: Request, stream: bool) -> HttpRequest:
        return HttpRequest(
            method="POST",
            url=f"{self.base_url}/messages",
            headers=self._headers(),
            json_body=self._payload(request, stream=stream),
            timeout=120.0 if stream else 60.0,
        )

    def parse_response(self, request: Request, response: HttpResponse) -> Response:
        data = response.json()
        parts: list[Part] = []

        for block in data.get("content", []):
            bt = block.get("type")
            if bt == "text":
                parts.append(TextPart(text=block.get("text", "")))
            elif bt == "tool_use":
                parts.append(ToolCallPart(
                    id=block.get("id", ""),
                    name=block.get("name", ""),
                    input=block.get("input", {}),
                ))
            elif bt == "thinking":
                parts.append(ThinkingPart(text=block.get("thinking", ""), redacted=False))
            elif bt == "redacted_thinking":
                parts.append(ThinkingPart(text="[redacted]", redacted=True))

        finish = "tool_call" if any(isinstance(p, ToolCallPart) for p in parts) else "stop"
        u = data.get("usage", {})
        usage = Usage(
            input_tokens=u.get("input_tokens", 0),
            output_tokens=u.get("output_tokens", 0),
            total_tokens=u.get("input_tokens", 0) + u.get("output_tokens", 0),
            cache_read_tokens=u.get("cache_read_input_tokens"),
            cache_write_tokens=u.get("cache_creation_input_tokens"),
        )

        return Response(
            id=data.get("id", ""),
            model=data.get("model", request.model),
            message=Message(role="assistant", parts=tuple(parts or [TextPart(text="")])),
            finish_reason=finish,
            usage=usage,
            provider_data=data,
        )

    def parse_stream_event(self, request: Request, raw_event: SSEEvent) -> StreamEvent | None:
        if not raw_event.data:
            return None

        payload = json.loads(raw_event.data)
        et = payload.get("type")

        if et == "message_start":
            msg = payload.get("message", {})
            return StreamEvent(type="start", id=msg.get("id"), model=msg.get("model"))

        if et == "content_block_start":
            block = payload.get("content_block", {})
            idx = payload.get("index", 0)
            if block.get("type") == "tool_use":
                return StreamEvent(
                    type="delta",
                    delta=Delta(
                        type="tool_call",
                        part_index=idx,
                        id=block.get("id"),
                        name=block.get("name"),
                        input=json.dumps(block.get("input", {})) if isinstance(block.get("input"), dict) else (block.get("input") or ""),
                    ),
                )
            return None

        if et == "content_block_delta":
            delta = payload.get("delta", {})
            idx = payload.get("index", 0)
            if delta.get("type") == "text_delta":
                return StreamEvent(type="delta", delta=Delta(type="text", part_index=idx, text=delta.get("text", "")))
            if delta.get("type") == "input_json_delta":
                return StreamEvent(type="delta", delta=Delta(type="tool_call", part_index=idx, input=delta.get("partial_json", "")))
            if delta.get("type") == "thinking_delta":
                return StreamEvent(type="delta", delta=Delta(type="thinking", part_index=idx, text=delta.get("thinking", "")))
            return None

        if et == "content_block_stop":
            return None

        if et == "message_stop":
            return StreamEvent(type="end", finish_reason="stop")

        if et == "message_delta":
            u = payload.get("usage", {})
            usage = Usage(
                input_tokens=u.get("input_tokens", 0),
                output_tokens=u.get("output_tokens", 0),
                total_tokens=u.get("input_tokens", 0) + u.get("output_tokens", 0),
            ) if u else None
            return StreamEvent(type="end", finish_reason="stop", usage=usage)

        if et == "error":
            e = payload.get("error")
            if isinstance(e, dict):
                provider_code = str(e.get("type") or e.get("code") or "provider")
                message = str(e.get("message") or "")
            else:
                provider_code = str(payload.get("code") or "provider")
                message = str(payload.get("message") or "")
            return StreamEvent(type="error", error=self._make_error_detail(provider_code, message))

        return None

    def file_upload(self, request: FileUploadRequest) -> FileUploadResponse:
        req = HttpRequest(
            method="POST",
            url=f"{self.base_url}/files",
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": self.api_version,
                "content-type": request.media_type,
                "x-filename": request.filename,
            },
            body=request.bytes_data,
            timeout=120.0,
        )
        resp = self.transport.request(req)
        if resp.status >= 400:
            raise self.normalize_error(resp.status, resp.text())
        data = resp.json()
        file_id = data.get("id") or (data.get("file") or {}).get("id") or ""
        return FileUploadResponse(id=file_id, provider_data=data)

    def batch_submit(self, request: BatchRequest) -> BatchResponse:
        payload: dict[str, Any] = {
            "requests": [
                {"custom_id": f"req_{i}", "params": self._payload(r, stream=False)}
                for i, r in enumerate(request.requests)
            ],
            **(request.extensions or {}),
        }
        req = HttpRequest(
            method="POST",
            url=f"{self.base_url}/messages/batches",
            headers=self._headers(),
            json_body=payload,
            timeout=120.0,
        )
        resp = self.transport.request(req)
        if resp.status >= 400:
            raise self.normalize_error(resp.status, resp.text())
        data = resp.json()
        batch_id = data.get("id") or f"batch_{uuid.uuid4().hex[:12]}"
        status = data.get("processing_status") or data.get("status") or "submitted"
        return BatchResponse(id=batch_id, status=status, provider_data=data)
