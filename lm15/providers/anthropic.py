from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
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
    UnsupportedModelError,
    canonical_error_code,
    map_http_error,
)
from ..features import EndpointSupport, ProviderManifest
from ..protocols import Capabilities
from ..sse import SSEEvent
from ..transports import Request as TransportRequest
from ..types import (
    BatchRequest,
    BatchResponse,
    BuiltinTool,
    CitationDelta,
    CitationPart,
    DocumentPart,
    ErrorDetail,
    FileUploadRequest,
    FileUploadResponse,
    FunctionTool,
    ImagePart,
    Message,
    Request,
    Response,
    StreamDeltaEvent,
    StreamEndEvent,
    StreamErrorEvent,
    StreamEvent,
    StreamStartEvent,
    TextDelta,
    TextPart,
    ThinkingDelta,
    ThinkingPart,
    ToolCallDelta,
    ToolCallPart,
    ToolResultPart,
    Usage,
)
from .base import BaseProviderLM, HttpResponse, SyncTransport, default_transport
from .common import anthropic_source, make_json_request, parts_to_text

# Canonical builtin tool name → Anthropic tool format
_ANTHROPIC_BUILTIN_MAP: dict[str, str] = {
    "web_search": "web_search_20250305",
    "code_execution": "code_execution_20250522",
}

ANTHROPIC_PROVIDER_EXECUTED_BLOCKS = {
    "server_tool_use",
    "web_search_tool_result",
    "code_execution_tool_result",
}


def _attach_unmapped(provider_data: dict[str, Any], unmapped: list[dict[str, str]]) -> dict[str, Any]:
    if not unmapped:
        return provider_data
    out = dict(provider_data)
    out["_lm15_unmapped"] = unmapped
    return out


def _record_unmapped(unmapped: list[dict[str, str]], path: str, typ: Any) -> None:
    unmapped.append({"path": path, "type": str(typ or "<missing>")})


def _builtin_to_anthropic(tool: BuiltinTool) -> dict[str, Any]:
    out: dict[str, Any] = {"type": _ANTHROPIC_BUILTIN_MAP.get(tool.name, tool.name), "name": tool.name}
    if tool.config:
        out.update(tool.config)
    return out


def _finish_reason(stop_reason: str | None, *, has_tool_call: bool = False) -> str:
    if has_tool_call:
        return "tool_call"
    reason = str(stop_reason or "").lower()
    if reason in {"max_tokens", "model_context_window_exceeded"}:
        return "length"
    if reason in {"tool_use", "pause_turn"}:
        return "tool_call"
    if reason in {"refusal", "safety", "content_filter"}:
        return "content_filter"
    return "stop"


def _batch_status(status: str) -> str:
    status = status.lower()
    if status in {"completed", "failed", "cancelled"}:
        return status
    if status in {"in_progress", "running", "processing"}:
        return "running"
    if status in {"queued", "validating"}:
        return "queued"
    return "submitted"


def _citation_from_anthropic(citation: dict[str, Any]) -> CitationPart | None:
    url = citation.get("url") or citation.get("uri")
    title = citation.get("title") or citation.get("document_title") or citation.get("source_title")
    text = citation.get("cited_text") or citation.get("text") or citation.get("quote")
    url_s = str(url) if url else None
    title_s = str(title) if title else None
    text_s = str(text) if text else None
    if url_s is None and title_s is None and text_s is None:
        return None
    return CitationPart(url=url_s, title=title_s, text=text_s)


@dataclass(slots=True)
class AnthropicLM(BaseProviderLM):
    api_key: str
    transport: SyncTransport = field(default_factory=default_transport)
    base_url: str = "https://api.anthropic.com/v1"
    api_version: str = "2023-06-01"

    provider: str = "anthropic"
    capabilities: Capabilities = Capabilities(
        input_modalities=frozenset({"text", "image", "document"}),
        output_modalities=frozenset({"text"}),
        features=frozenset({"streaming", "tools", "reasoning", "files", "batch"}),
    )
    supports: ClassVar[EndpointSupport] = EndpointSupport(
        complete=True, stream=True, files=True, batches=True
    )
    manifest: ClassVar[ProviderManifest] = ProviderManifest(
        provider="anthropic",
        supports=supports,
        auth_modes=("x-api-key",),
        env_keys=("ANTHROPIC_API_KEY",),
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
        lowered = msg.lower()
        return (
            "prompt is too long" in lowered
            or "too many tokens" in lowered
            or "context window" in lowered
            or "context length" in lowered
            or ("token" in lowered and ("limit" in lowered or "exceed" in lowered))
        )

    @staticmethod
    def _is_model_error(message: str) -> bool:
        lowered = message.lower()
        return "model" in lowered and any(
            marker in lowered
            for marker in (
                "not found",
                "does not exist",
                "not exist",
                "not supported",
                "unsupported",
                "not available",
                "unknown",
            )
        )

    def _error_detail(self, provider_code: str, message: str) -> ErrorDetail:
        cls = self._error_type_map.get(provider_code, ProviderError)
        if self._is_context_length_message(message):
            cls = ContextLengthError
        elif provider_code == "not_found_error" and self._is_model_error(message):
            cls = UnsupportedModelError
        return ErrorDetail(
            code=canonical_error_code(cls),
            message=message or provider_code or "provider error",
            provider_code=provider_code or "provider",
        )

    def normalize_error(self, status: int, body: str) -> ProviderError:
        try:
            data = json.loads(body)
            err = data.get("error", {}) if isinstance(data, dict) else {}
            msg = err.get("message", "") if isinstance(err, dict) else str(err)
            err_type = str(err.get("type") or "") if isinstance(err, dict) else ""
            request_id = str(data.get("request_id") or "") if isinstance(data, dict) else ""

            if self._is_context_length_message(msg):
                return self._provider_error(
                    ContextLengthError,
                    msg,
                    status=status,
                    provider_code=err_type or None,
                    request_id=request_id or None,
                )
            if err_type == "not_found_error" and self._is_model_error(msg):
                return self._provider_error(
                    UnsupportedModelError,
                    msg,
                    status=status,
                    provider_code=err_type,
                    request_id=request_id or None,
                )

            cls = self._error_type_map.get(err_type)
            if cls:
                return self._provider_error(
                    cls,
                    msg,
                    status=status,
                    provider_code=err_type or None,
                    request_id=request_id or None,
                )
            if err_type and err_type not in msg:
                msg = f"{msg} ({err_type})"
        except Exception:
            msg = body.strip()[:500] or f"HTTP {status}"
            err_type = ""
            request_id = ""
        return map_http_error(
            status,
            msg,
            provider=self.provider,
            env_keys=self.manifest.env_keys,
            provider_code=err_type or None,
            request_id=request_id or None,
        )

    def _headers(self, request: Request | None = None) -> dict[str, str]:
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": self.api_version,
            "content-type": "application/json",
        }
        if request is not None and any(
            isinstance(tool, BuiltinTool) and tool.name == "code_execution"
            for tool in request.tools
        ):
            headers["anthropic-beta"] = "code-execution-2025-05-22"
        return headers

    # ─── Request serialization ──────────────────────────────────────

    def _part(self, part) -> dict[str, Any]:
        if isinstance(part, TextPart):
            return {"type": "text", "text": part.text}
        if isinstance(part, ImagePart):
            return {"type": "image", "source": anthropic_source(part)}
        if isinstance(part, DocumentPart):
            return {"type": "document", "source": anthropic_source(part)}
        if isinstance(part, ToolCallPart):
            return {"type": "tool_use", "id": part.id, "name": part.name, "input": part.input}
        if isinstance(part, ToolResultPart):
            content_blocks = [self._tool_result_content(p) for p in part.content]
            out: dict[str, Any] = {"type": "tool_result", "tool_use_id": part.id}
            if content_blocks:
                # Anthropic accepts either a string or content blocks.  Blocks
                # preserve image/document tool outputs when present.
                if len(content_blocks) == 1 and content_blocks[0].get("type") == "text":
                    out["content"] = content_blocks[0]["text"]
                else:
                    out["content"] = content_blocks
            if part.is_error:
                out["is_error"] = True
            return out
        if isinstance(part, ThinkingPart):
            return {"type": "text", "text": part.text}
        return {"type": "text", "text": getattr(part, "text", "") or ""}

    def _tool_result_content(self, part) -> dict[str, Any]:
        if isinstance(part, TextPart):
            return {"type": "text", "text": part.text}
        if isinstance(part, ImagePart):
            return {"type": "image", "source": anthropic_source(part)}
        if isinstance(part, DocumentPart):
            return {"type": "document", "source": anthropic_source(part)}
        return {"type": "text", "text": getattr(part, "text", "") or ""}

    def _message(self, msg: Message) -> dict[str, Any]:
        role = "assistant" if msg.role == "assistant" else "user"
        parts = [self._part(part) for part in msg.parts]
        if msg.role == "developer":
            text = parts_to_text(msg.parts)
            parts = [{"type": "text", "text": f"[developer]\n{text}"}]
        return {"role": role, "content": parts}

    def _tool_choice_payload(self, request: Request) -> dict[str, Any] | None:
        tc = request.config.tool_choice
        if tc is None:
            return None
        
        payload: dict[str, Any] = {}
        if tc.mode == "none":
            payload["type"] = "none"
        elif tc.allowed:
            if len(tc.allowed) == 1:
                payload["type"] = "tool"
                payload["name"] = tc.allowed[0]
            else:
                payload["type"] = "any" if tc.mode == "required" else "auto"
        elif tc.mode == "required":
            payload["type"] = "any"
        else:
            payload["type"] = "auto"

        if tc.parallel is False and payload["type"] != "none":
            payload["disable_parallel_tool_use"] = True
            
        return payload

    def _payload(self, request: Request, stream: bool) -> dict[str, Any]:
        extensions = dict(request.config.extensions or {})
        prompt_caching = bool(extensions.get("prompt_caching"))

        messages = [self._message(m) for m in request.messages]
        if prompt_caching and len(messages) >= 2 and messages[-2].get("content"):
            messages[-2]["content"][-1].setdefault("cache_control", {"type": "ephemeral"})

        payload: dict[str, Any] = {
            "model": request.model,
            "messages": messages,
            "stream": stream,
            "max_tokens": request.config.max_tokens or 1024,
        }
        if request.system:
            system_text = request.system if isinstance(request.system, str) else parts_to_text(request.system)
            if prompt_caching:
                payload["system"] = [{"type": "text", "text": system_text, "cache_control": {"type": "ephemeral"}}]
            else:
                payload["system"] = system_text
        if request.config.temperature is not None:
            payload["temperature"] = request.config.temperature
        if request.config.top_p is not None:
            payload["top_p"] = request.config.top_p
        if request.config.top_k is not None:
            payload["top_k"] = request.config.top_k
        if request.config.stop:
            payload["stop_sequences"] = list(request.config.stop)
        if request.tools:
            tools_wire: list[dict[str, Any]] = []
            for tool in request.tools:
                if isinstance(tool, FunctionTool):
                    tools_wire.append({"name": tool.name, "description": tool.description, "input_schema": tool.parameters})
                elif isinstance(tool, BuiltinTool):
                    tools_wire.append(_builtin_to_anthropic(tool))
            payload["tools"] = tools_wire
        tool_choice = self._tool_choice_payload(request)
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice
        if request.config.reasoning and not request.config.reasoning.is_off:
            payload["thinking"] = {
                "type": "enabled",
                "budget_tokens": request.config.reasoning.thinking_budget or 1024,
            }
        if extensions:
            passthrough = {k: v for k, v in extensions.items() if k != "prompt_caching"}
            payload.update(passthrough)
        return payload

    def build_request(self, request: Request, stream: bool) -> TransportRequest:
        return make_json_request(
            method="POST",
            url=f"{self.base_url.rstrip('/')}/messages",
            headers=self._headers(request),
            payload=self._payload(request, stream=stream),
            read_timeout=120.0 if stream else 60.0,
        )

    # ─── Response parsing ───────────────────────────────────────────

    def parse_response(self, request: Request, response: HttpResponse) -> Response:
        data = response.json()
        parts: list[Any] = []
        unmapped: list[dict[str, str]] = []
        for block_index, block in enumerate(data.get("content", []) or []):
            if not isinstance(block, dict):
                _record_unmapped(unmapped, f"content[{block_index}]", type(block).__name__)
                continue
            block_type = block.get("type")
            if block_type == "text":
                parts.append(TextPart(text=str(block.get("text") or "")))
                for citation_payload in block.get("citations", []) or []:
                    if not isinstance(citation_payload, dict):
                        continue
                    citation = _citation_from_anthropic(citation_payload)
                    if citation is not None:
                        parts.append(citation)
            elif block_type == "tool_use":
                parts.append(ToolCallPart(
                    id=str(block.get("id") or f"tool_{len(parts)}"),
                    name=str(block.get("name") or "tool"),
                    input=block.get("input") if isinstance(block.get("input"), dict) else {},
                ))
            elif block_type == "thinking":
                parts.append(ThinkingPart(text=str(block.get("thinking") or block.get("text") or ""), redacted=False))
            elif block_type == "redacted_thinking":
                parts.append(ThinkingPart(text="[redacted]", redacted=True))
            elif block_type in ANTHROPIC_PROVIDER_EXECUTED_BLOCKS:
                continue
            else:
                _record_unmapped(unmapped, f"content[{block_index}]", block_type)

        if not parts:
            parts = [TextPart(text="")]

        usage_payload = data.get("usage", {}) or {}
        input_tokens = int(usage_payload.get("input_tokens", 0) or 0)
        output_tokens = int(usage_payload.get("output_tokens", 0) or 0)
        usage = Usage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            cache_read_tokens=usage_payload.get("cache_read_input_tokens"),
            cache_write_tokens=usage_payload.get("cache_creation_input_tokens"),
        )
        has_tool = any(isinstance(part, ToolCallPart) for part in parts)
        return Response(
            id=str(data.get("id")) if data.get("id") else None,
            model=str(data.get("model") or request.model),
            message=Message.assistant(tuple(parts)),
            finish_reason=_finish_reason(data.get("stop_reason"), has_tool_call=has_tool),
            usage=usage,
            provider_data=_attach_unmapped(data, unmapped),
        )

    def parse_stream_event(self, request: Request, raw_event: SSEEvent) -> StreamEvent | None:
        if not raw_event.data:
            return None
        payload = json.loads(raw_event.data)
        et = payload.get("type")
        if et == "message_start":
            msg = payload.get("message", {}) if isinstance(payload.get("message"), dict) else {}
            return StreamStartEvent(
                id=str(msg.get("id")) if msg.get("id") else None,
                model=str(msg.get("model") or request.model),
            )
        if et == "content_block_start":
            block = payload.get("content_block", {}) if isinstance(payload.get("content_block"), dict) else {}
            if block.get("type") == "tool_use":
                return StreamDeltaEvent(
                    delta=ToolCallDelta(
                        input=json.dumps(block.get("input", {}), separators=(",", ":")) if isinstance(block.get("input"), dict) else str(block.get("input") or ""),
                        part_index=int(payload.get("index", 0) or 0),
                        id=str(block.get("id") or "") or None,
                        name=str(block.get("name") or "") or None,
                    )
                )
            return None
        if et == "content_block_delta":
            delta = payload.get("delta", {}) if isinstance(payload.get("delta"), dict) else {}
            idx = int(payload.get("index", 0) or 0)
            dtype = delta.get("type")
            if dtype == "text_delta":
                return StreamDeltaEvent(delta=TextDelta(text=str(delta.get("text") or ""), part_index=idx))
            if dtype == "input_json_delta":
                return StreamDeltaEvent(delta=ToolCallDelta(input=str(delta.get("partial_json") or ""), part_index=idx))
            if dtype == "thinking_delta":
                return StreamDeltaEvent(delta=ThinkingDelta(text=str(delta.get("thinking") or ""), part_index=idx))
            if dtype in {"citation_delta", "citations_delta"}:
                citation = delta.get("citation", {}) if isinstance(delta.get("citation"), dict) else delta
                return StreamDeltaEvent(delta=CitationDelta(
                    part_index=idx,
                    text=str(citation.get("cited_text") or citation.get("text") or "") or None,
                    url=str(citation.get("url") or "") or None,
                    title=str(citation.get("title") or "") or None,
                ))
            return None
        if et == "message_stop":
            return StreamEndEvent(finish_reason="stop")
        if et == "error":
            err = payload.get("error")
            if isinstance(err, dict):
                provider_code = str(err.get("type") or err.get("code") or payload.get("code") or "provider")
                message = str(err.get("message") or payload.get("message") or "")
            else:
                provider_code = str(payload.get("code") or payload.get("error_type") or "provider")
                message = str(payload.get("message") or "")
            return StreamErrorEvent(error=self._error_detail(provider_code, message))
        return None

    # ─── Other endpoints ────────────────────────────────────────────

    def file_upload(self, request: FileUploadRequest) -> FileUploadResponse:
        req = TransportRequest(
            method="POST",
            url=f"{self.base_url.rstrip('/')}/files",
            headers=[
                ("x-api-key", self.api_key),
                ("anthropic-version", self.api_version),
                ("content-type", request.media_type),
                ("x-filename", request.filename),
            ],
            body=request.bytes,
            read_timeout=120.0,
        )
        resp = self._send(req)
        if resp.status >= 400:
            raise self.normalize_error(resp.status, resp.text())
        data = resp.json()
        file_id = data.get("id") or (data.get("file") or {}).get("id") or ""
        return FileUploadResponse(id=str(file_id), provider_data=data)

    def batch_submit(self, request: BatchRequest) -> BatchResponse:
        payload = {
            "requests": [
                {"custom_id": f"req_{i}", "params": self._payload(nested, stream=False)}
                for i, nested in enumerate(request.requests)
            ],
            **(request.extensions or {}),
        }
        resp = self._send(make_json_request(
            method="POST",
            url=f"{self.base_url.rstrip('/')}/messages/batches",
            headers=self._headers(),
            payload=payload,
            read_timeout=120.0,
        ))
        if resp.status >= 400:
            raise self.normalize_error(resp.status, resp.text())
        data = resp.json()
        batch_id = data.get("id") or f"batch_{uuid.uuid4().hex[:12]}"
        status = _batch_status(str(data.get("processing_status") or data.get("status") or "submitted"))
        return BatchResponse(id=str(batch_id), status=status, provider_data=data)
