"""
lm15.providers.gemini — Google Gemini API adapter.

Translates between universal Request/Response and Gemini's wire format.
Supports HTTP streaming and WebSocket (live/realtime) transport.
"""

from __future__ import annotations

import base64 as _base64
import hashlib
import json
import struct as _struct
import urllib.parse
import uuid
from dataclasses import dataclass, field
from typing import Any, ClassVar, Iterator

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
from ..live import WebSocketLiveSession, require_websocket_sync_connect
from ..protocols import Capabilities
from ..sse import SSEEvent
from ..transports.base import HttpRequest, HttpResponse, Transport
from ..types import (
    AudioGenerationRequest,
    AudioGenerationResponse,
    AudioPart,
    BatchRequest,
    BatchResponse,
    Config,
    Delta,
    DocumentPart,
    EmbeddingRequest,
    EmbeddingResponse,
    ErrorDetail,
    FileUploadRequest,
    FileUploadResponse,
    ImageGenerationRequest,
    ImageGenerationResponse,
    ImagePart,
    LiveClientEvent,
    LiveConfig,
    LiveServerEvent,
    Message,
    Part,
    Request,
    Response,
    Source,
    StreamEvent,
    TextPart,
    ToolCallPart,
    ToolResultPart,
    Usage,
    VideoPart,
)
from .base import BaseProviderAdapter

# Canonical builtin tool name → Gemini wire key
_GEMINI_BUILTIN_MAP: dict[str, str] = {
    "web_search": "googleSearch",
    "code_execution": "codeExecution",
}


def _builtin_to_gemini(tool: Any) -> dict[str, Any]:
    wire_key = _GEMINI_BUILTIN_MAP.get(tool.name)
    if wire_key:
        return {wire_key: tool.config or {}}
    return {tool.name: tool.config or {}}


@dataclass(slots=True)
class GeminiAdapter(BaseProviderAdapter):
    api_key: str
    transport: Transport
    base_url: str = "https://generativelanguage.googleapis.com/v1beta"
    upload_base_url: str = "https://generativelanguage.googleapis.com/upload/v1beta"
    _cached_content_ids: dict[str, str] = field(default_factory=dict, repr=False)

    provider: str = "gemini"
    capabilities: Capabilities = Capabilities(
        input_modalities=frozenset({"text", "image", "audio", "video", "document"}),
        output_modalities=frozenset({"text", "image", "audio"}),
        features=frozenset({"streaming", "tools", "json_output", "live", "embeddings", "files", "batch", "images", "audio"}),
    )
    supports: ClassVar[EndpointSupport] = EndpointSupport(
        complete=True, stream=True, live=True, embeddings=True,
        files=True, batches=True, images=True, audio=True,
    )
    manifest: ClassVar[ProviderManifest] = ProviderManifest(
        provider="gemini", supports=supports,
        auth_modes=("query-api-key", "bearer"),
        env_keys=("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    )

    _error_status_map: ClassVar[dict[str, type[ProviderError]]] = {
        "INVALID_ARGUMENT": InvalidRequestError,
        "FAILED_PRECONDITION": BillingError,
        "PERMISSION_DENIED": AuthError,
        "NOT_FOUND": InvalidRequestError,
        "RESOURCE_EXHAUSTED": RateLimitError,
        "INTERNAL": ServerError,
        "UNAVAILABLE": ServerError,
        "DEADLINE_EXCEEDED": TimeoutError,
    }

    @staticmethod
    def _is_context_length_message(msg: str) -> bool:
        m = msg.lower()
        return (
            ("token" in m and ("limit" in m or "exceed" in m))
            or "too long" in m
            or "context is too long" in m
            or "context length" in m
        )

    def _make_error_detail(self, provider_code: str, message: str) -> ErrorDetail:
        cls = self._error_status_map.get(provider_code, ProviderError)
        if self._is_context_length_message(message):
            cls = ContextLengthError
        return ErrorDetail(
            code=canonical_error_code(cls),
            message=message,
            provider_code=provider_code or "provider",
        )

    @staticmethod
    def _is_candidate_finish_error(finish_reason: str) -> bool:
        return finish_reason in {
            "SAFETY", "RECITATION", "LANGUAGE", "BLOCKLIST",
            "PROHIBITED_CONTENT", "SPII", "MALFORMED_FUNCTION_CALL",
            "IMAGE_SAFETY", "IMAGE_PROHIBITED_CONTENT", "IMAGE_OTHER",
            "NO_IMAGE", "IMAGE_RECITATION", "UNEXPECTED_TOOL_CALL",
            "TOO_MANY_TOOL_CALLS", "MISSING_THOUGHT_SIGNATURE",
            "MALFORMED_RESPONSE",
        }

    def _inband_error(self, data: dict[str, Any]) -> ProviderError | None:
        prompt_feedback = data.get("promptFeedback")
        if isinstance(prompt_feedback, dict):
            block_reason = str(prompt_feedback.get("blockReason") or "")
            if block_reason and block_reason != "BLOCK_REASON_UNSPECIFIED":
                return InvalidRequestError(f"Prompt blocked: {block_reason}")

        candidate = (data.get("candidates") or [{}])[0]
        finish_reason = str(candidate.get("finishReason") or "")
        if self._is_candidate_finish_error(finish_reason):
            finish_message = str(candidate.get("finishMessage") or "")
            return InvalidRequestError(finish_message or f"Candidate blocked: {finish_reason}")
        return None

    def normalize_error(self, status: int, body: str) -> ProviderError:
        try:
            data = json.loads(body)
            err = data.get("error", {})
            msg = err.get("message", "") if isinstance(err, dict) else str(err)
            err_status = err.get("status", "") if isinstance(err, dict) else ""

            if self._is_context_length_message(msg):
                return ContextLengthError(msg)

            cls = self._error_status_map.get(err_status)
            if cls:
                return cls(msg)

            if err_status and err_status not in msg:
                msg = f"{msg} ({err_status})"
        except Exception:
            msg = body.strip()[:200] or f"HTTP {status}"
        return map_http_error(status, msg)

    def _model_path(self, model: str) -> str:
        return model if model.startswith("models/") else f"models/{model}"

    def _part_to_wire(self, p: Part) -> dict[str, Any]:
        """Convert a Part to Gemini content part format."""
        if isinstance(p, TextPart):
            return {"text": p.text}
        if isinstance(p, (ImagePart, AudioPart, VideoPart, DocumentPart)):
            mime = p.source.media_type or "application/octet-stream"
            if p.source.type == "url":
                return {"fileData": {"mimeType": mime, "fileUri": p.source.url}}
            if p.source.type == "base64":
                return {"inlineData": {"mimeType": mime, "data": p.source.data}}
            if p.source.type == "file":
                return {"fileData": {"mimeType": mime, "fileUri": p.source.file_id}}
        if isinstance(p, ToolCallPart):
            out: dict[str, Any] = {"functionCall": {"name": p.name, "args": p.input}}
            if p.id:
                out["functionCall"]["id"] = p.id
            return out
        if isinstance(p, ToolResultPart):
            result_text = "".join(
                x.text for x in p.content if isinstance(x, TextPart) and x.text
            )
            fr: dict[str, Any] = {"name": p.name or "tool", "response": {"result": result_text}}
            if p.id:
                fr["id"] = p.id
            return {"functionResponse": fr}
        # Fallback
        return {"text": getattr(p, "text", "")}

    def _payload(self, request: Request) -> dict[str, Any]:
        ext = request.config.extensions or {}
        prompt_caching = bool(ext.get("prompt_caching"))

        payload: dict[str, Any] = {
            "contents": [
                {
                    "role": "model" if m.role == "assistant" else "user",
                    "parts": [self._part_to_wire(p) for p in m.parts],
                }
                for m in request.messages
            ]
        }

        if request.system:
            sys_text = request.system if isinstance(request.system, str) else "\n".join(
                p.text for p in request.system if isinstance(p, TextPart) and p.text
            )
            payload["systemInstruction"] = {"parts": [{"text": sys_text}]}

        cfg: dict[str, Any] = {}
        if request.config.temperature is not None:
            cfg["temperature"] = request.config.temperature
        if request.config.max_tokens is not None:
            cfg["maxOutputTokens"] = request.config.max_tokens
        if request.config.stop:
            cfg["stopSequences"] = list(request.config.stop)
        if request.config.response_format:
            cfg.update(request.config.response_format)
        if cfg:
            payload["generationConfig"] = cfg

        if request.tools:
            func_decls = [
                {"name": t.name, "description": t.description,
                 "parameters": t.parameters or {"type": "OBJECT", "properties": {}}}
                for t in request.tools if t.type == "function"
            ]
            tools_wire: list[dict[str, Any]] = []
            if func_decls:
                tools_wire.append({"functionDeclarations": func_decls})
            for t in request.tools:
                if t.type == "builtin":
                    tools_wire.append(_builtin_to_gemini(t))
            payload["tools"] = tools_wire

        if prompt_caching:
            self._apply_prompt_cache(request, payload)

        output_mode = ext.get("output")
        if output_mode == "image":
            payload.setdefault("generationConfig", {})["responseModalities"] = ["IMAGE"]
        elif output_mode == "audio":
            payload.setdefault("generationConfig", {})["responseModalities"] = ["AUDIO"]

        if ext:
            passthrough = {k: v for k, v in ext.items() if k not in {"prompt_caching", "output"}}
            payload.update(passthrough)

        return payload

    def _auth_headers(self, extra: dict[str, str] | None = None) -> dict[str, str]:
        headers = {"x-goog-api-key": self.api_key}
        if extra:
            headers.update(extra)
        return headers

    @staticmethod
    def _auth_params(extra: dict[str, str] | None = None) -> dict[str, str]:
        return dict(extra or {})

    def _apply_prompt_cache(self, request: Request, payload: dict[str, Any]) -> None:
        contents = payload.get("contents") or []
        if len(contents) < 2:
            return
        prefix = contents[:-1]
        key_payload = {
            "model": self._model_path(request.model),
            "systemInstruction": payload.get("systemInstruction"),
            "contents": prefix,
        }
        key = hashlib.sha256(json.dumps(key_payload, sort_keys=True).encode()).hexdigest()
        cache_id = self._cached_content_ids.get(key)

        if cache_id is None:
            body: dict[str, Any] = {"model": self._model_path(request.model), "contents": prefix}
            if payload.get("systemInstruction"):
                body["systemInstruction"] = payload["systemInstruction"]
            req = HttpRequest(
                method="POST", url=f"{self.base_url}/cachedContents",
                headers=self._auth_headers({"Content-Type": "application/json"}),
                params=self._auth_params(), json_body=body, timeout=60.0,
            )
            resp = self.transport.request(req)
            if resp.status < 400:
                data = resp.json()
                cache_id = data.get("name")
                if cache_id:
                    self._cached_content_ids[key] = cache_id

        if cache_id:
            payload["cachedContent"] = cache_id
            payload["contents"] = contents[-1:]
            payload.pop("systemInstruction", None)

    def build_request(self, request: Request, stream: bool) -> HttpRequest:
        endpoint = "streamGenerateContent" if stream else "generateContent"
        params = self._auth_params({"alt": "sse"} if stream else None)
        return HttpRequest(
            method="POST",
            url=f"{self.base_url}/{self._model_path(request.model)}:{endpoint}",
            headers=self._auth_headers({"Content-Type": "application/json"}),
            params=params,
            json_body=self._payload(request),
            timeout=120.0 if stream else 60.0,
        )

    def _parse_candidate_parts(self, parts_payload: list[dict[str, Any]]) -> list[Part]:
        parts: list[Part] = []
        for p in parts_payload:
            if "text" in p:
                parts.append(TextPart(text=p["text"]))
            elif "functionCall" in p:
                fc = p["functionCall"]
                parts.append(ToolCallPart(id=fc.get("id", "fc_0"), name=fc.get("name", ""), input=fc.get("args", {})))
            elif "inlineData" in p:
                inline = p["inlineData"]
                mime = inline.get("mimeType", "application/octet-stream")
                data = inline.get("data", "")
                if mime.startswith("image/"):
                    parts.append(ImagePart(source=Source(type="base64", media_type=mime, data=data)))
                elif mime.startswith("audio/"):
                    parts.append(AudioPart(source=Source(type="base64", media_type=mime, data=data)))
                else:
                    parts.append(DocumentPart(source=Source(type="base64", media_type=mime, data=data)))
            elif "fileData" in p:
                fd = p["fileData"]
                uri = fd.get("fileUri", "")
                mime = fd.get("mimeType", "application/octet-stream")
                if mime.startswith("image/"):
                    parts.append(ImagePart(source=Source(type="url", url=uri, media_type=mime)))
                elif mime.startswith("audio/"):
                    parts.append(AudioPart(source=Source(type="url", url=uri, media_type=mime)))
                else:
                    parts.append(DocumentPart(source=Source(type="url", url=uri, media_type=mime)))
        return parts

    def parse_response(self, request: Request, response: HttpResponse) -> Response:
        data = response.json()

        inband_err = self._inband_error(data)
        if inband_err is not None:
            raise inband_err

        candidate = (data.get("candidates") or [{}])[0]
        content = candidate.get("content", {})
        parts = self._parse_candidate_parts(content.get("parts", []))

        um = data.get("usageMetadata") or {}
        usage = Usage(
            input_tokens=um.get("promptTokenCount", 0),
            output_tokens=um.get("candidatesTokenCount", 0),
            total_tokens=um.get("totalTokenCount", 0),
            cache_read_tokens=um.get("cachedContentTokenCount"),
            reasoning_tokens=um.get("thoughtsTokenCount"),
        )

        if not parts:
            parts = [TextPart(text="")]

        return Response(
            id=data.get("responseId", ""),
            model=request.model,
            message=Message(role="assistant", parts=tuple(parts)),
            finish_reason="tool_call" if any(isinstance(p, ToolCallPart) for p in parts) else "stop",
            usage=usage,
            provider_data=data,
        )

    def parse_stream_event(self, request: Request, raw_event: SSEEvent) -> StreamEvent | None:
        if not raw_event.data:
            return None

        payload = json.loads(raw_event.data)

        if "error" in payload:
            e = payload["error"]
            provider_code = str(e.get("status") or e.get("code") or "provider") if isinstance(e, dict) else "provider"
            message = str(e.get("message", "")) if isinstance(e, dict) else ""
            return StreamEvent(type="error", error=self._make_error_detail(provider_code, message))

        inband_err = self._inband_error(payload)
        if inband_err is not None:
            return StreamEvent(
                type="error",
                error=ErrorDetail(
                    code=canonical_error_code(inband_err),
                    message=str(inband_err),
                    provider_code="inband_finish_reason",
                ),
            )

        cands = payload.get("candidates") or []
        if not cands:
            return None

        part = (cands[0].get("content", {}).get("parts") or [{}])[0]
        if "text" in part:
            return StreamEvent(type="delta", delta=Delta(type="text", text=part["text"]))
        if "functionCall" in part:
            fc = part["functionCall"]
            return StreamEvent(
                type="delta",
                delta=Delta(
                    type="tool_call",
                    id=fc.get("id", "fc_0"),
                    name=fc.get("name", ""),
                    input=json.dumps(fc.get("args", {})),
                ),
            )
        if "inlineData" in part:
            inline = part["inlineData"]
            mime = inline.get("mimeType", "application/octet-stream")
            if mime.startswith("audio/"):
                return StreamEvent(type="delta", delta=Delta(type="audio", data=inline.get("data", "")))
        return None

    # ─── WebSocket / Live transport ──────────────────────────────

    def stream(self, request: Request) -> Iterator[StreamEvent]:
        if self._should_use_live(request):
            yield from self._stream_via_live(request)
            return
        yield from super().stream(request)

    def _should_use_live(self, request: Request) -> bool:
        ext = request.config.extensions or {}
        transport_mode = str(ext.get("transport") or "").lower()
        if transport_mode in {"live", "websocket", "ws"}:
            return True
        model = request.model.lower()
        return "-live" in model or model.endswith("live")

    @staticmethod
    def _is_audio_native_live_model(model: str) -> bool:
        m = model.lower()
        return "live-preview" in m or "native-audio" in m

    @staticmethod
    def _wav_to_pcm(data: bytes) -> tuple[bytes, int]:
        if len(data) >= 44 and data[:4] == b"RIFF" and data[8:12] == b"WAVE":
            sample_rate = _struct.unpack_from("<I", data, 24)[0]
            pos = 12
            while pos + 8 <= len(data):
                chunk_id = data[pos:pos + 4]
                chunk_size = _struct.unpack_from("<I", data, pos + 4)[0]
                if chunk_id == b"data":
                    return data[pos + 8:pos + 8 + chunk_size], sample_rate
                pos += 8 + chunk_size
            return data[44:], sample_rate
        return data, 16000

    def _stream_via_live(self, request: Request) -> Iterator[StreamEvent]:
        ws = self._live_connect(self._live_url())
        saw_tool_call = False
        audio_native = self._is_audio_native_live_model(request.model)
        acc_usage = Usage()

        try:
            setup_payload = self._live_setup_payload_from_request(request)
            setup_inner = setup_payload.setdefault("setup", {})
            if not audio_native:
                setup_inner.setdefault("generationConfig", {}).setdefault("responseModalities", ["TEXT"])
            ws.send(json.dumps(setup_payload))
            self._wait_for_setup_complete(ws)

            for msg in self._live_client_content_from_request(request):
                ws.send(json.dumps(msg))

            yield StreamEvent(type="start", model=request.model)

            while True:
                raw = ws.recv()
                events, turn_complete, usage = self._decode_live_stream_events(raw)
                acc_usage = Usage(
                    input_tokens=max(acc_usage.input_tokens, usage.input_tokens),
                    output_tokens=max(acc_usage.output_tokens, usage.output_tokens),
                    total_tokens=max(acc_usage.total_tokens, usage.total_tokens),
                )
                for evt in events:
                    if evt.type == "delta":
                        d = evt.delta
                        if d is not None and d.type == "tool_call":
                            saw_tool_call = True
                        yield evt
                    elif evt.type == "error":
                        yield evt
                        return

                if turn_complete:
                    yield StreamEvent(type="end", finish_reason="tool_call" if saw_tool_call else "stop", usage=acc_usage)
                    return
        finally:
            try:
                ws.close()
            except Exception:
                pass

    def _live_setup_payload_from_request(self, request: Request) -> dict[str, Any]:
        ext = dict(request.config.extensions or {})
        ext.pop("transport", None)
        ext.pop("prompt_caching", None)
        ext.pop("output", None)
        cfg = LiveConfig(model=request.model, system=request.system, tools=request.tools, extensions=ext or None)
        payload = self._live_setup_payload(cfg)

        output = (request.config.extensions or {}).get("output")
        audio_native = self._is_audio_native_live_model(request.model)

        if output == "audio" or audio_native:
            setup = payload.setdefault("setup", {})
            setup.setdefault("generationConfig", {})["responseModalities"] = ["AUDIO"]
            if output != "audio":
                setup["outputAudioTranscription"] = {}
            has_media = any(
                isinstance(p, (AudioPart, VideoPart))
                for m in request.messages for p in m.parts
            )
            if has_media:
                setup.setdefault("realtimeInputConfig", {}).setdefault("automaticActivityDetection", {})["disabled"] = True
        elif output == "image":
            setup = payload.setdefault("setup", {})
            setup.setdefault("generationConfig", {})["responseModalities"] = ["IMAGE"]

        return payload

    def _live_client_content_from_request(self, request: Request) -> list[dict[str, Any]]:
        audio_native = self._is_audio_native_live_model(request.model)
        if audio_native:
            return self._build_realtime_input_payloads(request)

        if len(request.messages) == 1 and request.messages[0].role == "user":
            parts = request.messages[0].parts
            if all(isinstance(p, TextPart) for p in parts):
                combined_text = "\n".join(p.text for p in parts if isinstance(p, TextPart))
                return [{"realtimeInput": {"text": combined_text}}]

        turns = [
            {"role": "model" if m.role == "assistant" else "user",
             "parts": [self._part_to_wire(p) for p in m.parts]}
            for m in request.messages
        ]
        return [{"clientContent": {"turns": turns, "turnComplete": True}}]

    def _build_realtime_input_payloads(self, request: Request) -> list[dict[str, Any]]:
        text_payloads: list[dict[str, Any]] = []
        media_payloads: list[dict[str, Any]] = []
        content_parts: list[dict[str, Any]] = []
        sent_audio = False

        for msg in request.messages:
            for part in msg.parts:
                if isinstance(part, TextPart) and part.text:
                    text_payloads.append({"realtimeInput": {"text": part.text}})
                elif isinstance(part, AudioPart):
                    mime = part.source.media_type or ""
                    if part.source.type == "base64" and part.source.data:
                        if "wav" in mime or "wave" in mime:
                            pcm, rate = self._wav_to_pcm(part.source.bytes)
                            b64 = _base64.b64encode(pcm).decode("ascii")
                            media_payloads.append({"realtimeInput": {"audio": {"mimeType": f"audio/pcm;rate={rate}", "data": b64}}})
                        else:
                            media_payloads.append({"realtimeInput": {"audio": {"mimeType": mime or "audio/pcm", "data": part.source.data}}})
                        sent_audio = True
                elif isinstance(part, VideoPart) and part.source.data:
                    media_payloads.append({"realtimeInput": {"video": {"mimeType": part.source.media_type or "video/mp4", "data": part.source.data}}})
                elif isinstance(part, (ImagePart, DocumentPart)):
                    content_parts.append(self._part_to_wire(part))

        payloads: list[dict[str, Any]] = []
        if content_parts:
            payloads.append({"clientContent": {"turns": [{"role": "user", "parts": content_parts}], "turnComplete": False}})
        payloads.extend(text_payloads + media_payloads)

        if sent_audio:
            payloads.insert(0, {"realtimeInput": {"activityStart": {}}})
            payloads.append({"realtimeInput": {"activityEnd": {}}})

        if not payloads:
            payloads.append({"realtimeInput": {"text": ""}})
        return payloads

    def _decode_live_stream_events(self, raw: str | bytes) -> tuple[list[StreamEvent], bool, Usage]:
        try:
            payload = json.loads(raw.decode("utf-8") if isinstance(raw, bytes) else raw)
        except Exception:
            return [], False, Usage()
        if not isinstance(payload, dict):
            return [], False, Usage()

        if "error" in payload:
            e = payload["error"]
            provider_code = str(e.get("status") or e.get("code") or "provider") if isinstance(e, dict) else "provider"
            message = str(e.get("message", "")) if isinstance(e, dict) else ""
            return [StreamEvent(type="error", error=self._make_error_detail(provider_code, message))], False, Usage()

        events: list[StreamEvent] = []

        tool_call_data = payload.get("toolCall")
        if isinstance(tool_call_data, dict):
            for idx, fc in enumerate(tool_call_data.get("functionCalls") or []):
                if not isinstance(fc, dict):
                    continue
                events.append(StreamEvent(
                    type="delta",
                    delta=Delta(
                        type="tool_call", part_index=idx,
                        id=fc.get("id", f"fc_{idx}"),
                        name=fc.get("name", "tool"),
                        input=json.dumps(fc.get("args", {})),
                    ),
                ))

        server = payload.get("serverContent")
        if not isinstance(server, dict):
            return events, False, self._live_usage(payload, None)

        model_turn = server.get("modelTurn", {})
        if isinstance(model_turn, dict):
            for idx, part in enumerate(model_turn.get("parts", []) or []):
                if "text" in part:
                    events.append(StreamEvent(type="delta", delta=Delta(type="text", part_index=idx, text=str(part["text"]))))
                elif "functionCall" in part and isinstance(part["functionCall"], dict):
                    fc = part["functionCall"]
                    events.append(StreamEvent(
                        type="delta",
                        delta=Delta(
                            type="tool_call", part_index=idx,
                            id=fc.get("id", "fc_0"), name=fc.get("name", "tool"),
                            input=json.dumps(fc.get("args", {})),
                        ),
                    ))
                elif "inlineData" in part and isinstance(part["inlineData"], dict):
                    inline = part["inlineData"]
                    mime = str(inline.get("mimeType") or "")
                    if mime.startswith("audio/"):
                        events.append(StreamEvent(type="delta", delta=Delta(type="audio", part_index=idx, data=str(inline.get("data") or ""))))

        out_tx = server.get("outputTranscription")
        if isinstance(out_tx, dict) and out_tx.get("text"):
            events.append(StreamEvent(type="delta", delta=Delta(type="text", text=str(out_tx["text"]))))

        turn_complete = bool(server.get("turnComplete"))
        usage = self._live_usage(payload, server)
        return events, turn_complete, usage

    # ─── Live sessions ───────────────────────────────────────────

    def live(self, config: LiveConfig):
        ws = self._live_connect(self._live_url())
        payload = self._live_setup_payload(config)
        if self._is_audio_native_live_model(config.model):
            payload.setdefault("setup", {})["outputAudioTranscription"] = {}
        ws.send(json.dumps(payload))
        self._wait_for_setup_complete(ws)

        callable_registry = {
            t.name: t.fn for t in config.tools
            if t.type == "function" and callable(getattr(t, "fn", None))
        }
        audio_native = self._is_audio_native_live_model(config.model)

        def encode_event(event: LiveClientEvent) -> list[dict[str, Any]]:
            if audio_native and event.type == "text":
                return [{"realtimeInput": {"text": event.text or ""}}]
            return self._encode_live_client_event(event)

        return WebSocketLiveSession(
            ws=ws, encode_event=encode_event,
            decode_event=self._decode_live_server_event,
            callable_registry=callable_registry,
        )

    def _live_connect(self, url: str):
        connect = require_websocket_sync_connect()
        return connect(url)

    def _wait_for_setup_complete(self, ws: Any) -> None:
        while True:
            raw = ws.recv()
            try:
                payload = json.loads(raw.decode("utf-8") if isinstance(raw, bytes) else raw)
            except Exception:
                continue
            if isinstance(payload, dict) and "setupComplete" in payload:
                return
            if isinstance(payload, dict) and "error" in payload:
                err = payload["error"]
                msg = err.get("message", "") if isinstance(err, dict) else str(err)
                raise InvalidRequestError(f"Live setup failed: {msg}")

    def _live_url(self) -> str:
        parsed = urllib.parse.urlparse(self.base_url)
        scheme = "wss" if parsed.scheme == "https" else "ws"
        path = "/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent"
        query = urllib.parse.urlencode({"key": self.api_key})
        return urllib.parse.urlunparse((scheme, parsed.netloc, path, "", query, ""))

    def _live_setup_payload(self, config: LiveConfig) -> dict[str, Any]:
        setup: dict[str, Any] = {"model": self._model_path(config.model)}
        if config.system:
            if isinstance(config.system, str):
                sys_text = config.system
            else:
                sys_text = "\n".join(p.text for p in config.system if isinstance(p, TextPart) and p.text)
            setup["systemInstruction"] = {"parts": [{"text": sys_text}]}
        function_tools = [
            {"name": t.name, "description": t.description,
             "parameters": t.parameters or {"type": "object", "properties": {}}}
            for t in config.tools if t.type == "function"
        ]
        if function_tools:
            setup["tools"] = [{"functionDeclarations": function_tools}]
        generation_config: dict[str, Any] = {}
        if config.output_format is not None:
            generation_config["responseModalities"] = ["AUDIO"]
        elif self._is_audio_native_live_model(config.model):
            generation_config["responseModalities"] = ["AUDIO"]
        if config.voice:
            generation_config.setdefault("speechConfig", {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": config.voice}}})
        if generation_config:
            setup["generationConfig"] = generation_config
        if config.extensions:
            setup.update(config.extensions)
        return {"setup": setup}

    def _live_usage(self, payload: dict[str, Any], server: dict[str, Any] | None) -> Usage:
        u = payload.get("usageMetadata")
        if not isinstance(u, dict) and isinstance(server, dict):
            u = server.get("usageMetadata")
        u = u if isinstance(u, dict) else {}
        return Usage(
            input_tokens=int(u.get("promptTokenCount", 0) or 0),
            output_tokens=int(u.get("responseTokenCount", u.get("candidatesTokenCount", 0)) or 0),
            total_tokens=int(u.get("totalTokenCount", 0) or 0),
            cache_read_tokens=u.get("cachedContentTokenCount"),
            reasoning_tokens=u.get("thoughtsTokenCount"),
        )

    def _encode_live_client_event(self, event: LiveClientEvent) -> list[dict[str, Any]]:
        if event.type == "audio":
            return [{"realtimeInput": {"audio": {"mimeType": "audio/pcm", "data": event.data}}}]
        if event.type == "video":
            return [{"realtimeInput": {"video": {"mimeType": "video/mp4", "data": event.data}}}]
        if event.type == "interrupt":
            return [{"clientContent": {"turnComplete": True}}]
        if event.type == "end_audio":
            return [{"realtimeInput": {"audioStreamEnd": True}}]
        if event.type == "text":
            parts_wire: list[dict[str, Any]] = [{"text": event.text or ""}]
            parts_wire.extend(self._part_to_wire(p) for p in event.content)
            return [{"clientContent": {"turns": [{"role": "user", "parts": parts_wire}], "turnComplete": True}}]
        if event.type == "tool_result":
            response_parts: list[dict[str, Any]] = []
            for part in event.content:
                if isinstance(part, (ImagePart, AudioPart, VideoPart, DocumentPart)):
                    s = part.source
                    if s.type == "base64":
                        response_parts.append({"inlineData": {"mimeType": s.media_type, "data": s.data or ""}})
                    elif s.type == "url":
                        response_parts.append({"fileData": {"mimeType": s.media_type, "fileUri": s.url or ""}})
                    elif s.type == "file":
                        response_parts.append({"fileData": {"mimeType": s.media_type, "fileUri": s.file_id or ""}})
                else:
                    response_parts.append({"text": getattr(part, "text", "")})
            return [{"toolResponse": {"functionResponses": [{"id": event.id, "response": {"output": response_parts}}]}}]
        return []

    def _decode_live_server_event(self, raw: str | bytes) -> list[LiveServerEvent]:
        try:
            payload = json.loads(raw.decode("utf-8") if isinstance(raw, bytes) else raw)
        except Exception:
            return []
        if not isinstance(payload, dict):
            return []

        if "error" in payload:
            err = payload.get("error")
            provider_code = str(err.get("status") or err.get("code") or "provider") if isinstance(err, dict) else "provider"
            message = str(err.get("message") or "") if isinstance(err, dict) else ""
            return [LiveServerEvent(type="error", error=self._make_error_detail(provider_code, message))]

        events: list[LiveServerEvent] = []

        tool_call_data = payload.get("toolCall")
        if isinstance(tool_call_data, dict):
            for fc in tool_call_data.get("functionCalls") or []:
                if not isinstance(fc, dict):
                    continue
                call_id = str(fc.get("id") or "fc_0")
                name = str(fc.get("name") or "tool")
                args = fc.get("args") if isinstance(fc.get("args"), dict) else {}
                events.append(LiveServerEvent(type="tool_call", id=call_id, name=name, input=args))

        server = payload.get("serverContent")
        if not isinstance(server, dict):
            return events

        model_turn = server.get("modelTurn", {})
        if isinstance(model_turn, dict):
            for part in model_turn.get("parts", []) or []:
                if "text" in part:
                    events.append(LiveServerEvent(type="text", text=str(part["text"])))
                elif "inlineData" in part and isinstance(part["inlineData"], dict):
                    inline = part["inlineData"]
                    mime = str(inline.get("mimeType") or "")
                    if mime.startswith("audio/"):
                        events.append(LiveServerEvent(type="audio", data=str(inline.get("data") or "")))
                elif "functionCall" in part and isinstance(part["functionCall"], dict):
                    fc = part["functionCall"]
                    events.append(LiveServerEvent(
                        type="tool_call",
                        id=str(fc.get("id") or "fc_0"),
                        name=str(fc.get("name") or "tool"),
                        input=fc.get("args") if isinstance(fc.get("args"), dict) else {},
                    ))

        out_tx = server.get("outputTranscription")
        if isinstance(out_tx, dict) and out_tx.get("text"):
            events.append(LiveServerEvent(type="text", text=str(out_tx["text"])))

        if server.get("interrupted"):
            events.append(LiveServerEvent(type="interrupted"))
        if server.get("turnComplete"):
            events.append(LiveServerEvent(type="turn_end", usage=self._live_usage(payload, server)))

        return events

    # ─── Other endpoints ─────────────────────────────────────────

    def embeddings(self, request: EmbeddingRequest) -> EmbeddingResponse:
        model_path = self._model_path(request.model)
        if len(request.inputs) <= 1:
            payload = {
                "model": model_path,
                "content": {"parts": [{"text": request.inputs[0] if request.inputs else ""}]},
                **(request.extensions or {}),
            }
            req = HttpRequest(
                method="POST", url=f"{self.base_url}/{model_path}:embedContent",
                headers=self._auth_headers({"Content-Type": "application/json"}),
                params=self._auth_params(), json_body=payload, timeout=60.0,
            )
            resp = self.transport.request(req)
            if resp.status >= 400:
                raise self.normalize_error(resp.status, resp.text())
            data = resp.json()
            values = tuple(float(v) for v in (data.get("embedding", {}) or {}).get("values", []))
            return EmbeddingResponse(model=request.model, vectors=(values,), provider_data=data)

        payload = {
            "requests": [{"model": model_path, "content": {"parts": [{"text": x}]}} for x in request.inputs],
            **(request.extensions or {}),
        }
        req = HttpRequest(
            method="POST", url=f"{self.base_url}/{model_path}:batchEmbedContents",
            headers=self._auth_headers({"Content-Type": "application/json"}),
            params=self._auth_params(), json_body=payload, timeout=60.0,
        )
        resp = self.transport.request(req)
        if resp.status >= 400:
            raise self.normalize_error(resp.status, resp.text())
        data = resp.json()
        vectors = tuple(tuple(float(v) for v in (e.get("values") or [])) for e in data.get("embeddings", []))
        return EmbeddingResponse(model=request.model, vectors=vectors, provider_data=data)

    def file_upload(self, request: FileUploadRequest) -> FileUploadResponse:
        req = HttpRequest(
            method="POST", url=f"{self.upload_base_url}/files",
            headers=self._auth_headers({
                "X-Goog-Upload-Protocol": "raw",
                "X-Goog-Upload-File-Name": request.filename,
                "Content-Type": request.media_type,
            }),
            params=self._auth_params(request.extensions),
            body=request.bytes_data, timeout=120.0,
        )
        resp = self.transport.request(req)
        if resp.status >= 400:
            raise self.normalize_error(resp.status, resp.text())
        data = resp.json()
        file_name = (data.get("file") or {}).get("name") or data.get("name") or ""
        return FileUploadResponse(id=file_name, provider_data=data)

    def batch_submit(self, request: BatchRequest) -> BatchResponse:
        results: list[dict[str, Any]] = []
        for r in request.requests:
            resp = self.complete(r)
            results.append({
                "id": resp.id, "finish_reason": resp.finish_reason,
                "usage": {"input_tokens": resp.usage.input_tokens, "output_tokens": resp.usage.output_tokens, "total_tokens": resp.usage.total_tokens},
            })
        return BatchResponse(id=f"batch_{uuid.uuid4().hex[:12]}", status="completed", provider_data={"results": results})

    def image_generate(self, request: ImageGenerationRequest) -> ImageGenerationResponse:
        ext = {"generationConfig": {"responseModalities": ["IMAGE"]}, **(request.extensions or {})}
        lm_req = Request(model=request.model, messages=(Message.user(request.prompt),), config=Config(extensions=ext))
        resp = self.complete(lm_req)
        images = tuple(p.source for p in resp.message.parts if isinstance(p, ImagePart))
        return ImageGenerationResponse(images=images, provider_data=resp.provider_data)

    def audio_generate(self, request: AudioGenerationRequest) -> AudioGenerationResponse:
        ext: dict[str, Any] = {"generationConfig": {"responseModalities": ["AUDIO"]}, **(request.extensions or {})}
        if request.voice:
            ext.setdefault("speechConfig", {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": request.voice}}})
        lm_req = Request(model=request.model, messages=(Message.user(request.prompt),), config=Config(extensions=ext))
        resp = self.complete(lm_req)
        audio_parts = [p for p in resp.message.parts if isinstance(p, AudioPart)]
        if not audio_parts:
            raise ValueError("provider did not return audio data")
        return AudioGenerationResponse(audio=audio_parts[0].source, provider_data=resp.provider_data)


