"""
lm15.types — Core vocabulary for foundation model interaction.

The fundamental unit is the Part: an atomic, typed block of content.
Parts compose into Messages (attributed to a speaker).
Messages compose into Requests (sent to a model).
Models produce Responses (containing a Message).

Streams reveal Responses incrementally through Deltas — typed fragments
of parts being assembled.

Design principles:

1. Parts are a proper discriminated union.  Each Part variant is an
   independent frozen dataclass.  Fields that don't belong to a variant
   don't exist on it — accessing them raises AttributeError.  Check
   .type or use isinstance() before accessing variant-specific fields.

2. One representation per concept.  A Delta is always a typed Delta
   object, never a dict.  Tool call arguments are called "input"
   everywhere — in memory, in deltas, in serialization.

3. Frozen + slotted dataclasses throughout.  Objects are immutable
   after construction, validated once in __post_init__, and
   memory-efficient.

4. Universal structure, provider-specific values.  The shape of a
   Request is universal.  Provider-specific configuration flows
   through Config.extensions — clearly separated from universal knobs.
   Provider-specific response metadata lives in Response.provider_data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, TypeAlias


# ─── Literal vocabularies ────────────────────────────────────────────

Role = Literal["user", "assistant", "tool"]

PartType = Literal[
    "text", "image", "audio", "video", "document",
    "tool_call", "tool_result",
    "thinking", "refusal", "citation",
]

DeltaType = Literal[
    "text", "thinking", "audio",
    "tool_call", "image", "citation",
]

FinishReason = Literal["stop", "length", "tool_call", "content_filter", "error"]
ReasoningEffort = Literal["low", "medium", "high"]
ErrorCode = Literal[
    "auth", "billing", "rate_limit", "invalid_request",
    "context_length", "timeout", "server", "provider",
]
StreamEventType = Literal["start", "delta", "end", "error"]
ResponseFormatType = Literal["text", "json", "json_schema"]


# ─── Source ──────────────────────────────────────────────────────────
#
# How media content is addressed, independent of what kind of content
# it is.  Three modes: inline bytes (base64), remote reference (url),
# or provider-managed file (file_id).

@dataclass(frozen=True, slots=True)
class Source:
    """A reference to media content.

    Exactly one of data, url, or file_id must be set, matching type.
    """
    type: Literal["base64", "url", "file"]
    media_type: str
    data: str | None = None
    url: str | None = None
    file_id: str | None = None
    detail: Literal["low", "high", "auto"] | None = None

    def __post_init__(self) -> None:
        if self.type == "base64":
            if not self.data:
                raise ValueError("Source(type='base64') requires data")
            if not self.media_type:
                raise ValueError("Source(type='base64') requires media_type")
        elif self.type == "url":
            if not self.url:
                raise ValueError("Source(type='url') requires url")
        elif self.type == "file":
            if not self.file_id:
                raise ValueError("Source(type='file') requires file_id")
        else:
            raise ValueError(f"unsupported source type: {self.type}")

    @property
    def bytes(self) -> bytes:
        """Decode inline base64 data to bytes."""
        if self.type != "base64" or not self.data:
            raise ValueError(
                f"Source(type='{self.type}') has no inline bytes — "
                f"only base64 sources can be decoded"
            )
        import base64
        return base64.b64decode(self.data)

    @staticmethod
    def from_bytes(data: bytes, media_type: str, **kw: Any) -> "Source":
        """Create a base64 source from raw bytes."""
        import base64
        return Source(
            type="base64",
            media_type=media_type,
            data=base64.b64encode(data).decode("ascii"),
            **kw,
        )

    @staticmethod
    def from_url(url: str, media_type: str = "", **kw: Any) -> "Source":
        return Source(type="url", media_type=media_type, url=url, **kw)

    @staticmethod
    def from_file(file_id: str, media_type: str = "", **kw: Any) -> "Source":
        return Source(type="file", media_type=media_type, file_id=file_id, **kw)


# ─── Parts ───────────────────────────────────────────────────────────
#
# The atoms of content.  A discriminated union — check .type or use
# isinstance(), then access the fields that belong to that variant.
#
# There is no base class with __getattr__ fallbacks.  Accessing a field
# that doesn't exist on a variant raises AttributeError.  This is
# deliberate: it makes incorrect code fail loudly.

@dataclass(frozen=True, slots=True)
class TextPart:
    """A block of text content."""
    text: str
    metadata: dict[str, Any] | None = None
    type: Literal["text"] = field(default="text", init=False)


@dataclass(frozen=True, slots=True)
class ImagePart:
    """An image, referenced via Source."""
    source: Source
    metadata: dict[str, Any] | None = None
    type: Literal["image"] = field(default="image", init=False)

    @property
    def bytes(self) -> bytes:
        return self.source.bytes


@dataclass(frozen=True, slots=True)
class AudioPart:
    """Audio content, referenced via Source."""
    source: Source
    metadata: dict[str, Any] | None = None
    type: Literal["audio"] = field(default="audio", init=False)

    @property
    def bytes(self) -> bytes:
        return self.source.bytes


@dataclass(frozen=True, slots=True)
class VideoPart:
    """Video content, referenced via Source."""
    source: Source
    metadata: dict[str, Any] | None = None
    type: Literal["video"] = field(default="video", init=False)


@dataclass(frozen=True, slots=True)
class DocumentPart:
    """A document (PDF, etc.), referenced via Source."""
    source: Source
    metadata: dict[str, Any] | None = None
    type: Literal["document"] = field(default="document", init=False)


@dataclass(frozen=True, slots=True)
class ToolCallPart:
    """The model requests an external computation."""
    id: str
    name: str
    input: dict[str, Any]
    type: Literal["tool_call"] = field(default="tool_call", init=False)

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("ToolCallPart requires id")
        if not self.name:
            raise ValueError("ToolCallPart requires name")


@dataclass(frozen=True, slots=True)
class ToolResultPart:
    """The result of an external computation, sent back to the model."""
    id: str
    content: tuple["Part", ...]
    name: str | None = None
    is_error: bool = False
    type: Literal["tool_result"] = field(default="tool_result", init=False)

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("ToolResultPart requires id")


@dataclass(frozen=True, slots=True)
class ThinkingPart:
    """Model reasoning trace — may be redacted by the provider."""
    text: str
    redacted: bool = False
    type: Literal["thinking"] = field(default="thinking", init=False)


@dataclass(frozen=True, slots=True)
class RefusalPart:
    """Model explicitly refused to respond."""
    text: str
    type: Literal["refusal"] = field(default="refusal", init=False)


@dataclass(frozen=True, slots=True)
class CitationPart:
    """A reference to source material."""
    url: str | None = None
    title: str | None = None
    text: str | None = None
    type: Literal["citation"] = field(default="citation", init=False)


# The union type.  This IS the vocabulary of content.
Part: TypeAlias = (
    TextPart | ImagePart | AudioPart | VideoPart | DocumentPart
    | ToolCallPart | ToolResultPart
    | ThinkingPart | RefusalPart | CitationPart
)

# Runtime dispatch table
PART_TYPES: dict[str, type] = {
    "text": TextPart,
    "image": ImagePart,
    "audio": AudioPart,
    "video": VideoPart,
    "document": DocumentPart,
    "tool_call": ToolCallPart,
    "tool_result": ToolResultPart,
    "thinking": ThinkingPart,
    "refusal": RefusalPart,
    "citation": CitationPart,
}

# Media part types for isinstance checks
MediaPart: TypeAlias = ImagePart | AudioPart | VideoPart | DocumentPart
MEDIA_TYPES: tuple[type, ...] = (ImagePart, AudioPart, VideoPart, DocumentPart)


# ─── Part constructors ───────────────────────────────────────────────
#
# Factory functions for the common construction patterns.  These live
# at module level — there's no base class to hang them on.

def text(content: str, *, metadata: dict[str, Any] | None = None) -> TextPart:
    """Create a text part."""
    return TextPart(text=content, metadata=metadata)


def thinking(content: str, *, redacted: bool = False) -> ThinkingPart:
    return ThinkingPart(text=content, redacted=redacted)


def refusal(content: str) -> RefusalPart:
    return RefusalPart(text=content)


def citation(
    *,
    url: str | None = None,
    title: str | None = None,
    text: str | None = None,
) -> CitationPart:
    return CitationPart(url=url, title=title, text=text)


def _make_source(
    kind: str,
    *,
    url: str | None = None,
    data: bytes | str | None = None,
    file_id: str | None = None,
    media_type: str | None = None,
    detail: Literal["low", "high", "auto"] | None = None,
    default_media_type: str,
) -> Source:
    """Build a Source from exactly one of url/data/file_id."""
    provided = sum(1 for x in (url, data, file_id) if x is not None)
    if provided != 1:
        raise ValueError(f"{kind} requires exactly one of url, data, file_id")

    if url is not None:
        return Source(type="url", url=url, media_type=media_type or default_media_type, detail=detail)
    if file_id is not None:
        return Source(type="file", file_id=file_id, media_type=media_type or default_media_type, detail=detail)

    # data
    if isinstance(data, bytes):
        return Source.from_bytes(data, media_type=media_type or default_media_type, detail=detail)
    return Source(
        type="base64",
        data=data or "",
        media_type=media_type or default_media_type,
        detail=detail,
    )


def _cache_metadata(cache: bool | dict[str, Any] | None) -> dict[str, Any] | None:
    if cache is None:
        return None
    if cache is True:
        return {"cache": True}
    if isinstance(cache, dict):
        return {"cache": cache}
    return {"cache": bool(cache)}


def image(
    *,
    url: str | None = None,
    data: bytes | str | None = None,
    file_id: str | None = None,
    media_type: str | None = None,
    detail: Literal["low", "high", "auto"] | None = None,
    cache: bool | dict[str, Any] | None = None,
) -> ImagePart:
    source = _make_source("image", url=url, data=data, file_id=file_id,
                          media_type=media_type, detail=detail, default_media_type="image/png")
    return ImagePart(source=source, metadata=_cache_metadata(cache))


def audio(
    *,
    url: str | None = None,
    data: bytes | str | None = None,
    file_id: str | None = None,
    media_type: str | None = None,
    detail: Literal["low", "high", "auto"] | None = None,
    cache: bool | dict[str, Any] | None = None,
) -> AudioPart:
    source = _make_source("audio", url=url, data=data, file_id=file_id,
                          media_type=media_type, detail=detail, default_media_type="audio/wav")
    return AudioPart(source=source, metadata=_cache_metadata(cache))


def video(
    *,
    url: str | None = None,
    data: bytes | str | None = None,
    file_id: str | None = None,
    media_type: str | None = None,
    detail: Literal["low", "high", "auto"] | None = None,
    cache: bool | dict[str, Any] | None = None,
) -> VideoPart:
    source = _make_source("video", url=url, data=data, file_id=file_id,
                          media_type=media_type, detail=detail, default_media_type="video/mp4")
    return VideoPart(source=source, metadata=_cache_metadata(cache))


def document(
    *,
    url: str | None = None,
    data: bytes | str | None = None,
    file_id: str | None = None,
    media_type: str | None = None,
    detail: Literal["low", "high", "auto"] | None = None,
    cache: bool | dict[str, Any] | None = None,
) -> DocumentPart:
    source = _make_source("document", url=url, data=data, file_id=file_id,
                          media_type=media_type, detail=detail, default_media_type="application/pdf")
    return DocumentPart(source=source, metadata=_cache_metadata(cache))


def tool_call(id: str, name: str, input: dict[str, Any]) -> ToolCallPart:
    return ToolCallPart(id=id, name=name, input=input)


def tool_result(
    id: str,
    content: list[Part] | Part | str,
    *,
    name: str | None = None,
    is_error: bool = False,
) -> ToolResultPart:
    """Create a tool result part.

    content can be a list of parts, a single part, or a string
    (which becomes a TextPart).
    """
    if isinstance(content, str):
        parts = (TextPart(text=content),)
    elif isinstance(content, list):
        parts = tuple(content)
    else:
        parts = (content,)
    return ToolResultPart(id=id, content=parts, name=name, is_error=is_error)


# ─── Messages ────────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class Message:
    """A contribution to a conversation, attributed to a speaker.

    A message is a sequence of typed Parts.  That's it.
    """
    role: Role
    parts: tuple[Part, ...]

    def __post_init__(self) -> None:
        if self.role not in {"user", "assistant", "tool"}:
            raise ValueError(f"unsupported role: {self.role}")
        if not self.parts:
            raise ValueError("Message requires at least one part")

    @staticmethod
    def user(content: str | Part | list[Part]) -> "Message":
        return Message(role="user", parts=_normalize_parts(content))

    @staticmethod
    def assistant(content: str | Part | list[Part]) -> "Message":
        return Message(role="assistant", parts=_normalize_parts(content))

    @staticmethod
    def tool(results: list[ToolResultPart] | dict[str, str | Part | list[Part]]) -> "Message":
        """Create a tool message.

        Accepts either a list of ToolResultParts or a dict mapping
        call_id → output (str, Part, or list[Part]).
        """
        if isinstance(results, dict):
            parts: list[Part] = []
            for call_id, value in results.items():
                parts.append(tool_result(call_id, value))
            return Message(role="tool", parts=tuple(parts))
        return Message(role="tool", parts=tuple(results))

    @property
    def text(self) -> str | None:
        """Concatenated text from all TextParts, or None."""
        texts = [p.text for p in self.parts if isinstance(p, TextPart)]
        return "\n".join(texts) if texts else None


def _normalize_parts(content: str | Part | list[Part]) -> tuple[Part, ...]:
    if isinstance(content, str):
        return (TextPart(text=content),)
    if isinstance(content, list):
        return tuple(content)
    return (content,)


# ─── Deltas ──────────────────────────────────────────────────────────
#
# A Delta is a typed fragment of a Part being assembled during
# streaming.  There is ONE representation — no dict escape hatch.
#
# Fields are populated based on .type.  Accessing an irrelevant field
# gets None (these are all optional), but the invariant is enforced:
# a text delta MUST have .text, etc.

@dataclass(frozen=True, slots=True)
class Delta:
    """A typed fragment of a Part arriving during streaming.

    Fields populated by type:
      text, thinking  → .text
      audio           → .data (base64 chunk)
      tool_call       → .input (JSON fragment string), optionally .id, .name
      image           → .source (complete or partial)
      citation        → .url, .title, .text
    """
    type: DeltaType
    part_index: int = 0

    # Content fields
    text: str | None = None
    data: str | None = None         # base64 audio chunk
    input: str | None = None        # JSON string fragment for tool calls

    # Identity fields (tool calls, citations)
    id: str | None = None
    name: str | None = None
    url: str | None = None
    title: str | None = None

    # Media (image deltas)
    source: Source | None = None

    def __post_init__(self) -> None:
        if self.type in ("text", "thinking") and self.text is None:
            raise ValueError(f"Delta(type='{self.type}') requires text")
        if self.type == "audio" and self.data is None:
            raise ValueError("Delta(type='audio') requires data")
        if self.type == "tool_call" and self.input is None:
            raise ValueError("Delta(type='tool_call') requires input")


# ─── Stream Events ───────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class ErrorDetail:
    """Structured error information.  A dataclass, not a dict."""
    code: ErrorCode
    message: str
    provider_code: str | None = None


@dataclass(frozen=True, slots=True)
class StreamEvent:
    """An event in the streaming protocol.

    Four event types:
      start — the response begins (id, model)
      delta — a fragment of content arrives
      end   — the response is complete (usage, finish_reason)
      error — something went wrong
    """
    type: StreamEventType

    # start
    id: str | None = None
    model: str | None = None

    # delta
    delta: Delta | None = None

    # end
    finish_reason: FinishReason | None = None
    usage: "Usage | None" = None

    # error
    error: ErrorDetail | None = None

    # Provider-specific metadata (end events)
    provider_data: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.type == "delta" and self.delta is None:
            raise ValueError("StreamEvent(type='delta') requires delta")
        if self.type == "error" and self.error is None:
            raise ValueError("StreamEvent(type='error') requires error")


# ─── Tools ───────────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class FunctionTool:
    """A function the model can invoke."""
    name: str
    description: str | None = None
    parameters: dict[str, Any] = field(default_factory=lambda: {"type": "object", "properties": {}})
    fn: Any = None
    type: Literal["function"] = field(default="function", init=False)

    @staticmethod
    def from_fn(fn: Any) -> "FunctionTool":
        """Infer a FunctionTool from a callable's signature."""
        import inspect
        sig = inspect.signature(fn)
        hints = inspect.get_annotations(fn, eval_str=True)
        properties: dict[str, Any] = {}
        required: list[str] = []
        for name, param in sig.parameters.items():
            if param.kind not in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY):
                continue
            ann = hints.get(name, str)
            origin = getattr(ann, "__origin__", None)
            if origin in (list, tuple, set):
                json_type: dict[str, Any] = {"type": "array"}
            elif origin is dict:
                json_type = {"type": "object"}
            elif ann in (int,):
                json_type = {"type": "integer"}
            elif ann in (float,):
                json_type = {"type": "number"}
            elif ann in (bool,):
                json_type = {"type": "boolean"}
            else:
                json_type = {"type": "string"}
            properties[name] = json_type
            if param.default is inspect.Parameter.empty:
                required.append(name)
        schema: dict[str, Any] = {"type": "object", "properties": properties}
        if required:
            schema["required"] = required
        return FunctionTool(
            name=fn.__name__,
            description=(inspect.getdoc(fn) or "").strip() or None,
            parameters=schema,
        )


@dataclass(frozen=True, slots=True)
class BuiltinTool:
    """A provider-native tool (web search, code execution, etc.)."""
    name: str
    config: dict[str, Any] | None = None
    type: Literal["builtin"] = field(default="builtin", init=False)


Tool: TypeAlias = FunctionTool | BuiltinTool


# ─── Configuration ───────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class Reasoning:
    """Extended thinking / reasoning configuration."""
    enabled: bool = True
    budget: int | None = None
    effort: ReasoningEffort | None = None

    def __post_init__(self) -> None:
        if self.budget is not None and self.budget <= 0:
            raise ValueError("budget must be > 0")


@dataclass(frozen=True, slots=True)
class ToolChoice:
    """How the model should use tools."""
    mode: Literal["auto", "required", "none"] = "auto"
    allowed: tuple[str, ...] = ()
    parallel: bool | None = None


@dataclass(frozen=True, slots=True)
class Config:
    """Generation parameters.

    Universal fields are typed.  Provider-specific settings go in
    `extensions` — a clearly-separated namespace that never pretends
    to be part of the universal schema.
    """
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    stop: tuple[str, ...] = ()
    response_format: dict[str, Any] | None = None
    tool_choice: ToolChoice | None = None
    reasoning: Reasoning | None = None
    extensions: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.max_tokens is not None and self.max_tokens <= 0:
            raise ValueError("max_tokens must be > 0")
        if self.temperature is not None and self.temperature < 0:
            raise ValueError("temperature must be >= 0")
        if self.top_p is not None and not (0 <= self.top_p <= 1):
            raise ValueError("top_p must be in [0, 1]")


# ─── Request ─────────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class Request:
    """A complete request to a foundation model.

    The composed artifact sent to the model — conversation history,
    system instructions, available tools, and generation config.
    """
    model: str
    messages: tuple[Message, ...]
    system: str | tuple[Part, ...] | None = None
    tools: tuple[Tool, ...] = ()
    config: Config = field(default_factory=Config)

    def __post_init__(self) -> None:
        if not self.model:
            raise ValueError("model is required")
        if not self.messages:
            raise ValueError("at least one message is required")
        if isinstance(self.system, tuple) and not self.system:
            raise ValueError("system parts cannot be empty")


# ─── Usage ───────────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class Usage:
    """Token usage.  Universal trio + optional detail fields."""
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cache_read_tokens: int | None = None
    cache_write_tokens: int | None = None
    reasoning_tokens: int | None = None
    input_audio_tokens: int | None = None
    output_audio_tokens: int | None = None


# ─── Response ────────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class Response:
    """The composed artifact returned by a foundation model.

    Properties provide convenient access to common content shapes
    without requiring callers to walk the parts list.
    """
    id: str
    model: str
    message: Message
    finish_reason: FinishReason
    usage: Usage
    provider_data: dict[str, Any] | None = None

    @property
    def text(self) -> str | None:
        return self.message.text

    @property
    def tool_calls(self) -> list[ToolCallPart]:
        return [p for p in self.message.parts if isinstance(p, ToolCallPart)]

    @property
    def image(self) -> ImagePart | None:
        return next((p for p in self.message.parts if isinstance(p, ImagePart)), None)

    @property
    def images(self) -> list[ImagePart]:
        return [p for p in self.message.parts if isinstance(p, ImagePart)]

    @property
    def audio(self) -> AudioPart | None:
        return next((p for p in self.message.parts if isinstance(p, AudioPart)), None)

    @property
    def thinking(self) -> str | None:
        texts = [p.text for p in self.message.parts if isinstance(p, ThinkingPart)]
        return "\n".join(texts) if texts else None

    @property
    def citations(self) -> list[CitationPart]:
        return [p for p in self.message.parts if isinstance(p, CitationPart)]

    @property
    def json(self) -> Any:
        import json as _json
        t = self.text
        if t is None:
            raise ValueError(
                "Cannot parse response as JSON: no text content. "
                f"Parts: {[p.type for p in self.message.parts]}"
            )
        try:
            return _json.loads(t)
        except _json.JSONDecodeError as e:
            preview = t[:200] + ("..." if len(t) > 200 else "")
            raise ValueError(f"Cannot parse response as JSON: {e}\nRaw text: {preview}") from e

    @property
    def image_bytes(self) -> bytes:
        img = self.image
        if img is None:
            raise ValueError(f"Response contains no image. Parts: {[p.type for p in self.message.parts]}")
        return img.bytes

    @property
    def audio_bytes(self) -> bytes:
        aud = self.audio
        if aud is None:
            raise ValueError(f"Response contains no audio. Parts: {[p.type for p in self.message.parts]}")
        return aud.bytes


# ─── Embeddings ──────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class EmbeddingRequest:
    model: str
    inputs: tuple[str, ...]
    extensions: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class EmbeddingResponse:
    model: str
    vectors: tuple[tuple[float, ...], ...]
    usage: Usage = field(default_factory=Usage)
    provider_data: dict[str, Any] | None = None


# ─── File Upload ─────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class FileUploadRequest:
    model: str | None = None
    filename: str = "file.bin"
    bytes_data: bytes = b""
    media_type: str = "application/octet-stream"
    extensions: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class FileUploadResponse:
    id: str
    provider_data: dict[str, Any] | None = None


# ─── Batch ───────────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class BatchRequest:
    model: str
    requests: tuple[Request, ...]
    extensions: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class BatchResponse:
    id: str
    status: str
    provider_data: dict[str, Any] | None = None


# ─── Image Generation ────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class ImageGenerationRequest:
    model: str
    prompt: str
    size: str | None = None
    extensions: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class ImageGenerationResponse:
    images: tuple[Source, ...]
    provider_data: dict[str, Any] | None = None


# ─── Audio Generation ────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class AudioGenerationRequest:
    model: str
    prompt: str
    voice: str | None = None
    format: str | None = None
    extensions: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class AudioGenerationResponse:
    audio: Source
    provider_data: dict[str, Any] | None = None


# ─── Audio Format ────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class AudioFormat:
    encoding: Literal["pcm16", "opus", "mp3", "aac"]
    sample_rate: int
    channels: int = 1

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be > 0")
        if self.channels <= 0:
            raise ValueError("channels must be > 0")


# ─── Live (Realtime) ─────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class LiveConfig:
    model: str
    system: str | tuple[Part, ...] | None = None
    tools: tuple[Tool, ...] = ()
    voice: str | None = None
    input_format: AudioFormat | None = None
    output_format: AudioFormat | None = None
    extensions: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.model:
            raise ValueError("model is required")
        if isinstance(self.system, tuple) and not self.system:
            raise ValueError("system parts cannot be empty")


@dataclass(frozen=True, slots=True)
class LiveClientEvent:
    type: Literal["audio", "video", "text", "tool_result", "interrupt", "end_audio"]
    data: str | None = None
    text: str | None = None
    id: str | None = None
    content: tuple[Part, ...] = ()

    def __post_init__(self) -> None:
        if self.type in {"audio", "video"} and self.data is None:
            raise ValueError(f"LiveClientEvent(type='{self.type}') requires data")
        if self.type == "text" and self.text is None:
            raise ValueError("LiveClientEvent(type='text') requires text")
        if self.type == "tool_result":
            if not self.id:
                raise ValueError("LiveClientEvent(type='tool_result') requires id")
            if not self.content:
                raise ValueError("LiveClientEvent(type='tool_result') requires content")


@dataclass(frozen=True, slots=True)
class LiveServerEvent:
    type: Literal["audio", "text", "tool_call", "interrupted", "turn_end", "error"]
    data: str | None = None
    text: str | None = None
    id: str | None = None
    name: str | None = None
    input: dict[str, Any] | None = None
    usage: Usage | None = None
    error: ErrorDetail | None = None

    def __post_init__(self) -> None:
        if self.type == "audio" and self.data is None:
            raise ValueError("LiveServerEvent(type='audio') requires data")
        if self.type == "text" and self.text is None:
            raise ValueError("LiveServerEvent(type='text') requires text")
        if self.type == "tool_call":
            if not self.id or not self.name or self.input is None:
                raise ValueError("LiveServerEvent(type='tool_call') requires id, name, input")
        if self.type == "turn_end" and self.usage is None:
            raise ValueError("LiveServerEvent(type='turn_end') requires usage")
        if self.type == "error" and self.error is None:
            raise ValueError("LiveServerEvent(type='error') requires error")


# ─── ToolCallInfo (for callbacks) ────────────────────────────────────

@dataclass(frozen=True, slots=True)
class ToolCallInfo:
    id: str
    name: str
    input: dict[str, Any]
