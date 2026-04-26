"""
lm15.types — Core vocabulary for foundation model interaction.

The fundamental unit is the Part: an atomic, typed block of content.
Parts compose into Messages (attributed to a speaker).
Messages compose into Requests (sent to a model).
Models produce Responses (containing a Message).

Streams reveal Responses incrementally through Deltas — typed fragments
of parts being assembled.

Design principles:

1. Parts and Deltas are proper discriminated unions.  Each variant is
   an independent frozen dataclass.  Fields that don't belong to a
   variant don't exist on it — accessing them raises AttributeError.
   Check .type or use isinstance() before accessing variant-specific
   fields.

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

5. Runtime validation is deliberately narrow.  Constructors enforce the
   invariants that make objects meaningful (required identities, valid
   media addresses, non-negative token counts, JSON-shaped extension
   fields) while adapters remain responsible for normalizing provider
   quirks before constructing these types.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, TypeAlias, TypeVar


# ─── Literal vocabularies ────────────────────────────────────────────

Role = Literal["user", "assistant", "tool", "developer"]

PartType = Literal[
    "text",
    "image",
    "audio",
    "video",
    "document",
    "tool_call",
    "tool_result",
    "thinking",
    "refusal",
    "citation",
]

FinishReason = Literal["stop", "length", "tool_call", "content_filter", "error"]

ReasoningEffort = Literal[
    "off", "adaptive", "minimal", "low", "medium", "high", "xhigh"
]

ErrorCode = Literal[
    "auth",
    "billing",
    "rate_limit",
    "invalid_request",
    "context_length",
    "timeout",
    "server",
    "provider",
]

StreamEventType = Literal["start", "delta", "end", "error"]

_P = TypeVar("_P")

# JSON-compatible values used for model inputs, tool schemas, provider
# extensions, and provider metadata.  Keep these small and boring: the
# adapter layer can map richer Python objects into this vocabulary before
# they enter the core model.
JsonPrimitive: TypeAlias = None | bool | int | float | str
JsonValue: TypeAlias = JsonPrimitive | list["JsonValue"] | dict[str, "JsonValue"]
JsonArray: TypeAlias = list[JsonValue]
JsonObject: TypeAlias = dict[str, JsonValue]


def _is_json_value(value: Any) -> bool:
    """Return True if value is representable as standard JSON."""
    if value is None or isinstance(value, (bool, int, str)):
        return True
    if isinstance(value, float):
        import math

        return math.isfinite(value)
    if isinstance(value, list):
        return all(_is_json_value(x) for x in value)
    if isinstance(value, dict):
        return all(isinstance(k, str) and _is_json_value(v) for k, v in value.items())
    return False


def _validate_json_object(value: Any, *, field_name: str) -> None:
    """Validate that value is a JSON object, if present."""
    if value is None:
        return
    if not isinstance(value, dict) or not _is_json_value(value):
        raise TypeError(f"{field_name} must be a JSON object")


def _validate_positive(value: int | None, *, field_name: str) -> None:
    if value is not None and value <= 0:
        raise ValueError(f"{field_name} must be > 0")


def _validate_non_negative(value: int | None, *, field_name: str) -> None:
    if value is not None and value < 0:
        raise ValueError(f"{field_name} must be >= 0")


def _validate_part_index(part_index: int) -> None:
    if part_index < 0:
        raise ValueError("part_index must be >= 0")


# ─── Media helpers ────────────────────────────────────────────────────


def _validate_media(
    part_type: str, data: str | None, url: str | None, file_id: str | None
) -> None:
    """Validate that exactly one non-empty media address is set."""
    provided = [
        (name, value)
        for name, value in (("data", data), ("url", url), ("file_id", file_id))
        if value is not None
    ]
    if len(provided) != 1:
        raise ValueError(f"{part_type} requires exactly one of data, url, or file_id")
    name, value = provided[0]
    if value == "":
        raise ValueError(f"{part_type} {name} cannot be empty")


def _decode_data(part_type: str, data: str | None) -> bytes:
    """Decode base64 data to bytes."""
    if not data:
        raise ValueError(
            f"{part_type} has no inline data — only data parts can be decoded"
        )
    import base64

    return base64.b64decode(data, validate=True)


class _MediaMixin:
    """Shared validation and byte access for inline-capable media parts."""

    __slots__ = ()

    type: str
    media_type: str
    data: str | None
    url: str | None
    file_id: str | None

    def __post_init__(self) -> None:
        if not self.media_type:
            raise ValueError(f"{self.__class__.__name__} requires media_type")
        _validate_media(self.__class__.__name__, self.data, self.url, self.file_id)

    @property
    def bytes(self) -> bytes:
        return _decode_data(self.__class__.__name__, self.data)


# ─── Parts ───────────────────────────────────────────────────────────
#
# The atoms of content.  A discriminated union — check .type or use
# isinstance(), then access the fields that belong to that variant.


@dataclass(frozen=True, slots=True)
class TextPart:
    """A block of text content."""

    text: str
    type: Literal["text"] = field(default="text", init=False)


@dataclass(frozen=True, slots=True)
class ImagePart(_MediaMixin):
    """An image, addressed by exactly one of data/url/file_id."""

    media_type: str = "image/png"
    data: str | None = None
    url: str | None = None
    file_id: str | None = None
    detail: Literal["low", "high", "auto"] | None = None
    type: Literal["image"] = field(default="image", init=False)


@dataclass(frozen=True, slots=True)
class AudioPart(_MediaMixin):
    """Audio content, addressed by exactly one of data/url/file_id."""

    media_type: str = "audio/wav"
    data: str | None = None
    url: str | None = None
    file_id: str | None = None
    type: Literal["audio"] = field(default="audio", init=False)


@dataclass(frozen=True, slots=True)
class VideoPart(_MediaMixin):
    """Video content, addressed by exactly one of data/url/file_id."""

    media_type: str = "video/mp4"
    data: str | None = None
    url: str | None = None
    file_id: str | None = None
    type: Literal["video"] = field(default="video", init=False)


@dataclass(frozen=True, slots=True)
class DocumentPart(_MediaMixin):
    """A document (PDF, etc.), addressed by exactly one of data/url/file_id."""

    media_type: str = "application/pdf"
    data: str | None = None
    url: str | None = None
    file_id: str | None = None
    type: Literal["document"] = field(default="document", init=False)


@dataclass(frozen=True, slots=True)
class ToolCallPart:
    """The model requests an external computation."""

    id: str
    name: str
    input: JsonObject
    type: Literal["tool_call"] = field(default="tool_call", init=False)

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("ToolCallPart requires id")
        if not self.name:
            raise ValueError("ToolCallPart requires name")
        _validate_json_object(self.input, field_name="input")


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
        object.__setattr__(self, "content", tuple(self.content))
        if not all(_is_part(p) for p in self.content):
            raise TypeError("ToolResultPart.content must contain Part objects")


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

    def __post_init__(self) -> None:
        if self.url is None and self.title is None and self.text is None:
            raise ValueError("CitationPart requires at least one of url, title, or text")


# The union type.  This IS the vocabulary of content.
Part: TypeAlias = (
    TextPart
    | ImagePart
    | AudioPart
    | VideoPart
    | DocumentPart
    | ToolCallPart
    | ToolResultPart
    | ThinkingPart
    | RefusalPart
    | CitationPart
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

# Runtime tuples for isinstance checks.
PART_CLASSES: tuple[type, ...] = tuple(PART_TYPES.values())


def _is_part(value: object) -> bool:
    """Return True if value is one of lm15's concrete Part variants."""
    return isinstance(value, PART_CLASSES)


MediaPart: TypeAlias = ImagePart | AudioPart | VideoPart | DocumentPart
MEDIA_TYPES: tuple[type, ...] = (ImagePart, AudioPart, VideoPart, DocumentPart)

# Shared endpoint metadata/content aliases
Extensions: TypeAlias = JsonObject
ProviderData: TypeAlias = JsonObject
PartInput: TypeAlias = str | Part | Sequence[Part]
ToolResultContent: TypeAlias = str | Part | Sequence[Part]
SystemContent: TypeAlias = str | tuple[Part, ...]


# ─── Part constructors ───────────────────────────────────────────────
#
# Factory functions for the common construction patterns.  These live
# at module level — there's no base class to hang them on.


def text(content: str) -> TextPart:
    """Create a text part."""
    return TextPart(text=content)


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


def _encode_data(data: bytes | str) -> str:
    """Ensure data is a base64 string."""
    if isinstance(data, bytes):
        import base64

        return base64.b64encode(data).decode("ascii")
    return data


def image(
    *,
    url: str | None = None,
    data: bytes | str | None = None,
    file_id: str | None = None,
    media_type: str | None = None,
    detail: Literal["low", "high", "auto"] | None = None,
) -> ImagePart:
    return ImagePart(
        media_type=media_type or "image/png",
        data=_encode_data(data) if data is not None else None,
        url=url,
        file_id=file_id,
        detail=detail,
    )


def audio(
    *,
    url: str | None = None,
    data: bytes | str | None = None,
    file_id: str | None = None,
    media_type: str | None = None,
) -> AudioPart:
    return AudioPart(
        media_type=media_type or "audio/wav",
        data=_encode_data(data) if data is not None else None,
        url=url,
        file_id=file_id,
    )


def video(
    *,
    url: str | None = None,
    data: bytes | str | None = None,
    file_id: str | None = None,
    media_type: str | None = None,
) -> VideoPart:
    return VideoPart(
        media_type=media_type or "video/mp4",
        data=_encode_data(data) if data is not None else None,
        url=url,
        file_id=file_id,
    )


def document(
    *,
    url: str | None = None,
    data: bytes | str | None = None,
    file_id: str | None = None,
    media_type: str | None = None,
) -> DocumentPart:
    return DocumentPart(
        media_type=media_type or "application/pdf",
        data=_encode_data(data) if data is not None else None,
        url=url,
        file_id=file_id,
    )


def tool_call(id: str, name: str, input: JsonObject) -> ToolCallPart:
    return ToolCallPart(id=id, name=name, input=input)


def tool_result(
    id: str,
    content: ToolResultContent,
    *,
    name: str | None = None,
    is_error: bool = False,
) -> ToolResultPart:
    """Create a tool result part.

    content can be a sequence of parts, a single part, or a string
    (which becomes a TextPart).
    """
    if isinstance(content, str):
        parts = (TextPart(text=content),)
    elif _is_part(content):
        parts = (content,)
    elif isinstance(content, Sequence):
        parts = tuple(content)
    else:
        raise TypeError("tool_result content must be a string, Part, or sequence of Parts")
    return ToolResultPart(id=id, content=parts, name=name, is_error=is_error)


# ─── Messages ────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class Message:
    """A contribution to a conversation, attributed to a speaker.

    A message is a sequence of typed Parts.  Roles:

    - ``user``      — end-user input
    - ``assistant``  — model output
    - ``tool``      — tool execution results
    - ``developer`` — high-authority instructions from the application
                      developer.  Can appear mid-conversation to inject
                      new instructions without invalidating the KV-cache
                      prefix.  On OpenAI this maps to the native
                      ``developer`` role; on other providers the adapter
                      converts it to a user message with a clear prefix.
    """

    role: Role
    parts: tuple[Part, ...]

    def __post_init__(self) -> None:
        if self.role not in {"user", "assistant", "tool", "developer"}:
            raise ValueError(f"unsupported role: {self.role}")
        parts = (self.parts,) if _is_part(self.parts) else tuple(self.parts)
        object.__setattr__(self, "parts", parts)
        if not self.parts:
            raise ValueError("Message requires at least one part")
        if not all(_is_part(p) for p in self.parts):
            raise TypeError("Message.parts must contain Part objects")

    @staticmethod
    def user(content: PartInput) -> "Message":
        return Message(role="user", parts=_normalize_parts(content))

    @staticmethod
    def assistant(content: PartInput) -> "Message":
        return Message(role="assistant", parts=_normalize_parts(content))

    @staticmethod
    def developer(content: PartInput) -> "Message":
        """Create a developer message.

        Developer messages carry instructions with higher authority than
        user messages (equivalent to OpenAI's ``developer`` role).  They
        can appear anywhere in the conversation — including mid-conversation
        — which is useful for injecting new instructions without
        invalidating the KV-cache prefix built from earlier turns.

        On providers that don't natively support a developer role
        (Anthropic, Gemini), the adapter converts these to user messages
        with a clear ``[developer]`` prefix so the model still sees the
        instruction boundary.
        """
        return Message(role="developer", parts=_normalize_parts(content))

    @staticmethod
    def tool(
        results: Sequence[ToolResultPart] | dict[str, ToolResultContent],
    ) -> "Message":
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

    def parts_of(self, cls: type[_P]) -> list[_P]:
        """Return all parts that are instances of ``cls``."""
        return [p for p in self.parts if isinstance(p, cls)]

    def first(self, cls: type[_P]) -> _P | None:
        """Return the first part that is an instance of ``cls``, if any."""
        return next((p for p in self.parts if isinstance(p, cls)), None)

    @property
    def text(self) -> str | None:
        """Concatenated text from all TextParts, or None."""
        texts = [p.text for p in self.parts_of(TextPart)]
        return "\n".join(texts) if texts else None


def _normalize_parts(content: PartInput) -> tuple[Part, ...]:
    if isinstance(content, str):
        return (TextPart(text=content),)
    if _is_part(content):
        return (content,)
    if isinstance(content, Sequence):
        parts = tuple(content)
        if not all(_is_part(p) for p in parts):
            raise TypeError("content sequence must contain Part objects")
        return parts
    raise TypeError("content must be a string, Part, or sequence of Parts")


def _require_fields(owner: str, obj: object, fields: tuple[str, ...]) -> None:
    """Validate that required optional fields have been populated."""
    for field_name in fields:
        if getattr(obj, field_name) is None:
            raise ValueError(f"{owner} requires {field_name}")


# ─── Deltas ──────────────────────────────────────────────────────────
#
# A Delta is a typed fragment of a Part being assembled during
# streaming.  Like Part, Delta is a discriminated union: fields that
# don't belong to a variant don't exist on it.

DeltaType = Literal["text", "thinking", "audio", "image", "tool_call", "citation"]


@dataclass(frozen=True, slots=True)
class TextDelta:
    """A text fragment arriving during streaming."""

    text: str
    part_index: int = 0
    type: Literal["text"] = field(default="text", init=False)

    def __post_init__(self) -> None:
        _validate_part_index(self.part_index)


@dataclass(frozen=True, slots=True)
class ThinkingDelta:
    """A reasoning/thinking fragment arriving during streaming."""

    text: str
    part_index: int = 0
    type: Literal["thinking"] = field(default="thinking", init=False)

    def __post_init__(self) -> None:
        _validate_part_index(self.part_index)


@dataclass(frozen=True, slots=True)
class AudioDelta:
    """An audio data fragment arriving during streaming."""

    data: str
    part_index: int = 0
    media_type: str | None = None
    type: Literal["audio"] = field(default="audio", init=False)

    def __post_init__(self) -> None:
        _validate_part_index(self.part_index)


@dataclass(frozen=True, slots=True)
class ImageDelta:
    """An image fragment, addressed by exactly one of data/url/file_id."""

    data: str | None = None
    url: str | None = None
    file_id: str | None = None
    part_index: int = 0
    media_type: str | None = None
    type: Literal["image"] = field(default="image", init=False)

    def __post_init__(self) -> None:
        _validate_part_index(self.part_index)
        _validate_media("ImageDelta", self.data, self.url, self.file_id)


@dataclass(frozen=True, slots=True)
class ToolCallDelta:
    """A tool-call input fragment, optionally carrying call identity."""

    input: str
    part_index: int = 0
    id: str | None = None
    name: str | None = None
    type: Literal["tool_call"] = field(default="tool_call", init=False)

    def __post_init__(self) -> None:
        _validate_part_index(self.part_index)


@dataclass(frozen=True, slots=True)
class CitationDelta:
    """A citation fragment arriving during streaming."""

    text: str | None = None
    url: str | None = None
    title: str | None = None
    part_index: int = 0
    type: Literal["citation"] = field(default="citation", init=False)

    def __post_init__(self) -> None:
        _validate_part_index(self.part_index)
        if self.text is None and self.url is None and self.title is None:
            raise ValueError("CitationDelta requires at least one of text, url, or title")


Delta: TypeAlias = (
    TextDelta
    | ThinkingDelta
    | AudioDelta
    | ImageDelta
    | ToolCallDelta
    | CitationDelta
)

DELTA_TYPES: dict[str, type] = {
    "text": TextDelta,
    "thinking": ThinkingDelta,
    "audio": AudioDelta,
    "image": ImageDelta,
    "tool_call": ToolCallDelta,
    "citation": CitationDelta,
}


# ─── Stream Events ───────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class ErrorDetail:
    """Structured error information.  A dataclass, not a dict."""

    code: ErrorCode
    message: str
    provider_code: str | None = None

    def __post_init__(self) -> None:
        if self.code not in {
            "auth",
            "billing",
            "rate_limit",
            "invalid_request",
            "context_length",
            "timeout",
            "server",
            "provider",
        }:
            raise ValueError(f"unsupported error code: {self.code}")


_STREAM_EVENT_REQUIRED_FIELDS: dict[str, tuple[str, ...]] = {
    "delta": ("delta",),
    "error": ("error",),
}


@dataclass(frozen=True, slots=True)
class StreamEvent:
    """An event in the streaming protocol.

    Four event types:
      start — the response begins, optionally carrying id/model
      delta — a fragment of content arrives
      end   — the response is complete, optionally carrying usage/finish_reason
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
    provider_data: ProviderData | None = None

    def __post_init__(self) -> None:
        if self.type not in {"start", "delta", "end", "error"}:
            raise ValueError(f"unsupported stream event type: {self.type}")
        _require_fields(
            f"StreamEvent(type={self.type!r})",
            self,
            _STREAM_EVENT_REQUIRED_FIELDS.get(self.type, ()),
        )
        _validate_json_object(self.provider_data, field_name="provider_data")


# ─── Tools ───────────────────────────────────────────────────────────

_JSON_SCHEMA_TYPES: dict[Any, str] = {
    int: "integer",
    float: "number",
    bool: "boolean",
    str: "string",
    list: "array",
    tuple: "array",
    set: "array",
    dict: "object",
}


@dataclass(frozen=True, slots=True)
class FunctionTool:
    """Serializable function tool specification sent to the model."""

    name: str
    description: str | None = None
    parameters: JsonObject = field(
        default_factory=lambda: {"type": "object", "properties": {}}
    )
    type: Literal["function"] = field(default="function", init=False)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("FunctionTool requires name")
        _validate_json_object(self.parameters, field_name="parameters")

    @staticmethod
    def from_fn(fn: Callable[..., Any]) -> "FunctionTool":
        """Infer a serializable tool spec from a callable's signature."""
        import inspect

        sig = inspect.signature(fn)
        hints = inspect.get_annotations(fn, eval_str=True)
        properties: JsonObject = {}
        required: list[str] = []
        for name, param in sig.parameters.items():
            if param.kind not in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            ):
                continue
            ann = hints.get(name, str)
            origin = getattr(ann, "__origin__", None)
            json_schema_type = (
                _JSON_SCHEMA_TYPES.get(origin)
                or _JSON_SCHEMA_TYPES.get(ann)
                or "string"
            )
            properties[name] = {"type": json_schema_type}
            if param.default is inspect.Parameter.empty:
                required.append(name)
        schema: JsonObject = {"type": "object", "properties": properties}
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
    config: JsonObject | None = None
    type: Literal["builtin"] = field(default="builtin", init=False)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("BuiltinTool requires name")
        _validate_json_object(self.config, field_name="config")


Tool: TypeAlias = FunctionTool | BuiltinTool
ToolRegistry: TypeAlias = dict[str, Callable[..., Any]]


# ─── Configuration ───────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class Reasoning:
    """Extended thinking / reasoning configuration.

    effort controls the model's reasoning depth:
      - "off"      → no reasoning (provider will skip/disable thinking)
      - "adaptive" → model decides whether to think based on complexity
      - "low"      → light reasoning
      - "medium"   → moderate reasoning (typical default when reasoning is on)
      - "high"     → deep reasoning

    thinking_budget is an optional hard cap on reasoning tokens.
    When set, it limits how many tokens the model spends on internal
    reasoning — independent of the visible response length.

    total_budget caps the combined output (thinking + response tokens).
    When set alongside Config.max_tokens, both limits are enforced:
    the response won't exceed max_tokens, and the total won't exceed
    total_budget.

    Not all providers support every knob. The adapter maps to the
    closest available mechanism and reports degradation via warnings.
    """

    effort: ReasoningEffort = "medium"
    thinking_budget: int | None = None
    total_budget: int | None = None

    def __post_init__(self) -> None:
        if self.effort not in {"off", "adaptive", "minimal", "low", "medium", "high", "xhigh"}:
            raise ValueError(f"unsupported reasoning effort: {self.effort}")
        _validate_positive(self.thinking_budget, field_name="thinking_budget")
        _validate_positive(self.total_budget, field_name="total_budget")


@dataclass(frozen=True, slots=True)
class ToolChoice:
    """How the model should use tools."""

    mode: Literal["auto", "required", "none"] = "auto"
    allowed: tuple[str, ...] = ()
    parallel: bool | None = None

    def __post_init__(self) -> None:
        if self.mode not in {"auto", "required", "none"}:
            raise ValueError(f"unsupported tool choice mode: {self.mode}")
        object.__setattr__(self, "allowed", tuple(self.allowed))
        if any(not name for name in self.allowed):
            raise ValueError("ToolChoice.allowed cannot contain empty tool names")


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
    response_format: JsonObject | None = None
    tool_choice: ToolChoice | None = None
    reasoning: Reasoning | None = None
    extensions: Extensions | None = None

    def __post_init__(self) -> None:
        if self.stop is None:
            stop = ()
        elif isinstance(self.stop, str):
            stop = (self.stop,)
        else:
            stop = tuple(self.stop)
        object.__setattr__(self, "stop", stop)
        if self.max_tokens is not None and self.max_tokens <= 0:
            raise ValueError("max_tokens must be > 0")
        if self.temperature is not None and self.temperature < 0:
            raise ValueError("temperature must be >= 0")
        if self.top_p is not None and not (0 <= self.top_p <= 1):
            raise ValueError("top_p must be in [0, 1]")
        if self.top_k is not None and self.top_k <= 0:
            raise ValueError("top_k must be > 0")
        if any(not isinstance(s, str) or not s for s in self.stop):
            raise ValueError("stop must contain non-empty strings")
        if self.tool_choice is not None and not isinstance(self.tool_choice, ToolChoice):
            raise TypeError("tool_choice must be a ToolChoice")
        if self.reasoning is not None and not isinstance(self.reasoning, Reasoning):
            raise TypeError("reasoning must be a Reasoning")
        _validate_json_object(self.response_format, field_name="response_format")
        _validate_json_object(self.extensions, field_name="extensions")


# ─── Request ─────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class _ModelRequest:
    """Base for endpoint requests/configs that require a model."""

    model: str

    def __post_init__(self) -> None:
        if not self.model:
            raise ValueError("model is required")


@dataclass(frozen=True, slots=True)
class Request(_ModelRequest):
    """A complete request to a foundation model.

    The composed artifact sent to the model — conversation history,
    system instructions, available tools, and generation config.

    cache — enable prompt caching (default ``True``).  When enabled,
    the adapter places cache breakpoints on the system prompt and
    conversation history so that stable prefixes are reused across
    requests.  On Anthropic this translates to ``cache_control``
    annotations; on OpenAI and Gemini caching is automatic.
    Set to ``False`` only for known one-shot requests where the
    25% Anthropic write surcharge matters.
    """

    messages: tuple[Message, ...]
    system: SystemContent | None = None
    tools: tuple[Tool, ...] = ()
    config: Config = field(default_factory=Config)
    cache: bool = True

    def __post_init__(self) -> None:
        _ModelRequest.__post_init__(self)
        object.__setattr__(self, "messages", tuple(self.messages))
        object.__setattr__(self, "tools", tuple(self.tools))
        if not self.messages:
            raise ValueError("at least one message is required")
        if not all(isinstance(m, Message) for m in self.messages):
            raise TypeError("Request.messages must contain Message objects")
        if not all(isinstance(t, (FunctionTool, BuiltinTool)) for t in self.tools):
            raise TypeError("Request.tools must contain Tool objects")
        if not isinstance(self.config, Config):
            raise TypeError("Request.config must be a Config")
        if isinstance(self.system, list):
            object.__setattr__(self, "system", tuple(self.system))
        if isinstance(self.system, tuple):
            if not self.system:
                raise ValueError("system parts cannot be empty")
            if not all(_is_part(p) for p in self.system):
                raise TypeError("system parts must contain Part objects")


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

    def __post_init__(self) -> None:
        for field_name in (
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "cache_read_tokens",
            "cache_write_tokens",
            "reasoning_tokens",
            "input_audio_tokens",
            "output_audio_tokens",
        ):
            _validate_non_negative(getattr(self, field_name), field_name=field_name)


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
    provider_data: ProviderData | None = None

    def __post_init__(self) -> None:
        if not self.model:
            raise ValueError("Response requires model")
        if not isinstance(self.message, Message):
            raise TypeError("Response.message must be a Message")
        if self.finish_reason not in {"stop", "length", "tool_call", "content_filter", "error"}:
            raise ValueError(f"unsupported finish reason: {self.finish_reason}")
        if not isinstance(self.usage, Usage):
            raise TypeError("Response.usage must be a Usage")
        _validate_json_object(self.provider_data, field_name="provider_data")

    def _require_part(self, cls: type[_P], label: str) -> _P:
        part = self.message.first(cls)
        if part is None:
            raise ValueError(
                f"Response contains no {label}. "
                f"Parts: {[p.type for p in self.message.parts]}"
            )
        return part

    @property
    def text(self) -> str | None:
        return self.message.text

    @property
    def tool_calls(self) -> list[ToolCallPart]:
        return self.message.parts_of(ToolCallPart)

    @property
    def image(self) -> ImagePart | None:
        return self.message.first(ImagePart)

    @property
    def images(self) -> list[ImagePart]:
        return self.message.parts_of(ImagePart)

    @property
    def audio(self) -> AudioPart | None:
        return self.message.first(AudioPart)

    @property
    def video(self) -> VideoPart | None:
        return self.message.first(VideoPart)

    @property
    def videos(self) -> list[VideoPart]:
        return self.message.parts_of(VideoPart)

    @property
    def document(self) -> DocumentPart | None:
        return self.message.first(DocumentPart)

    @property
    def documents(self) -> list[DocumentPart]:
        return self.message.parts_of(DocumentPart)

    @property
    def thinking(self) -> str | None:
        texts = [p.text for p in self.message.parts_of(ThinkingPart)]
        return "\n".join(texts) if texts else None

    @property
    def citations(self) -> list[CitationPart]:
        return self.message.parts_of(CitationPart)

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
            raise ValueError(
                f"Cannot parse response as JSON: {e}\nRaw text: {preview}"
            ) from e

    @property
    def image_bytes(self) -> bytes:
        return self._require_part(ImagePart, "image").bytes

    @property
    def audio_bytes(self) -> bytes:
        return self._require_part(AudioPart, "audio").bytes

    @property
    def video_bytes(self) -> bytes:
        return self._require_part(VideoPart, "video").bytes

    @property
    def document_bytes(self) -> bytes:
        return self._require_part(DocumentPart, "document").bytes


# ─── Embeddings ──────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class EmbeddingRequest(_ModelRequest):
    inputs: tuple[str, ...]
    extensions: Extensions | None = None

    def __post_init__(self) -> None:
        _ModelRequest.__post_init__(self)
        object.__setattr__(self, "inputs", tuple(self.inputs))
        if not self.inputs:
            raise ValueError("inputs cannot be empty")
        if any(not isinstance(x, str) or x == "" for x in self.inputs):
            raise ValueError("inputs must contain non-empty strings")
        _validate_json_object(self.extensions, field_name="extensions")


@dataclass(frozen=True, slots=True)
class EmbeddingResponse:
    model: str
    vectors: tuple[tuple[float, ...], ...]
    usage: Usage = field(default_factory=Usage)
    provider_data: ProviderData | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "vectors", tuple(tuple(v) for v in self.vectors))
        _validate_json_object(self.provider_data, field_name="provider_data")


# ─── File Upload ─────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class FileUploadRequest:
    model: str | None = None
    filename: str = "file.bin"
    bytes_data: bytes = b""
    media_type: str = "application/octet-stream"
    extensions: Extensions | None = None

    def __post_init__(self) -> None:
        if self.model == "":
            raise ValueError("model cannot be empty")
        if not self.filename:
            raise ValueError("filename is required")
        if not self.media_type:
            raise ValueError("media_type is required")
        _validate_json_object(self.extensions, field_name="extensions")


@dataclass(frozen=True, slots=True)
class FileUploadResponse:
    id: str
    provider_data: ProviderData | None = None

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("FileUploadResponse requires id")
        _validate_json_object(self.provider_data, field_name="provider_data")


# ─── Batch ───────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class BatchRequest(_ModelRequest):
    requests: tuple[Request, ...]
    extensions: Extensions | None = None

    def __post_init__(self) -> None:
        _ModelRequest.__post_init__(self)
        object.__setattr__(self, "requests", tuple(self.requests))
        if not self.requests:
            raise ValueError("requests cannot be empty")
        if not all(isinstance(r, Request) for r in self.requests):
            raise TypeError("BatchRequest.requests must contain Request objects")
        _validate_json_object(self.extensions, field_name="extensions")


@dataclass(frozen=True, slots=True)
class BatchResponse:
    id: str
    status: str
    provider_data: ProviderData | None = None

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("BatchResponse requires id")
        if not self.status:
            raise ValueError("BatchResponse requires status")
        _validate_json_object(self.provider_data, field_name="provider_data")


@dataclass(frozen=True, slots=True)
class _PromptRequest(_ModelRequest):
    """Base for endpoint requests that require a non-empty text prompt."""

    prompt: str

    def __post_init__(self) -> None:
        _ModelRequest.__post_init__(self)
        if not self.prompt:
            raise ValueError("prompt is required")


# ─── Image Generation ────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class ImageGenerationRequest(_PromptRequest):
    size: str | None = None
    extensions: Extensions | None = None

    def __post_init__(self) -> None:
        _PromptRequest.__post_init__(self)
        _validate_json_object(self.extensions, field_name="extensions")


@dataclass(frozen=True, slots=True)
class ImageGenerationResponse:
    images: tuple[ImagePart, ...]
    provider_data: ProviderData | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "images", tuple(self.images))
        if not self.images:
            raise ValueError("ImageGenerationResponse requires at least one image")
        if not all(isinstance(img, ImagePart) for img in self.images):
            raise TypeError("images must contain ImagePart objects")
        _validate_json_object(self.provider_data, field_name="provider_data")


# ─── Audio Generation ────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class AudioGenerationRequest(_PromptRequest):
    voice: str | None = None
    format: str | None = None
    extensions: Extensions | None = None

    def __post_init__(self) -> None:
        _PromptRequest.__post_init__(self)
        _validate_json_object(self.extensions, field_name="extensions")


@dataclass(frozen=True, slots=True)
class AudioGenerationResponse:
    audio: AudioPart
    provider_data: ProviderData | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.audio, AudioPart):
            raise TypeError("audio must be an AudioPart")
        _validate_json_object(self.provider_data, field_name="provider_data")


EndpointRequest: TypeAlias = (
    Request
    | EmbeddingRequest
    | FileUploadRequest
    | BatchRequest
    | ImageGenerationRequest
    | AudioGenerationRequest
)

EndpointResponse: TypeAlias = (
    Response
    | EmbeddingResponse
    | FileUploadResponse
    | BatchResponse
    | ImageGenerationResponse
    | AudioGenerationResponse
)


# ─── Audio Format ────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class AudioFormat:
    encoding: Literal["pcm16", "opus", "mp3", "aac"]
    sample_rate: int
    channels: int = 1

    def __post_init__(self) -> None:
        if self.encoding not in {"pcm16", "opus", "mp3", "aac"}:
            raise ValueError(f"unsupported audio encoding: {self.encoding}")
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be > 0")
        if self.channels <= 0:
            raise ValueError("channels must be > 0")


# ─── Live (Realtime) ─────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class LiveConfig(_ModelRequest):
    system: SystemContent | None = None
    tools: tuple[Tool, ...] = ()
    voice: str | None = None
    input_format: AudioFormat | None = None
    output_format: AudioFormat | None = None
    extensions: Extensions | None = None

    def __post_init__(self) -> None:
        _ModelRequest.__post_init__(self)
        object.__setattr__(self, "tools", tuple(self.tools))
        if not all(isinstance(t, (FunctionTool, BuiltinTool)) for t in self.tools):
            raise TypeError("LiveConfig.tools must contain Tool objects")
        if isinstance(self.system, list):
            object.__setattr__(self, "system", tuple(self.system))
        if isinstance(self.system, tuple):
            if not self.system:
                raise ValueError("system parts cannot be empty")
            if not all(_is_part(p) for p in self.system):
                raise TypeError("system parts must contain Part objects")
        if self.input_format is not None and not isinstance(self.input_format, AudioFormat):
            raise TypeError("input_format must be an AudioFormat")
        if self.output_format is not None and not isinstance(self.output_format, AudioFormat):
            raise TypeError("output_format must be an AudioFormat")
        _validate_json_object(self.extensions, field_name="extensions")


_LIVE_CLIENT_REQUIRED_FIELDS: dict[str, tuple[str, ...]] = {
    "audio": ("data",),
    "video": ("data",),
    "text": ("text",),
}

_LIVE_SERVER_REQUIRED_FIELDS: dict[str, tuple[str, ...]] = {
    "audio": ("data",),
    "text": ("text",),
    "tool_call": ("input",),
    "turn_end": ("usage",),
    "error": ("error",),
}


@dataclass(frozen=True, slots=True)
class LiveClientEvent:
    type: Literal["audio", "video", "text", "tool_result", "interrupt", "end_audio"]
    data: str | None = None
    text: str | None = None
    id: str | None = None
    content: tuple[Part, ...] = ()

    def __post_init__(self) -> None:
        if self.type not in {"audio", "video", "text", "tool_result", "interrupt", "end_audio"}:
            raise ValueError(f"unsupported live client event type: {self.type}")
        object.__setattr__(self, "content", tuple(self.content))
        _require_fields(
            f"LiveClientEvent(type={self.type!r})",
            self,
            _LIVE_CLIENT_REQUIRED_FIELDS.get(self.type, ()),
        )
        if self.type == "tool_result":
            if not self.id:
                raise ValueError("LiveClientEvent(type='tool_result') requires id")
            if not self.content:
                raise ValueError("LiveClientEvent(type='tool_result') requires content")
            if not all(_is_part(p) for p in self.content):
                raise TypeError("LiveClientEvent.content must contain Part objects")


@dataclass(frozen=True, slots=True)
class LiveServerEvent:
    type: Literal["audio", "text", "tool_call", "interrupted", "turn_end", "error"]
    data: str | None = None
    text: str | None = None
    id: str | None = None
    name: str | None = None
    input: JsonObject | None = None
    usage: Usage | None = None
    error: ErrorDetail | None = None

    def __post_init__(self) -> None:
        if self.type not in {"audio", "text", "tool_call", "interrupted", "turn_end", "error"}:
            raise ValueError(f"unsupported live server event type: {self.type}")
        _require_fields(
            f"LiveServerEvent(type={self.type!r})",
            self,
            _LIVE_SERVER_REQUIRED_FIELDS.get(self.type, ()),
        )
        if self.type == "tool_call":
            if not self.id:
                raise ValueError("LiveServerEvent(type='tool_call') requires id")
            if not self.name:
                raise ValueError("LiveServerEvent(type='tool_call') requires name")
            _validate_json_object(self.input, field_name="input")


# ─── ToolCallInfo (for callbacks) ────────────────────────────────────
#
# Lightweight callback payload: same identity/input shape as ToolCallPart,
# but without the content-part discriminator.


@dataclass(frozen=True, slots=True)
class ToolCallInfo:
    id: str
    name: str
    input: JsonObject

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("ToolCallInfo requires id")
        if not self.name:
            raise ValueError("ToolCallInfo requires name")
        _validate_json_object(self.input, field_name="input")
