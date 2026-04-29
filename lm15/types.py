"""
lm15.types — Core vocabulary for foundation model interaction.

The fundamental unit is the Part: an atomic, typed block of content.
Parts compose into Messages (attributed to a speaker).
Messages compose into Requests (sent to a model).
Models produce Responses (containing a Message).

Streams reveal Responses incrementally through Deltas — typed fragments
of streamable response parts.  Not every Part is streamable; use
StreamablePart / NonStreamablePart to make that boundary explicit.

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
   after construction, validated once in __post_init__, hashable when
   their scalar fields are hashable, and memory-efficient.  JSON
   dict/list payloads are recursively frozen into dict/list-compatible
   read-only containers with stable hashes.

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

import base64 as _base64
import binascii as _binascii
import inspect as _inspect
import json as _json
import math as _math
import mimetypes as _mimetypes
import re as _re
import warnings as _warnings
from collections.abc import Sequence
from dataclasses import dataclass, field
from os import PathLike
from pathlib import Path
from types import NoneType, UnionType
from typing import Annotated, Any, Callable, Literal, TypeAlias, TypeVar, Union, get_args, get_origin


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

# FinishReason values are a separate namespace from PartType values even when
# a token such as "tool_call" appears in both.
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
ERROR_CODES = frozenset(get_args(ErrorCode))

StreamEventType = Literal["start", "delta", "end", "error"]
BatchStatus = Literal["submitted", "queued", "running", "completed", "failed", "cancelled"]
AudioEncoding = Literal["pcm16", "opus", "mp3", "aac"]
ToolChoiceMode = Literal["auto", "required", "none"]
LiveClientEventType = Literal["audio", "video", "text", "tool_result", "interrupt", "end_audio"]
LiveServerEventType = Literal["audio", "text", "tool_call", "tool_call_delta", "interrupted", "turn_end", "error"]

ROLE_VALUES = frozenset(get_args(Role))
FINISH_REASONS = frozenset(get_args(FinishReason))
REASONING_EFFORTS = frozenset(get_args(ReasoningEffort))
STREAM_EVENT_TYPES = frozenset(get_args(StreamEventType))
BATCH_STATUSES = frozenset(get_args(BatchStatus))
AUDIO_ENCODINGS = frozenset(get_args(AudioEncoding))
TOOL_CHOICE_MODES = frozenset(get_args(ToolChoiceMode))
LIVE_CLIENT_EVENT_TYPES = frozenset(get_args(LiveClientEventType))
LIVE_SERVER_EVENT_TYPES = frozenset(get_args(LiveServerEventType))

_P = TypeVar("_P")

# JSON-compatible values used for model inputs, tool schemas, provider
# extensions, and provider metadata.  Keep these small and boring: the
# adapter layer can map richer Python objects into this vocabulary before
# they enter the core model.
JsonPrimitive: TypeAlias = None | bool | int | float | str
JsonValue: TypeAlias = JsonPrimitive | list["JsonValue"] | dict[str, "JsonValue"]
JsonArray: TypeAlias = list[JsonValue]
JsonObject: TypeAlias = dict[str, JsonValue]


def _hashable_json_value(value: JsonValue) -> object:
    """Return a hashable representation matching JSON equality semantics."""
    if isinstance(value, dict):
        return tuple(sorted((key, _hashable_json_value(item)) for key, item in value.items()))
    if isinstance(value, list):
        return tuple(_hashable_json_value(item) for item in value)
    return value


class _FrozenJsonObject(dict[str, JsonValue]):
    """A JSON object that remains dict-compatible but cannot be mutated."""

    __slots__ = ("_hash",)

    def unwrap(self) -> JsonObject:
        """Return a mutable deep copy using plain JSON containers."""
        return _thaw_json_value(self)  # type: ignore[return-value]

    def __hash__(self) -> int:
        try:
            return self._hash
        except AttributeError:
            value = hash(_hashable_json_value(self))
            object.__setattr__(self, "_hash", value)
            return value

    def _readonly(self, *args: Any, **kwargs: Any) -> None:
        raise TypeError("JSON objects on lm15 types are immutable")

    __setitem__ = _readonly
    __delitem__ = _readonly
    __ior__ = _readonly
    __setattr__ = _readonly
    clear = _readonly
    pop = _readonly
    popitem = _readonly
    setdefault = _readonly
    update = _readonly


class _FrozenJsonArray(list[JsonValue]):
    """A JSON array that remains list-compatible but cannot be mutated."""

    __slots__ = ("_hash",)

    def unwrap(self) -> JsonArray:
        """Return a mutable deep copy using plain JSON containers."""
        return _thaw_json_value(self)  # type: ignore[return-value]

    def __hash__(self) -> int:
        try:
            return self._hash
        except AttributeError:
            value = hash(_hashable_json_value(self))
            object.__setattr__(self, "_hash", value)
            return value

    def _readonly(self, *args: Any, **kwargs: Any) -> None:
        raise TypeError("JSON arrays on lm15 types are immutable")

    __setitem__ = _readonly
    __delitem__ = _readonly
    __iadd__ = _readonly
    __imul__ = _readonly
    __setattr__ = _readonly
    append = _readonly
    clear = _readonly
    extend = _readonly
    insert = _readonly
    pop = _readonly
    remove = _readonly
    reverse = _readonly
    sort = _readonly


def _is_json_value(value: Any) -> bool:
    """Return True if value is representable as standard JSON.

    ``bool`` is matched before ``int`` because ``bool`` is a subclass of
    ``int`` in Python; both serialize correctly, but the explicit ordering
    documents the intent.
    """
    if value is None or isinstance(value, bool) or isinstance(value, (int, str)):
        return True
    if isinstance(value, float):
        return _math.isfinite(value)
    if isinstance(value, list):
        return all(_is_json_value(x) for x in value)
    if isinstance(value, dict):
        return all(isinstance(k, str) and _is_json_value(v) for k, v in value.items())
    return False


def _freeze_json_value(value: JsonValue) -> JsonValue:
    """Recursively freeze JSON containers while keeping JSON-compatible shapes."""
    if isinstance(value, _FrozenJsonObject | _FrozenJsonArray):
        return value
    if isinstance(value, dict):
        return _FrozenJsonObject(
            {key: _freeze_json_value(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return _FrozenJsonArray([_freeze_json_value(item) for item in value])
    return value


def _thaw_json_value(value: JsonValue) -> JsonValue:
    """Recursively copy frozen JSON containers into mutable JSON containers."""
    if isinstance(value, dict):
        return {key: _thaw_json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_thaw_json_value(item) for item in value]
    return value


def _freeze_json_object(
    value: Any, *, field_name: str, required: bool
) -> JsonObject | None:
    """Validate and freeze a JSON object."""
    if value is None:
        if required:
            raise TypeError(f"{field_name} must be a JSON object")
        return None
    if not isinstance(value, dict):
        raise TypeError(f"{field_name} must be a JSON object")
    if not _is_json_value(value):
        raise TypeError(f"{field_name} must contain only JSON-compatible values")
    return _freeze_json_value(value)  # type: ignore[return-value]


def _freeze_field(obj: object, field_name: str, *, required: bool = False) -> None:
    """Freeze a JSON object field in-place on a frozen dataclass."""
    object.__setattr__(
        obj,
        field_name,
        _freeze_json_object(
            getattr(obj, field_name), field_name=field_name, required=required
        ),
    )


def _freeze_extensions_field(obj: object) -> None:
    """Freeze an extensions field, normalizing an empty mapping to None."""
    if getattr(obj, "extensions") == {}:
        object.__setattr__(obj, "extensions", None)
        return
    _freeze_field(obj, "extensions")


def _validate_int(value: Any, *, field_name: str) -> None:
    """Reject non-int values, including bool (which subclasses int)."""
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an int")


def _validate_positive(value: int | None, *, field_name: str) -> None:
    _validate_int(value, field_name=field_name)
    if value is not None and value <= 0:
        raise ValueError(f"{field_name} must be > 0")


def _validate_non_negative(value: int | None, *, field_name: str) -> None:
    _validate_int(value, field_name=field_name)
    if value is not None and value < 0:
        raise ValueError(f"{field_name} must be >= 0")


def _validate_part_index(part_index: int) -> None:
    _validate_int(part_index, field_name="part_index")
    if part_index < 0:
        raise ValueError("part_index must be >= 0")


def _validate_text(value: Any, *, field_name: str, allow_empty: bool = True) -> None:
    """Validate that value is a string (and optionally non-empty)."""
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    if not allow_empty and value == "":
        raise ValueError(f"{field_name} cannot be empty")


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
    if not isinstance(value, str):
        raise TypeError(f"{part_type} {name} must be a string")
    if value == "":
        raise ValueError(f"{part_type} {name} cannot be empty")


_BASE64_RE = _re.compile(r"^[A-Za-z0-9+/]*={0,2}$")


def _base64_payload(part_type: str, data: str | None) -> str:
    """Return the base64 payload from a raw base64 string or data URI."""
    if data is None:
        raise ValueError(
            f"{part_type} has no inline data; fetch url/file_id-addressed media before decoding"
        )
    if not isinstance(data, str):
        raise TypeError(f"{part_type}.data must be a base64 string")
    if data == "":
        raise ValueError(f"{part_type}.data cannot be empty")
    if data.startswith("data:") and ";base64," in data:
        data = data.split(";base64,", 1)[1]
    return "".join(data.split())


def _validate_base64_data(part_type: str, data: str | None) -> None:
    """Validate base64 shape without eagerly decoding large payloads."""
    payload = _base64_payload(part_type, data)
    if len(payload) % 4 != 0 or not _BASE64_RE.fullmatch(payload):
        raise ValueError(f"{part_type}.data must be a valid base64 string")


def _decode_data(part_type: str, data: str | None) -> bytes:
    """Decode base64 data to bytes."""
    payload = _base64_payload(part_type, data)
    try:
        return _base64.b64decode(payload, validate=True)
    except (_binascii.Error, ValueError) as e:
        raise ValueError(f"{part_type}.data must be a valid base64 string") from e


def _base64_summary(data: str | None) -> str | None:
    """Return a short repr-safe summary for base64 data."""
    if data is None:
        return None
    payload = _base64_payload("media", data)
    return f"<base64: {len(payload)} chars>"


def _bytes_summary(data: bytes | bytearray) -> str:
    """Return a short repr-safe summary for raw bytes."""
    return f"<bytes: {len(data)} bytes>"


@dataclass(frozen=True, slots=True, repr=False)
class _MediaMixin:
    """Shared fields, validation, repr, and byte access for media parts."""

    media_type: str
    data: str | None = None
    url: str | None = None
    file_id: str | None = None
    _decoded_bytes: bytes | None = field(default=None, init=False, repr=False, compare=False, hash=False)

    def __post_init__(self) -> None:
        if not isinstance(self.media_type, str) or self.media_type == "":
            raise ValueError(f"{self.__class__.__name__} requires media_type")
        _validate_media(self.__class__.__name__, self.data, self.url, self.file_id)
        if self.data is not None:
            _validate_base64_data(self.__class__.__name__, self.data)

    def __repr__(self) -> str:
        fields = [
            ("media_type", self.media_type),
            ("data", _base64_summary(self.data)),
            ("url", self.url),
            ("file_id", self.file_id),
        ]
        detail = getattr(self, "detail", None)
        if detail is not None:
            fields.append(("detail", detail))
        args = ", ".join(f"{name}={value!r}" for name, value in fields)
        return f"{self.__class__.__name__}({args})"

    @property
    def bytes(self) -> bytes:
        if self._decoded_bytes is None:
            object.__setattr__(self, "_decoded_bytes", _decode_data(self.__class__.__name__, self.data))
        return self._decoded_bytes


# ─── Parts ───────────────────────────────────────────────────────────
#
# The atoms of content.  A discriminated union — check .type or use
# isinstance(), then access the fields that belong to that variant.


@dataclass(frozen=True, slots=True)
class TextPart:
    """A block of text content."""

    text: str
    type: Literal["text"] = field(default="text", init=False)

    def __post_init__(self) -> None:
        _validate_text(self.text, field_name="TextPart.text")


@dataclass(frozen=True, slots=True, repr=False)
class ImagePart(_MediaMixin):
    """An image, addressed by exactly one of data/url/file_id."""

    media_type: str = "image/png"
    detail: Literal["low", "high", "auto"] | None = None
    type: Literal["image"] = field(default="image", init=False)


@dataclass(frozen=True, slots=True, repr=False)
class AudioPart(_MediaMixin):
    """Audio content, addressed by exactly one of data/url/file_id."""

    media_type: str = "audio/wav"
    type: Literal["audio"] = field(default="audio", init=False)


@dataclass(frozen=True, slots=True, repr=False)
class VideoPart(_MediaMixin):
    """Video content, addressed by exactly one of data/url/file_id."""

    media_type: str = "video/mp4"
    type: Literal["video"] = field(default="video", init=False)


@dataclass(frozen=True, slots=True, repr=False)
class DocumentPart(_MediaMixin):
    """A document (PDF, etc.), addressed by exactly one of data/url/file_id."""

    media_type: str = "application/pdf"
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
        _freeze_field(self, "input", required=True)


@dataclass(frozen=True, slots=True)
class ToolResultPart:
    """The result of an external computation, sent back to the model."""

    id: str
    content: tuple[ToolResultContentPart, ...]
    name: str | None = None
    is_error: bool = False
    type: Literal["tool_result"] = field(default="tool_result", init=False)

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("ToolResultPart requires id")
        object.__setattr__(self, "content", tuple(self.content))
        if not self.content:
            raise ValueError("ToolResultPart requires content")
        if not all(_is_part(p) for p in self.content):
            raise TypeError("ToolResultPart.content must contain Part objects")
        if not isinstance(self.is_error, bool):
            raise TypeError("ToolResultPart.is_error must be a bool")
        # Tool results may not contain protocol parts: tool calls, nested
        # tool results, model reasoning traces, or refusals.  Only the
        # presentational variants from ToolResultContentPart are allowed.
        if any(isinstance(p, _TOOL_RESULT_FORBIDDEN_PARTS) for p in self.content):
            raise TypeError(
                "ToolResultPart.content cannot contain tool calls, nested tool "
                "results, thinking parts, or refusals"
            )


@dataclass(frozen=True, slots=True)
class ThinkingPart:
    """Model reasoning trace — may be redacted by the provider."""

    text: str
    redacted: bool = False
    type: Literal["thinking"] = field(default="thinking", init=False)

    def __post_init__(self) -> None:
        _validate_text(self.text, field_name="ThinkingPart.text")
        if not isinstance(self.redacted, bool):
            raise TypeError("ThinkingPart.redacted must be a bool")


@dataclass(frozen=True, slots=True)
class RefusalPart:
    """Model explicitly refused to respond.

    Refusals require non-empty text because they are final semantic content;
    empty ``TextPart``/``ThinkingPart`` values remain allowed for streaming
    reassembly and provider redaction edge cases.
    """

    text: str
    type: Literal["refusal"] = field(default="refusal", init=False)

    def __post_init__(self) -> None:
        _validate_text(self.text, field_name="RefusalPart.text", allow_empty=False)


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


_TOOL_RESULT_FORBIDDEN_PARTS: tuple[type, ...] = (
    ToolCallPart,
    ToolResultPart,
    ThinkingPart,
    RefusalPart,
)


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

# Runtime dispatch table, derived from the union so adding a Part variant has
# one source of truth: the variant class plus the Part union.
def _variant_type(cls: type) -> str:
    return cls.__dataclass_fields__["type"].default  # type: ignore[attr-defined]


PART_CLASSES: tuple[type, ...] = get_args(Part)
PART_TYPES: dict[str, type] = {_variant_type(cls): cls for cls in PART_CLASSES}


def _is_part(value: object) -> bool:
    """Return True if value is one of lm15's concrete Part variants."""
    return isinstance(value, PART_CLASSES)


MediaPart: TypeAlias = ImagePart | AudioPart | VideoPart | DocumentPart
MEDIA_TYPES: tuple[type, ...] = get_args(MediaPart)

# Shared endpoint metadata/content aliases
Extensions: TypeAlias = JsonObject
ProviderData: TypeAlias = JsonObject

# Parts allowed in tool result content: presentational variants only.
# (ToolCallPart, ToolResultPart, ThinkingPart, and RefusalPart are excluded
# both at the type level and in ``ToolResultPart.__post_init__``.)
ToolResultContentPart: TypeAlias = (
    TextPart | ImagePart | AudioPart | VideoPart | DocumentPart | CitationPart
)
ToolResultContent: TypeAlias = str | ToolResultContentPart | Sequence[ToolResultContentPart]

# Parts allowed in prompts (user/developer messages and system content).
# Excludes model/tool protocol parts which are produced by the model or
# tool runtime, never authored by the caller.
PromptPart: TypeAlias = (
    TextPart | ImagePart | AudioPart | VideoPart | DocumentPart
)
PartInput: TypeAlias = str | PromptPart | Sequence[PromptPart]
SystemContent: TypeAlias = str | PromptPart | Sequence[PromptPart]

# Parts forbidden in prompts (user/developer messages and system content).
# Defined once and reused by every prompt-side validator.
_PROMPT_FORBIDDEN_PARTS: tuple[type, ...] = (
    ToolCallPart,
    ToolResultPart,
    ThinkingPart,
    RefusalPart,
    CitationPart,
)


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
        return _base64.b64encode(data).decode("ascii")
    return data


def _prepare_media_factory_input(
    part_type: str,
    *,
    url: str | None,
    data: bytes | str | None,
    file_id: str | None,
    path: str | PathLike[str] | None,
    media_type: str | None,
    default_media_type: str,
) -> tuple[str | None, str]:
    provided = [
        name
        for name, value in (
            ("data", data),
            ("url", url),
            ("file_id", file_id),
            ("path", path),
        )
        if value is not None
    ]
    if len(provided) != 1:
        raise ValueError(
            f"{part_type} requires exactly one of data, url, file_id, or path"
        )
    if path is not None:
        if str(path) == "":
            raise ValueError(f"{part_type} path cannot be empty")
        media_path = Path(path)
        data = media_path.read_bytes()
        media_type = media_type or _mimetypes.guess_type(str(media_path))[0]
    encoded = _encode_data(data) if data is not None else None
    return encoded, media_type or default_media_type


def image(
    *,
    url: str | None = None,
    data: bytes | str | None = None,
    path: str | PathLike[str] | None = None,
    file_id: str | None = None,
    media_type: str | None = None,
    detail: Literal["low", "high", "auto"] | None = None,
) -> ImagePart:
    encoded_data, resolved_media_type = _prepare_media_factory_input(
        "ImagePart",
        url=url,
        data=data,
        file_id=file_id,
        path=path,
        media_type=media_type,
        default_media_type="image/png",
    )
    return ImagePart(
        media_type=resolved_media_type,
        data=encoded_data,
        url=url,
        file_id=file_id,
        detail=detail,
    )


def audio(
    *,
    url: str | None = None,
    data: bytes | str | None = None,
    path: str | PathLike[str] | None = None,
    file_id: str | None = None,
    media_type: str | None = None,
) -> AudioPart:
    encoded_data, resolved_media_type = _prepare_media_factory_input(
        "AudioPart",
        url=url,
        data=data,
        file_id=file_id,
        path=path,
        media_type=media_type,
        default_media_type="audio/wav",
    )
    return AudioPart(
        media_type=resolved_media_type,
        data=encoded_data,
        url=url,
        file_id=file_id,
    )


def video(
    *,
    url: str | None = None,
    data: bytes | str | None = None,
    path: str | PathLike[str] | None = None,
    file_id: str | None = None,
    media_type: str | None = None,
) -> VideoPart:
    encoded_data, resolved_media_type = _prepare_media_factory_input(
        "VideoPart",
        url=url,
        data=data,
        file_id=file_id,
        path=path,
        media_type=media_type,
        default_media_type="video/mp4",
    )
    return VideoPart(
        media_type=resolved_media_type,
        data=encoded_data,
        url=url,
        file_id=file_id,
    )


def document(
    *,
    url: str | None = None,
    data: bytes | str | None = None,
    path: str | PathLike[str] | None = None,
    file_id: str | None = None,
    media_type: str | None = None,
) -> DocumentPart:
    encoded_data, resolved_media_type = _prepare_media_factory_input(
        "DocumentPart",
        url=url,
        data=data,
        file_id=file_id,
        path=path,
        media_type=media_type,
        default_media_type="application/pdf",
    )
    return DocumentPart(
        media_type=resolved_media_type,
        data=encoded_data,
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
    (which becomes a TextPart).  Binary tool results must be wrapped in an
    appropriate media part (ImagePart/AudioPart/VideoPart/DocumentPart) rather
    than passed as raw bytes.
    """
    parts = _normalize_parts(content)  # type: ignore[arg-type]
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
        if self.role not in ROLE_VALUES:
            raise ValueError(f"unsupported role: {self.role}")
        if isinstance(self.parts, str):
            raise TypeError("Message.parts must be Part objects; use Message.user('text') for strings")
        parts = (self.parts,) if _is_part(self.parts) else tuple(self.parts)
        object.__setattr__(self, "parts", parts)
        if not self.parts:
            raise ValueError("Message requires at least one part")
        if not all(_is_part(p) for p in self.parts):
            raise TypeError("Message.parts must contain Part objects")
        _validate_message_parts(self.role, self.parts)

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
        parts = tuple(results)
        if not all(isinstance(p, ToolResultPart) for p in parts):
            raise TypeError("Message.tool() requires ToolResultPart objects")
        return Message(role="tool", parts=parts)

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
        if not parts:
            raise ValueError("content sequence cannot be empty")
        if not all(_is_part(p) for p in parts):
            raise TypeError("content sequence must contain Part objects")
        return parts
    raise TypeError("content must be a string, Part, or sequence of Parts")


def _validate_message_parts(role: Role, parts: tuple[Part, ...]) -> None:
    if role == "tool":
        if not all(isinstance(p, ToolResultPart) for p in parts):
            raise TypeError("tool messages may only contain ToolResultPart objects")
        return
    if role == "assistant":
        if any(isinstance(p, ToolResultPart) for p in parts):
            raise TypeError(
                "assistant messages cannot contain ToolResultPart objects"
            )
        return
    # User and developer messages carry prompts/instructions, not model/tool
    # protocol parts and not model-emitted artifacts like citations.
    if any(isinstance(p, _PROMPT_FORBIDDEN_PARTS) for p in parts):
        raise TypeError(f"{role} messages cannot contain model/tool protocol parts")


def _normalize_system(system: SystemContent | None) -> str | tuple[PromptPart, ...] | None:
    if system is None:
        return None
    if isinstance(system, str):
        if system == "":
            raise ValueError("system cannot be empty")
        return system
    parts = _normalize_parts(system)
    if any(isinstance(p, _PROMPT_FORBIDDEN_PARTS) for p in parts):
        raise TypeError("system parts cannot contain model/tool protocol parts")
    return parts


def _field_default(obj: object, field_name: str) -> Any:
    """Return the declared default for a dataclass field.

    Used to distinguish "populated" from "left at its default" in the
    flat tagged-union dataclasses (StreamEvent, LiveClientEvent,
    LiveServerEvent).  ``MISSING`` is returned via a unique sentinel so
    fields without a default are treated as required and never count as
    matching the default.
    """
    import dataclasses

    fields = getattr(obj, "__dataclass_fields__", {})
    f = fields.get(field_name)
    if f is None:
        return _MISSING
    if f.default is not dataclasses.MISSING:
        return f.default
    if f.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
        return f.default_factory()  # type: ignore[misc]
    return _MISSING


_MISSING: Any = object()


def _field_is_set(obj: object, field_name: str) -> bool:
    """Return True if a tagged-union field has been populated past its default.

    Defaults (``None``, ``()``, etc.) count as "unset"; any other value
    counts as explicitly populated by the caller.
    """
    value = getattr(obj, field_name)
    default = _field_default(obj, field_name)
    if default is _MISSING:
        return value is not None
    if default == () and isinstance(value, (list, tuple)) and len(value) == 0:
        return False
    return value != default


def _require_fields(owner: str, obj: object, fields: tuple[str, ...]) -> None:
    """Validate that required tagged-union fields have been populated."""
    for field_name in fields:
        if not _field_is_set(obj, field_name):
            raise ValueError(f"{owner} requires {field_name}")


def _forbid_fields(owner: str, obj: object, allowed: tuple[str, ...]) -> None:
    """Reject any tagged-union field outside ``allowed`` that has been set.

    The set of candidate fields is derived from ``obj``'s dataclass fields
    so that adding a new optional field to the variant automatically
    extends the forbidden-field check on every other variant.
    """
    fields = getattr(obj, "__dataclass_fields__", None)
    if not fields:
        return
    allowed_set = set(allowed) | {"type"}
    for field_name in fields:
        if field_name in allowed_set:
            continue
        if _field_is_set(obj, field_name):
            raise ValueError(f"{owner} cannot include {field_name}")


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


@dataclass(frozen=True, slots=True, repr=False)
class AudioDelta:
    """An audio data fragment arriving during streaming."""

    data: str
    part_index: int = 0
    media_type: str | None = None
    type: Literal["audio"] = field(default="audio", init=False)

    def __post_init__(self) -> None:
        _validate_part_index(self.part_index)
        if not isinstance(self.data, str) or self.data == "":
            raise ValueError("AudioDelta data must be a non-empty base64 string")
        _decode_data("AudioDelta", self.data)
        if self.media_type is not None and (
            not isinstance(self.media_type, str) or self.media_type == ""
        ):
            raise ValueError("AudioDelta media_type cannot be empty")

    def __repr__(self) -> str:
        return (
            "AudioDelta("
            f"data={_base64_summary(self.data)!r}, "
            f"part_index={self.part_index!r}, "
            f"media_type={self.media_type!r})"
        )


@dataclass(frozen=True, slots=True, repr=False)
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
        if self.media_type is not None and (
            not isinstance(self.media_type, str) or self.media_type == ""
        ):
            raise ValueError("ImageDelta media_type cannot be empty")
        _validate_media("ImageDelta", self.data, self.url, self.file_id)
        if self.data is not None:
            _decode_data("ImageDelta", self.data)

    def __repr__(self) -> str:
        return (
            "ImageDelta("
            f"data={_base64_summary(self.data)!r}, "
            f"url={self.url!r}, "
            f"file_id={self.file_id!r}, "
            f"part_index={self.part_index!r}, "
            f"media_type={self.media_type!r})"
        )


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

DELTA_CLASSES: tuple[type, ...] = get_args(Delta)
DELTA_TYPES: dict[str, type] = {_variant_type(cls): cls for cls in DELTA_CLASSES}

# Streaming boundary: every Delta variant assembles into exactly one
# StreamablePart, and NonStreamablePart variants must have no Delta
# representation.  The runtime consistency check below ties these manual
# unions to the derived DELTA_TYPES/PART_TYPES tables so adding a Part or
# Delta variant fails fast at module import if the boundary drifts.
StreamablePart: TypeAlias = TextPart | ThinkingPart | ImagePart | AudioPart | ToolCallPart | CitationPart
NonStreamablePart: TypeAlias = VideoPart | DocumentPart | ToolResultPart | RefusalPart

_STREAMABLE_PART_CLASSES: tuple[type, ...] = get_args(StreamablePart)
_NON_STREAMABLE_PART_CLASSES: tuple[type, ...] = get_args(NonStreamablePart)


def _check_streamable_partition() -> None:
    streamable = {_variant_type(cls) for cls in _STREAMABLE_PART_CLASSES}
    non_streamable = {_variant_type(cls) for cls in _NON_STREAMABLE_PART_CLASSES}
    delta_types = set(DELTA_TYPES)
    part_types = set(PART_TYPES)

    overlap = streamable & non_streamable
    if overlap:
        raise RuntimeError(
            f"StreamablePart and NonStreamablePart overlap on: {sorted(overlap)}"
        )
    union = streamable | non_streamable
    if union != part_types:
        missing = part_types - union
        extra = union - part_types
        raise RuntimeError(
            "StreamablePart ∪ NonStreamablePart must equal Part. "
            f"missing={sorted(missing)} extra={sorted(extra)}"
        )
    if delta_types != streamable:
        raise RuntimeError(
            "Delta variants must match StreamablePart. "
            f"delta_types={sorted(delta_types)} streamable={sorted(streamable)}"
        )


_check_streamable_partition()


# ─── Stream Events ───────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class ErrorDetail:
    """Structured error information.  A dataclass, not a dict."""

    code: ErrorCode
    message: str
    provider_code: str | None = None

    def __post_init__(self) -> None:
        if self.code not in ERROR_CODES:
            raise ValueError(f"unsupported error code: {self.code}")


_STREAM_EVENT_ALLOWED_FIELDS: dict[str, tuple[str, ...]] = {
    "start": ("id", "model"),
    "delta": ("delta",),
    "end": ("finish_reason", "usage", "provider_data"),
    "error": ("error",),
}

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
        if self.type not in STREAM_EVENT_TYPES:
            raise ValueError(f"unsupported stream event type: {self.type}")
        owner = f"StreamEvent(type={self.type!r})"
        _require_fields(owner, self, _STREAM_EVENT_REQUIRED_FIELDS.get(self.type, ()))
        _forbid_fields(owner, self, _STREAM_EVENT_ALLOWED_FIELDS[self.type])
        if self.delta is not None and not isinstance(self.delta, DELTA_CLASSES):
            raise TypeError("StreamEvent.delta must be a Delta")
        if self.usage is not None and not isinstance(self.usage, Usage):
            raise TypeError("StreamEvent.usage must be a Usage")
        if self.error is not None and not isinstance(self.error, ErrorDetail):
            raise TypeError("StreamEvent.error must be an ErrorDetail")
        _freeze_field(self, "provider_data")


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


def _json_schema_for_annotation(annotation: Any) -> JsonObject:
    """Best-effort JSON Schema for common Python annotations.

    Unknown annotations intentionally become ``{}`` (unconstrained) rather
    than pretending to be strings.
    """
    if annotation is Any:
        return {}

    origin = get_origin(annotation)
    args = get_args(annotation)

    if origin is Annotated:
        return _json_schema_for_annotation(args[0])

    if origin in (Union, UnionType):
        schemas = [_json_schema_for_annotation(arg) for arg in args]
        if any(arg is NoneType for arg in args):
            non_null = [s for arg, s in zip(args, schemas) if arg is not NoneType]
            if len(non_null) == 1:
                schema = dict(non_null[0])
                typ = schema.get("type")
                if isinstance(typ, str):
                    schema["type"] = [typ, "null"]
                return schema  # type: ignore[return-value]
        return {"anyOf": schemas}

    if origin is Literal:
        values = list(args)
        schema: JsonObject = {"enum": values}
        value_types = {type(value) for value in values if value is not None}
        if len(value_types) == 1:
            schema_type = _JSON_SCHEMA_TYPES.get(next(iter(value_types)))
            if schema_type is not None:
                schema["type"] = schema_type
        return schema

    if origin in (list, tuple, set, Sequence):
        schema: JsonObject = {"type": "array"}
        if args:
            schema["items"] = _json_schema_for_annotation(args[0])
        return schema

    if origin is dict:
        schema = {"type": "object"}
        if len(args) == 2 and args[1] is not Any:
            schema["additionalProperties"] = _json_schema_for_annotation(args[1])
        return schema

    json_schema_type = _JSON_SCHEMA_TYPES.get(origin) or _JSON_SCHEMA_TYPES.get(annotation)
    if json_schema_type is not None:
        return {"type": json_schema_type}
    return {}


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
        _freeze_field(self, "parameters", required=True)

    @staticmethod
    def from_fn(fn: Callable[..., Any]) -> "FunctionTool":
        """Infer a serializable tool spec from a callable's signature."""
        sig = _inspect.signature(fn)
        hints = _inspect.get_annotations(fn, eval_str=True)
        properties: JsonObject = {}
        required: list[str] = []
        for name, param in sig.parameters.items():
            if param.kind not in (
                _inspect.Parameter.POSITIONAL_OR_KEYWORD,
                _inspect.Parameter.KEYWORD_ONLY,
            ):
                continue
            ann = hints.get(name, str)
            properties[name] = _json_schema_for_annotation(ann)
            if param.default is _inspect.Parameter.empty:
                required.append(name)
        schema: JsonObject = {"type": "object", "properties": properties}
        if required:
            schema["required"] = required
        return FunctionTool(
            name=fn.__name__,
            description=(_inspect.getdoc(fn) or "").strip() or None,
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
        _freeze_field(self, "config")


Tool: TypeAlias = FunctionTool | BuiltinTool
ToolRegistry: TypeAlias = dict[str, Callable[..., Any]]


# ─── Configuration ───────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class Reasoning:
    """Extended thinking / reasoning configuration.

    effort controls the model's reasoning depth:
      - "off"      → no reasoning (provider will skip/disable thinking)
      - "adaptive" → model decides whether to think based on complexity
      - "minimal"  → the smallest provider-supported reasoning effort
      - "low"      → light reasoning
      - "medium"   → moderate reasoning
      - "high"     → deep reasoning
      - "xhigh"    → extra-high effort where supported

    thinking_budget is an optional hard cap on reasoning tokens.
    When set, it limits how many tokens the model spends on internal
    reasoning — independent of the visible response length.  Budgets
    are only meaningful when reasoning is enabled: passing a budget
    together with effort="off" raises ValueError instead of silently
    discarding the budget.

    total_budget caps the combined output (thinking + response tokens).
    When set alongside Config.max_tokens, both limits are enforced:
    the response won't exceed max_tokens, and the total won't exceed
    total_budget.

    ``Config(reasoning=None)`` means "do not send an explicit reasoning
    preference"; ``Config(reasoning=Reasoning())`` means "explicitly force
    reasoning off."  This tri-state is intentional because some providers
    and models have their own defaults.

    Not all providers support every knob. The adapter maps to the
    closest available mechanism and reports degradation via warnings.
    """

    effort: ReasoningEffort = "off"
    thinking_budget: int | None = None
    total_budget: int | None = None

    def __post_init__(self) -> None:
        if self.effort not in REASONING_EFFORTS:
            raise ValueError(f"unsupported reasoning effort: {self.effort}")
        _validate_positive(self.thinking_budget, field_name="thinking_budget")
        _validate_positive(self.total_budget, field_name="total_budget")
        if self.effort == "off" and (
            self.thinking_budget is not None or self.total_budget is not None
        ):
            raise ValueError(
                "Reasoning(effort='off') cannot specify thinking_budget or total_budget"
            )

    @property
    def is_off(self) -> bool:
        return self.effort == "off"


@dataclass(frozen=True, slots=True)
class ToolChoice:
    """How the model should use tools.

    ``allowed`` is stored and exposed as tool names.  Passing Tool objects is
    accepted as constructor sugar, but they are immediately normalized to
    names; use ``ToolChoice.from_tools(...)`` when you want that conversion to
    be explicit at the call site.
    """

    mode: ToolChoiceMode = "auto"
    allowed: tuple[str, ...] = ()
    parallel: bool | None = None

    def __post_init__(self) -> None:
        if self.mode not in TOOL_CHOICE_MODES:
            raise ValueError(f"unsupported tool choice mode: {self.mode}")
        if isinstance(self.allowed, str):
            raise TypeError("ToolChoice.allowed must be a sequence of tool names, not a string")
        raw_allowed = (
            (self.allowed,)
            if isinstance(self.allowed, (FunctionTool, BuiltinTool))
            else tuple(self.allowed)
        )
        if any(isinstance(item, (FunctionTool, BuiltinTool)) for item in raw_allowed):
            _warnings.warn(
                "Passing Tool objects to ToolChoice.allowed is deprecated; "
                "pass tool names or use ToolChoice.from_tools() instead",
                DeprecationWarning,
                stacklevel=2,
            )
        allowed = tuple(
            item.name if isinstance(item, (FunctionTool, BuiltinTool)) else item
            for item in raw_allowed
        )
        object.__setattr__(self, "allowed", allowed)
        if any(not isinstance(name, str) or not name for name in self.allowed):
            raise ValueError("ToolChoice.allowed must contain non-empty tool names")
        if self.mode == "none" and (self.allowed or self.parallel is not None):
            raise ValueError("ToolChoice(mode='none') cannot specify allowed or parallel")

    @classmethod
    def from_tools(
        cls,
        allowed: Tool | Sequence[Tool | str],
        *,
        mode: ToolChoiceMode = "auto",
        parallel: bool | None = None,
    ) -> "ToolChoice":
        """Create a choice by explicitly converting Tool objects to names."""
        if isinstance(allowed, str):
            names = (allowed,)
        elif isinstance(allowed, (FunctionTool, BuiltinTool)):
            names = (allowed.name,)
        else:
            names = tuple(item.name if isinstance(item, (FunctionTool, BuiltinTool)) else item for item in allowed)
        return cls(mode=mode, allowed=names, parallel=parallel)


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
        stop = (self.stop,) if isinstance(self.stop, str) else tuple(self.stop or ())
        object.__setattr__(self, "stop", stop)
        _validate_positive(self.max_tokens, field_name="max_tokens")
        _validate_positive(self.top_k, field_name="top_k")
        for field_name in ("temperature", "top_p"):
            value = getattr(self, field_name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, (int, float))
            ):
                raise TypeError(f"{field_name} must be numeric")
        if self.temperature is not None and self.temperature < 0:
            raise ValueError("temperature must be >= 0")
        if self.top_p is not None and not (0 <= self.top_p <= 1):
            raise ValueError("top_p must be in [0, 1]")
        if any(not isinstance(s, str) or not s for s in self.stop):
            raise ValueError("stop must contain non-empty strings")
        if self.tool_choice is not None and not isinstance(self.tool_choice, ToolChoice):
            raise TypeError("tool_choice must be a ToolChoice")
        if self.reasoning is not None and not isinstance(self.reasoning, Reasoning):
            raise TypeError("reasoning must be a Reasoning")
        _freeze_field(self, "response_format")
        _freeze_extensions_field(self)


# ─── Request ─────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class _ModelRequest:
    """Base for endpoint requests/configs that require a model."""

    model: str

    def __post_init__(self) -> None:
        if not isinstance(self.model, str) or not self.model:
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
    system: str | tuple[PromptPart, ...] | None = None
    tools: tuple[Tool, ...] = ()
    config: Config = field(default_factory=Config)
    cache: bool = True

    def __post_init__(self) -> None:
        _ModelRequest.__post_init__(self)
        object.__setattr__(self, "messages", tuple(self.messages))
        object.__setattr__(self, "tools", tuple(self.tools))
        object.__setattr__(self, "system", _normalize_system(self.system))
        if not self.messages:
            raise ValueError("at least one message is required")
        if not all(isinstance(m, Message) for m in self.messages):
            raise TypeError("Request.messages must contain Message objects")
        if not all(isinstance(t, (FunctionTool, BuiltinTool)) for t in self.tools):
            raise TypeError("Request.tools must contain Tool objects")
        tool_names = [t.name for t in self.tools]
        if len(set(tool_names)) != len(tool_names):
            raise ValueError("Request.tools cannot contain duplicate tool names")
        if not isinstance(self.config, Config):
            raise TypeError("Request.config must be a Config")
        if self.config.tool_choice is not None and self.config.tool_choice.allowed:
            missing = set(self.config.tool_choice.allowed) - set(tool_names)
            if missing:
                raise ValueError(
                    f"ToolChoice.allowed contains tools not present in Request.tools: {sorted(missing)}"
                )


# ─── Usage ───────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class Usage:
    """Token usage.

    ``input_tokens`` and ``output_tokens`` are the canonical billing
    dimensions: every other count (``cache_*``, ``reasoning_tokens``,
    ``*_audio_tokens``) decomposes one of those two and is exposed only
    for telemetry.

    ``total_tokens`` defaults to ``input_tokens + output_tokens``.  When
    supplied explicitly it must equal that sum; adapters are expected to
    fold any provider-side reasoning/audio counts back into
    ``input_tokens``/``output_tokens`` before constructing ``Usage`` so
    the invariant always holds.
    """

    # ``None`` means "compute from input + output".  After
    # ``__post_init__`` runs, ``total_tokens`` is stored as a
    # non-negative ``int`` equal to ``input_tokens + output_tokens``.
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int | None = None
    cache_read_tokens: int | None = None
    cache_write_tokens: int | None = None
    reasoning_tokens: int | None = None
    input_audio_tokens: int | None = None
    output_audio_tokens: int | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "input_tokens",
            "output_tokens",
            "cache_read_tokens",
            "cache_write_tokens",
            "reasoning_tokens",
            "input_audio_tokens",
            "output_audio_tokens",
        ):
            _validate_non_negative(getattr(self, field_name), field_name=field_name)
        _validate_int(self.total_tokens, field_name="total_tokens")
        computed_total = self.input_tokens + self.output_tokens
        if self.total_tokens is None:
            object.__setattr__(self, "total_tokens", computed_total)
        elif self.total_tokens != computed_total:
            raise ValueError("total_tokens must equal input_tokens + output_tokens")


# ─── Response ────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class Response:
    """The composed artifact returned by a foundation model.

    ``Response`` keeps only minimal convenience properties.  Use
    ``response.message.first(...)`` and ``response.message.parts_of(...)``
    for variant-specific content access.
    """

    id: str | None
    model: str
    message: Message
    finish_reason: FinishReason
    usage: Usage
    provider_data: ProviderData | None = None
    _parsed_json: Any = field(default=_MISSING, init=False, repr=False, compare=False, hash=False)

    def __post_init__(self) -> None:
        if self.id == "":
            raise ValueError("Response.id cannot be empty; use None when unavailable")
        if not self.model:
            raise ValueError("Response requires model")
        if not isinstance(self.message, Message):
            raise TypeError("Response.message must be a Message")
        if self.message.role != "assistant":
            raise ValueError("Response.message must have role 'assistant'")
        if self.finish_reason not in FINISH_REASONS:
            raise ValueError(f"unsupported finish reason: {self.finish_reason}")
        if not isinstance(self.usage, Usage):
            raise TypeError("Response.usage must be a Usage")
        _freeze_field(self, "provider_data")

    @property
    def text(self) -> str | None:
        return self.message.text

    @property
    def tool_calls(self) -> list[ToolCallPart]:
        return self.message.parts_of(ToolCallPart)

    def parse_json(self, *, default: Any = _MISSING) -> Any:
        """Parse text content as JSON.

        Raises ``ValueError`` by default.  Pass ``default=...`` to receive a
        fallback instead of an exception.  Successful parses are cached.
        """
        if self._parsed_json is not _MISSING:
            return self._parsed_json

        t = self.text
        if t is None:
            if default is not _MISSING:
                return default
            raise ValueError(
                "Cannot parse response as JSON: no text content. "
                f"Parts: {[p.type for p in self.message.parts]}"
            )
        stripped = t.strip()
        match = _re.search(
            r"```[ \t]*(?:json)?[ \t\r\n]+(.*?)\s*```",
            stripped,
            _re.DOTALL | _re.IGNORECASE,
        )
        if match:
            stripped = match.group(1).strip()
        try:
            parsed = _json.loads(stripped)
        except _json.JSONDecodeError as e:
            if default is not _MISSING:
                return default
            preview = stripped[:200] + ("..." if len(stripped) > 200 else "")
            raise ValueError(
                f"Cannot parse response as JSON: {e}\nRaw text: {preview}"
            ) from e
        object.__setattr__(self, "_parsed_json", parsed)
        return parsed

    @property
    def json(self) -> Any:
        """Best-effort parsed JSON, or ``None`` when parsing is impossible.

        Use ``parse_json()`` when parse failures should be reported.
        """
        return self.parse_json(default=None)


# ─── Embeddings ──────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class EmbeddingRequest(_ModelRequest):
    inputs: tuple[str, ...]
    extensions: Extensions | None = None

    def __post_init__(self) -> None:
        _ModelRequest.__post_init__(self)
        inputs = (self.inputs,) if isinstance(self.inputs, str) else tuple(self.inputs)
        object.__setattr__(self, "inputs", inputs)
        if not self.inputs:
            raise ValueError("inputs cannot be empty")
        if any(not isinstance(x, str) or x == "" for x in self.inputs):
            raise ValueError("inputs must contain non-empty strings")
        _freeze_extensions_field(self)


@dataclass(frozen=True, slots=True)
class EmbeddingResponse:
    model: str
    vectors: tuple[tuple[float, ...], ...]
    usage: Usage = field(default_factory=Usage)
    provider_data: ProviderData | None = None

    def __post_init__(self) -> None:
        if not self.model:
            raise ValueError("EmbeddingResponse requires model")
        if not isinstance(self.usage, Usage):
            raise TypeError("EmbeddingResponse.usage must be a Usage")
        vectors = tuple(tuple(v) for v in self.vectors)
        if not vectors:
            raise ValueError("EmbeddingResponse requires at least one vector")
        for vector in vectors:
            if not vector:
                raise ValueError("EmbeddingResponse vectors cannot be empty")
            for value in vector:
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise TypeError(
                        "EmbeddingResponse vector elements must be numeric"
                    )
                if isinstance(value, float) and not _math.isfinite(value):
                    raise ValueError(
                        "EmbeddingResponse vector elements must be finite"
                    )
        object.__setattr__(self, "vectors", vectors)
        _freeze_field(self, "provider_data")


# ─── File Upload ─────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True, repr=False)
class FileUploadRequest:
    """A file upload request.

    Unlike most endpoint requests, ``model`` is optional because some
    providers scope file uploads to the account, not a specific model.
    The non-default fields (``filename``, ``bytes_data``) are required
    and have no defaults: a default-constructed FileUploadRequest is
    not meaningful.
    """

    filename: str
    bytes_data: bytes
    media_type: str = "application/octet-stream"
    model: str | None = None
    extensions: Extensions | None = None

    def __post_init__(self) -> None:
        if self.model == "":
            raise ValueError("model cannot be empty")
        if not self.filename:
            raise ValueError("filename is required")
        if not isinstance(self.bytes_data, (bytes, bytearray)):
            raise TypeError("bytes_data must be bytes")
        if not self.bytes_data:
            raise ValueError("bytes_data is required")
        if isinstance(self.bytes_data, bytearray):
            object.__setattr__(self, "bytes_data", bytes(self.bytes_data))
        if not self.media_type:
            raise ValueError("media_type is required")
        _freeze_extensions_field(self)

    def __repr__(self) -> str:
        return (
            "FileUploadRequest("
            f"filename={self.filename!r}, "
            f"bytes_data={_bytes_summary(self.bytes_data)!r}, "
            f"media_type={self.media_type!r}, "
            f"model={self.model!r}, "
            f"extensions={self.extensions!r})"
        )


@dataclass(frozen=True, slots=True)
class FileUploadResponse:
    id: str
    provider_data: ProviderData | None = None

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("FileUploadResponse requires id")
        _freeze_field(self, "provider_data")


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
        mismatched = [r.model for r in self.requests if r.model != self.model]
        if mismatched:
            raise ValueError("BatchRequest.model must match every nested Request.model")
        _freeze_extensions_field(self)


@dataclass(frozen=True, slots=True)
class BatchResponse:
    id: str
    status: BatchStatus
    provider_data: ProviderData | None = None

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("BatchResponse requires id")
        if self.status not in BATCH_STATUSES:
            raise ValueError(f"unsupported batch status: {self.status}")
        _freeze_field(self, "provider_data")


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
        if self.size is not None and (
            not isinstance(self.size, str) or self.size == ""
        ):
            raise ValueError("size cannot be empty")
        _freeze_extensions_field(self)


@dataclass(frozen=True, slots=True)
class ImageGenerationResponse:
    images: tuple[ImagePart, ...]
    id: str | None = None
    model: str | None = None
    usage: Usage = field(default_factory=Usage)
    provider_data: ProviderData | None = None

    def __post_init__(self) -> None:
        if self.id == "":
            raise ValueError("ImageGenerationResponse.id cannot be empty; use None when unavailable")
        if self.model == "":
            raise ValueError("ImageGenerationResponse.model cannot be empty; use None when unavailable")
        if not isinstance(self.usage, Usage):
            raise TypeError("ImageGenerationResponse.usage must be a Usage")
        object.__setattr__(self, "images", tuple(self.images))
        if not self.images:
            raise ValueError("ImageGenerationResponse requires at least one image")
        if not all(isinstance(img, ImagePart) for img in self.images):
            raise TypeError("images must contain ImagePart objects")
        _freeze_field(self, "provider_data")


# ─── Audio Generation ────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class AudioGenerationRequest(_PromptRequest):
    voice: str | None = None
    format: str | None = None
    extensions: Extensions | None = None

    def __post_init__(self) -> None:
        _PromptRequest.__post_init__(self)
        for field_name in ("voice", "format"):
            value = getattr(self, field_name)
            if value is not None and (not isinstance(value, str) or value == ""):
                raise ValueError(f"{field_name} cannot be empty")
        _freeze_extensions_field(self)


@dataclass(frozen=True, slots=True)
class AudioGenerationResponse:
    audio: AudioPart
    id: str | None = None
    model: str | None = None
    usage: Usage = field(default_factory=Usage)
    provider_data: ProviderData | None = None

    def __post_init__(self) -> None:
        if self.id == "":
            raise ValueError("AudioGenerationResponse.id cannot be empty; use None when unavailable")
        if self.model == "":
            raise ValueError("AudioGenerationResponse.model cannot be empty; use None when unavailable")
        if not isinstance(self.usage, Usage):
            raise TypeError("AudioGenerationResponse.usage must be a Usage")
        if not isinstance(self.audio, AudioPart):
            raise TypeError("audio must be an AudioPart")
        _freeze_field(self, "provider_data")


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
    encoding: AudioEncoding
    sample_rate: int
    channels: int = 1

    def __post_init__(self) -> None:
        if self.encoding not in AUDIO_ENCODINGS:
            raise ValueError(f"unsupported audio encoding: {self.encoding}")
        _validate_positive(self.sample_rate, field_name="sample_rate")
        _validate_positive(self.channels, field_name="channels")


# ─── Live (Realtime) ─────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class LiveConfig(_ModelRequest):
    system: str | tuple[PromptPart, ...] | None = None
    tools: tuple[Tool, ...] = ()
    voice: str | None = None
    input_format: AudioFormat | None = None
    output_format: AudioFormat | None = None
    extensions: Extensions | None = None

    def __post_init__(self) -> None:
        _ModelRequest.__post_init__(self)
        object.__setattr__(self, "tools", tuple(self.tools))
        object.__setattr__(self, "system", _normalize_system(self.system))
        if not all(isinstance(t, (FunctionTool, BuiltinTool)) for t in self.tools):
            raise TypeError("LiveConfig.tools must contain Tool objects")
        tool_names = [t.name for t in self.tools]
        if len(set(tool_names)) != len(tool_names):
            raise ValueError("LiveConfig.tools cannot contain duplicate tool names")
        if self.input_format is not None and not isinstance(self.input_format, AudioFormat):
            raise TypeError("input_format must be an AudioFormat")
        if self.output_format is not None and not isinstance(self.output_format, AudioFormat):
            raise TypeError("output_format must be an AudioFormat")
        _freeze_extensions_field(self)


_LIVE_CLIENT_ALLOWED_FIELDS: dict[str, tuple[str, ...]] = {
    "audio": ("data",),
    "video": ("data",),
    "text": ("text",),
    "tool_result": ("id", "content"),
    "interrupt": (),
    "end_audio": (),
}

_LIVE_CLIENT_REQUIRED_FIELDS: dict[str, tuple[str, ...]] = {
    "audio": ("data",),
    "video": ("data",),
    "text": ("text",),
    "tool_result": ("id", "content"),
}

_LIVE_SERVER_ALLOWED_FIELDS: dict[str, tuple[str, ...]] = {
    "audio": ("data",),
    "text": ("text",),
    "tool_call": ("id", "name", "input"),
    "tool_call_delta": ("id", "name", "input_delta"),
    "interrupted": (),
    "turn_end": ("usage",),
    "error": ("error",),
}

_LIVE_SERVER_REQUIRED_FIELDS: dict[str, tuple[str, ...]] = {
    "audio": ("data",),
    "text": ("text",),
    "tool_call": ("id", "name", "input"),
    # id/name are optional on deltas because some realtime transports send
    # them only on the first fragment of a tool call; adapters maintain that
    # association while clients can still consume empty input_delta chunks.
    "tool_call_delta": ("input_delta",),
    "turn_end": ("usage",),
    "error": ("error",),
}


@dataclass(frozen=True, slots=True)
class LiveClientEvent:
    type: LiveClientEventType
    data: str | None = None
    text: str | None = None
    id: str | None = None
    content: tuple[ToolResultContentPart, ...] = ()

    def __post_init__(self) -> None:
        if self.type not in LIVE_CLIENT_EVENT_TYPES:
            raise ValueError(f"unsupported live client event type: {self.type}")
        object.__setattr__(self, "content", tuple(self.content))
        owner = f"LiveClientEvent(type={self.type!r})"
        _require_fields(owner, self, _LIVE_CLIENT_REQUIRED_FIELDS.get(self.type, ()))
        _forbid_fields(owner, self, _LIVE_CLIENT_ALLOWED_FIELDS[self.type])
        if self.type == "tool_result":
            if not all(_is_part(p) for p in self.content):
                raise TypeError("LiveClientEvent.content must contain Part objects")
            if any(isinstance(p, _TOOL_RESULT_FORBIDDEN_PARTS) for p in self.content):
                raise TypeError(
                    "LiveClientEvent.content cannot contain model or protocol parts"
                )


@dataclass(frozen=True, slots=True)
class LiveServerEvent:
    type: LiveServerEventType
    data: str | None = None
    text: str | None = None
    id: str | None = None
    name: str | None = None
    input: JsonObject | None = None
    input_delta: str | None = None
    usage: Usage | None = None
    error: ErrorDetail | None = None

    def __post_init__(self) -> None:
        if self.type not in LIVE_SERVER_EVENT_TYPES:
            raise ValueError(f"unsupported live server event type: {self.type}")
        owner = f"LiveServerEvent(type={self.type!r})"
        _require_fields(owner, self, _LIVE_SERVER_REQUIRED_FIELDS.get(self.type, ()))
        _forbid_fields(owner, self, _LIVE_SERVER_ALLOWED_FIELDS[self.type])
        if self.usage is not None and not isinstance(self.usage, Usage):
            raise TypeError("LiveServerEvent.usage must be a Usage")
        if self.error is not None and not isinstance(self.error, ErrorDetail):
            raise TypeError("LiveServerEvent.error must be an ErrorDetail")
        if self.type == "tool_call":
            _freeze_field(self, "input", required=True)


# ─── ToolCallInfo (for callbacks) ────────────────────────────────────
#
# Lightweight callback payload: same identity/input shape as ToolCallPart,
# but without the content-part discriminator.


@dataclass(frozen=True, slots=True)
class ToolCallInfo:
    """Backward-compatible callback view of a tool call.

    ``ToolCallPart`` is the canonical tool-call value object.  This wrapper
    exists for callback APIs that historically omitted the part discriminator;
    convert explicitly with ``from_part()``/``to_part()`` to avoid shape drift.
    """

    id: str
    name: str
    input: JsonObject

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("ToolCallInfo requires id")
        if not self.name:
            raise ValueError("ToolCallInfo requires name")
        _freeze_field(self, "input", required=True)

    @classmethod
    def from_part(cls, part: ToolCallPart) -> "ToolCallInfo":
        if not isinstance(part, ToolCallPart):
            raise TypeError("ToolCallInfo.from_part() requires a ToolCallPart")
        return cls(id=part.id, name=part.name, input=part.input)

    def to_part(self) -> ToolCallPart:
        return ToolCallPart(id=self.id, name=self.name, input=self.input)


def _check_literal_vocabularies() -> None:
    checks = (
        ("PartType", set(get_args(PartType)), set(PART_TYPES)),
        ("DeltaType", set(get_args(DeltaType)), set(DELTA_TYPES)),
        ("ErrorCode", set(get_args(ErrorCode)), set(ERROR_CODES)),
        ("FinishReason", set(get_args(FinishReason)), set(FINISH_REASONS)),
        ("Role", set(get_args(Role)), set(ROLE_VALUES)),
        ("StreamEventType", set(get_args(StreamEventType)), set(STREAM_EVENT_TYPES)),
        ("BatchStatus", set(get_args(BatchStatus)), set(BATCH_STATUSES)),
        ("AudioEncoding", set(get_args(AudioEncoding)), set(AUDIO_ENCODINGS)),
        ("ToolChoiceMode", set(get_args(ToolChoiceMode)), set(TOOL_CHOICE_MODES)),
        ("ReasoningEffort", set(get_args(ReasoningEffort)), set(REASONING_EFFORTS)),
        ("LiveClientEventType", set(get_args(LiveClientEventType)), set(LIVE_CLIENT_EVENT_TYPES)),
        ("LiveServerEventType", set(get_args(LiveServerEventType)), set(LIVE_SERVER_EVENT_TYPES)),
    )
    for name, literal_values, runtime_values in checks:
        if literal_values != runtime_values:
            raise RuntimeError(
                f"{name} literal values must match runtime vocabulary: "
                f"literal={sorted(literal_values)} runtime={sorted(runtime_values)}"
            )


_check_literal_vocabularies()
