from __future__ import annotations

import base64
import shutil
import subprocess
from types import SimpleNamespace
from typing import Any

import pytest
from google.protobuf import descriptor_pb2, descriptor_pool, message_factory
from google.protobuf.json_format import MessageToDict, ParseDict

import lm15.protobuf as lm15_proto
from lm15.types import (
    AudioDelta,
    AudioFormat,
    AudioGenerationRequest,
    AudioGenerationResponse,
    AudioPart,
    BatchRequest,
    BatchResponse,
    BuiltinTool,
    CitationDelta,
    CitationPart,
    Config,
    DocumentPart,
    EmbeddingRequest,
    EmbeddingResponse,
    ErrorDetail,
    FileUploadRequest,
    FileUploadResponse,
    FunctionTool,
    ImageDelta,
    ImageGenerationRequest,
    ImageGenerationResponse,
    ImagePart,
    LiveClientEvent,
    LiveConfig,
    LiveServerEvent,
    Message,
    Part,
    Reasoning,
    RefusalPart,
    Request,
    Response,
    StreamEvent,
    TextDelta,
    TextPart,
    ThinkingDelta,
    ThinkingPart,
    Tool,
    ToolCallDelta,
    ToolCallInfo,
    ToolCallPart,
    ToolChoice,
    ToolResultPart,
    Usage,
    VideoPart,
)

PROTO = "proto/lm15/v1/lm15.proto"


def _b64(value: bytes) -> str:
    return base64.b64encode(value).decode("ascii")


@pytest.fixture(scope="session")
def pb(tmp_path_factory):
    protoc = shutil.which("protoc")
    if protoc is None:
        pytest.skip("protoc is not installed")

    out = tmp_path_factory.mktemp("lm15_proto_descriptor") / "lm15.pb"
    subprocess.run(
        [
            protoc,
            "--proto_path=proto",
            "--proto_path=/usr/include",
            "--include_imports",
            f"--descriptor_set_out={out}",
            PROTO,
        ],
        check=True,
    )

    fds = descriptor_pb2.FileDescriptorSet()
    fds.ParseFromString(out.read_bytes())

    pool = descriptor_pool.DescriptorPool()
    for file_descriptor in fds.file:
        pool.Add(file_descriptor)

    def cls(name: str):
        descriptor = pool.FindMessageTypeByName(f"lm15.v1.{name}")
        return message_factory.GetMessageClass(descriptor)

    def enum_value(enum_name: str, value_name: str) -> int:
        enum = pool.FindEnumTypeByName(f"lm15.v1.{enum_name}")
        return enum.values_by_name[value_name].number

    names = [
        "Part",
        "MediaSource",
        "TextPart",
        "ImagePart",
        "AudioPart",
        "VideoPart",
        "DocumentPart",
        "ToolCallPart",
        "ToolResultPart",
        "ThinkingPart",
        "RefusalPart",
        "CitationPart",
        "Message",
        "SystemContent",
        "Delta",
        "StreamEvent",
        "Request",
        "Response",
        "Usage",
        "Tool",
        "ToolChoice",
        "Reasoning",
        "Config",
        "ErrorDetail",
        "EmbeddingRequest",
        "EmbeddingResponse",
        "Vector",
        "FileUploadRequest",
        "FileUploadResponse",
        "BatchRequest",
        "BatchResponse",
        "ImageGenerationRequest",
        "ImageGenerationResponse",
        "AudioGenerationRequest",
        "AudioGenerationResponse",
        "EndpointRequest",
        "EndpointResponse",
        "AudioFormat",
        "LiveConfig",
        "LiveClientEvent",
        "LiveServerEvent",
        "ToolCallInfo",
    ]
    ns = SimpleNamespace(**{name: cls(name) for name in names})

    enum_names = {
        "ROLE_UNSPECIFIED": ("Role", "ROLE_UNSPECIFIED"),
        "ROLE_USER": ("Role", "ROLE_USER"),
        "ROLE_ASSISTANT": ("Role", "ROLE_ASSISTANT"),
        "ROLE_TOOL": ("Role", "ROLE_TOOL"),
        "ROLE_DEVELOPER": ("Role", "ROLE_DEVELOPER"),
        "FINISH_REASON_UNSPECIFIED": ("FinishReason", "FINISH_REASON_UNSPECIFIED"),
        "FINISH_REASON_STOP": ("FinishReason", "FINISH_REASON_STOP"),
        "FINISH_REASON_LENGTH": ("FinishReason", "FINISH_REASON_LENGTH"),
        "FINISH_REASON_TOOL_CALL": ("FinishReason", "FINISH_REASON_TOOL_CALL"),
        "FINISH_REASON_CONTENT_FILTER": ("FinishReason", "FINISH_REASON_CONTENT_FILTER"),
        "FINISH_REASON_ERROR": ("FinishReason", "FINISH_REASON_ERROR"),
        "REASONING_EFFORT_OFF": ("ReasoningEffort", "REASONING_EFFORT_OFF"),
        "REASONING_EFFORT_ADAPTIVE": ("ReasoningEffort", "REASONING_EFFORT_ADAPTIVE"),
        "REASONING_EFFORT_MINIMAL": ("ReasoningEffort", "REASONING_EFFORT_MINIMAL"),
        "REASONING_EFFORT_LOW": ("ReasoningEffort", "REASONING_EFFORT_LOW"),
        "REASONING_EFFORT_MEDIUM": ("ReasoningEffort", "REASONING_EFFORT_MEDIUM"),
        "REASONING_EFFORT_HIGH": ("ReasoningEffort", "REASONING_EFFORT_HIGH"),
        "REASONING_EFFORT_XHIGH": ("ReasoningEffort", "REASONING_EFFORT_XHIGH"),
        "ERROR_CODE_AUTH": ("ErrorCode", "ERROR_CODE_AUTH"),
        "ERROR_CODE_BILLING": ("ErrorCode", "ERROR_CODE_BILLING"),
        "ERROR_CODE_RATE_LIMIT": ("ErrorCode", "ERROR_CODE_RATE_LIMIT"),
        "ERROR_CODE_INVALID_REQUEST": ("ErrorCode", "ERROR_CODE_INVALID_REQUEST"),
        "ERROR_CODE_CONTEXT_LENGTH": ("ErrorCode", "ERROR_CODE_CONTEXT_LENGTH"),
        "ERROR_CODE_TIMEOUT": ("ErrorCode", "ERROR_CODE_TIMEOUT"),
        "ERROR_CODE_SERVER": ("ErrorCode", "ERROR_CODE_SERVER"),
        "ERROR_CODE_PROVIDER": ("ErrorCode", "ERROR_CODE_PROVIDER"),
        "IMAGE_DETAIL_LOW": ("ImageDetail", "IMAGE_DETAIL_LOW"),
        "IMAGE_DETAIL_HIGH": ("ImageDetail", "IMAGE_DETAIL_HIGH"),
        "IMAGE_DETAIL_AUTO": ("ImageDetail", "IMAGE_DETAIL_AUTO"),
        "TOOL_CHOICE_MODE_AUTO": ("ToolChoiceMode", "TOOL_CHOICE_MODE_AUTO"),
        "TOOL_CHOICE_MODE_REQUIRED": ("ToolChoiceMode", "TOOL_CHOICE_MODE_REQUIRED"),
        "TOOL_CHOICE_MODE_NONE": ("ToolChoiceMode", "TOOL_CHOICE_MODE_NONE"),
        "AUDIO_ENCODING_PCM16": ("AudioEncoding", "AUDIO_ENCODING_PCM16"),
        "AUDIO_ENCODING_OPUS": ("AudioEncoding", "AUDIO_ENCODING_OPUS"),
        "AUDIO_ENCODING_MP3": ("AudioEncoding", "AUDIO_ENCODING_MP3"),
        "AUDIO_ENCODING_AAC": ("AudioEncoding", "AUDIO_ENCODING_AAC"),
    }
    for attr, (enum_name, value_name) in enum_names.items():
        setattr(ns, attr, enum_value(enum_name, value_name))
    return ns


# ─── Generic protobuf helpers ────────────────────────────────────────


def _set_wrapper(wrapper: Any, value: Any) -> None:
    wrapper.value = value


def _wrapper_value(msg: Any, field: str) -> Any | None:
    return getattr(msg, field).value if msg.HasField(field) else None


def _copy_struct(target: Any, value: dict[str, Any] | None) -> None:
    if value is not None:
        ParseDict(value, target)


def _struct_value(msg: Any, field: str) -> dict[str, Any] | None:
    if not msg.HasField(field):
        return None
    return MessageToDict(getattr(msg, field), preserving_proto_field_name=True)


# ─── Enum maps ───────────────────────────────────────────────────────


def _maps(pb):
    role_to_proto = {
        "user": pb.ROLE_USER,
        "assistant": pb.ROLE_ASSISTANT,
        "tool": pb.ROLE_TOOL,
        "developer": pb.ROLE_DEVELOPER,
    }
    finish_to_proto = {
        "stop": pb.FINISH_REASON_STOP,
        "length": pb.FINISH_REASON_LENGTH,
        "tool_call": pb.FINISH_REASON_TOOL_CALL,
        "content_filter": pb.FINISH_REASON_CONTENT_FILTER,
        "error": pb.FINISH_REASON_ERROR,
    }
    effort_to_proto = {
        "off": pb.REASONING_EFFORT_OFF,
        "adaptive": pb.REASONING_EFFORT_ADAPTIVE,
        "minimal": pb.REASONING_EFFORT_MINIMAL,
        "low": pb.REASONING_EFFORT_LOW,
        "medium": pb.REASONING_EFFORT_MEDIUM,
        "high": pb.REASONING_EFFORT_HIGH,
        "xhigh": pb.REASONING_EFFORT_XHIGH,
    }
    error_to_proto = {
        "auth": pb.ERROR_CODE_AUTH,
        "billing": pb.ERROR_CODE_BILLING,
        "rate_limit": pb.ERROR_CODE_RATE_LIMIT,
        "invalid_request": pb.ERROR_CODE_INVALID_REQUEST,
        "context_length": pb.ERROR_CODE_CONTEXT_LENGTH,
        "timeout": pb.ERROR_CODE_TIMEOUT,
        "server": pb.ERROR_CODE_SERVER,
        "provider": pb.ERROR_CODE_PROVIDER,
    }
    detail_to_proto = {
        "low": pb.IMAGE_DETAIL_LOW,
        "high": pb.IMAGE_DETAIL_HIGH,
        "auto": pb.IMAGE_DETAIL_AUTO,
    }
    mode_to_proto = {
        "auto": pb.TOOL_CHOICE_MODE_AUTO,
        "required": pb.TOOL_CHOICE_MODE_REQUIRED,
        "none": pb.TOOL_CHOICE_MODE_NONE,
    }
    encoding_to_proto = {
        "pcm16": pb.AUDIO_ENCODING_PCM16,
        "opus": pb.AUDIO_ENCODING_OPUS,
        "mp3": pb.AUDIO_ENCODING_MP3,
        "aac": pb.AUDIO_ENCODING_AAC,
    }
    return {
        "role": (role_to_proto, {v: k for k, v in role_to_proto.items()}),
        "finish": (finish_to_proto, {v: k for k, v in finish_to_proto.items()}),
        "effort": (effort_to_proto, {v: k for k, v in effort_to_proto.items()}),
        "error": (error_to_proto, {v: k for k, v in error_to_proto.items()}),
        "detail": (detail_to_proto, {v: k for k, v in detail_to_proto.items()}),
        "mode": (mode_to_proto, {v: k for k, v in mode_to_proto.items()}),
        "encoding": (encoding_to_proto, {v: k for k, v in encoding_to_proto.items()}),
    }


# ─── Python -> protobuf ──────────────────────────────────────────────


def _media_source_to_proto(pb, part: ImagePart | AudioPart | VideoPart | DocumentPart):
    out = pb.MediaSource(media_type=part.media_type)
    if part.data is not None:
        out.data = base64.b64decode(part.data)
    elif part.url is not None:
        out.url = part.url
    elif part.file_id is not None:
        out.file_id = part.file_id
    return out


def _part_to_proto(pb, part: Part):
    maps = _maps(pb)
    out = pb.Part()
    if isinstance(part, TextPart):
        out.text.text = part.text
    elif isinstance(part, ImagePart):
        out.image.source.CopyFrom(_media_source_to_proto(pb, part))
        if part.detail is not None:
            out.image.detail = maps["detail"][0][part.detail]
    elif isinstance(part, AudioPart):
        out.audio.source.CopyFrom(_media_source_to_proto(pb, part))
    elif isinstance(part, VideoPart):
        out.video.source.CopyFrom(_media_source_to_proto(pb, part))
    elif isinstance(part, DocumentPart):
        out.document.source.CopyFrom(_media_source_to_proto(pb, part))
    elif isinstance(part, ToolCallPart):
        out.tool_call.id = part.id
        out.tool_call.name = part.name
        _copy_struct(out.tool_call.input, part.input)
    elif isinstance(part, ToolResultPart):
        out.tool_result.id = part.id
        out.tool_result.content.extend(_part_to_proto(pb, p) for p in part.content)
        if part.name is not None:
            _set_wrapper(out.tool_result.name, part.name)
        out.tool_result.is_error = part.is_error
    elif isinstance(part, ThinkingPart):
        out.thinking.text = part.text
        out.thinking.redacted = part.redacted
    elif isinstance(part, RefusalPart):
        out.refusal.text = part.text
    elif isinstance(part, CitationPart):
        if part.url is not None:
            _set_wrapper(out.citation.url, part.url)
        if part.title is not None:
            _set_wrapper(out.citation.title, part.title)
        if part.text is not None:
            _set_wrapper(out.citation.text, part.text)
    else:
        raise TypeError(type(part))
    return out


def _message_to_proto(pb, message: Message):
    out = pb.Message(role=_maps(pb)["role"][0][message.role])
    out.parts.extend(_part_to_proto(pb, p) for p in message.parts)
    return out


def _system_to_proto(pb, system: str | tuple[Part, ...] | None):
    if system is None:
        return None
    out = pb.SystemContent()
    if isinstance(system, str):
        out.text = system
    else:
        out.parts.parts.extend(_part_to_proto(pb, p) for p in system)
    return out


def _tool_to_proto(pb, tool: Tool):
    out = pb.Tool()
    if isinstance(tool, FunctionTool):
        out.function.name = tool.name
        if tool.description is not None:
            _set_wrapper(out.function.description, tool.description)
        _copy_struct(out.function.parameters, tool.parameters)
    elif isinstance(tool, BuiltinTool):
        out.builtin.name = tool.name
        _copy_struct(out.builtin.config, tool.config)
    else:
        raise TypeError(type(tool))
    return out


def _tool_choice_to_proto(pb, choice: ToolChoice):
    out = pb.ToolChoice(mode=_maps(pb)["mode"][0][choice.mode])
    out.allowed.extend(choice.allowed)
    if choice.parallel is not None:
        _set_wrapper(out.parallel, choice.parallel)
    return out


def _reasoning_to_proto(pb, reasoning: Reasoning):
    out = pb.Reasoning(effort=_maps(pb)["effort"][0][reasoning.effort])
    if reasoning.thinking_budget is not None:
        _set_wrapper(out.thinking_budget, reasoning.thinking_budget)
    if reasoning.total_budget is not None:
        _set_wrapper(out.total_budget, reasoning.total_budget)
    return out


def _config_to_proto(pb, config: Config):
    out = pb.Config()
    if config.max_tokens is not None:
        _set_wrapper(out.max_tokens, config.max_tokens)
    if config.temperature is not None:
        _set_wrapper(out.temperature, config.temperature)
    if config.top_p is not None:
        _set_wrapper(out.top_p, config.top_p)
    if config.top_k is not None:
        _set_wrapper(out.top_k, config.top_k)
    out.stop.extend(config.stop)
    _copy_struct(out.response_format, config.response_format)
    if config.tool_choice is not None:
        out.tool_choice.CopyFrom(_tool_choice_to_proto(pb, config.tool_choice))
    if config.reasoning is not None:
        out.reasoning.CopyFrom(_reasoning_to_proto(pb, config.reasoning))
    _copy_struct(out.extensions, config.extensions)
    return out


def _usage_to_proto(pb, usage: Usage):
    out = pb.Usage(
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        total_tokens=usage.total_tokens,
    )
    for field in (
        "cache_read_tokens",
        "cache_write_tokens",
        "reasoning_tokens",
        "input_audio_tokens",
        "output_audio_tokens",
    ):
        value = getattr(usage, field)
        if value is not None:
            _set_wrapper(getattr(out, field), value)
    return out


def _error_to_proto(pb, error: ErrorDetail):
    out = pb.ErrorDetail(
        code=_maps(pb)["error"][0][error.code],
        message=error.message,
    )
    if error.provider_code is not None:
        _set_wrapper(out.provider_code, error.provider_code)
    return out


def _delta_to_proto(pb, delta):
    out = pb.Delta()
    if isinstance(delta, TextDelta):
        out.text.text = delta.text
        out.text.part_index = delta.part_index
    elif isinstance(delta, ThinkingDelta):
        out.thinking.text = delta.text
        out.thinking.part_index = delta.part_index
    elif isinstance(delta, AudioDelta):
        out.audio.data = base64.b64decode(delta.data)
        out.audio.part_index = delta.part_index
        if delta.media_type is not None:
            _set_wrapper(out.audio.media_type, delta.media_type)
    elif isinstance(delta, ImageDelta):
        out.image.part_index = delta.part_index
        if delta.media_type is not None:
            _set_wrapper(out.image.media_type, delta.media_type)
        if delta.data is not None:
            out.image.data = base64.b64decode(delta.data)
        elif delta.url is not None:
            out.image.url = delta.url
        elif delta.file_id is not None:
            out.image.file_id = delta.file_id
    elif isinstance(delta, ToolCallDelta):
        out.tool_call.input = delta.input
        out.tool_call.part_index = delta.part_index
        if delta.id is not None:
            _set_wrapper(out.tool_call.id, delta.id)
        if delta.name is not None:
            _set_wrapper(out.tool_call.name, delta.name)
    elif isinstance(delta, CitationDelta):
        if delta.text is not None:
            _set_wrapper(out.citation.text, delta.text)
        if delta.url is not None:
            _set_wrapper(out.citation.url, delta.url)
        if delta.title is not None:
            _set_wrapper(out.citation.title, delta.title)
        out.citation.part_index = delta.part_index
    else:
        raise TypeError(type(delta))
    return out


def _stream_event_to_proto(pb, event: StreamEvent):
    out = pb.StreamEvent()
    if event.type == "start":
        out.start.SetInParent()
        if event.id is not None:
            _set_wrapper(out.start.id, event.id)
        if event.model is not None:
            _set_wrapper(out.start.model, event.model)
    elif event.type == "delta":
        assert event.delta is not None
        out.delta.delta.CopyFrom(_delta_to_proto(pb, event.delta))
    elif event.type == "end":
        out.end.SetInParent()
        if event.finish_reason is not None:
            out.end.finish_reason = _maps(pb)["finish"][0][event.finish_reason]
        if event.usage is not None:
            out.end.usage.CopyFrom(_usage_to_proto(pb, event.usage))
        _copy_struct(out.end.provider_data, event.provider_data)
    elif event.type == "error":
        assert event.error is not None
        out.error.error.CopyFrom(_error_to_proto(pb, event.error))
    else:
        raise ValueError(event.type)
    return out


def _request_to_proto(pb, request: Request):
    out = pb.Request(model=request.model)
    out.messages.extend(_message_to_proto(pb, m) for m in request.messages)
    system = _system_to_proto(pb, request.system)
    if system is not None:
        out.system.CopyFrom(system)
    out.tools.extend(_tool_to_proto(pb, t) for t in request.tools)
    out.config.CopyFrom(_config_to_proto(pb, request.config))
    _set_wrapper(out.cache, request.cache)
    return out


def _response_to_proto(pb, response: Response):
    out = pb.Response(
        id=response.id,
        model=response.model,
        finish_reason=_maps(pb)["finish"][0][response.finish_reason],
    )
    out.message.CopyFrom(_message_to_proto(pb, response.message))
    out.usage.CopyFrom(_usage_to_proto(pb, response.usage))
    _copy_struct(out.provider_data, response.provider_data)
    return out


def _audio_format_to_proto(pb, audio_format: AudioFormat):
    return pb.AudioFormat(
        encoding=_maps(pb)["encoding"][0][audio_format.encoding],
        sample_rate=audio_format.sample_rate,
        channels=audio_format.channels,
    )


def _live_config_to_proto(pb, config: LiveConfig):
    out = pb.LiveConfig(model=config.model)
    system = _system_to_proto(pb, config.system)
    if system is not None:
        out.system.CopyFrom(system)
    out.tools.extend(_tool_to_proto(pb, t) for t in config.tools)
    if config.voice is not None:
        _set_wrapper(out.voice, config.voice)
    if config.input_format is not None:
        out.input_format.CopyFrom(_audio_format_to_proto(pb, config.input_format))
    if config.output_format is not None:
        out.output_format.CopyFrom(_audio_format_to_proto(pb, config.output_format))
    _copy_struct(out.extensions, config.extensions)
    return out


def _live_client_event_to_proto(pb, event: LiveClientEvent):
    out = pb.LiveClientEvent()
    if event.type == "audio":
        out.audio.data = base64.b64decode(event.data or "")
    elif event.type == "video":
        out.video.data = base64.b64decode(event.data or "")
    elif event.type == "text":
        out.text.text = event.text or ""
    elif event.type == "tool_result":
        out.tool_result.id = event.id or ""
        out.tool_result.content.extend(_part_to_proto(pb, p) for p in event.content)
    elif event.type == "interrupt":
        out.interrupt.SetInParent()
    elif event.type == "end_audio":
        out.end_audio.SetInParent()
    else:
        raise ValueError(event.type)
    return out


def _live_server_event_to_proto(pb, event: LiveServerEvent):
    out = pb.LiveServerEvent()
    if event.type == "audio":
        out.audio.data = base64.b64decode(event.data or "")
    elif event.type == "text":
        out.text.text = event.text or ""
    elif event.type == "tool_call":
        out.tool_call.id = event.id or ""
        out.tool_call.name = event.name or ""
        _copy_struct(out.tool_call.input, event.input)
    elif event.type == "interrupted":
        out.interrupted.SetInParent()
    elif event.type == "turn_end":
        assert event.usage is not None
        out.turn_end.usage.CopyFrom(_usage_to_proto(pb, event.usage))
    elif event.type == "error":
        assert event.error is not None
        out.error.error.CopyFrom(_error_to_proto(pb, event.error))
    else:
        raise ValueError(event.type)
    return out


# ─── protobuf -> Python ──────────────────────────────────────────────


def _media_source_from_proto(msg):
    source = msg.WhichOneof("source")
    kwargs: dict[str, Any] = {"media_type": msg.media_type}
    if source == "data":
        kwargs["data"] = _b64(msg.data)
    elif source == "url":
        kwargs["url"] = msg.url
    elif source == "file_id":
        kwargs["file_id"] = msg.file_id
    else:
        raise ValueError("media source missing")
    return kwargs


def _part_from_proto(pb, msg) -> Part:
    maps = _maps(pb)
    kind = msg.WhichOneof("kind")
    if kind == "text":
        return TextPart(text=msg.text.text)
    if kind == "image":
        detail = maps["detail"][1].get(msg.image.detail)
        return ImagePart(**_media_source_from_proto(msg.image.source), detail=detail)
    if kind == "audio":
        return AudioPart(**_media_source_from_proto(msg.audio.source))
    if kind == "video":
        return VideoPart(**_media_source_from_proto(msg.video.source))
    if kind == "document":
        return DocumentPart(**_media_source_from_proto(msg.document.source))
    if kind == "tool_call":
        return ToolCallPart(
            id=msg.tool_call.id,
            name=msg.tool_call.name,
            input=_struct_value(msg.tool_call, "input") or {},
        )
    if kind == "tool_result":
        return ToolResultPart(
            id=msg.tool_result.id,
            content=tuple(_part_from_proto(pb, p) for p in msg.tool_result.content),
            name=_wrapper_value(msg.tool_result, "name"),
            is_error=msg.tool_result.is_error,
        )
    if kind == "thinking":
        return ThinkingPart(text=msg.thinking.text, redacted=msg.thinking.redacted)
    if kind == "refusal":
        return RefusalPart(text=msg.refusal.text)
    if kind == "citation":
        return CitationPart(
            url=_wrapper_value(msg.citation, "url"),
            title=_wrapper_value(msg.citation, "title"),
            text=_wrapper_value(msg.citation, "text"),
        )
    raise ValueError("part kind missing")


def _message_from_proto(pb, msg):
    return Message(
        role=_maps(pb)["role"][1][msg.role],
        parts=tuple(_part_from_proto(pb, p) for p in msg.parts),
    )


def _system_from_proto(pb, msg) -> str | tuple[Part, ...] | None:
    kind = msg.WhichOneof("kind")
    if kind is None:
        return None
    if kind == "text":
        return msg.text
    return tuple(_part_from_proto(pb, p) for p in msg.parts.parts)


def _tool_from_proto(pb, msg):
    kind = msg.WhichOneof("kind")
    if kind == "function":
        return FunctionTool(
            name=msg.function.name,
            description=_wrapper_value(msg.function, "description"),
            parameters=_struct_value(msg.function, "parameters") or {},
        )
    if kind == "builtin":
        return BuiltinTool(
            name=msg.builtin.name,
            config=_struct_value(msg.builtin, "config"),
        )
    raise ValueError("tool kind missing")


def _tool_choice_from_proto(pb, msg):
    return ToolChoice(
        mode=_maps(pb)["mode"][1][msg.mode],
        allowed=tuple(msg.allowed),
        parallel=_wrapper_value(msg, "parallel"),
    )


def _reasoning_from_proto(pb, msg):
    return Reasoning(
        effort=_maps(pb)["effort"][1][msg.effort],
        thinking_budget=_wrapper_value(msg, "thinking_budget"),
        total_budget=_wrapper_value(msg, "total_budget"),
    )


def _config_from_proto(pb, msg):
    return Config(
        max_tokens=_wrapper_value(msg, "max_tokens"),
        temperature=_wrapper_value(msg, "temperature"),
        top_p=_wrapper_value(msg, "top_p"),
        top_k=_wrapper_value(msg, "top_k"),
        stop=tuple(msg.stop),
        response_format=_struct_value(msg, "response_format"),
        tool_choice=_tool_choice_from_proto(pb, msg.tool_choice)
        if msg.HasField("tool_choice")
        else None,
        reasoning=_reasoning_from_proto(pb, msg.reasoning)
        if msg.HasField("reasoning")
        else None,
        extensions=_struct_value(msg, "extensions"),
    )


def _usage_from_proto(pb, msg):
    return Usage(
        input_tokens=msg.input_tokens,
        output_tokens=msg.output_tokens,
        total_tokens=msg.total_tokens,
        cache_read_tokens=_wrapper_value(msg, "cache_read_tokens"),
        cache_write_tokens=_wrapper_value(msg, "cache_write_tokens"),
        reasoning_tokens=_wrapper_value(msg, "reasoning_tokens"),
        input_audio_tokens=_wrapper_value(msg, "input_audio_tokens"),
        output_audio_tokens=_wrapper_value(msg, "output_audio_tokens"),
    )


def _error_from_proto(pb, msg):
    return ErrorDetail(
        code=_maps(pb)["error"][1][msg.code],
        message=msg.message,
        provider_code=_wrapper_value(msg, "provider_code"),
    )


def _delta_from_proto(pb, msg):
    kind = msg.WhichOneof("kind")
    if kind == "text":
        return TextDelta(text=msg.text.text, part_index=msg.text.part_index)
    if kind == "thinking":
        return ThinkingDelta(text=msg.thinking.text, part_index=msg.thinking.part_index)
    if kind == "audio":
        return AudioDelta(
            data=_b64(msg.audio.data),
            part_index=msg.audio.part_index,
            media_type=_wrapper_value(msg.audio, "media_type"),
        )
    if kind == "image":
        source = msg.image.WhichOneof("source")
        kwargs = {
            "part_index": msg.image.part_index,
            "media_type": _wrapper_value(msg.image, "media_type"),
        }
        if source == "data":
            kwargs["data"] = _b64(msg.image.data)
        elif source == "url":
            kwargs["url"] = msg.image.url
        elif source == "file_id":
            kwargs["file_id"] = msg.image.file_id
        return ImageDelta(**kwargs)
    if kind == "tool_call":
        return ToolCallDelta(
            input=msg.tool_call.input,
            part_index=msg.tool_call.part_index,
            id=_wrapper_value(msg.tool_call, "id"),
            name=_wrapper_value(msg.tool_call, "name"),
        )
    if kind == "citation":
        return CitationDelta(
            text=_wrapper_value(msg.citation, "text"),
            url=_wrapper_value(msg.citation, "url"),
            title=_wrapper_value(msg.citation, "title"),
            part_index=msg.citation.part_index,
        )
    raise ValueError("delta kind missing")


def _stream_event_from_proto(pb, msg):
    kind = msg.WhichOneof("event")
    if kind == "start":
        return StreamEvent(
            type="start",
            id=_wrapper_value(msg.start, "id"),
            model=_wrapper_value(msg.start, "model"),
        )
    if kind == "delta":
        return StreamEvent(type="delta", delta=_delta_from_proto(pb, msg.delta.delta))
    if kind == "end":
        finish_reason = _maps(pb)["finish"][1].get(msg.end.finish_reason)
        return StreamEvent(
            type="end",
            finish_reason=finish_reason,
            usage=_usage_from_proto(pb, msg.end.usage) if msg.end.HasField("usage") else None,
            provider_data=_struct_value(msg.end, "provider_data"),
        )
    if kind == "error":
        return StreamEvent(type="error", error=_error_from_proto(pb, msg.error.error))
    raise ValueError("event kind missing")


def _request_from_proto(pb, msg):
    return Request(
        model=msg.model,
        messages=tuple(_message_from_proto(pb, m) for m in msg.messages),
        system=_system_from_proto(pb, msg.system) if msg.HasField("system") else None,
        tools=tuple(_tool_from_proto(pb, t) for t in msg.tools),
        config=_config_from_proto(pb, msg.config) if msg.HasField("config") else Config(),
        cache=_wrapper_value(msg, "cache") if msg.HasField("cache") else True,
    )


def _response_from_proto(pb, msg):
    return Response(
        id=msg.id,
        model=msg.model,
        message=_message_from_proto(pb, msg.message),
        finish_reason=_maps(pb)["finish"][1][msg.finish_reason],
        usage=_usage_from_proto(pb, msg.usage),
        provider_data=_struct_value(msg, "provider_data"),
    )


def _audio_format_from_proto(pb, msg):
    return AudioFormat(
        encoding=_maps(pb)["encoding"][1][msg.encoding],
        sample_rate=msg.sample_rate,
        channels=msg.channels,
    )


def _live_config_from_proto(pb, msg):
    return LiveConfig(
        model=msg.model,
        system=_system_from_proto(pb, msg.system) if msg.HasField("system") else None,
        tools=tuple(_tool_from_proto(pb, t) for t in msg.tools),
        voice=_wrapper_value(msg, "voice"),
        input_format=_audio_format_from_proto(pb, msg.input_format)
        if msg.HasField("input_format")
        else None,
        output_format=_audio_format_from_proto(pb, msg.output_format)
        if msg.HasField("output_format")
        else None,
        extensions=_struct_value(msg, "extensions"),
    )


def _live_client_event_from_proto(pb, msg):
    kind = msg.WhichOneof("event")
    if kind == "audio":
        return LiveClientEvent(type="audio", data=_b64(msg.audio.data))
    if kind == "video":
        return LiveClientEvent(type="video", data=_b64(msg.video.data))
    if kind == "text":
        return LiveClientEvent(type="text", text=msg.text.text)
    if kind == "tool_result":
        return LiveClientEvent(
            type="tool_result",
            id=msg.tool_result.id,
            content=tuple(_part_from_proto(pb, p) for p in msg.tool_result.content),
        )
    if kind == "interrupt":
        return LiveClientEvent(type="interrupt")
    if kind == "end_audio":
        return LiveClientEvent(type="end_audio")
    raise ValueError("live client event missing")


def _live_server_event_from_proto(pb, msg):
    kind = msg.WhichOneof("event")
    if kind == "audio":
        return LiveServerEvent(type="audio", data=_b64(msg.audio.data))
    if kind == "text":
        return LiveServerEvent(type="text", text=msg.text.text)
    if kind == "tool_call":
        return LiveServerEvent(
            type="tool_call",
            id=msg.tool_call.id,
            name=msg.tool_call.name,
            input=_struct_value(msg.tool_call, "input") or {},
        )
    if kind == "interrupted":
        return LiveServerEvent(type="interrupted")
    if kind == "turn_end":
        return LiveServerEvent(type="turn_end", usage=_usage_from_proto(pb, msg.turn_end.usage))
    if kind == "error":
        return LiveServerEvent(type="error", error=_error_from_proto(pb, msg.error.error))
    raise ValueError("live server event missing")


# ─── Round-trip assertions ───────────────────────────────────────────


def _binary_roundtrip(msg):
    clone = msg.__class__()
    clone.ParseFromString(msg.SerializeToString())
    return clone


def _assert_py_proto_py(value, to_proto, from_proto, pb) -> None:
    proto = to_proto(pb, value)
    proto_after_wire = _binary_roundtrip(proto)
    value_after_wire = from_proto(pb, proto_after_wire)
    proto_after_python = to_proto(pb, value_after_wire)

    if proto.DESCRIPTOR.full_name == "lm15.v1.EndpointRequest":
        production_proto = lm15_proto.endpoint_request_to_proto(value, pb)
        production_value = lm15_proto.endpoint_request_from_proto(production_proto, pb)
    elif proto.DESCRIPTOR.full_name == "lm15.v1.EndpointResponse":
        production_proto = lm15_proto.endpoint_response_to_proto(value, pb)
        production_value = lm15_proto.endpoint_response_from_proto(production_proto, pb)
    else:
        production_proto = lm15_proto.to_proto(value, pb)
        production_value = lm15_proto.from_proto(production_proto, pb)

    assert value_after_wire == value
    assert proto_after_python == proto_after_wire
    assert production_proto == proto
    assert production_value == value


def test_part_roundtrips_cover_every_part_variant(pb) -> None:
    parts: list[Part] = [
        TextPart("hello"),
        ImagePart(media_type="image/png", data=_b64(b"image"), detail="low"),
        ImagePart(url="https://example.com/image.png"),
        ImagePart(file_id="file_image"),
        AudioPart(media_type="audio/wav", data=_b64(b"audio")),
        VideoPart(media_type="video/mp4", data=_b64(b"video")),
        DocumentPart(media_type="application/pdf", data=_b64(b"pdf")),
        ToolCallPart(id="call_1", name="lookup", input={"q": "cats", "n": 2}),
        ToolResultPart(id="call_1", content=(TextPart("ok"),), name="lookup", is_error=True),
        ThinkingPart("hidden", redacted=True),
        RefusalPart("no"),
        CitationPart(url="https://example.com", title="Example", text="source"),
    ]

    for part in parts:
        _assert_py_proto_py(part, _part_to_proto, _part_from_proto, pb)


def test_delta_and_stream_event_roundtrips_cover_every_variant(pb) -> None:
    deltas = [
        TextDelta("hi", part_index=0),
        ThinkingDelta("think", part_index=1),
        AudioDelta(data=_b64(b"pcm"), part_index=2, media_type="audio/wav"),
        ImageDelta(data=_b64(b"img"), part_index=3, media_type="image/png"),
        ImageDelta(url="https://example.com/i.png", part_index=4),
        ToolCallDelta(input='{"x":1}', part_index=5, id="call_1", name="fn"),
        CitationDelta(text="quote", url="https://example.com", title="T", part_index=6),
    ]
    for delta in deltas:
        _assert_py_proto_py(delta, _delta_to_proto, _delta_from_proto, pb)

    events = [
        StreamEvent(type="start"),
        StreamEvent(type="start", id="r1", model="m"),
        StreamEvent(type="delta", delta=TextDelta("hi")),
        StreamEvent(
            type="end",
            finish_reason="stop",
            usage=Usage(input_tokens=1, output_tokens=2, total_tokens=3),
            provider_data={"provider": "ok"},
        ),
        StreamEvent(type="error", error=ErrorDetail(code="timeout", message="slow")),
    ]
    for event in events:
        _assert_py_proto_py(event, _stream_event_to_proto, _stream_event_from_proto, pb)


def test_request_and_response_roundtrip(pb) -> None:
    request = Request(
        model="model-1",
        system=(TextPart("system"),),
        messages=(
            Message.developer("be terse"),
            Message.user([
                TextPart("describe"),
                ImagePart(data=_b64(b"image"), detail="auto"),
            ]),
            Message.assistant([ToolCallPart(id="call_1", name="lookup", input={"q": "x"})]),
            Message.tool({"call_1": "result"}),
        ),
        tools=(
            FunctionTool(
                name="lookup",
                description="Lookup a thing",
                parameters={
                    "type": "object",
                    "properties": {"q": {"type": "string"}},
                    "required": ["q"],
                },
            ),
            BuiltinTool("web_search", config={"search_context_size": "low"}),
        ),
        config=Config(
            max_tokens=100,
            temperature=0.2,
            top_p=0.9,
            top_k=5,
            stop=("STOP",),
            response_format={"type": "json_object"},
            tool_choice=ToolChoice(mode="required", allowed=("lookup",), parallel=False),
            reasoning=Reasoning(effort="high", thinking_budget=32, total_budget=64),
            extensions={"openai": {"store": False}},
        ),
        cache=False,
    )
    _assert_py_proto_py(request, _request_to_proto, _request_from_proto, pb)
    assert lm15_proto.from_proto_bytes(
        "Request",
        lm15_proto.to_proto_bytes(request),
    ) == request

    response = Response(
        id="r1",
        model="model-1",
        message=Message.assistant([
            ThinkingPart("hidden"),
            TextPart("hello"),
            CitationPart(url="https://example.com"),
        ]),
        finish_reason="stop",
        usage=Usage(
            input_tokens=10,
            output_tokens=5,
            total_tokens=15,
            cache_read_tokens=3,
            reasoning_tokens=2,
        ),
        provider_data={"id": "native-r1"},
    )
    _assert_py_proto_py(response, _response_to_proto, _response_from_proto, pb)


def test_endpoint_request_response_roundtrips_cover_auxiliary_endpoints(pb) -> None:
    request = Request(model="m", messages=(Message.user("hi"),))
    values = [
        (EmbeddingRequest(model="embed", inputs=("a", "b"), extensions={"task": "search"}), None),
        (
            FileUploadRequest(
                model="m",
                filename="x.pdf",
                bytes_data=b"pdf",
                media_type="application/pdf",
            ),
            None,
        ),
        (BatchRequest(model="m", requests=(request,), extensions={"tier": "batch"}), None),
        (ImageGenerationRequest(model="img", prompt="draw", size="1024x1024"), None),
        (AudioGenerationRequest(model="tts", prompt="say hi", voice="alloy", format="mp3"), None),
    ]

    # Direct request types.
    request_values = [value for value, _ in values]
    _assert_py_proto_py(
        values[0][0], _embedding_request_to_proto, _embedding_request_from_proto, pb
    )
    _assert_py_proto_py(
        values[1][0], _file_upload_request_to_proto, _file_upload_request_from_proto, pb
    )
    _assert_py_proto_py(
        values[2][0], _batch_request_to_proto, _batch_request_from_proto, pb
    )
    _assert_py_proto_py(
        values[3][0],
        _image_generation_request_to_proto,
        _image_generation_request_from_proto,
        pb,
    )
    _assert_py_proto_py(
        values[4][0],
        _audio_generation_request_to_proto,
        _audio_generation_request_from_proto,
        pb,
    )
    for value in (request, *request_values):
        _assert_py_proto_py(value, _endpoint_request_to_proto, _endpoint_request_from_proto, pb)

    responses = [
        (
            EmbeddingResponse(
                model="embed",
                vectors=((1.0, 2.0), (3.0,)),
                usage=Usage(input_tokens=2, output_tokens=1, total_tokens=3),
                provider_data={"native": "ok"},
            ),
            _embedding_response_to_proto,
            _embedding_response_from_proto,
        ),
        (
            FileUploadResponse(id="file_1", provider_data={"size": 3}),
            _file_upload_response_to_proto,
            _file_upload_response_from_proto,
        ),
        (
            BatchResponse(id="batch_1", status="submitted"),
            _batch_response_to_proto,
            _batch_response_from_proto,
        ),
        (
            ImageGenerationResponse(images=(ImagePart(data=_b64(b"img")),), provider_data={"n": 1}),
            _image_generation_response_to_proto,
            _image_generation_response_from_proto,
        ),
        (
            AudioGenerationResponse(
                audio=AudioPart(data=_b64(b"wav")),
                provider_data={"voice": "v"},
            ),
            _audio_generation_response_to_proto,
            _audio_generation_response_from_proto,
        ),
    ]
    for value, to_proto, from_proto in responses:
        _assert_py_proto_py(value, to_proto, from_proto, pb)
    response = Response(
        id="r1",
        model="m",
        message=Message.assistant("ok"),
        finish_reason="stop",
        usage=Usage(),
    )
    for value in (response, *(value for value, _, _ in responses)):
        _assert_py_proto_py(value, _endpoint_response_to_proto, _endpoint_response_from_proto, pb)


def test_live_config_and_events_roundtrip(pb) -> None:
    config = LiveConfig(
        model="live-model",
        system="be quick",
        tools=(BuiltinTool("code_execution"),),
        voice="verse",
        input_format=AudioFormat("pcm16", sample_rate=16000, channels=1),
        output_format=AudioFormat("opus", sample_rate=24000, channels=1),
        extensions={"native": {"x": True}},
    )
    _assert_py_proto_py(config, _live_config_to_proto, _live_config_from_proto, pb)

    client_events = [
        LiveClientEvent(type="audio", data=_b64(b"audio")),
        LiveClientEvent(type="video", data=_b64(b"video")),
        LiveClientEvent(type="text", text="hello"),
        LiveClientEvent(type="tool_result", id="call_1", content=(TextPart("ok"),)),
        LiveClientEvent(type="interrupt"),
        LiveClientEvent(type="end_audio"),
    ]
    for event in client_events:
        _assert_py_proto_py(event, _live_client_event_to_proto, _live_client_event_from_proto, pb)

    server_events = [
        LiveServerEvent(type="audio", data=_b64(b"audio")),
        LiveServerEvent(type="text", text="hello"),
        LiveServerEvent(type="tool_call", id="call_1", name="lookup", input={"q": "x"}),
        LiveServerEvent(type="interrupted"),
        LiveServerEvent(
            type="turn_end",
            usage=Usage(input_tokens=4, output_tokens=5, total_tokens=9),
        ),
        LiveServerEvent(type="error", error=ErrorDetail(code="provider", message="bad")),
    ]
    for event in server_events:
        _assert_py_proto_py(event, _live_server_event_to_proto, _live_server_event_from_proto, pb)


def test_standalone_config_tool_usage_error_and_audio_format_roundtrip(pb) -> None:
    values = [
        (FunctionTool("lookup"), _tool_to_proto, _tool_from_proto),
        (BuiltinTool("web_search", config={}), _tool_to_proto, _tool_from_proto),
        (
            ToolChoice(mode="auto", allowed=("lookup",), parallel=True),
            _tool_choice_to_proto,
            _tool_choice_from_proto,
        ),
        (
            Reasoning(effort="minimal", thinking_budget=1, total_budget=2),
            _reasoning_to_proto,
            _reasoning_from_proto,
        ),
        (Config(response_format={}, extensions={}), _config_to_proto, _config_from_proto),
        (
            Usage(input_tokens=1, output_tokens=2, total_tokens=3),
            _usage_to_proto,
            _usage_from_proto,
        ),
        (
            ErrorDetail(code="auth", message="bad key", provider_code="401"),
            _error_to_proto,
            _error_from_proto,
        ),
        (
            AudioFormat("aac", sample_rate=44100, channels=2),
            _audio_format_to_proto,
            _audio_format_from_proto,
        ),
    ]
    for value, to_proto, from_proto in values:
        _assert_py_proto_py(value, to_proto, from_proto, pb)


def test_tool_call_info_roundtrip(pb) -> None:
    value = ToolCallInfo(id="call_1", name="lookup", input={"q": "cats"})
    proto = pb.ToolCallInfo(id=value.id, name=value.name)
    _copy_struct(proto.input, value.input)
    clone = _binary_roundtrip(proto)
    production_proto = lm15_proto.tool_call_info_to_proto(value, pb)

    assert (
        ToolCallInfo(
            id=clone.id,
            name=clone.name,
            input=_struct_value(clone, "input") or {},
        )
        == value
    )
    assert production_proto == proto
    assert lm15_proto.tool_call_info_from_proto(production_proto, pb) == value


# Auxiliary endpoint converters live below the tests so the main core
# conversion logic above stays readable.


def _endpoint_request_to_proto(pb, value):
    out = pb.EndpointRequest()
    if isinstance(value, Request):
        out.request.CopyFrom(_request_to_proto(pb, value))
    elif isinstance(value, EmbeddingRequest):
        out.embedding_request.CopyFrom(_embedding_request_to_proto(pb, value))
    elif isinstance(value, FileUploadRequest):
        out.file_upload_request.CopyFrom(_file_upload_request_to_proto(pb, value))
    elif isinstance(value, BatchRequest):
        out.batch_request.CopyFrom(_batch_request_to_proto(pb, value))
    elif isinstance(value, ImageGenerationRequest):
        out.image_generation_request.CopyFrom(_image_generation_request_to_proto(pb, value))
    elif isinstance(value, AudioGenerationRequest):
        out.audio_generation_request.CopyFrom(_audio_generation_request_to_proto(pb, value))
    else:
        raise TypeError(type(value))
    return out


def _endpoint_request_from_proto(pb, msg):
    kind = msg.WhichOneof("kind")
    if kind == "request":
        return _request_from_proto(pb, msg.request)
    if kind == "embedding_request":
        return _embedding_request_from_proto(pb, msg.embedding_request)
    if kind == "file_upload_request":
        return _file_upload_request_from_proto(pb, msg.file_upload_request)
    if kind == "batch_request":
        return _batch_request_from_proto(pb, msg.batch_request)
    if kind == "image_generation_request":
        return _image_generation_request_from_proto(pb, msg.image_generation_request)
    if kind == "audio_generation_request":
        return _audio_generation_request_from_proto(pb, msg.audio_generation_request)
    raise ValueError("endpoint request kind missing")


def _endpoint_response_to_proto(pb, value):
    out = pb.EndpointResponse()
    if isinstance(value, Response):
        out.response.CopyFrom(_response_to_proto(pb, value))
    elif isinstance(value, EmbeddingResponse):
        out.embedding_response.CopyFrom(_embedding_response_to_proto(pb, value))
    elif isinstance(value, FileUploadResponse):
        out.file_upload_response.CopyFrom(_file_upload_response_to_proto(pb, value))
    elif isinstance(value, BatchResponse):
        out.batch_response.CopyFrom(_batch_response_to_proto(pb, value))
    elif isinstance(value, ImageGenerationResponse):
        out.image_generation_response.CopyFrom(_image_generation_response_to_proto(pb, value))
    elif isinstance(value, AudioGenerationResponse):
        out.audio_generation_response.CopyFrom(_audio_generation_response_to_proto(pb, value))
    else:
        raise TypeError(type(value))
    return out


def _endpoint_response_from_proto(pb, msg):
    kind = msg.WhichOneof("kind")
    if kind == "response":
        return _response_from_proto(pb, msg.response)
    if kind == "embedding_response":
        return _embedding_response_from_proto(pb, msg.embedding_response)
    if kind == "file_upload_response":
        return _file_upload_response_from_proto(pb, msg.file_upload_response)
    if kind == "batch_response":
        return _batch_response_from_proto(pb, msg.batch_response)
    if kind == "image_generation_response":
        return _image_generation_response_from_proto(pb, msg.image_generation_response)
    if kind == "audio_generation_response":
        return _audio_generation_response_from_proto(pb, msg.audio_generation_response)
    raise ValueError("endpoint response kind missing")


def _embedding_request_to_proto(pb, value: EmbeddingRequest):
    out = pb.EmbeddingRequest(model=value.model)
    out.inputs.extend(value.inputs)
    _copy_struct(out.extensions, value.extensions)
    return out


def _embedding_request_from_proto(pb, msg):
    return EmbeddingRequest(
        model=msg.model,
        inputs=tuple(msg.inputs),
        extensions=_struct_value(msg, "extensions"),
    )


def _embedding_response_to_proto(pb, value: EmbeddingResponse):
    out = pb.EmbeddingResponse(model=value.model)
    for vector in value.vectors:
        v = out.vectors.add()
        v.values.extend(vector)
    out.usage.CopyFrom(_usage_to_proto(pb, value.usage))
    _copy_struct(out.provider_data, value.provider_data)
    return out


def _embedding_response_from_proto(pb, msg):
    return EmbeddingResponse(
        model=msg.model,
        vectors=tuple(tuple(v.values) for v in msg.vectors),
        usage=_usage_from_proto(pb, msg.usage) if msg.HasField("usage") else Usage(),
        provider_data=_struct_value(msg, "provider_data"),
    )


def _file_upload_request_to_proto(pb, value: FileUploadRequest):
    out = pb.FileUploadRequest(
        filename=value.filename,
        bytes_data=value.bytes_data,
        media_type=value.media_type,
    )
    if value.model is not None:
        _set_wrapper(out.model, value.model)
    _copy_struct(out.extensions, value.extensions)
    return out


def _file_upload_request_from_proto(pb, msg):
    return FileUploadRequest(
        model=_wrapper_value(msg, "model"),
        filename=msg.filename,
        bytes_data=msg.bytes_data,
        media_type=msg.media_type,
        extensions=_struct_value(msg, "extensions"),
    )


def _file_upload_response_to_proto(pb, value: FileUploadResponse):
    out = pb.FileUploadResponse(id=value.id)
    _copy_struct(out.provider_data, value.provider_data)
    return out


def _file_upload_response_from_proto(pb, msg):
    return FileUploadResponse(id=msg.id, provider_data=_struct_value(msg, "provider_data"))


def _batch_request_to_proto(pb, value: BatchRequest):
    out = pb.BatchRequest(model=value.model)
    out.requests.extend(_request_to_proto(pb, r) for r in value.requests)
    _copy_struct(out.extensions, value.extensions)
    return out


def _batch_request_from_proto(pb, msg):
    return BatchRequest(
        model=msg.model,
        requests=tuple(_request_from_proto(pb, r) for r in msg.requests),
        extensions=_struct_value(msg, "extensions"),
    )


def _batch_response_to_proto(pb, value: BatchResponse):
    out = pb.BatchResponse(id=value.id, status=value.status)
    _copy_struct(out.provider_data, value.provider_data)
    return out


def _batch_response_from_proto(pb, msg):
    return BatchResponse(
        id=msg.id,
        status=msg.status,
        provider_data=_struct_value(msg, "provider_data"),
    )


def _image_generation_request_to_proto(pb, value: ImageGenerationRequest):
    out = pb.ImageGenerationRequest(model=value.model, prompt=value.prompt)
    if value.size is not None:
        _set_wrapper(out.size, value.size)
    _copy_struct(out.extensions, value.extensions)
    return out


def _image_generation_request_from_proto(pb, msg):
    return ImageGenerationRequest(
        model=msg.model,
        prompt=msg.prompt,
        size=_wrapper_value(msg, "size"),
        extensions=_struct_value(msg, "extensions"),
    )


def _image_generation_response_to_proto(pb, value: ImageGenerationResponse):
    out = pb.ImageGenerationResponse()
    for image in value.images:
        out.images.add().CopyFrom(_part_to_proto(pb, image).image)
    _copy_struct(out.provider_data, value.provider_data)
    return out


def _image_generation_response_from_proto(pb, msg):
    return ImageGenerationResponse(
        images=tuple(
            ImagePart(
                **_media_source_from_proto(img.source),
                detail=_maps(pb)["detail"][1].get(img.detail),
            )
            for img in msg.images
        ),
        provider_data=_struct_value(msg, "provider_data"),
    )


def _audio_generation_request_to_proto(pb, value: AudioGenerationRequest):
    out = pb.AudioGenerationRequest(model=value.model, prompt=value.prompt)
    if value.voice is not None:
        _set_wrapper(out.voice, value.voice)
    if value.format is not None:
        _set_wrapper(out.format, value.format)
    _copy_struct(out.extensions, value.extensions)
    return out


def _audio_generation_request_from_proto(pb, msg):
    return AudioGenerationRequest(
        model=msg.model,
        prompt=msg.prompt,
        voice=_wrapper_value(msg, "voice"),
        format=_wrapper_value(msg, "format"),
        extensions=_struct_value(msg, "extensions"),
    )


def _audio_generation_response_to_proto(pb, value: AudioGenerationResponse):
    out = pb.AudioGenerationResponse()
    out.audio.CopyFrom(_part_to_proto(pb, value.audio).audio)
    _copy_struct(out.provider_data, value.provider_data)
    return out


def _audio_generation_response_from_proto(pb, msg):
    return AudioGenerationResponse(
        audio=AudioPart(**_media_source_from_proto(msg.audio.source)),
        provider_data=_struct_value(msg, "provider_data"),
    )
