from __future__ import annotations

import shutil
import subprocess

import pytest
from google.protobuf import descriptor_pb2

import lm15.protobuf as lm15_proto

PROTO = "proto/lm15/v1/lm15.proto"


def _compile_descriptor_set(tmp_path, *, include_imports: bool = False):
    protoc = shutil.which("protoc")
    if protoc is None:
        pytest.skip("protoc is not installed")

    out = tmp_path / "lm15.pb"
    command = [
        protoc,
        "--proto_path=proto",
        "--proto_path=/usr/include",
        f"--descriptor_set_out={out}",
        PROTO,
    ]
    if include_imports:
        command.insert(-2, "--include_imports")
    subprocess.run(command, check=True)

    fds = descriptor_pb2.FileDescriptorSet()
    fds.ParseFromString(out.read_bytes())
    return fds


def _compile_descriptor(tmp_path):
    fds = _compile_descriptor_set(tmp_path)
    return next(f for f in fds.file if f.name == "lm15/v1/lm15.proto")


def _messages(file):
    return {message.name: message for message in file.message_type}


def _enums(file):
    return {enum.name: enum for enum in file.enum_type}


def _oneof_fields(message, oneof_name: str) -> list[str]:
    index = next(i for i, oneof in enumerate(message.oneof_decl) if oneof.name == oneof_name)
    return [
        field.name
        for field in message.field
        if field.HasField("oneof_index") and field.oneof_index == index
    ]


def _field_names(message) -> list[str]:
    return [field.name for field in message.field]


def test_embedded_descriptor_matches_proto_file(tmp_path) -> None:
    compiled = _compile_descriptor_set(tmp_path, include_imports=True)
    embedded = descriptor_pb2.FileDescriptorSet()
    embedded.ParseFromString(lm15_proto.descriptor_set_bytes())

    compiled_lm15 = next(f for f in compiled.file if f.name == "lm15/v1/lm15.proto")
    embedded_lm15 = next(f for f in embedded.file if f.name == "lm15/v1/lm15.proto")
    assert embedded_lm15 == compiled_lm15


def test_proto_compiles_and_models_core_unions(tmp_path) -> None:
    file = _compile_descriptor(tmp_path)
    messages = _messages(file)

    assert _oneof_fields(messages["Part"], "kind") == [
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
    assert _oneof_fields(messages["Delta"], "kind") == [
        "text",
        "thinking",
        "audio",
        "image",
        "tool_call",
        "citation",
    ]
    assert _oneof_fields(messages["StreamEvent"], "event") == [
        "start",
        "delta",
        "end",
        "error",
    ]


def test_proto_preserves_media_and_request_shape(tmp_path) -> None:
    file = _compile_descriptor(tmp_path)
    messages = _messages(file)

    assert _oneof_fields(messages["MediaSource"], "source") == [
        "data",
        "url",
        "file_id",
    ]
    assert _field_names(messages["Request"]) == [
        "model",
        "messages",
        "system",
        "tools",
        "config",
        "cache",
    ]
    assert _oneof_fields(messages["SystemContent"], "kind") == ["text", "parts"]


def test_proto_has_language_neutral_endpoint_and_live_unions(tmp_path) -> None:
    file = _compile_descriptor(tmp_path)
    messages = _messages(file)

    assert _oneof_fields(messages["EndpointRequest"], "kind") == [
        "request",
        "embedding_request",
        "file_upload_request",
        "batch_request",
        "image_generation_request",
        "audio_generation_request",
    ]
    assert _oneof_fields(messages["EndpointResponse"], "kind") == [
        "response",
        "embedding_response",
        "file_upload_response",
        "batch_response",
        "image_generation_response",
        "audio_generation_response",
    ]
    assert _oneof_fields(messages["LiveClientEvent"], "event") == [
        "audio",
        "video",
        "text",
        "tool_result",
        "interrupt",
        "end_audio",
    ]
    assert _oneof_fields(messages["LiveServerEvent"], "event") == [
        "audio",
        "text",
        "tool_call",
        "interrupted",
        "turn_end",
        "error",
        "tool_call_delta",
    ]


def test_proto_enums_cover_python_literals(tmp_path) -> None:
    file = _compile_descriptor(tmp_path)
    enums = _enums(file)

    assert [value.name for value in enums["Role"].value] == [
        "ROLE_UNSPECIFIED",
        "ROLE_USER",
        "ROLE_ASSISTANT",
        "ROLE_TOOL",
        "ROLE_DEVELOPER",
    ]
    assert [value.name for value in enums["FinishReason"].value] == [
        "FINISH_REASON_UNSPECIFIED",
        "FINISH_REASON_STOP",
        "FINISH_REASON_LENGTH",
        "FINISH_REASON_TOOL_CALL",
        "FINISH_REASON_CONTENT_FILTER",
        "FINISH_REASON_ERROR",
    ]
    assert [value.name for value in enums["ReasoningEffort"].value] == [
        "REASONING_EFFORT_UNSPECIFIED",
        "REASONING_EFFORT_OFF",
        "REASONING_EFFORT_ADAPTIVE",
        "REASONING_EFFORT_MINIMAL",
        "REASONING_EFFORT_LOW",
        "REASONING_EFFORT_MEDIUM",
        "REASONING_EFFORT_HIGH",
        "REASONING_EFFORT_XHIGH",
    ]
    assert [value.name for value in enums["ErrorCode"].value] == [
        "ERROR_CODE_UNSPECIFIED",
        "ERROR_CODE_AUTH",
        "ERROR_CODE_BILLING",
        "ERROR_CODE_RATE_LIMIT",
        "ERROR_CODE_INVALID_REQUEST",
        "ERROR_CODE_CONTEXT_LENGTH",
        "ERROR_CODE_TIMEOUT",
        "ERROR_CODE_SERVER",
        "ERROR_CODE_PROVIDER",
        "ERROR_CODE_UNSUPPORTED_MODEL",
        "ERROR_CODE_UNSUPPORTED_FEATURE",
        "ERROR_CODE_NOT_CONFIGURED",
        "ERROR_CODE_TRANSPORT",
    ]
