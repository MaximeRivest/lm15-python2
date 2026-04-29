import base64

import pytest

from lm15.serde import (
    config_from_dict,
    config_to_dict,
    delta_from_dict,
    delta_to_dict,
    tool_to_dict,
)
from lm15.types import (
    AudioDelta,
    AudioPart,
    CitationPart,
    Config,
    DocumentPart,
    EmbeddingRequest,
    EmbeddingResponse,
    FileUploadRequest,
    FunctionTool,
    ImageDelta,
    ImageGenerationRequest,
    ImagePart,
    LiveServerEvent,
    Message,
    Reasoning,
    RefusalPart,
    Response,
    StreamEvent,
    TextDelta,
    TextPart,
    ThinkingPart,
    ToolCallDelta,
    ToolCallPart,
    ToolChoice,
    ToolResultPart,
    Usage,
    VideoPart,
    tool_call,
)


def _b64(value: bytes) -> str:
    return base64.b64encode(value).decode("ascii")


def test_media_parts_share_validation_and_byte_access() -> None:
    assert ImagePart(data=_b64(b"image")).bytes == b"image"
    assert AudioPart(data=_b64(b"audio")).bytes == b"audio"
    assert DocumentPart(data=_b64(b"doc")).bytes == b"doc"

    with pytest.raises(ValueError, match="requires exactly one"):
        ImagePart(data=_b64(b"image"), url="https://example.com/image.png")


def test_message_filters_power_response_helpers() -> None:
    image = ImagePart(data=_b64(b"image"))
    video = VideoPart(data=_b64(b"video"))
    document = DocumentPart(data=_b64(b"doc"))
    message = Message.assistant(
        [
            TextPart("hello"),
            ThinkingPart("hidden"),
            image,
            video,
            document,
            CitationPart(url="https://example.com"),
        ]
    )
    response = Response(
        id="r1",
        model="m",
        message=message,
        finish_reason="stop",
        usage=Usage(),
    )

    assert message.parts_of(TextPart) == [TextPart("hello")]
    assert message.first(ImagePart) is image
    assert response.text == "hello"
    assert response.thinking == "hidden"
    assert response.image is image
    assert response.images == [image]
    assert response.image_bytes == b"image"
    assert response.video is video
    assert response.videos == [video]
    assert response.video_bytes == b"video"
    assert response.document is document
    assert response.documents == [document]
    assert response.document_bytes == b"doc"
    assert response.citations == [CitationPart(url="https://example.com")]


def test_delta_variants_are_proper_unions() -> None:
    delta = TextDelta("hello")

    assert delta.type == "text"
    assert delta.text == "hello"
    assert not hasattr(delta, "data")


def test_image_delta_requires_exactly_one_media_address() -> None:
    with pytest.raises(ValueError, match="requires exactly one"):
        ImageDelta()

    assert (
        ImageDelta(url="https://example.com/image.png").url
        == "https://example.com/image.png"
    )


def test_delta_serde_roundtrips_variant_types() -> None:
    tool_delta = ToolCallDelta(input='{"x": 1}', id="call_1", name="lookup")
    image_delta = ImageDelta(url="https://example.com/image.png", media_type="image/png")

    assert delta_from_dict(delta_to_dict(tool_delta)) == tool_delta
    assert delta_from_dict(delta_to_dict(image_delta)) == image_delta
    assert delta_from_dict(delta_to_dict(TextDelta(""))) == TextDelta("")


def test_function_tool_is_serializable_spec_without_callable() -> None:
    def lookup(query: str) -> str:
        """Look up a query."""
        return query

    tool = FunctionTool.from_fn(lookup)

    assert tool.name == "lookup"
    assert tool.description == "Look up a query."
    assert not hasattr(tool, "fn")
    assert "fn" not in tool_to_dict(tool)


def test_endpoint_request_bases_validate_model_and_prompt() -> None:
    with pytest.raises(ValueError, match="model is required"):
        EmbeddingRequest(model="", inputs=("hello",))

    with pytest.raises(ValueError, match="prompt is required"):
        ImageGenerationRequest(model="m", prompt="")


def test_reasoning_serde_uses_current_fields_and_reads_legacy_budget() -> None:
    config = Config(
        reasoning=Reasoning(effort="high", thinking_budget=12, total_budget=100)
    )

    assert config_to_dict(config)["reasoning"] == {
        "effort": "high",
        "thinking_budget": 12,
        "total_budget": 100,
    }

    # Legacy payloads with enabled=False + budget collapse to effort="off";
    # the budget is dropped because reasoning is disabled.
    legacy = config_from_dict({"reasoning": {"enabled": False, "budget": 7}})
    assert legacy.reasoning == Reasoning(effort="off")

    with pytest.raises(ValueError, match="effort='off'"):
        Reasoning(effort="off", thinking_budget=7)


def test_json_fields_are_validated() -> None:
    with pytest.raises(TypeError, match="input"):
        tool_call("call_1", "lookup", {"bad": object()})  # type: ignore[dict-item]

    with pytest.raises(TypeError, match="response_format"):
        Config(response_format={"enum": ("a", "b")})  # type: ignore[dict-item]

    with pytest.raises(TypeError, match="extensions"):
        Config(extensions={"bad": object()})  # type: ignore[dict-item]


def test_sequence_inputs_are_normalized_to_tuples() -> None:
    msg = Message.user((TextPart("a"), TextPart("b")))
    choice = ToolChoice(allowed=["lookup"])  # type: ignore[arg-type]
    config = Config(stop="END")  # type: ignore[arg-type]

    assert msg.parts == (TextPart("a"), TextPart("b"))
    assert choice.allowed == ("lookup",)
    assert config.stop == ("END",)


def test_stream_events_validate_type_specific_fields() -> None:
    with pytest.raises(ValueError, match="requires delta"):
        StreamEvent(type="delta")

    with pytest.raises(ValueError, match="requires error"):
        StreamEvent(type="error")

    # Providers do not always expose ids or usage at stream boundaries;
    # the materializer can fill sane defaults from the request.
    assert StreamEvent(type="start").type == "start"
    assert StreamEvent(type="end").type == "end"


def test_numeric_budgets_and_usage_cannot_be_negative() -> None:
    with pytest.raises(ValueError, match="thinking_budget"):
        Reasoning(thinking_budget=0)

    with pytest.raises(ValueError, match="input_tokens"):
        Usage(input_tokens=-1)


def test_frozen_json_object_blocks_in_place_or_assignment() -> None:
    """Provider/tool-call payloads must remain immutable through `|=` aliases."""
    call = ToolCallPart(id="c", name="f", input={"a": 1})
    aliased = call.input
    with pytest.raises(TypeError):
        aliased |= {"mutated": True}
    assert "mutated" not in call.input


def test_tool_result_rejects_thinking_and_protocol_parts() -> None:
    """ToolResultPart rejects parts outside ToolResultContentPart at runtime."""
    with pytest.raises(TypeError, match="thinking parts"):
        ToolResultPart(id="c", content=(ThinkingPart("internal"),))


def test_user_messages_reject_citations() -> None:
    """User and developer messages cannot carry model-emitted citation parts."""
    with pytest.raises(TypeError, match="protocol parts"):
        Message.user(CitationPart(url="https://example.com"))
    with pytest.raises(TypeError, match="protocol parts"):
        Message.developer(CitationPart(url="https://example.com"))


def test_text_parts_reject_non_string_text() -> None:
    with pytest.raises(TypeError, match="TextPart.text"):
        TextPart(text=123)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="ThinkingPart.text"):
        ThinkingPart(text=None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="RefusalPart.text"):
        RefusalPart(text="")


def test_numeric_validators_reject_bool() -> None:
    """`bool` subclasses `int`; numeric fields should still reject it."""
    with pytest.raises(TypeError, match="max_tokens"):
        Config(max_tokens=True)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="part_index"):
        TextDelta(text="x", part_index=True)  # type: ignore[arg-type]


def test_audio_delta_validates_base64_data() -> None:
    with pytest.raises(ValueError, match="base64"):
        AudioDelta(data="not base64!@#")
    valid = AudioDelta(data=base64.b64encode(b"hi").decode("ascii"))
    assert valid.data == base64.b64encode(b"hi").decode("ascii")


def test_reasoning_off_rejects_budgets() -> None:
    with pytest.raises(ValueError, match="effort='off'"):
        Reasoning(effort="off", thinking_budget=10)
    with pytest.raises(ValueError, match="effort='off'"):
        Reasoning(effort="off", total_budget=20)


def test_embedding_response_validates_payload() -> None:
    with pytest.raises(ValueError, match="requires model"):
        EmbeddingResponse(model="", vectors=((1.0,),))
    with pytest.raises(ValueError, match="at least one vector"):
        EmbeddingResponse(model="m", vectors=())
    with pytest.raises(ValueError, match="vectors cannot be empty"):
        EmbeddingResponse(model="m", vectors=((),))
    with pytest.raises(ValueError, match="finite"):
        EmbeddingResponse(model="m", vectors=((float("nan"),),))


def test_generated_media_optional_strings_reject_empty() -> None:
    with pytest.raises(ValueError, match="size"):
        ImageGenerationRequest(model="m", prompt="draw", size="")


def test_file_upload_request_requires_payload() -> None:
    with pytest.raises(TypeError):
        FileUploadRequest()  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="bytes_data"):
        FileUploadRequest(filename="f", bytes_data=b"")


def test_live_and_stream_tool_call_deltas_are_symmetric_on_empty_input() -> None:
    """Both code paths must accept empty fragments equally."""
    ToolCallDelta(input="")  # accepted
    LiveServerEvent(type="tool_call_delta", input_delta="")  # accepted


def test_forbid_fields_is_derived_from_dataclass_fields() -> None:
    """Adding a field to one variant should still be rejected on others."""
    with pytest.raises(ValueError, match="cannot include"):
        StreamEvent(type="start", delta=TextDelta(text="x"))
    with pytest.raises(ValueError, match="cannot include"):
        LiveServerEvent(type="interrupted", text="oops")
