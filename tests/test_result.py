from __future__ import annotations

import base64

import pytest

from lm15.result import (
    Result,
    _invoke_tool,
    _normalize_tool_output,
    response_to_events,
)
from lm15.types import (
    AudioPart,
    DocumentPart,
    ImageDelta,
    ImagePart,
    Message,
    RefusalPart,
    Request,
    Response,
    TextPart,
    Usage,
    VideoPart,
)


def _b64(value: bytes) -> str:
    return base64.b64encode(value).decode("ascii")


def _response_with(part) -> Response:
    return Response(
        id="r1",
        model="m",
        message=Message.assistant(part),
        finish_reason="stop",
        usage=Usage(),
    )


def test_response_to_events_preserves_image_file_ids() -> None:
    image = ImagePart(file_id="file_123")
    events = list(response_to_events(_response_with(image)))

    assert events[0].type == "start"
    assert isinstance(events[1].delta, ImageDelta)
    assert events[1].delta.file_id == "file_123"
    assert events[-1].type == "end"


@pytest.mark.parametrize(
    "part",
    [
        # AudioPart by reference is non-streamable: AudioDelta requires inline data.
        AudioPart(url="https://example.com/audio.wav"),
        VideoPart(data=_b64(b"video")),
        DocumentPart(data=_b64(b"pdf")),
        RefusalPart("no"),
    ],
)
def test_response_to_events_raises_for_parts_without_delta_variants(part) -> None:
    with pytest.raises(TypeError, match="Cannot convert"):
        list(response_to_events(_response_with(part)))


def test_result_delegates_video_and_document_helpers() -> None:
    video = VideoPart(data=_b64(b"video"))
    document = DocumentPart(data=_b64(b"doc"))
    response = Response(
        id="r1",
        model="m",
        message=Message.assistant([video, document]),
        finish_reason="stop",
        usage=Usage(),
    )
    result = Result(events=iter(()), request=Request(model="m", messages=(Message.user("hi"),)))
    result._response = response
    result._done = True

    assert result.video is video
    assert result.videos == [video]
    assert result.video_bytes == b"video"
    assert result.document is document
    assert result.documents == [document]
    assert result.document_bytes == b"doc"


def test_tool_output_normalization_accepts_any_part_sequence() -> None:
    video = VideoPart(data=_b64(b"video"))
    document = DocumentPart(data=_b64(b"doc"))

    assert _normalize_tool_output((video, document)) == [video, document]
    assert _normalize_tool_output("ok") == [TextPart("ok")]


def test_tool_invocation_fallback_does_not_swallow_internal_type_errors() -> None:
    def accepts_payload(payload):
        return payload["x"]

    def raises_inside(x):
        raise TypeError("internal bug")

    assert _invoke_tool(accepts_payload, {"x": 1}) == 1
    with pytest.raises(TypeError, match="internal bug"):
        _invoke_tool(raises_inside, {"x": 1})
