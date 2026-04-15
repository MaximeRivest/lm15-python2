"""
lm15.providers.common — Shared helpers for provider adapters.

Translation from universal Part/Source types to provider-specific
wire formats.
"""

from __future__ import annotations

from typing import Any

from ..types import (
    Part,
    Source,
    TextPart,
    ImagePart,
    AudioPart,
    VideoPart,
    DocumentPart,
    ToolResultPart,
    ThinkingPart,
    RefusalPart,
)


def parts_to_text(parts: tuple[Part, ...]) -> str:
    """Extract concatenated text from text-like parts."""
    return "\n".join(
        p.text
        for p in parts
        if isinstance(p, (TextPart, ThinkingPart, RefusalPart)) and p.text
    )


# ─── OpenAI Responses API input format ──────────────────────────────

def part_to_openai_input(part: Part) -> dict[str, Any]:
    """Convert a Part to an OpenAI Responses API input content block."""
    if isinstance(part, TextPart):
        return {"type": "input_text", "text": part.text}

    if isinstance(part, ImagePart):
        s = part.source
        if s.type == "url":
            payload: dict[str, Any] = {"type": "input_image", "image_url": s.url}
            if s.detail:
                payload["detail"] = s.detail
            return payload
        if s.type == "base64":
            return {"type": "input_image", "image_url": f"data:{s.media_type};base64,{s.data}"}
        if s.type == "file":
            return {"type": "input_image", "file_id": s.file_id}

    if isinstance(part, AudioPart):
        s = part.source
        if s.type == "base64":
            media = (s.media_type or "audio/wav").split("/")[-1]
            return {"type": "input_audio", "audio": s.data, "format": media}
        if s.type == "url":
            return {"type": "input_audio", "audio_url": s.url}
        if s.type == "file":
            return {"type": "input_audio", "file_id": s.file_id}

    if isinstance(part, DocumentPart):
        s = part.source
        if s.type == "url":
            return {"type": "input_file", "file_url": s.url}
        if s.type == "base64":
            return {"type": "input_file", "file_data": f"data:{s.media_type};base64,{s.data}"}
        if s.type == "file":
            return {"type": "input_file", "file_id": s.file_id}

    if isinstance(part, VideoPart):
        s = part.source
        if s.type == "url":
            return {"type": "input_video", "video_url": s.url}
        if s.type == "base64":
            return {"type": "input_video", "video_data": f"data:{s.media_type};base64,{s.data}"}
        if s.type == "file":
            return {"type": "input_video", "file_id": s.file_id}

    if isinstance(part, ToolResultPart):
        return {"type": "input_text", "text": parts_to_text(part.content)}

    # Fallback for thinking, refusal, citation — extract text
    if isinstance(part, (ThinkingPart, RefusalPart)):
        return {"type": "input_text", "text": part.text}

    return {"type": "input_text", "text": ""}


# ─── Anthropic source format ────────────────────────────────────────

def source_to_anthropic(s: Source) -> dict[str, Any]:
    """Convert a Source to Anthropic's source wire format."""
    if s.type == "url":
        return {"type": "url", "url": s.url}
    if s.type == "file":
        return {"type": "file", "file_id": s.file_id}
    return {"type": "base64", "media_type": s.media_type, "data": s.data}
