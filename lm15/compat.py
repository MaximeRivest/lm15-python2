"""
lm15.compat — Typed provider/API compatibility policies.

Compatibility policies describe how a provider adapter should serialize a
canonical lm15 request for a specific API dialect. They are intentionally
separate from lm15.types: Request/Response describe *what* the caller wants;
compat profiles describe provider wire-format quirks.

Fields default to None, which means "inherit from the parent profile". The
string value "auto" is an explicit policy: ask the adapter to use its automatic
heuristic for that field. Keeping None distinct from "auto" matters because
profiles are layered.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Literal, TypeAlias, get_args

from .types import JsonObject


# ─── Shared helpers ──────────────────────────────────────────────────


def _check_literal_or_none(value: object, literal_alias: object, field_name: str) -> None:
    if value is not None and value not in get_args(literal_alias):  # type: ignore[arg-type]
        raise ValueError(f"unsupported {field_name}: {value!r}")


def _check_json_object_or_none(value: object, field_name: str) -> None:
    if value is None:
        return
    if not isinstance(value, dict):
        raise TypeError(f"{field_name} must be a JSON object or None")


def _merge_json_object(a: JsonObject | None, b: JsonObject | None) -> JsonObject | None:
    if a is None:
        return b
    if b is None:
        return a
    return {**a, **b}


# ─── OpenAI Responses API compatibility ──────────────────────────────

OpenAIResponsesDeveloperRole = Literal["auto", "developer", "system"]
OpenAIResponsesMaxOutputTokensField = Literal[
    "auto",
    "max_output_tokens",
    "max_completion_tokens",
    "max_tokens",
]
OpenAIResponsesReasoningFormat = Literal[
    "auto",
    "none",
    "responses_reasoning",
    "reasoning_effort",
    "openrouter",
    "deepseek",
    "qwen",
    "qwen_chat_template",
    "zai",
]
OpenAIToolResultName = Literal["auto", "include", "omit"]
OpenAIStrictTools = Literal["auto", "include", "omit"]
OpenAICacheControl = Literal["auto", "none", "openai", "anthropic"]


@dataclass(frozen=True, slots=True)
class OpenAIResponsesCompat:
    """Partial compatibility policy for OpenAI Responses-family APIs.

    None means "inherit". Non-None values override parent profiles. The value
    "auto" means "explicitly use adapter auto-detection".
    """

    developer_role: OpenAIResponsesDeveloperRole | None = None
    max_output_tokens_field: OpenAIResponsesMaxOutputTokensField | None = None
    reasoning_format: OpenAIResponsesReasoningFormat | None = None
    tool_result_name: OpenAIToolResultName | None = None
    strict_tools: OpenAIStrictTools | None = None
    cache_control: OpenAICacheControl | None = None
    routing: JsonObject | None = None
    extensions: JsonObject | None = None

    def __post_init__(self) -> None:
        _check_literal_or_none(self.developer_role, OpenAIResponsesDeveloperRole, "developer_role")
        _check_literal_or_none(
            self.max_output_tokens_field,
            OpenAIResponsesMaxOutputTokensField,
            "max_output_tokens_field",
        )
        _check_literal_or_none(self.reasoning_format, OpenAIResponsesReasoningFormat, "reasoning_format")
        _check_literal_or_none(self.tool_result_name, OpenAIToolResultName, "tool_result_name")
        _check_literal_or_none(self.strict_tools, OpenAIStrictTools, "strict_tools")
        _check_literal_or_none(self.cache_control, OpenAICacheControl, "cache_control")
        _check_json_object_or_none(self.routing, "routing")
        _check_json_object_or_none(self.extensions, "extensions")

    @classmethod
    def preset(cls, name: str) -> "OpenAIResponsesCompat":
        key = name.lower().replace("-", "_").replace(" ", "_")

        if key in {"openai", "responses", "openai_responses"}:
            return cls(
                developer_role="developer",
                max_output_tokens_field="max_output_tokens",
                reasoning_format="responses_reasoning",
                tool_result_name="omit",
                strict_tools="omit",
                cache_control="openai",
            )

        if key == "openrouter":
            return cls(
                developer_role="developer",
                max_output_tokens_field="max_tokens",
                reasoning_format="openrouter",
                tool_result_name="omit",
                strict_tools="omit",
                cache_control="openai",
            )

        if key in {"ollama", "lmstudio", "lm_studio"}:
            return cls(
                developer_role="system",
                max_output_tokens_field="max_tokens",
                reasoning_format="none",
                tool_result_name="omit",
                strict_tools="omit",
                cache_control="none",
            )

        if key in {"vllm", "sglang"}:
            return cls(
                developer_role="system",
                max_output_tokens_field="max_tokens",
                reasoning_format="reasoning_effort",
                tool_result_name="omit",
                strict_tools="omit",
                cache_control="none",
            )

        if key in {"qwen", "dashscope_qwen"}:
            return cls(
                developer_role="system",
                max_output_tokens_field="max_tokens",
                reasoning_format="qwen",
                tool_result_name="omit",
                strict_tools="omit",
                cache_control="none",
            )

        if key == "deepseek":
            return cls(
                developer_role="system",
                max_output_tokens_field="max_tokens",
                reasoning_format="deepseek",
                tool_result_name="omit",
                strict_tools="omit",
                cache_control="none",
            )

        if key in {"zai", "z_ai"}:
            return cls(
                developer_role="system",
                max_output_tokens_field="max_tokens",
                reasoning_format="zai",
                tool_result_name="omit",
                strict_tools="omit",
                cache_control="none",
            )

        raise ValueError(f"unknown OpenAIResponsesCompat preset: {name!r}")


@dataclass(frozen=True, slots=True)
class ResolvedOpenAIResponsesCompat:
    """Fully resolved OpenAI Responses compatibility policy."""

    developer_role: Literal["developer", "system"] = "developer"
    max_output_tokens_field: Literal["max_output_tokens", "max_completion_tokens", "max_tokens"] = "max_output_tokens"
    reasoning_format: Literal[
        "none",
        "responses_reasoning",
        "reasoning_effort",
        "openrouter",
        "deepseek",
        "qwen",
        "qwen_chat_template",
        "zai",
    ] = "responses_reasoning"
    tool_result_name: Literal["include", "omit"] = "omit"
    strict_tools: Literal["include", "omit"] = "omit"
    cache_control: Literal["none", "openai", "anthropic"] = "openai"
    routing: JsonObject | None = None
    extensions: JsonObject | None = None


# ─── OpenAI Chat Completions compatibility ───────────────────────────

OpenAIChatInstructionRole = Literal["auto", "developer", "system"]
OpenAIChatMaxTokensField = Literal["auto", "max_completion_tokens", "max_tokens"]
OpenAIChatStreamUsage = Literal["auto", "include", "omit"]
OpenAIChatAssistantAfterToolResult = Literal["auto", "insert", "omit"]
OpenAIChatThinkingReplay = Literal["auto", "native", "as_text", "omit"]
OpenAIChatAssistantReasoningContent = Literal["auto", "include_empty", "omit"]
OpenAIChatThinkingFormat = Literal[
    "auto",
    "none",
    "reasoning_effort",
    "openrouter",
    "deepseek",
    "qwen",
    "qwen_chat_template",
    "zai",
]


@dataclass(frozen=True, slots=True)
class OpenAIChatCompat:
    """Partial compatibility policy for OpenAI Chat Completions-family APIs.

    This class is provided now so profiles can describe chat-completions style
    endpoints without overloading OpenAIResponsesCompat. It is not consumed by
    the current OpenAILM Responses serializer.
    """

    instruction_role: OpenAIChatInstructionRole | None = None
    max_tokens_field: OpenAIChatMaxTokensField | None = None
    stream_usage: OpenAIChatStreamUsage | None = None
    tool_result_name: OpenAIToolResultName | None = None
    assistant_after_tool_result: OpenAIChatAssistantAfterToolResult | None = None
    thinking_format: OpenAIChatThinkingFormat | None = None
    thinking_replay: OpenAIChatThinkingReplay | None = None
    assistant_reasoning_content: OpenAIChatAssistantReasoningContent | None = None
    strict_tools: OpenAIStrictTools | None = None
    cache_control: OpenAICacheControl | None = None
    routing: JsonObject | None = None
    extensions: JsonObject | None = None

    def __post_init__(self) -> None:
        _check_literal_or_none(self.instruction_role, OpenAIChatInstructionRole, "instruction_role")
        _check_literal_or_none(self.max_tokens_field, OpenAIChatMaxTokensField, "max_tokens_field")
        _check_literal_or_none(self.stream_usage, OpenAIChatStreamUsage, "stream_usage")
        _check_literal_or_none(self.tool_result_name, OpenAIToolResultName, "tool_result_name")
        _check_literal_or_none(
            self.assistant_after_tool_result,
            OpenAIChatAssistantAfterToolResult,
            "assistant_after_tool_result",
        )
        _check_literal_or_none(self.thinking_format, OpenAIChatThinkingFormat, "thinking_format")
        _check_literal_or_none(self.thinking_replay, OpenAIChatThinkingReplay, "thinking_replay")
        _check_literal_or_none(
            self.assistant_reasoning_content,
            OpenAIChatAssistantReasoningContent,
            "assistant_reasoning_content",
        )
        _check_literal_or_none(self.strict_tools, OpenAIStrictTools, "strict_tools")
        _check_literal_or_none(self.cache_control, OpenAICacheControl, "cache_control")
        _check_json_object_or_none(self.routing, "routing")
        _check_json_object_or_none(self.extensions, "extensions")


CompatProfile: TypeAlias = OpenAIResponsesCompat | OpenAIChatCompat


# ─── Merge helpers ──────────────────────────────────────────────────


def merge_openai_responses_compat(
    base: OpenAIResponsesCompat,
    override: OpenAIResponsesCompat | None,
) -> OpenAIResponsesCompat:
    """Merge partial OpenAI Responses compat objects.

    None fields inherit. Non-None fields, including "auto", override.
    """
    if override is None:
        return base

    kwargs = {}
    for f in fields(OpenAIResponsesCompat):
        value = getattr(override, f.name)
        if f.name == "extensions":
            kwargs[f.name] = _merge_json_object(base.extensions, override.extensions)
        else:
            kwargs[f.name] = getattr(base, f.name) if value is None else value
    return OpenAIResponsesCompat(**kwargs)


def resolve_openai_responses_compat(partial: OpenAIResponsesCompat) -> ResolvedOpenAIResponsesCompat:
    """Resolve a partial compat object into concrete serializer policy."""
    developer_role = partial.developer_role
    if developer_role in {None, "auto"}:
        developer_role = "developer"

    max_field = partial.max_output_tokens_field
    if max_field in {None, "auto"}:
        max_field = "max_output_tokens"

    reasoning_format = partial.reasoning_format
    if reasoning_format in {None, "auto"}:
        reasoning_format = "responses_reasoning"

    tool_result_name = partial.tool_result_name
    if tool_result_name in {None, "auto"}:
        tool_result_name = "omit"

    strict_tools = partial.strict_tools
    if strict_tools in {None, "auto"}:
        strict_tools = "omit"

    cache_control = partial.cache_control
    if cache_control in {None, "auto"}:
        cache_control = "openai"

    return ResolvedOpenAIResponsesCompat(
        developer_role=developer_role,  # type: ignore[arg-type]
        max_output_tokens_field=max_field,  # type: ignore[arg-type]
        reasoning_format=reasoning_format,  # type: ignore[arg-type]
        tool_result_name=tool_result_name,  # type: ignore[arg-type]
        strict_tools=strict_tools,  # type: ignore[arg-type]
        cache_control=cache_control,  # type: ignore[arg-type]
        routing=partial.routing,
        extensions=partial.extensions,
    )
