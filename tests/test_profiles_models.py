import json
from dataclasses import replace

from lm15.compat import OpenAIResponsesCompat
from lm15.model_registry import ModelRegistry
from lm15.models import InferenceModelInfo, InferencePricing, ModelInfo
from lm15.profiles import ProviderProfile, resolve_openai_responses_compat
from lm15.providers import OpenAILM
from lm15.types import Config, FunctionTool, Message, Reasoning, Request, ToolResultPart, TextPart

from .test_providers import _FakeTransport


def test_openai_responses_compat_none_inherits_but_auto_overrides() -> None:
    profile = ProviderProfile.inference(
        provider="local",
        api_family="openai_responses",
        compat=OpenAIResponsesCompat.preset("ollama"),
        models=(
            ModelInfo(
                id="m",
                provider="local",
                api_family="openai_responses",
                compat=OpenAIResponsesCompat(reasoning_format="auto"),
            ),
        ),
    )

    compat = resolve_openai_responses_compat(
        base_url="http://localhost:11434/v1",
        model="m",
        profile=profile,
        request_extensions=None,
    )

    # Provider preset is inherited where the model did not override.
    assert compat.developer_role == "system"
    # The explicit "auto" model override resolves to adapter default rather
    # than inheriting provider-level reasoning_format="none".
    assert compat.reasoning_format == "responses_reasoning"


def test_model_registry_resolves_aliases_and_estimates_cost() -> None:
    pricing = InferencePricing(input_per_million=1.0, output_per_million=2.0)
    model = ModelInfo(
        id="qwen3-coder",
        provider="local",
        api_family="openai_responses",
        aliases=("qwen",),
        inference=InferenceModelInfo(
            input_modalities=("text",),
            context_window=128000,
            pricing=pricing,
        ),
    )
    registry = ModelRegistry()
    registry.add(model)

    assert registry.resolve("qwen", provider="local") is model
    assert registry.resolve("qwen3-coder") is model
    assert pricing.estimate(input_tokens=1_000_000, output_tokens=500_000) == 2.0


def test_openai_profile_controls_responses_payload() -> None:
    profile = ProviderProfile.inference(
        provider="local-qwen",
        api_family="openai_responses",
        compat=OpenAIResponsesCompat.preset("qwen"),
    )
    lm = OpenAILM(api_key="local", base_url="http://localhost:8000/v1", profile=profile, transport=_FakeTransport())
    request = Request(
        model="qwen3",
        messages=(Message.developer("Follow policy."), Message.user("Hi")),
        tools=(FunctionTool(name="lookup"),),
        config=Config(max_tokens=123, reasoning=Reasoning(effort="high"), cache=None),
    )

    payload = json.loads(lm.build_request(request, stream=True).body)

    assert payload["max_tokens"] == 123
    assert "max_output_tokens" not in payload
    assert payload["enable_thinking"] is True
    assert payload["input"][0]["role"] == "system"
    assert "strict" not in payload["tools"][0]


def test_request_level_openai_responses_compat_override_is_not_passed_through() -> None:
    lm = OpenAILM(api_key="sk-test", transport=_FakeTransport())
    request = Request(
        model="m",
        messages=(Message.user("Hi"),),
        config=Config(
            max_tokens=7,
            extensions={
                "openai_responses_compat": {"max_output_tokens_field": "max_tokens"},
                "metadata": {"user_id": "u1"},
            },
        ),
    )

    payload = json.loads(lm.build_request(request, stream=False).body)

    assert payload["max_tokens"] == 7
    assert "max_output_tokens" not in payload
    assert "openai_responses_compat" not in payload
    assert payload["metadata"] == {"user_id": "u1"}


def test_openai_tool_result_name_compat() -> None:
    profile = ProviderProfile.inference(
        provider="proxy",
        api_family="openai_responses",
        compat=OpenAIResponsesCompat(tool_result_name="include"),
    )
    lm = OpenAILM(api_key="sk-test", profile=profile, transport=_FakeTransport())
    request = Request(
        model="m",
        messages=(
            Message(
                role="tool",
                parts=(ToolResultPart(id="call_1", name="lookup", content=(TextPart("ok"),)),),
            ),
        ),
    )

    payload = json.loads(lm.build_request(request, stream=False).body)

    assert payload["input"][0]["name"] == "lookup"
