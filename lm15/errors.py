"""
lm15.errors — Canonical error taxonomy.

Every provider maps its idiosyncratic error shapes onto this hierarchy.
Error classes are the primary signal; canonical string codes exist for
serialization and wire formats.

Hierarchy:
    LM15Error
    ├── TransportError          (network/connection failures)
    └── ProviderError           (provider returned an error)
        ├── AuthError           (401/403 — bad or missing API key)
        ├── BillingError        (402 — payment/quota issue)
        ├── RateLimitError      (429 — too many requests)
        ├── InvalidRequestError (400/404/422 — bad request shape)
        │   └── ContextLengthError  (input too long for model)
        ├── TimeoutError        (408/504 — request timed out)
        ├── ServerError         (5xx — provider-side failure)
        ├── UnsupportedModelError
        └── UnsupportedFeatureError
"""

from __future__ import annotations


class LM15Error(Exception):
    """Base for all lm15 errors."""


class TransportError(LM15Error):
    """Network or connection failure."""


class ProviderError(LM15Error):
    """The provider returned an error."""


class AuthError(ProviderError):
    """Authentication failed — invalid, expired, or missing API key."""

    def __init__(self, message: str = "") -> None:
        guidance = (
            "\n\n"
            "  To fix, do one of:\n"
            "    1. Check that your API key is correct and not expired\n"
            "    2. Set it in your environment: export OPENAI_API_KEY=sk-...\n"
            "    3. Pass it directly: lm15.call(..., api_key='sk-...')\n"
            "    4. Add it to a .env file and call lm15.configure(env='.env')\n"
        )
        if guidance.strip() not in message:
            message = message.rstrip() + guidance
        super().__init__(message)


class RateLimitError(ProviderError):
    """Rate limited by the provider (HTTP 429)."""

    def __init__(self, message: str = "") -> None:
        guidance = (
            "\n\n"
            "  To fix:\n"
            "    - Wait a moment and retry\n"
            "    - Use retries= on model objects: lm15.model(..., retries=3)\n"
            "    - Reduce request rate or upgrade your API plan\n"
        )
        if guidance.strip() not in message:
            message = message.rstrip() + guidance
        super().__init__(message)


class BillingError(ProviderError):
    """402 — billing or payment issue."""


class TimeoutError(ProviderError):
    """Request timed out."""


class InvalidRequestError(ProviderError):
    """Bad request shape (400/404/422)."""


class ContextLengthError(InvalidRequestError):
    """The input exceeds the model's context window."""

    def __init__(self, message: str = "") -> None:
        guidance = (
            "\n\n"
            "  To fix:\n"
            "    - Reduce the prompt or system prompt length\n"
            "    - Clear conversation history\n"
            "    - Use a model with a larger context window\n"
            "    - Lower max_tokens to leave more room for input\n"
        )
        if guidance.strip() not in message:
            message = message.rstrip() + guidance
        super().__init__(message)


class ServerError(ProviderError):
    """Provider-side failure (5xx)."""


class UnsupportedModelError(ProviderError):
    """Model not found or not supported."""


class UnsupportedFeatureError(ProviderError):
    """Feature not supported by this provider."""


class NotConfiguredError(ProviderError):
    """No API key found for a provider."""


# ─── HTTP status → error class mapping ───────────────────────────────

def map_http_error(status: int, message: str) -> ProviderError:
    """Map HTTP status + message to a typed ProviderError.

    LMs extract the human-readable message from the provider's
    error body in their normalize_error override.  This function
    only maps status codes.
    """
    if status in (401, 403):
        return AuthError(message)
    if status == 402:
        return BillingError(message)
    if status in (408, 504):
        return TimeoutError(message)
    if status == 429:
        return RateLimitError(message)
    if status in (400, 404, 409, 413, 422):
        return InvalidRequestError(message)
    if 500 <= status <= 599:
        return ServerError(message)
    return ProviderError(message)


# ─── Canonical error codes ───────────────────────────────────────────

# Bidirectional mapping between error classes and string codes.
# Codes are provider-agnostic and stable across LMs.

_CLASS_TO_CODE: dict[type[ProviderError], str] = {
    ContextLengthError: "context_length",
    AuthError: "auth",
    BillingError: "billing",
    RateLimitError: "rate_limit",
    InvalidRequestError: "invalid_request",
    TimeoutError: "timeout",
    ServerError: "server",
    ProviderError: "provider",
}

_CODE_TO_CLASS: dict[str, type[ProviderError]] = {v: k for k, v in _CLASS_TO_CODE.items()}


def canonical_error_code(error: type[ProviderError] | ProviderError) -> str:
    """Return the canonical string code for an error class or instance."""
    cls = error if isinstance(error, type) else type(error)
    for check_cls, code in _CLASS_TO_CODE.items():
        if issubclass(cls, check_cls):
            return code
    return "provider"


def error_class_for_code(code: str) -> type[ProviderError]:
    """Return the ProviderError subclass for a canonical string code."""
    return _CODE_TO_CLASS.get(code, ProviderError)


# Retryable errors — used by Result for automatic retries
RETRYABLE_ERRORS = (RateLimitError, TimeoutError, ServerError, TransportError)
