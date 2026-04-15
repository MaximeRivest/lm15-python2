"""
lm15 — Universal foundation model interface.

Core types:
    Part (union):  TextPart, ImagePart, AudioPart, VideoPart, DocumentPart,
                   ToolCallPart, ToolResultPart, ThinkingPart, RefusalPart, CitationPart
    Message:       Parts attributed to a speaker (user/assistant/tool)
    Request:       A complete request to a model
    Response:      The model's composed response
    Delta:         A typed fragment of a Part during streaming
    StreamEvent:   The streaming protocol (start/delta/end/error)

Part constructors:
    text(), image(), audio(), video(), document(),
    tool_call(), tool_result(), thinking(), refusal(), citation()
"""

from .types import (
    # Part variants
    TextPart,
    ImagePart,
    AudioPart,
    VideoPart,
    DocumentPart,
    ToolCallPart,
    ToolResultPart,
    ThinkingPart,
    RefusalPart,
    CitationPart,
    # Part union
    Part,
    MediaPart,
    MEDIA_TYPES,
    PART_TYPES,
    # Part constructors
    text,
    image,
    audio,
    video,
    document,
    tool_call,
    tool_result,
    thinking,
    refusal,
    citation,
    # Source
    Source,
    # Messages
    Message,
    # Stream types
    Delta,
    StreamEvent,
    ErrorDetail,
    # Tools
    FunctionTool,
    BuiltinTool,
    Tool,
    ToolCallInfo,
    # Config
    Config,
    Reasoning,
    ToolChoice,
    # Request / Response
    Request,
    Response,
    Usage,
    # Other requests
    EmbeddingRequest,
    EmbeddingResponse,
    FileUploadRequest,
    FileUploadResponse,
    BatchRequest,
    BatchResponse,
    ImageGenerationRequest,
    ImageGenerationResponse,
    AudioGenerationRequest,
    AudioGenerationResponse,
    # Live
    AudioFormat,
    LiveConfig,
    LiveClientEvent,
    LiveServerEvent,
    # JSON types
    JsonObject,
    JsonValue,
    JsonArray,
    # Literal types
    Role,
    PartType,
    DeltaType,
    FinishReason,
    ReasoningEffort,
    ErrorCode,
    StreamEventType,
)

from .errors import (
    LM15Error,
    TransportError,
    ProviderError,
    AuthError,
    RateLimitError,
    BillingError,
    TimeoutError,
    InvalidRequestError,
    ContextLengthError,
    ServerError,
    UnsupportedModelError,
    UnsupportedFeatureError,
    NotConfiguredError,
)

from .result import Result, AsyncResult, StreamChunk

__all__ = [
    # Part variants
    "TextPart", "ImagePart", "AudioPart", "VideoPart", "DocumentPart",
    "ToolCallPart", "ToolResultPart", "ThinkingPart", "RefusalPart", "CitationPart",
    "Part", "MediaPart", "MEDIA_TYPES", "PART_TYPES",
    # Part constructors
    "text", "image", "audio", "video", "document",
    "tool_call", "tool_result", "thinking", "refusal", "citation",
    # Source
    "Source",
    # Messages
    "Message",
    # Stream
    "Delta", "StreamEvent", "ErrorDetail",
    # Tools
    "FunctionTool", "BuiltinTool", "Tool", "ToolCallInfo",
    # Config
    "Config", "Reasoning", "ToolChoice",
    # Request / Response
    "Request", "Response", "Usage",
    # Result
    "Result", "AsyncResult", "StreamChunk",
    # Errors
    "LM15Error", "TransportError", "ProviderError",
    "AuthError", "RateLimitError", "BillingError", "TimeoutError",
    "InvalidRequestError", "ContextLengthError", "ServerError",
    "UnsupportedModelError", "UnsupportedFeatureError", "NotConfiguredError",
]
