"""
lm15.providers.base — Base adapter with default implementations.

The adapter contract:
    build_request(Request, stream) → HttpRequest
    parse_response(Request, HttpResponse) → Response
    parse_stream_event(Request, SSEEvent) → StreamEvent | None
    normalize_error(status, body) → ProviderError

Three methods for the core flow.  One for errors.
The adapter doesn't own HTTP — the Transport does.
The adapter doesn't own streaming assembly — Result does.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Iterator, Protocol

from ..errors import ProviderError, UnsupportedFeatureError, map_http_error
from ..features import EndpointSupport, ProviderManifest
from ..protocols import Capabilities, LiveSession
from ..sse import SSEEvent, parse_sse
from ..transports.base import HttpRequest, HttpResponse, Transport
from ..types import (
    AudioGenerationRequest,
    AudioGenerationResponse,
    BatchRequest,
    BatchResponse,
    EmbeddingRequest,
    EmbeddingResponse,
    FileUploadRequest,
    FileUploadResponse,
    ImageGenerationRequest,
    ImageGenerationResponse,
    LiveConfig,
    Request,
    Response,
    StreamEvent,
)


class ProviderAdapter(Protocol):
    """The contract every provider adapter fulfills."""
    provider: str
    capabilities: Capabilities
    supports: EndpointSupport
    manifest: ProviderManifest

    def build_request(self, request: Request, stream: bool) -> HttpRequest: ...
    def parse_response(self, request: Request, response: HttpResponse) -> Response: ...
    def parse_stream_event(self, request: Request, raw_event: SSEEvent) -> StreamEvent | None: ...
    def normalize_error(self, status: int, body: str) -> ProviderError: ...


@dataclass(slots=True)
class BaseProviderAdapter:
    """Base with default complete/stream implementations.

    Subclasses implement:
      - build_request
      - parse_response
      - parse_stream_event
    And optionally override normalize_error, live, embeddings, etc.
    """
    transport: Transport
    supports: ClassVar[EndpointSupport] = EndpointSupport()
    manifest: ClassVar[ProviderManifest] = ProviderManifest(provider="unknown", supports=EndpointSupport())

    def complete(self, request: Request) -> Response:
        req = self.build_request(request, stream=False)
        resp = self.transport.request(req)
        if resp.status >= 400:
            raise self.normalize_error(resp.status, resp.text())
        return self.parse_response(request, resp)

    def stream(self, request: Request) -> Iterator[StreamEvent]:
        req = self.build_request(request, stream=True)
        for raw in parse_sse(self.transport.stream(req)):
            evt = self.parse_stream_event(request, raw)
            if evt is not None:
                yield evt

    def normalize_error(self, status: int, body: str) -> ProviderError:
        return map_http_error(status, body)

    # ─── Optional endpoints (raise by default) ───────────────────

    def live(self, config: LiveConfig) -> LiveSession:
        raise UnsupportedFeatureError(f"{self.provider}: live not supported")

    def embeddings(self, request: EmbeddingRequest) -> EmbeddingResponse:
        raise UnsupportedFeatureError(f"{self.provider}: embeddings not supported")

    def file_upload(self, request: FileUploadRequest) -> FileUploadResponse:
        raise UnsupportedFeatureError(f"{self.provider}: file upload not supported")

    def batch_submit(self, request: BatchRequest) -> BatchResponse:
        raise UnsupportedFeatureError(f"{self.provider}: batch submit not supported")

    def image_generate(self, request: ImageGenerationRequest) -> ImageGenerationResponse:
        raise UnsupportedFeatureError(f"{self.provider}: image generation not supported")

    def audio_generate(self, request: AudioGenerationRequest) -> AudioGenerationResponse:
        raise UnsupportedFeatureError(f"{self.provider}: audio generation not supported")
