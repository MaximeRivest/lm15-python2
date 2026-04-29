from __future__ import annotations

import json
from dataclasses import dataclass
from typing import ClassVar, Iterator, Protocol

from ..errors import (
    ProviderError,
    TransportError as LM15TransportError,
    UnsupportedFeatureError,
    map_http_error,
)
from ..features import EndpointSupport, ProviderManifest
from ..protocols import Capabilities, LiveSession
from ..sse import SSEEvent, parse_sse
from ..transports import Request as TransportRequest
from ..transports import Response as TransportResponse
from ..transports import StdlibTransport
from ..transports import TransportError as NetworkTransportError
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


class SyncTransport(Protocol):
    """Minimal sync transport surface used by provider LMs."""

    def stream(self, request: TransportRequest) -> TransportResponse: ...


def default_transport() -> SyncTransport:
    """Create the default sync transport for standalone provider LMs."""
    return StdlibTransport()


@dataclass(frozen=True, slots=True)
class HttpResponse:
    """Buffered provider-level HTTP response.

    The stdlib transport is streaming-first.  LMs that implement ordinary
    request/response endpoints buffer the body into this small value object so
    their parsing code can stay pure and easy to test.
    """

    status: int
    reason: str
    headers: list[tuple[str, str]]
    body: bytes
    http_version: str = "HTTP/1.1"

    def header(self, name: str) -> str | None:
        lname = name.lower()
        for key, value in self.headers:
            if key.lower() == lname:
                return value
        return None

    def headers_all(self, name: str) -> list[str]:
        lname = name.lower()
        return [value for key, value in self.headers if key.lower() == lname]

    def text(self) -> str:
        return self.body.decode("utf-8", errors="replace")

    def json(self):
        return json.loads(self.body)


class ProviderLM(Protocol):
    provider: str
    capabilities: Capabilities
    supports: EndpointSupport
    manifest: ProviderManifest

    def build_request(self, request: Request, stream: bool) -> TransportRequest: ...

    def parse_response(self, request: Request, response: HttpResponse) -> Response: ...

    def parse_stream_event(self, request: Request, raw_event: SSEEvent) -> StreamEvent | None: ...

    def normalize_error(self, status: int, body: str) -> ProviderError: ...


class BaseProviderLM:
    """Shared synchronous provider LM implementation."""

    transport: SyncTransport
    provider: str = "unknown"
    capabilities: Capabilities = Capabilities()
    supports: ClassVar[EndpointSupport] = EndpointSupport()
    manifest: ClassVar[ProviderManifest] = ProviderManifest(
        provider="unknown", supports=EndpointSupport()
    )

    def complete(self, request: Request) -> Response:
        req = self.build_request(request, stream=False)
        resp = self._send(req)
        if resp.status >= 400:
            raise self.normalize_error(resp.status, resp.text())
        return self.parse_response(request, resp)

    def stream(self, request: Request) -> Iterator[StreamEvent]:
        req = self.build_request(request, stream=True)
        try:
            with self.transport.stream(req) as resp:
                if resp.status >= 400:
                    body = resp.read()
                    raise self.normalize_error(
                        resp.status, body.decode("utf-8", errors="replace")
                    )
                lines = resp.iter_lines() if hasattr(resp, "iter_lines") else _iter_lines(resp)
                for raw in parse_sse(lines):
                    parse_many = getattr(self, "parse_stream_events", None)
                    if parse_many is not None:
                        for event in parse_many(request, raw):
                            if event is not None:
                                yield event
                    else:
                        event = self.parse_stream_event(request, raw)
                        if event is not None:
                            yield event
        except NetworkTransportError as exc:
            raise LM15TransportError(str(exc)) from exc

    def _send(self, request: TransportRequest) -> HttpResponse:
        try:
            with self.transport.stream(request) as resp:
                body = resp.read()
                return HttpResponse(
                    status=resp.status,
                    reason=resp.reason,
                    headers=resp.headers,
                    body=body,
                    http_version=resp.http_version,
                )
        except NetworkTransportError as exc:
            raise LM15TransportError(str(exc)) from exc

    def normalize_error(self, status: int, body: str) -> ProviderError:
        return map_http_error(status, body.strip()[:500] or f"HTTP {status}")

    def close(self) -> None:
        close = getattr(self.transport, "close", None)
        if callable(close):
            close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

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


class UnsupportedLiveSession:
    def send(self, event):
        raise UnsupportedFeatureError("live session not supported")

    def recv(self):
        raise UnsupportedFeatureError("live session not supported")

    def close(self) -> None:
        return


def _iter_lines(chunks: Iterator[bytes]) -> Iterator[bytes]:
    """Split arbitrary byte chunks into newline-terminated lines for SSE."""

    buf = bytearray()
    for chunk in chunks:
        if not chunk:
            continue
        buf.extend(chunk)
        while True:
            idx = buf.find(b"\n")
            if idx < 0:
                break
            line = bytes(buf[: idx + 1])
            del buf[: idx + 1]
            yield line
    if buf:
        yield bytes(buf)
