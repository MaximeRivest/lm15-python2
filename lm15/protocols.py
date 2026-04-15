"""
lm15.protocols — Protocol definitions for adapters and sessions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Protocol

from .features import EndpointSupport, ProviderManifest
from .types import (
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
    LiveClientEvent,
    LiveConfig,
    LiveServerEvent,
    Request,
    Response,
    StreamEvent,
)


@dataclass(frozen=True, slots=True)
class Capabilities:
    input_modalities: frozenset[str] = field(default_factory=frozenset)
    output_modalities: frozenset[str] = field(default_factory=frozenset)
    features: frozenset[str] = field(default_factory=frozenset)


class LiveSession(Protocol):
    def send(self, event: LiveClientEvent) -> None: ...
    def recv(self) -> LiveServerEvent: ...
    def close(self) -> None: ...


class LMAdapter(Protocol):
    provider: str
    capabilities: Capabilities
    supports: EndpointSupport
    manifest: ProviderManifest

    def complete(self, request: Request) -> Response: ...
    def stream(self, request: Request) -> Iterator[StreamEvent]: ...
    def live(self, config: LiveConfig) -> LiveSession: ...
    def embeddings(self, request: EmbeddingRequest) -> EmbeddingResponse: ...
    def file_upload(self, request: FileUploadRequest) -> FileUploadResponse: ...
    def batch_submit(self, request: BatchRequest) -> BatchResponse: ...
    def image_generate(self, request: ImageGenerationRequest) -> ImageGenerationResponse: ...
    def audio_generate(self, request: AudioGenerationRequest) -> AudioGenerationResponse: ...
