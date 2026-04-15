"""
lm15.features — Provider capability and endpoint declarations.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class EndpointSupport:
    complete: bool = True
    stream: bool = True
    live: bool = False
    embeddings: bool = False
    files: bool = False
    batches: bool = False
    images: bool = False
    audio: bool = False
    responses_api: bool = False


@dataclass(frozen=True, slots=True)
class ProviderManifest:
    provider: str
    supports: EndpointSupport
    auth_modes: tuple[str, ...] = field(default_factory=tuple)
    enterprise_variants: tuple[str, ...] = field(default_factory=tuple)
    env_keys: tuple[str, ...] = field(default_factory=tuple)
