"""
Transport-level request/response models.

These are intentionally minimal — they're the bytes-in/bytes-out interface
between the adapter layer (which speaks the lm15 type system) and the
HTTP transport.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import AsyncIterator, Iterator


@dataclass(slots=True)
class Request:
    method: str
    url: str
    headers: list[tuple[str, str]] = field(default_factory=list)
    body: bytes = b""
    # Per-request timeout overrides (None = use transport default)
    connect_timeout: float | None = None
    read_timeout: float | None = None
    write_timeout: float | None = None


class Response:
    """Sync streaming response.

    Iterating yields body chunks as bytes.  Must be used as a context manager
    so the connection is properly returned to the pool (or closed) even on
    early exit.
    """

    status: int
    reason: str
    headers: list[tuple[str, str]]
    http_version: str

    def __init__(
        self,
        *,
        status: int,
        reason: str,
        headers: list[tuple[str, str]],
        http_version: str,
        chunks: Iterator[bytes],
        release: "callable",
    ) -> None:
        self.status = status
        self.reason = reason
        self.headers = headers
        self.http_version = http_version
        self._chunks = chunks
        self._release = release
        self._released = False
        self._complete = False

    def header(self, name: str) -> str | None:
        lname = name.lower()
        for k, v in self.headers:
            if k.lower() == lname:
                return v
        return None

    def headers_all(self, name: str) -> list[str]:
        lname = name.lower()
        return [v for k, v in self.headers if k.lower() == lname]

    def __iter__(self) -> Iterator[bytes]:
        try:
            for chunk in self._chunks:
                if chunk:
                    yield chunk
            self._complete = True
        finally:
            self._release_once(body_consumed=self._complete)

    def read(self) -> bytes:
        return b"".join(self)

    def close(self) -> None:
        self._release_once(body_consumed=self._complete)

    def _release_once(self, *, body_consumed: bool) -> None:
        if self._released:
            return
        self._released = True
        try:
            self._release(body_consumed)
        except Exception:
            pass

    def __enter__(self) -> "Response":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


class AsyncResponse:
    """Async streaming response.  Async-iterate to get body chunks."""

    status: int
    reason: str
    headers: list[tuple[str, str]]
    http_version: str

    def __init__(
        self,
        *,
        status: int,
        reason: str,
        headers: list[tuple[str, str]],
        http_version: str,
        chunks: AsyncIterator[bytes],
        release: "callable",
    ) -> None:
        self.status = status
        self.reason = reason
        self.headers = headers
        self.http_version = http_version
        self._chunks = chunks
        self._release = release  # async callable(body_consumed: bool)
        self._released = False
        self._complete = False

    def header(self, name: str) -> str | None:
        lname = name.lower()
        for k, v in self.headers:
            if k.lower() == lname:
                return v
        return None

    def headers_all(self, name: str) -> list[str]:
        lname = name.lower()
        return [v for k, v in self.headers if k.lower() == lname]

    def __aiter__(self) -> AsyncIterator[bytes]:
        return self._iter()

    async def _iter(self) -> AsyncIterator[bytes]:
        try:
            async for chunk in self._chunks:
                if chunk:
                    yield chunk
            self._complete = True
        finally:
            await self._release_once(body_consumed=self._complete)

    async def read(self) -> bytes:
        buf = bytearray()
        async for c in self:
            buf.extend(c)
        return bytes(buf)

    async def aclose(self) -> None:
        await self._release_once(body_consumed=self._complete)

    async def _release_once(self, *, body_consumed: bool) -> None:
        if self._released:
            return
        self._released = True
        try:
            await self._release(body_consumed)
        except Exception:
            pass

    async def __aenter__(self) -> "AsyncResponse":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.aclose()
