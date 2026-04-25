"""
lm15.transports — Minimal stdlib-only HTTP/1.1 transports.

Public API:
    Request, Response         — transport-level request/response models
    StdlibTransport           — sync transport (blocking, socket-based)
    StdlibAsyncTransport      — async transport (asyncio-native)
    TransportError + subclasses
"""

from ._exceptions import (
    ConnectError,
    ConnectTimeout,
    ProtocolError,
    ReadError,
    ReadTimeout,
    TransportError,
    WriteError,
    WriteTimeout,
)
from ._types import Request, Response, AsyncResponse
from ._sync import StdlibTransport
from ._async import StdlibAsyncTransport

__all__ = [
    "Request",
    "Response",
    "AsyncResponse",
    "StdlibTransport",
    "StdlibAsyncTransport",
    "TransportError",
    "ConnectError",
    "ConnectTimeout",
    "ReadError",
    "ReadTimeout",
    "WriteError",
    "WriteTimeout",
    "ProtocolError",
]
