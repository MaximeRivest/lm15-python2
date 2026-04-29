"""
Transport-level exceptions.

These are distinct from lm15.errors (which are provider-level error taxonomy).
Transport errors get wrapped/translated by the LM layer into the
ProviderError hierarchy.
"""
from __future__ import annotations


class TransportError(Exception):
    """Base transport error — network or protocol failure below the LM layer."""


class ConnectError(TransportError):
    """Failed to establish a TCP (or TLS) connection."""


class ConnectTimeout(ConnectError):
    """Connection establishment timed out."""


class ReadError(TransportError):
    """Failed to read from an open socket (EOF, reset, etc.)."""


class ReadTimeout(ReadError):
    """Reading from the socket timed out."""


class WriteError(TransportError):
    """Failed to write to an open socket."""


class WriteTimeout(WriteError):
    """Writing to the socket timed out."""


class ProtocolError(TransportError):
    """The server sent something that isn't valid HTTP/1.1."""
