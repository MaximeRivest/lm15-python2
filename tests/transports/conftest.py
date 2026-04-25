"""Shared fixtures for transport tests.

Provides an in-process HTTP/1.1 server that can simulate:
    - simple Content-Length responses
    - chunked transfer encoding
    - SSE (Server-Sent Events)
    - slow-drip responses (for timeout/cancellation tests)
    - server-initiated disconnect (stale keepalive test)
    - TLS via a self-signed cert
"""
from __future__ import annotations

import socket
import ssl as _ssl
import threading
import time
import os
import tempfile
import subprocess
from dataclasses import dataclass, field
from typing import Callable

import pytest


# ─── request model as the server sees it ─────────────────────────────


@dataclass
class ReceivedRequest:
    method: str
    target: str
    headers: list[tuple[str, str]]
    body: bytes

    def header(self, name: str) -> str | None:
        lname = name.lower()
        for k, v in self.headers:
            if k.lower() == lname:
                return v
        return None


# ─── the test server ─────────────────────────────────────────────────


@dataclass
class ServerContext:
    """Knobs the test can tweak mid-flight."""
    handler: Callable[[ReceivedRequest, socket.socket], None] | None = None
    requests: list[ReceivedRequest] = field(default_factory=list)
    request_count: int = 0


class _TestServer:
    def __init__(self, use_tls: bool = False) -> None:
        self.use_tls = use_tls
        self.ctx = ServerContext()
        self._sock: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self.host = "127.0.0.1"
        self.port = 0
        # TLS bits
        self._cert_file: str | None = None
        self._key_file: str | None = None
        self._ssl_ctx: _ssl.SSLContext | None = None

    def start(self) -> None:
        if self.use_tls:
            self._provision_tls()

        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((self.host, 0))
        self._sock.listen(128)
        self.port = self._sock.getsockname()[1]
        self._sock.settimeout(0.2)

        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        try:
            if self._sock:
                self._sock.close()
        except Exception:
            pass
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._cert_file and os.path.exists(self._cert_file):
            os.unlink(self._cert_file)
        if self._key_file and os.path.exists(self._key_file):
            os.unlink(self._key_file)

    def base_url(self) -> str:
        scheme = "https" if self.use_tls else "http"
        return f"{scheme}://{self.host}:{self.port}"

    def _provision_tls(self) -> None:
        tmp = tempfile.mkdtemp()
        self._cert_file = os.path.join(tmp, "cert.pem")
        self._key_file = os.path.join(tmp, "key.pem")
        # Generate a self-signed cert
        subprocess.run(
            [
                "openssl", "req", "-x509", "-newkey", "rsa:2048",
                "-keyout", self._key_file, "-out", self._cert_file,
                "-days", "1", "-nodes", "-subj",
                f"/CN={self.host}",
                "-addext", f"subjectAltName=IP:{self.host}",
            ],
            check=True, capture_output=True,
        )
        self._ssl_ctx = _ssl.SSLContext(_ssl.PROTOCOL_TLS_SERVER)
        self._ssl_ctx.load_cert_chain(self._cert_file, self._key_file)

    def ca_bundle_path(self) -> str:
        assert self._cert_file is not None
        return self._cert_file

    def _serve(self) -> None:
        while not self._stop.is_set():
            try:
                assert self._sock is not None
                client, _addr = self._sock.accept()
            except (socket.timeout, TimeoutError):
                continue
            except OSError:
                return

            if self.use_tls:
                assert self._ssl_ctx is not None
                try:
                    client = self._ssl_ctx.wrap_socket(client, server_side=True)
                except Exception:
                    client.close()
                    continue

            threading.Thread(
                target=self._handle, args=(client,), daemon=True
            ).start()

    def _handle(self, client: socket.socket) -> None:
        """Handle a single keepalive connection: may process multiple requests."""
        try:
            client.settimeout(5.0)
            buf = b""
            while not self._stop.is_set():
                # Read headers
                while b"\r\n\r\n" not in buf:
                    try:
                        chunk = client.recv(4096)
                    except (socket.timeout, TimeoutError):
                        return
                    except OSError:
                        return
                    if not chunk:
                        return
                    buf += chunk

                head, _, rest = buf.partition(b"\r\n\r\n")
                lines = head.split(b"\r\n")
                request_line = lines[0].decode("iso-8859-1")
                method, target, _version = request_line.split(" ", 2)

                headers: list[tuple[str, str]] = []
                for line in lines[1:]:
                    if not line:
                        continue
                    k, _, v = line.decode("iso-8859-1").partition(":")
                    headers.append((k.strip(), v.strip()))

                # Read body based on Content-Length (simple case for tests)
                content_length = 0
                for k, v in headers:
                    if k.lower() == "content-length":
                        content_length = int(v)
                        break

                while len(rest) < content_length:
                    try:
                        chunk = client.recv(4096)
                    except OSError:
                        return
                    if not chunk:
                        return
                    rest += chunk
                body = rest[:content_length]
                buf = rest[content_length:]

                req = ReceivedRequest(method=method, target=target, headers=headers, body=body)
                self.ctx.request_count += 1
                self.ctx.requests.append(req)

                handler = self.ctx.handler or _default_handler
                try:
                    handler(req, client)
                except (BrokenPipeError, ConnectionResetError, OSError):
                    return

                # If the handler closed the client socket, stop.
                try:
                    client.fileno()
                except (OSError, ValueError):
                    return

                # Close connection if the handler asked to, or client signaled
                conn_hdr = req.header("connection") or ""
                if conn_hdr.lower() == "close":
                    return
        finally:
            try:
                client.close()
            except Exception:
                pass


def _default_handler(req: ReceivedRequest, client: socket.socket) -> None:
    body = b"ok"
    response = (
        b"HTTP/1.1 200 OK\r\n"
        b"Content-Type: text/plain\r\n"
        b"Content-Length: " + str(len(body)).encode("ascii") + b"\r\n"
        b"\r\n" + body
    )
    client.sendall(response)


# ─── pytest fixtures ─────────────────────────────────────────────────


@pytest.fixture
def server():
    s = _TestServer(use_tls=False)
    s.start()
    try:
        yield s
    finally:
        s.stop()


@pytest.fixture
def tls_server():
    s = _TestServer(use_tls=True)
    s.start()
    try:
        yield s
    finally:
        s.stop()


# ─── helpers for handlers ────────────────────────────────────────────


def reply_bytes(client: socket.socket, status: int, body: bytes,
                headers: list[tuple[str, str]] | None = None) -> None:
    reason = {200: "OK", 400: "Bad Request", 401: "Unauthorized",
              429: "Too Many Requests", 500: "Internal Server Error"}.get(status, "OK")
    out = f"HTTP/1.1 {status} {reason}\r\n".encode("ascii")
    hdrs = list(headers or [])
    has_len = any(k.lower() == "content-length" for k, _ in hdrs)
    has_te = any(k.lower() == "transfer-encoding" for k, _ in hdrs)
    if not has_len and not has_te:
        hdrs.append(("Content-Length", str(len(body))))
    for k, v in hdrs:
        out += f"{k}: {v}\r\n".encode("ascii")
    out += b"\r\n" + body
    client.sendall(out)


def reply_chunked(client: socket.socket, chunks: list[bytes],
                  headers: list[tuple[str, str]] | None = None,
                  chunk_delay: float = 0.0) -> None:
    out = b"HTTP/1.1 200 OK\r\n"
    hdrs = list(headers or [])
    hdrs.append(("Transfer-Encoding", "chunked"))
    for k, v in hdrs:
        out += f"{k}: {v}\r\n".encode("ascii")
    out += b"\r\n"
    client.sendall(out)
    for chunk in chunks:
        client.sendall(f"{len(chunk):X}\r\n".encode("ascii") + chunk + b"\r\n")
        if chunk_delay:
            time.sleep(chunk_delay)
    client.sendall(b"0\r\n\r\n")
