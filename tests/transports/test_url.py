"""Unit tests for the minimal URL parser."""
from __future__ import annotations

import pytest

from lm15.transports._url import ParsedURL, parse_url


class TestParseURL:
    def test_http_default_port(self) -> None:
        u = parse_url("http://example.com/foo")
        assert u == ParsedURL(scheme="http", host="example.com", port=80, target="/foo", is_tls=False)

    def test_https_default_port(self) -> None:
        u = parse_url("https://example.com/")
        assert u == ParsedURL(scheme="https", host="example.com", port=443, target="/", is_tls=True)

    def test_explicit_port(self) -> None:
        u = parse_url("https://api.openai.com:8443/v1/responses")
        assert u.port == 8443
        assert u.target == "/v1/responses"

    def test_no_path_defaults_to_slash(self) -> None:
        u = parse_url("http://example.com")
        assert u.target == "/"

    def test_query_string_preserved(self) -> None:
        u = parse_url("http://example.com/x?a=1&b=2")
        assert u.target == "/x?a=1&b=2"

    def test_empty_query_preserved(self) -> None:
        u = parse_url("http://example.com/x?")
        assert u.target == "/x?"

    def test_ipv4_host(self) -> None:
        u = parse_url("http://127.0.0.1:18080/")
        assert u.host == "127.0.0.1"
        assert u.port == 18080

    def test_ipv6_host(self) -> None:
        u = parse_url("http://[::1]:8080/foo")
        assert u.host == "::1"
        assert u.port == 8080
        assert u.target == "/foo"

    def test_ipv6_default_port(self) -> None:
        u = parse_url("https://[2001:db8::1]/")
        assert u.host == "2001:db8::1"
        assert u.port == 443

    def test_unsupported_scheme(self) -> None:
        with pytest.raises(ValueError, match="scheme"):
            parse_url("ftp://example.com/x")

    def test_missing_scheme(self) -> None:
        with pytest.raises(ValueError):
            parse_url("example.com/x")

    def test_host_authority_only(self) -> None:
        u = parse_url("https://example.com:443")
        assert u.target == "/"

    def test_fragment_stripped(self) -> None:
        # Fragments are not sent to servers; strip them.
        u = parse_url("http://example.com/x#frag")
        assert u.target == "/x"

    def test_host_hostname_property(self) -> None:
        """For SNI we need the hostname unwrapped (without brackets)."""
        u = parse_url("http://[::1]:8080/")
        assert u.host == "::1"

    def test_origin_key(self) -> None:
        """Connections are pooled by (scheme, host, port)."""
        u1 = parse_url("https://api.openai.com/v1/a")
        u2 = parse_url("https://api.openai.com/v1/b")
        assert u1.origin() == u2.origin()
        u3 = parse_url("https://api.openai.com:8443/v1/a")
        assert u1.origin() != u3.origin()
