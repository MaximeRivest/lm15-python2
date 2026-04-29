# lm15-python2 conformance

This directory is the active foundation for lm15 compatibility work.

For now, **lm15-python2 is the reference implementation**. Other language ports
are frozen until this conformance suite is stable. Future ports should be built
against these fixtures and only considered compatible when they pass the same
checks.

## What lives here

```text
conformance/
├── provider_requests/      # live-tested provider curl fixtures
│   ├── cases/              # expected provider HTTP requests
│   ├── features.yaml       # provider feature inventory and lm15/provider scope
│   └── results/            # saved live-test summaries and response bodies
├── cross_sdk/              # canonical logical cases + SDK dump adapters
│   ├── test_cases.json     # logical lm15 requests generated from fixtures
│   └── dump_request.py     # lm15-python2 logical case -> provider HTTP request
├── check_request_fixtures.py
└── reports/                # generated local reports, ignored by git
```

## Primary check

```bash
python3 conformance/check_request_fixtures.py
```

This compares the provider HTTP request built by lm15-python2 against the
corresponding curl fixture.

Useful options:

```bash
python3 conformance/check_request_fixtures.py --strict
python3 conformance/check_request_fixtures.py --case openai.basic_text
python3 conformance/check_request_fixtures.py --json conformance/reports/request-fixtures.json
```

## Fixture model

- `provider_requests/cases/**.json` are the provider wire truth: method, URL,
  headers, and request body known to work against the real API.
- `cross_sdk/test_cases.json` contains canonical logical lm15 inputs derived
  from those provider fixtures.
- `cross_sdk/dump_request.py` is the reference adapter that turns logical input
  into a normalized provider HTTP request without sending it.

## Future ports

Future Go/Rust/TypeScript/Julia ports should implement a tiny dump command with
this contract:

```bash
<port-dump-command> '<logical-case-json>'
```

It must emit normalized JSON with this shape:

```json
{
  "method": "POST",
  "url": "https://.../path",
  "params": {},
  "headers": {"content-type": "application/json"},
  "body": {}
}
```

The conformance runner can then compare each port against the same provider
fixtures and against lm15-python2.

## Protobuf

The protobuf schema in `../proto/` remains an optional machine-readable wire
format. Canonical JSON fixtures are the primary conformance mechanism while the
API is still evolving.
