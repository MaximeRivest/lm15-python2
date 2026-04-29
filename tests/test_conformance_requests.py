from __future__ import annotations

from conformance.check_request_fixtures import compare_case, load_logical_cases


def test_python_provider_requests_match_curl_fixtures() -> None:
    results = [compare_case(case) for case in load_logical_cases()]
    failures = [result for result in results if result.status != "pass"]

    assert not failures, [
        {"id": result.case_id, "status": result.status, "reason": result.reason}
        for result in failures
    ]
