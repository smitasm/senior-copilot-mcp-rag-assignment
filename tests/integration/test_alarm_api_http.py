"""HTTP-level tests for the Alarm API simulator (FastAPI TestClient).

Covers what the service tests cannot: auth, trace headers, error envelope,
query/body validation, fault injection and the Postman chains over real HTTP."""

import json
import logging
import time

import pytest
from fastapi.testclient import TestClient

from alarm_api.main import Settings, access_log, create_app

TOKEN = "test-token"
AUTH = {"Authorization": f"Bearer {TOKEN}"}
WINDOW = {"start_time": "2026-05-01T00:00:00Z", "end_time": "2026-07-01T00:00:00Z"}


@pytest.fixture(scope="module")
def client():
    return TestClient(create_app(Settings(api_token=TOKEN, enable_faults=True)))


def assert_error(resp, status, code):
    assert resp.status_code == status, resp.text
    body = resp.json()
    assert set(body) >= {"error", "trace_id"} and body["error"]["code"] == code and body["error"]["message"]
    return body


# ---- health & auth ------------------------------------------------------------
def test_health_needs_no_auth(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


@pytest.mark.parametrize("headers", [{}, {"Authorization": "Basic abc"}, {"Authorization": "Bearer"},
                                     {"Authorization": "Bearer wrong-token"}])
def test_protected_routes_reject_missing_or_bad_tokens(client, headers):
    r = client.get("/assets/search", params={"query": "pump"}, headers=headers)
    assert_error(r, 401, "unauthorized")
    assert r.headers["www-authenticate"] == "Bearer"


def test_every_route_except_health_requires_auth(client):
    for path, verb in [("/alarms", "get"), ("/alarms/ALM-000001", "get"), ("/analytics/kpi-definitions", "get"),
                       ("/alarms/summary", "post"), ("/recommendations/operator-actions", "post")]:
        assert getattr(client, verb)(path).status_code == 401, path


# ---- trace propagation -------------------------------------------------------------
def test_trace_headers_are_echoed_on_success_and_on_errors(client):
    h = {**AUTH, "trace_id": "trace-abc", "x-client-id": "copilot", "x-metadata-tag": "unit-test"}
    ok = client.get("/analytics/kpi-definitions", headers=h)
    assert (ok.headers["trace_id"], ok.headers["x-client-id"], ok.headers["x-metadata-tag"]) == \
           ("trace-abc", "copilot", "unit-test")
    err = client.get("/alarms/ALM-NOPE", headers=h)
    assert err.headers["trace_id"] == "trace-abc" and err.json()["trace_id"] == "trace-abc"


def test_trace_id_is_generated_when_absent(client):
    r = client.get("/health")
    assert len(r.headers["trace_id"]) >= 32


def test_access_log_has_trace_and_never_the_token(client):
    lines: list[str] = []

    class Grab(logging.Handler):
        def emit(self, record):
            lines.append(record.getMessage())

    handler = Grab()
    access_log.addHandler(handler)
    try:
        client.get("/alarms", params={"page_size": 1}, headers={**AUTH, "trace_id": "trace-log-1"})
    finally:
        access_log.removeHandler(handler)
    entry = json.loads(lines[-1])
    assert entry["trace_id"] == "trace-log-1" and entry["status"] == 200 and entry["path"] == "/alarms"
    assert TOKEN not in "".join(lines)


# ---- error envelope ------------------------------------------------------------------------
def test_not_found_uses_the_same_envelope_everywhere(client):
    assert_error(client.get("/alarms/ALM-NOPE", headers=AUTH), 404, "not_found")
    assert_error(client.get("/assets/AST-9999/metadata", headers=AUTH), 404, "not_found")
    assert_error(client.get("/no/such/route", headers=AUTH), 404, "not_found")


def test_query_validation_is_422_with_details(client):
    body = assert_error(client.get("/alarms", params={"page_size": 500}, headers=AUTH), 422, "validation_error")
    assert "page_size" in body["error"]["details"][0]["loc"]
    assert_error(client.get("/alarms", params={"bogus": 1}, headers=AUTH), 422, "validation_error")
    assert_error(client.get("/assets/search", headers=AUTH), 422, "validation_error")  # query required


@pytest.mark.parametrize("path,payload", [
    ("/alarms/summary", {"time_range": {"start_time": WINDOW["end_time"], "end_time": WINDOW["start_time"]}}),
    ("/alarms/priority-score", {"alarm_id": "ALM-000001", "typo": 1}),
    ("/alarms/correlation", {"asset_ids": [], "time_range": WINDOW}),
    ("/alarms/trends", {"time_range": WINDOW, "bucket": "yearly"}),
])
def test_invalid_bodies_are_422(client, path, payload):
    assert_error(client.post(path, json=payload, headers=AUTH), 422, "validation_error")


def test_business_rule_violations_are_400_and_unknown_ids_404(client):
    assert_error(client.get("/assets/search", params={"query": "   "}, headers=AUTH), 400, "invalid_request")
    r = client.post("/alarms/correlation", json={"asset_ids": ["AST-9999"], "time_range": WINDOW}, headers=AUTH)
    assert_error(r, 404, "not_found")
    assert_error(client.post("/alarms/priority-score", json={"alarm_id": "ALM-NOPE"}, headers=AUTH), 404, "not_found")


# ---- the Postman chains, over HTTP ----------------------------------------------------------------
def test_chain01_asset_to_summary_to_rationalization(client):
    aid = client.get("/assets/search", params={"query": "Boiler Feed Pump 101", "limit": 5},
                     headers=AUTH).json()["results"][0]["asset_id"]
    s = client.post("/alarms/summary", headers={**AUTH, "trace_id": "t1", "x-client-id": "c", "x-metadata-tag": "x"},
                    json={"asset_ids": [aid], "time_range": WINDOW, "severity": ["high", "critical"],
                          "group_by": ["alarm_name"], "kpis": ["alarm_count", "recurring_rate", "avg_ack_delay"]})
    assert s.status_code == 200 and s.json()["groups"]
    r = client.post("/alarms/rationalization-candidates", headers=AUTH,
                    json={"asset_ids": [aid], "time_range": WINDOW, "recurrence_threshold": 5})
    assert r.status_code == 200 and r.json()["candidates"]


def test_chain02_flood_window_timestamps_chain_into_raw_query_strings(client):
    """Postman pastes window start/end straight into a URL. That only works if
    timestamps are UTC 'Z' (a '+00:00' suffix would turn '+' into a space)."""
    flood = client.post("/alarms/flood-analysis", headers=AUTH,
                        json={"unit": "Unit 2", "time_range": WINDOW, "threshold_count": 10,
                              "rolling_window_minutes": 10}).json()
    w = flood["flood_windows"][0]
    assert w["start"].endswith("Z") and w["end"].endswith("Z")
    r = client.get(f"/alarms?unit=Unit%202&start_time={w['start']}&end_time={w['end']}&page=1&page_size=200",
                   headers=AUTH)
    assert r.status_code == 200 and r.json()["pagination"]["total"] == w["alarm_count"]


def test_chain09_east_active_alarm_to_priority_to_recommendations(client):
    rows = client.get("/alarms", headers=AUTH, params={"site": "EastRefinery", "status": "active",
                                                       "sort_by": "severity", "sort_order": "desc"}).json()["data"]
    top = rows[0]
    assert (top["asset_name"], top["alarm_name"]) == ("Recycle Gas Compressor K-201", "High Vibration")
    assert client.get(f"/alarms/{top['alarm_id']}", headers=AUTH).json() == top
    score = client.post("/alarms/priority-score", json={"alarm_id": top["alarm_id"]}, headers=AUTH).json()
    assert score["band"] == "critical"
    rec = client.post("/recommendations/operator-actions", headers=AUTH,
                      json={"alarm_id": top["alarm_id"], "include_related": True, "include_asset_context": True,
                            "include_historical_pattern": True}).json()
    assert rec["actions"] and rec["related_alarms"] and rec["asset_context"]["asset_id"] == top["asset_id"]


def test_chain04_generate_then_execute_calculation(client):
    f = {"unit": "Unit 3", **WINDOW}
    gen = client.post("/calculation-code/generate", headers=AUTH,
                      json={"calculation_type": "critical_alarm_density", "filters": f}).json()
    res = client.post("/calculation-code/execute", headers=AUTH,
                      json={"calculation_id": gen["calculation_id"], "filters": f})
    assert res.status_code == 200 and res.json()["value"] > 0


# ---- fault injection (used later to test MCP retry/timeout) ------------------------------------------
def test_transient_fault_fails_n_times_then_succeeds_for_same_trace(client):
    h = {**AUTH, "trace_id": "trace-flaky", "x-simulate-fault": "503:2"}
    codes = [client.get("/analytics/kpi-definitions", headers=h).status_code for _ in range(4)]
    assert codes == [503, 503, 200, 200]
    other = client.get("/analytics/kpi-definitions", headers={**h, "trace_id": "another"})
    assert other.status_code == 503  # counter is per trace_id


def test_fault_response_is_a_proper_error_with_retry_after(client):
    r = client.get("/alarms", headers={**AUTH, "x-simulate-fault": "503"})
    assert_error(r, 503, "simulated_fault")
    assert r.headers["retry-after"] == "1"


def test_delay_injection_slows_the_response(client):
    t0 = time.perf_counter()
    client.get("/analytics/kpi-definitions", headers={**AUTH, "x-simulate-delay-ms": "150"})
    assert time.perf_counter() - t0 >= 0.15


def test_faults_are_ignored_unless_explicitly_enabled():
    safe = TestClient(create_app(Settings(api_token=TOKEN, enable_faults=False)))
    r = safe.get("/analytics/kpi-definitions", headers={**AUTH, "x-simulate-fault": "500"})
    assert r.status_code == 200


# ---- contract surface ------------------------------------------------------------------------------------
def test_openapi_exposes_all_fifteen_operations(client):
    paths = client.get("/openapi.json").json()["paths"]
    ops = {(verb.upper(), p) for p, v in paths.items() for verb in v}
    assert len(ops) == 15
    assert ("GET", "/assets/search") in ops and ("POST", "/recommendations/operator-actions") in ops