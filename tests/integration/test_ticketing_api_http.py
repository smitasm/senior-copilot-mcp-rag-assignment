"""HTTP-level tests for the ticketing simulator, plus one cross-service test
that runs the Alarm API and Ticketing API together - the actual shape of the
workflow the MCP orchestrator will drive later."""

import pytest
from fastapi.testclient import TestClient

from alarm_api.main import Settings as AlarmSettings
from alarm_api.main import create_app as create_alarm_app
from ticketing_api.main import Settings, create_app

TOKEN = "test-token"
AUTH = {"Authorization": f"Bearer {TOKEN}"}


@pytest.fixture(scope="module")
def client():
    return TestClient(create_app(Settings(api_token=TOKEN, enable_faults=True)))


def assert_error(resp, status, code):
    assert resp.status_code == status, resp.text
    body = resp.json()
    assert body["error"]["code"] == code and body["trace_id"]
    return body


# ---- health & auth (same contract as the Alarm API) ------------------------------
def test_health_needs_no_auth(client):
    assert client.get("/health").json()["status"] == "ok"


def test_protected_routes_require_bearer_token(client):
    assert_error(client.get("/tickets"), 401, "unauthorized")
    assert client.get("/tickets", headers={"Authorization": "Bearer wrong"}).status_code == 401


def test_trace_id_is_echoed(client):
    r = client.get("/tickets", headers={**AUTH, "trace_id": "t-123"})
    assert r.headers["trace_id"] == "t-123"


# ---- validation & errors -------------------------------------------------------------------
def test_unknown_ticket_is_404(client):
    assert_error(client.get("/tickets/INC9999999", headers=AUTH), 404, "not_found")


def test_bad_query_param_is_422(client):
    assert_error(client.get("/tickets", params={"page_size": 999}, headers=AUTH), 422, "validation_error")


def test_similar_tickets_with_no_signal_is_422(client):
    assert_error(client.post("/tickets/similar", json={}, headers=AUTH), 422, "validation_error")


def test_create_ticket_unknown_asset_is_404(client):
    body = {"short_description": "x", "description": "y", "priority": "P3", "asset_id": "AST-9999",
           "reported_by": "unit-test"}
    assert_error(client.post("/tickets", json=body, headers=AUTH), 404, "not_found")


# ---- create + duplicate prevention, over HTTP -------------------------------------------------
def test_create_then_get_round_trips(client):
    asset_id = client.get("/tickets", headers=AUTH, params={"alarm_name": "High Vibration", "page_size": 1}) \
        .json()["data"][0]["asset_id"]
    body = {"short_description": "New vibration alarm", "description": "Test description.",
           "priority": "P2", "asset_id": asset_id, "reported_by": "unit-test", "alarm_name": "Brand New Alarm Name"}
    created = client.post("/tickets", json=body, headers=AUTH)
    assert created.status_code == 201 and created.json()["duplicate_of"] is None
    ticket_id = created.json()["ticket"]["ticket_id"]
    fetched = client.get(f"/tickets/{ticket_id}", headers=AUTH)
    assert fetched.status_code == 200 and fetched.json()["ticket_id"] == ticket_id


def test_create_ticket_for_open_alarm_returns_existing_ticket_not_a_new_one(client):
    k202 = client.get("/tickets", headers=AUTH,
                      params={"alarm_name": "Low Lube Oil Pressure", "page_size": 1}).json()["data"][0]["asset_id"]
    body = {"short_description": "Lube oil low again", "description": "y", "priority": "P2",
           "asset_id": k202, "reported_by": "unit-test", "alarm_name": "Low Lube Oil Pressure"}
    r1 = client.post("/tickets", json=body, headers=AUTH).json()
    r2 = client.post("/tickets", json=body, headers=AUTH).json()
    assert r1["duplicate_of"] == r2["duplicate_of"] == r1["ticket"]["ticket_id"]


# ---- similar tickets, over HTTP ----------------------------------------------------------
def test_similar_endpoint_surfaces_matched_terms(client):
    r = client.post("/tickets/similar", json={"query": "compressor vibration bearing"}, headers=AUTH)
    assert r.status_code == 200
    top = r.json()["candidates"][0]
    assert top["matched_terms"] and 0 < top["score"] <= 1


# ---- fault injection (mirrors the Alarm API) ------------------------------------------------
def test_fault_injection_and_reset(client):
    h = {**AUTH, "trace_id": "flaky-ticket", "x-simulate-fault": "503:1"}
    codes = [client.get("/tickets", headers=h).status_code for _ in range(2)]
    assert codes == [503, 200]


# ---- cross-service: the actual Chain-09-style workflow, two simulators together -------------
def test_active_east_alarm_to_similar_tickets_end_to_end():
    """Reproduces the copilot's real path: find the top-priority active alarm
    on the Alarm API, then ask the Ticketing API for similar tickets, and
    confirm the historical evidence for the incident draft is really there."""
    alarm_client = TestClient(create_alarm_app(AlarmSettings(api_token=TOKEN)))
    ticket_client = TestClient(create_app(Settings(api_token=TOKEN)))

    top = alarm_client.get("/alarms", headers=AUTH, params={"site": "EastRefinery", "status": "active",
                                                             "sort_by": "severity"}).json()["data"][0]
    assert (top["asset_name"], top["alarm_name"]) == ("Recycle Gas Compressor K-201", "High Vibration")

    sim = ticket_client.post("/tickets/similar", headers=AUTH,
                             json={"asset_ids": [top["asset_id"]], "alarm_name": top["alarm_name"], "limit": 5})
    candidates = sim.json()["candidates"]
    # asset_ids/alarm_name are ranking signals, not a hard filter (a ticket on a
    # DIFFERENT asset can still be useful context) - but the grounding evidence
    # for THIS alarm's own asset must rank at the very top, with a strong score.
    assert candidates[0]["ticket"]["asset_id"] == top["asset_id"] and candidates[0]["score"] >= 0.9
    assert any("bearing" in c["ticket"]["resolution_notes"].lower() or
              "coupling" in c["ticket"]["resolution_notes"].lower() for c in candidates
              if c["ticket"]["asset_id"] == top["asset_id"])


def test_openapi_exposes_all_four_ticket_operations_plus_health(client):
    paths = client.get("/openapi.json").json()["paths"]
    ops = {(verb.upper(), p) for p, v in paths.items() for verb in v}
    assert len(ops) == 5 and ("GET", "/health") in ops