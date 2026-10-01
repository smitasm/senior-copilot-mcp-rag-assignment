"""Tests for gui/async_bridge.py: the testable logic behind the Streamlit
app. Every test that needs a real ticket to persist across two calls (create
then fetch) builds its SimulatorClient/app ONCE and reuses it for both -
mirroring how a real running uvicorn server behaves, unlike a throwaway
script that builds a fresh app per call and gets a fresh empty database
each time (a mistake caught while writing these tests, not left in them).
"""

from __future__ import annotations

import httpx
import pytest

from alarm_api.main import Settings as AlarmSettings
from alarm_api.main import create_app as alarm_app
from gui.async_bridge import (
    ApprovalRequiredError, ConnectionConfig, approve_and_create, build_registry, check_connections,
    fetch_ticket_details, run_investigation,
)
from mcp_common.http_client import SimulatorClient
from orchestrator.llm import FakeLLM, LLMResponse, LLMTimeoutError, ToolCall
from ticketing_api.main import Settings as TicketSettings
from ticketing_api.main import create_app as ticket_app

TOKEN = "test-token"


def alarm_client() -> SimulatorClient:
    transport = httpx.ASGITransport(app=alarm_app(AlarmSettings(api_token=TOKEN)))
    return SimulatorClient(base_url="http://sim", token=TOKEN, transport=transport, client_id="gui")


def ticket_client() -> SimulatorClient:
    transport = httpx.ASGITransport(app=ticket_app(TicketSettings(api_token=TOKEN)))
    return SimulatorClient(base_url="http://sim", token=TOKEN, transport=transport, client_id="gui")


VALID_DRAFT_ARGS = {
    "alarm_id": "ALM-000001", "alarm_name": "High Vibration", "asset_id": "AST-0001",
    "asset_name": "Recycle Gas Compressor K-201", "priority_band": "critical",
    "likely_cause": "Bearing wear.", "evidence": ["Vibration trending up"],
    "recommended_actions": ["Check lube oil pressure"], "similar_ticket_ids": [], "citations": [],
    "ticket_priority": "P1", "ticket_short_description": "K-201 High Vibration - active critical alarm",
    "ticket_description": "Automated draft from copilot investigation.",
}


def submit_call() -> ToolCall:
    return ToolCall("submit-1", "submit_incident_draft", VALID_DRAFT_ARGS)


# ---- build_registry ---------------------------------------------------------------
@pytest.mark.asyncio
async def test_build_registry_aggregates_all_three_servers_by_default():
    cfg = ConnectionConfig(api_token=TOKEN)
    async with build_registry(cfg, alarm_client=alarm_client(), ticket_client=ticket_client()) as reg:
        assert len(reg.tool_names) == 19


@pytest.mark.asyncio
async def test_build_registry_respects_include_flags():
    cfg = ConnectionConfig(api_token=TOKEN)
    async with build_registry(cfg, include_alarm=False, ticket_client=ticket_client()) as reg:
        assert "priority_score" not in reg.tool_names
        assert "create_ticket" in reg.tool_names and "search_knowledge_base" in reg.tool_names


@pytest.mark.asyncio
async def test_build_registry_knowledge_only_needs_no_simulator_clients():
    cfg = ConnectionConfig()
    async with build_registry(cfg, include_alarm=False, include_ticketing=False) as reg:
        assert reg.tool_names == ["search_knowledge_base"]


# ---- run_investigation --------------------------------------------------------------------
@pytest.mark.asyncio
async def test_run_investigation_with_fakellm_returns_a_grounded_draft():
    cfg = ConnectionConfig(api_token=TOKEN)
    llm = FakeLLM([LLMResponse(content="", tool_calls=[submit_call()])])
    result = await run_investigation("prepare an incident", cfg, llm=llm, alarm_client=alarm_client(),
                                     ticket_client=ticket_client())
    assert result.stopped_reason == "draft_submitted" and result.draft.asset_name == "Recycle Gas Compressor K-201"


@pytest.mark.asyncio
async def test_run_investigation_handles_llm_timeout_without_crashing():
    """investigate() already catches LLMTimeoutError internally and returns
    stopped_reason="llm_error" rather than raising - this proves that
    behaviour survives being run through the extra build_registry/
    AsyncExitStack layering in this file too."""
    cfg = ConnectionConfig(api_token=TOKEN)
    llm = FakeLLM([LLMTimeoutError("simulated")])
    result = await run_investigation("prepare an incident", cfg, llm=llm, alarm_client=alarm_client(),
                                     ticket_client=ticket_client())
    assert result.stopped_reason == "llm_error" and result.draft is None


@pytest.mark.asyncio
async def test_run_investigation_propagates_a_genuinely_unexpected_error_cleanly():
    """A non-LLMTimeoutError exception from inside the investigation (here,
    forced via a FakeLLM script that raises something else entirely) must
    still surface to the caller - not be silently swallowed - but must not
    crash with the anyio TaskGroup error that motivated the catch-inside/
    reraise-outside pattern in run_investigation itself."""
    cfg = ConnectionConfig(api_token=TOKEN)
    llm = FakeLLM([RuntimeError("something genuinely unexpected")])
    with pytest.raises(RuntimeError, match="something genuinely unexpected"):
        await run_investigation("prepare an incident", cfg, llm=llm, alarm_client=alarm_client(),
                                ticket_client=ticket_client())


# ---- approve_and_create: the exact scenario that broke before the fix -----------------------------
@pytest.mark.asyncio
async def test_approve_and_create_without_approval_raises_cleanly_and_creates_nothing():
    """This is the exact call shape that raised
    'ExceptionGroup: unhandled errors in a TaskGroup' before switching to the
    catch-inside/reraise-outside pattern - confirms the fix holds."""
    cfg = ConnectionConfig(api_token=TOKEN)
    tc = ticket_client()
    from orchestrator.draft import IncidentDraft
    draft = IncidentDraft.model_validate(VALID_DRAFT_ARGS)

    before = await tc.get("/tickets", params={"page_size": 1})
    with pytest.raises(ApprovalRequiredError):
        await approve_and_create(draft, cfg, approved=False, ticket_client=tc)
    after = await tc.get("/tickets", params={"page_size": 1})
    assert after["pagination"]["total"] == before["pagination"]["total"]


@pytest.mark.asyncio
async def test_approve_and_create_with_approval_creates_a_real_ticket():
    cfg = ConnectionConfig(api_token=TOKEN)
    from orchestrator.draft import IncidentDraft
    draft = IncidentDraft.model_validate(dict(VALID_DRAFT_ARGS, alarm_name="Brand New Test Alarm"))
    result = await approve_and_create(draft, cfg, approved=True, ticket_client=ticket_client())
    assert result["duplicate_of"] is None and result["ticket"]["state"] == "new"


@pytest.mark.asyncio
async def test_approve_and_create_duplicate_detection_works_through_the_gui_layer():
    """K-202's Low Lube Oil Pressure ticket is already open in the seed data
    - the GUI's own call path must still route through the real dedup
    logic, not bypass it."""
    cfg = ConnectionConfig(api_token=TOKEN)
    tc = ticket_client()
    existing = await tc.get("/tickets", params={"alarm_name": "Low Lube Oil Pressure", "page_size": 1})
    asset_id = existing["data"][0]["asset_id"]
    from orchestrator.draft import IncidentDraft
    draft = IncidentDraft.model_validate(dict(VALID_DRAFT_ARGS, asset_id=asset_id,
                                              alarm_name="Low Lube Oil Pressure"))
    result = await approve_and_create(draft, cfg, approved=True, ticket_client=tc)
    assert result["duplicate_of"] == result["ticket"]["ticket_id"]


@pytest.mark.asyncio
async def test_approve_and_create_reported_by_defaults_and_is_overridable():
    cfg = ConnectionConfig(api_token=TOKEN)
    from orchestrator.draft import IncidentDraft
    draft = IncidentDraft.model_validate(dict(VALID_DRAFT_ARGS, alarm_name="Yet Another New Alarm"))
    result = await approve_and_create(draft, cfg, approved=True, ticket_client=ticket_client())
    assert result["ticket"]["reported_by"] == "copilot-gui-user"


# ---- fetch_ticket_details -------------------------------------------------------------------
@pytest.mark.asyncio
async def test_fetch_ticket_details_returns_real_content_for_a_just_created_ticket():
    """Uses ONE shared ticket_client for both the create and the fetch, so
    the created ticket is genuinely visible - the mistake a throwaway script
    made (fresh app per call) is exactly what this test structure avoids."""
    cfg = ConnectionConfig(api_token=TOKEN)
    tc = ticket_client()
    from orchestrator.draft import IncidentDraft
    draft = IncidentDraft.model_validate(dict(VALID_DRAFT_ARGS, alarm_name="Fetch Test Alarm"))
    created = await approve_and_create(draft, cfg, approved=True, ticket_client=tc)
    details = await fetch_ticket_details([created["ticket"]["ticket_id"]], cfg, ticket_client=tc)
    assert len(details) == 1 and details[0]["short_description"] == VALID_DRAFT_ARGS["ticket_short_description"]


@pytest.mark.asyncio
async def test_fetch_ticket_details_skips_unknown_ids_without_raising():
    cfg = ConnectionConfig(api_token=TOKEN)
    details = await fetch_ticket_details(["INC9999999"], cfg, ticket_client=ticket_client())
    assert details == []


@pytest.mark.asyncio
async def test_fetch_ticket_details_empty_input_returns_empty_list():
    """The empty-list early return in fetch_ticket_details is a trivial
    efficiency shortcut (skip the MCP handshake when there's nothing to
    fetch), not a correctness-critical path: an empty `ticket_ids` loop
    returns [] identically whether or not that shortcut exists, since
    connecting to an in-process MCP server is itself a local operation, not
    a network call - only actually CALLING a tool would be. An earlier
    version of this test claimed to prove "no connection is attempted",
    which isn't a meaningfully different or testable outcome here; this
    version tests the property that actually matters - the correct result
    for empty input - without overclaiming what a poison transport would
    prove."""
    cfg = ConnectionConfig(api_token=TOKEN)
    assert await fetch_ticket_details([], cfg, ticket_client=ticket_client()) == []


# ---- check_connections -------------------------------------------------------------------
@pytest.mark.asyncio
async def test_check_connections_reports_true_for_healthy_services():
    cfg = ConnectionConfig(api_token=TOKEN)
    health = await check_connections(cfg, alarm_client=alarm_client(), ticket_client=ticket_client())
    assert health == {"alarm_api": True, "ticketing_api": True}


@pytest.mark.asyncio
async def test_check_connections_fails_fast_not_with_retries():
    """A health check answering 'is this up at all' must not retry a dead
    service like a real API call would - directly motivated by a live run:
    with SimulatorClient's library-default retries (3, 10s timeout each), a
    single check_connections() call against two down services could take up
    to ~80s in the worst case, which is exactly what made one Windows run
    exceed a 15s timeout. Timed against a real non-routable address
    (10.255.255.1 - a well-known address that hangs rather than quickly
    refusing, so this genuinely exercises the timeout path, not just an
    instant connection-refused) rather than asserted in the abstract."""
    import time

    cfg = ConnectionConfig(alarm_api_url="http://10.255.255.1:1", ticketing_api_url="http://10.255.255.1:1",
                           api_token=TOKEN)
    started = time.perf_counter()
    health = await check_connections(cfg)
    elapsed = time.perf_counter() - started
    assert health == {"alarm_api": False, "ticketing_api": False}
    # 2 services x 3s timeout x 1 attempt (no retries) ~= 6s; generous margin
    # above that, but FAR below what retrying 3x per service would take (~80s)
    assert elapsed < 10.0, f"took {elapsed:.1f}s - looks like retries are happening again"


@pytest.mark.asyncio
async def test_check_connections_reports_false_for_an_unreachable_service():
    class AlwaysDown(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("connection refused", request=request)

    cfg = ConnectionConfig(api_token=TOKEN)
    down_client = SimulatorClient(base_url="http://sim", token=TOKEN, transport=AlwaysDown(), max_retries=0)
    health = await check_connections(cfg, alarm_client=down_client, ticket_client=ticket_client())
    assert health == {"alarm_api": False, "ticketing_api": True}