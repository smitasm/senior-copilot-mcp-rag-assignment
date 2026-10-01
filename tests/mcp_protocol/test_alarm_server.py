"""Alarm MCP server tests: real MCP protocol (ClientSession) on one side,
the REAL FastAPI simulator (via ASGI transport, in-process) on the other.
No mocking of either the MCP wire format or the HTTP API."""

import httpx
import pytest

from alarm_api.main import Settings, create_app
from mcp_common.http_client import SimulatorClient
from mcp_servers.alarm_server import build_alarm_server
from tests.mcp_protocol.helpers import call_tool, list_tools

TOKEN = "test-token"


@pytest.fixture
def server():
    transport = httpx.ASGITransport(app=create_app(Settings(api_token=TOKEN)))
    client = SimulatorClient(base_url="http://sim", token=TOKEN, transport=transport)
    return build_alarm_server(client)


# ---- discovery --------------------------------------------------------------
@pytest.mark.asyncio
async def test_all_fourteen_tools_are_registered_with_descriptions(server):
    tools = await list_tools(server)
    names = {t.name for t in tools}
    assert len(tools) == 14
    assert names == {
        "search_assets", "get_asset_metadata", "list_alarms", "get_alarm", "alarm_summary", "alarm_trends",
        "correlate_alarms", "flood_analysis", "rationalization_candidates", "priority_score",
        "operator_recommendations", "generate_calculation", "execute_calculation", "kpi_definitions",
    }
    assert all(t.description for t in tools)


# ---- happy paths, structured output ---------------------------------------------
@pytest.mark.asyncio
async def test_search_assets_returns_structured_results(server):
    res = await call_tool(server, "search_assets", {"req": {"query": "Boiler Feed Pump 101"}})
    assert not res.is_error
    assert res.structured_content["results"][0]["name"] == "Boiler Feed Pump 101"


@pytest.mark.asyncio
async def test_list_alarms_default_sort_is_not_priority(server):
    res = await call_tool(server, "list_alarms", {"req": {"site": "EastRefinery", "status": "active"}})
    assert not res.is_error
    assert res.structured_content["data"][0]["alarm_name"] != "High Vibration"


@pytest.mark.asyncio
async def test_priority_score_then_get_alarm_agree_on_the_top_east_alarm(server):
    listing = await call_tool(server, "list_alarms",
                              {"req": {"site": "EastRefinery", "status": "active", "sort_by": "severity"}})
    top = listing.structured_content["data"][0]
    scored = await call_tool(server, "priority_score", {"req": {"alarm_id": top["alarm_id"]}})
    assert scored.structured_content["band"] == "critical"
    fetched = await call_tool(server, "get_alarm", {"req": {"alarm_id": top["alarm_id"]}})
    assert fetched.structured_content["alarm_name"] == "High Vibration"


@pytest.mark.asyncio
async def test_operator_recommendations_with_full_context(server):
    listing = await call_tool(server, "list_alarms",
                              {"req": {"site": "EastRefinery", "status": "active", "sort_by": "severity"}})
    top = listing.structured_content["data"][0]
    rec = await call_tool(server, "operator_recommendations", {
        "req": {"alarm_id": top["alarm_id"], "include_related": True, "include_asset_context": True,
               "include_historical_pattern": True}})
    body = rec.structured_content
    assert body["actions"] and body["related_alarms"] and body["asset_context"]["asset_id"] == top["asset_id"]
    assert body["historical_pattern"]["occurrences_total"] >= 5


@pytest.mark.asyncio
async def test_generate_then_execute_calculation(server):
    gen = await call_tool(server, "generate_calculation", {
        "req": {"calculation_type": "critical_alarm_density",
               "filters": {"unit": "Unit 3", "start_time": "2026-05-01T00:00:00Z", "end_time": "2026-07-01T00:00:00Z"}}})
    cid = gen.structured_content["calculation_id"]
    ex = await call_tool(server, "execute_calculation", {"req": {"calculation_id": cid}})
    assert ex.structured_content["value"] > 0


@pytest.mark.asyncio
async def test_kpi_definitions_takes_no_arguments(server):
    res = await call_tool(server, "kpi_definitions", {})
    assert not res.is_error and res.structured_content["kpis"]


# ---- error mapping: ApiError -> ToolError, message preserved ------------------------------
@pytest.mark.asyncio
async def test_unknown_alarm_id_is_a_clean_tool_error_not_a_crash(server):
    res = await call_tool(server, "get_alarm", {"req": {"alarm_id": "ALM-NOPE"}})
    assert res.is_error
    text = res.content[0].text
    assert "not_found" in text and "ALM-NOPE" in text
    assert "Traceback" not in text  # no internal detail leaked


@pytest.mark.asyncio
async def test_unknown_asset_in_correlation_is_a_clean_tool_error(server):
    res = await call_tool(server, "correlate_alarms", {
        "req": {"asset_ids": ["AST-9999"],
               "time_range": {"start_time": "2026-05-01T00:00:00Z", "end_time": "2026-07-01T00:00:00Z"}}})
    assert res.is_error and "not_found" in res.content[0].text


# ---- schema rejects a malformed call before it ever reaches the simulator -------------------
@pytest.mark.asyncio
async def test_malformed_arguments_are_rejected_by_schema_validation(server):
    res = await call_tool(server, "search_assets", {"req": {"query": ""}})  # violates min_length=1
    assert res.is_error


@pytest.mark.asyncio
async def test_flood_analysis_end_to_end_yields_a_window(server):
    res = await call_tool(server, "flood_analysis", {
        "req": {"unit": "Unit 2",
               "time_range": {"start_time": "2026-05-01T00:00:00Z", "end_time": "2026-07-01T00:00:00Z"}}})
    assert res.structured_content["flood_windows"]