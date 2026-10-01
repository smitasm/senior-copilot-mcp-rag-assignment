"""Ticketing MCP server tests. The centerpiece is create_ticket's approval
gate: it must be enforced by the tool itself, not merely suggested in its
description - these tests prove the ticketing API is never even called
when approved is missing or false."""

import httpx
import pytest

from mcp_common.http_client import SimulatorClient
from mcp_servers.ticketing_server import build_ticketing_server
from ticketing_api.main import Settings, create_app
from tests.mcp_protocol.helpers import call_tool, list_tools

TOKEN = "test-token"


@pytest.fixture
def server():
    transport = httpx.ASGITransport(app=create_app(Settings(api_token=TOKEN)))
    client = SimulatorClient(base_url="http://sim", token=TOKEN, transport=transport)
    return build_ticketing_server(client)


@pytest.mark.asyncio
async def test_all_four_tools_are_registered(server):
    names = {t.name for t in await list_tools(server)}
    assert names == {"list_tickets", "get_ticket", "find_similar_tickets", "create_ticket"}


@pytest.mark.asyncio
async def test_list_and_get_ticket_round_trip(server):
    listing = await call_tool(server, "list_tickets", {"req": {"page_size": 1}})
    ticket_id = listing.structured_content["data"][0]["ticket_id"]
    fetched = await call_tool(server, "get_ticket", {"req": {"ticket_id": ticket_id}})
    assert fetched.structured_content["ticket_id"] == ticket_id


@pytest.mark.asyncio
async def test_find_similar_tickets_surfaces_k201_history(server):
    res = await call_tool(server, "find_similar_tickets",
                          {"req": {"query": "compressor vibration bearing"}})
    assert res.structured_content["candidates"]


# ---- the approval gate itself ------------------------------------------------------
@pytest.mark.asyncio
async def test_create_ticket_without_approved_field_is_refused(server):
    res = await call_tool(server, "create_ticket", {"req": {"ticket": {
        "short_description": "x", "description": "y", "priority": "P3",
        "asset_id": "AST-0001", "reported_by": "unit-test"}}})
    assert res.is_error and "approv" in res.content[0].text.lower()


@pytest.mark.asyncio
async def test_create_ticket_with_approved_false_is_refused(server):
    res = await call_tool(server, "create_ticket", {"req": {
        "approved": False,
        "ticket": {"short_description": "x", "description": "y", "priority": "P3",
                  "asset_id": "AST-0001", "reported_by": "unit-test"}}})
    assert res.is_error and "approv" in res.content[0].text.lower()


@pytest.mark.asyncio
async def test_refused_call_never_reaches_the_ticketing_api(server):
    """The refusal must happen before any HTTP call - proven by counting how
    many tickets exist before and after a refused attempt."""
    before = await call_tool(server, "list_tickets", {"req": {"page_size": 1}})
    total_before = before.structured_content["pagination"]["total"]
    await call_tool(server, "create_ticket", {"req": {"ticket": {
        "short_description": "should not be created", "description": "y", "priority": "P3",
        "asset_id": "AST-0001", "reported_by": "unit-test"}}})
    after = await call_tool(server, "list_tickets", {"req": {"page_size": 1}})
    assert after.structured_content["pagination"]["total"] == total_before


@pytest.mark.asyncio
async def test_create_ticket_with_approved_true_succeeds(server):
    res = await call_tool(server, "create_ticket", {"req": {
        "approved": True,
        "ticket": {"short_description": "New test ticket", "description": "y", "priority": "P2",
                  "asset_id": "AST-0001", "reported_by": "unit-test", "alarm_name": "Brand New Test Alarm"}}})
    assert not res.is_error
    body = res.structured_content
    assert body["duplicate_of"] is None and body["ticket"]["state"] == "new"


@pytest.mark.asyncio
async def test_create_ticket_for_an_already_open_alarm_returns_the_existing_ticket(server):
    """K-202's Low Lube Oil Pressure ticket is already open in the seed data;
    approval alone must not create a duplicate."""
    k202 = await call_tool(server, "list_tickets", {"req": {"alarm_name": "Low Lube Oil Pressure", "page_size": 1}})
    asset_id = k202.structured_content["data"][0]["asset_id"]
    res = await call_tool(server, "create_ticket", {"req": {
        "approved": True,
        "ticket": {"short_description": "Lube oil low again", "description": "y", "priority": "P2",
                  "asset_id": asset_id, "reported_by": "unit-test", "alarm_name": "Low Lube Oil Pressure"}}})
    body = res.structured_content
    assert body["duplicate_of"] == body["ticket"]["ticket_id"]


@pytest.mark.asyncio
async def test_create_ticket_unknown_asset_is_a_clean_tool_error(server):
    res = await call_tool(server, "create_ticket", {"req": {
        "approved": True,
        "ticket": {"short_description": "x", "description": "y", "priority": "P3",
                  "asset_id": "AST-9999", "reported_by": "unit-test"}}})
    assert res.is_error and "not_found" in res.content[0].text