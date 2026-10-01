"""ToolRegistry tests, against real MCP servers (alarm + ticketing) connected
in-process - not mocks. The auto-wrap shim is the part most worth proving:
a weak local model WILL sometimes pass flat arguments for a req-wrapped tool
(confirmed directly in the Phase-5 smoke test), and a call that could have
succeeded must not fail just because of that shape mismatch.
"""

import httpx
import pytest

from alarm_api.main import Settings as AlarmSettings
from alarm_api.main import create_app as alarm_app
from mcp_common.http_client import SimulatorClient
from mcp_servers.alarm_server import build_alarm_server
from mcp_servers.knowledge_server import build_knowledge_server
from mcp_servers.ticketing_server import build_ticketing_server
from orchestrator.mcp_client import ToolCallError, ToolRegistry, connect_inprocess
from ticketing_api.main import Settings as TicketSettings
from ticketing_api.main import create_app as ticket_app

TOKEN = "test-token"


async def _alarm_registry():
    transport = httpx.ASGITransport(app=alarm_app(AlarmSettings(api_token=TOKEN)))
    client = SimulatorClient(base_url="http://sim", token=TOKEN, transport=transport)
    return build_alarm_server(client)


async def _ticket_registry():
    transport = httpx.ASGITransport(app=ticket_app(TicketSettings(api_token=TOKEN)))
    client = SimulatorClient(base_url="http://sim", token=TOKEN, transport=transport)
    return build_ticketing_server(client)


@pytest.mark.asyncio
async def test_add_aggregates_tools_from_multiple_servers_into_one_registry():
    alarm_srv, ticket_srv, know_srv = await _alarm_registry(), await _ticket_registry(), build_knowledge_server()
    async with connect_inprocess(alarm_srv) as s1, connect_inprocess(ticket_srv) as s2, connect_inprocess(know_srv) as s3:
        reg = ToolRegistry()
        await reg.add(s1)
        await reg.add(s2)
        await reg.add(s3)
        assert len(reg.tool_names) == 19  # 14 alarm + 4 ticketing + 1 knowledge
        assert "priority_score" in reg.tool_names and "create_ticket" in reg.tool_names
        assert "search_knowledge_base" in reg.tool_names


@pytest.mark.asyncio
async def test_llm_tools_returns_openai_style_function_specs():
    alarm_srv = await _alarm_registry()
    async with connect_inprocess(alarm_srv) as session:
        reg = ToolRegistry()
        await reg.add(session)
        specs = reg.llm_tools()
        assert all(s["type"] == "function" and "name" in s["function"] and "parameters" in s["function"]
                  for s in specs)


@pytest.mark.asyncio
async def test_duplicate_tool_name_across_servers_is_rejected():
    alarm_srv = await _alarm_registry()
    async with connect_inprocess(alarm_srv) as s1, connect_inprocess(alarm_srv) as s2:
        reg = ToolRegistry()
        await reg.add(s1)
        with pytest.raises(ValueError, match="duplicate tool name"):
            await reg.add(s2)


# ---- the auto-wrap shim -----------------------------------------------------------------------
@pytest.mark.asyncio
async def test_call_auto_wraps_flat_arguments_for_a_req_shaped_tool():
    alarm_srv = await _alarm_registry()
    async with connect_inprocess(alarm_srv) as session:
        reg = ToolRegistry()
        await reg.add(session)
        # deliberately FLAT: {"alarm_id": ...} instead of {"req": {"alarm_id": ...}}
        result = await reg.call("priority_score", {"alarm_id": "ALM-000001"})
        assert 0 <= result["score"] <= 100


@pytest.mark.asyncio
async def test_call_still_works_when_arguments_are_already_correctly_wrapped():
    alarm_srv = await _alarm_registry()
    async with connect_inprocess(alarm_srv) as session:
        reg = ToolRegistry()
        await reg.add(session)
        result = await reg.call("priority_score", {"req": {"alarm_id": "ALM-000001"}})
        assert 0 <= result["score"] <= 100


@pytest.mark.asyncio
async def test_auto_wrap_only_applies_to_tools_whose_schema_expects_exactly_req():
    """kpi_definitions takes NO arguments - the shim must not wrap {} into
    {"req": {}} for a tool that doesn't have a `req` parameter at all."""
    alarm_srv = await _alarm_registry()
    async with connect_inprocess(alarm_srv) as session:
        reg = ToolRegistry()
        await reg.add(session)
        result = await reg.call("kpi_definitions", {})
        assert result["kpis"]


# ---- error propagation -----------------------------------------------------------------------
@pytest.mark.asyncio
async def test_unknown_tool_name_raises_with_the_available_tools_listed():
    alarm_srv = await _alarm_registry()
    async with connect_inprocess(alarm_srv) as session:
        reg = ToolRegistry()
        await reg.add(session)
        with pytest.raises(ToolCallError, match="Unknown tool 'not_a_real_tool'"):
            await reg.call("not_a_real_tool", {})


@pytest.mark.asyncio
async def test_a_real_service_error_surfaces_as_toolcallerror_with_a_clean_message():
    alarm_srv = await _alarm_registry()
    async with connect_inprocess(alarm_srv) as session:
        reg = ToolRegistry()
        await reg.add(session)
        with pytest.raises(ToolCallError, match="not_found"):
            await reg.call("get_alarm", {"req": {"alarm_id": "ALM-NOPE"}})


@pytest.mark.asyncio
async def test_call_returns_plain_text_when_a_tool_explicitly_opts_out_of_structured_output():
    """Every real tool in this project uses structured_output=True, so this
    proves the text-content fallback path in call() itself works, using a
    throwaway tool that explicitly opts out. (The SDK auto-generates
    structured content by default even without opting in - confirmed before
    writing this - so `structured_output=False` is required to exercise this
    path at all.)"""
    from mcp.server.mcpserver import MCPServer

    srv = MCPServer("plain-text-test")

    @srv.tool(structured_output=False)
    async def echo(text: str) -> str:
        """Echo back the given text."""
        return f"echo: {text}"

    async with connect_inprocess(srv) as session:
        reg = ToolRegistry()
        await reg.add(session)
        result = await reg.call("echo", {"text": "hello"})
        assert result == "echo: hello"