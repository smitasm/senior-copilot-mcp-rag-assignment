"""Agent loop tests.

Two styles, deliberately: most tests here use a tiny throwaway MCP server
with one dummy tool, so the LOOP's control flow (recovery, retries, turn
limits) can be tested precisely without needing real alarm/ticket data. One
integration-style test at the end runs the loop against the REAL alarm,
ticketing and knowledge servers (connected to the real simulators via ASGI
transport) to prove a realistic investigation genuinely produces a grounded
draft - the single most demo-relevant test in this project.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

import httpx
import pytest
from mcp.server.mcpserver import MCPServer
from pydantic import BaseModel

from alarm_api.main import Settings as AlarmSettings
from alarm_api.main import create_app as alarm_app
from mcp_common.http_client import SimulatorClient
from mcp_servers.alarm_server import build_alarm_server
from mcp_servers.knowledge_server import build_knowledge_server
from mcp_servers.ticketing_server import build_ticketing_server
from orchestrator.agent import SUBMIT_DRAFT_TOOL_NAME, investigate
from orchestrator.llm import FakeLLM, LLMResponse, LLMTimeoutError, ToolCall
from orchestrator.mcp_client import ToolRegistry, connect_inprocess
from ticketing_api.main import Settings as TicketSettings
from ticketing_api.main import create_app as ticket_app

TOKEN = "test-token"

VALID_DRAFT_ARGS = {
    "alarm_id": "ALM-000001", "alarm_name": "High Vibration", "asset_id": "AST-0001",
    "asset_name": "Recycle Gas Compressor K-201",
    "priority_band": "critical", "likely_cause": "Bearing wear, consistent with rising vibration trend.",
    "evidence": ["Vibration trending up over several shifts"], "recommended_actions": ["Check lube oil pressure"],
    "similar_ticket_ids": [], "citations": [], "ticket_priority": "P1",
    "ticket_short_description": "K-201 High Vibration - active critical alarm",
    "ticket_description": "Automated draft from copilot investigation.",
}


def submit_call(call_id: str = "submit-1", args: dict | None = None) -> ToolCall:
    return ToolCall(call_id, SUBMIT_DRAFT_TOOL_NAME, args or VALID_DRAFT_ARGS)


class _EchoResult(BaseModel):
    echo: str


@asynccontextmanager
async def dummy_registry():
    """A minimal real MCP server (one tool) for testing the agent loop's
    control flow in isolation from real alarm/ticket data. Deliberately a
    plain async context manager, NOT a pytest async-generator fixture: an
    async-generator fixture wrapping connect_inprocess() tears down in a
    different task than it opened in under pytest-asyncio, which the MCP
    SDK's internal TaskGroup cancel scope rejects ("Attempted to exit cancel
    scope in a different task than it was entered in") - confirmed by
    hitting exactly that error before switching to this pattern. Opening and
    closing within a single test function's own task, as done everywhere
    else in this project, avoids it entirely."""
    srv = MCPServer("dummy")

    @srv.tool(structured_output=True)
    async def dummy_tool(query: str) -> _EchoResult:
        """A dummy tool that just echoes its query."""
        return _EchoResult(echo=query)

    async with connect_inprocess(srv) as session:
        reg = ToolRegistry()
        await reg.add(session)
        yield reg


# ---- happy paths -------------------------------------------------------------
@pytest.mark.asyncio
async def test_immediate_submit_with_no_tool_calls_first_still_works():
    async with dummy_registry() as reg:
        llm = FakeLLM([LLMResponse(content="", tool_calls=[submit_call()])])
        result = await investigate("prepare an incident", llm, reg)
    assert result.stopped_reason == "draft_submitted" and result.draft is not None
    assert result.draft.alarm_id == "ALM-000001"
    assert result.tool_calls_made == []  # submit itself is not counted as a "real" tool call


@pytest.mark.asyncio
async def test_tool_call_then_submit_is_recorded_in_tool_calls_made():
    async with dummy_registry() as reg:
        llm = FakeLLM([
            LLMResponse(content="", tool_calls=[ToolCall("c1", "dummy_tool", {"query": "K-201"})]),
            LLMResponse(content="", tool_calls=[submit_call()]),
        ])
        result = await investigate("prepare an incident", llm, reg)
    assert result.tool_calls_made == [("dummy_tool", {"query": "K-201"})]
    assert result.turns_used == 2 and result.stopped_reason == "draft_submitted"


# ---- resilience: a weak model's mistakes must not crash the run ------------------------
@pytest.mark.asyncio
async def test_hallucinated_tool_name_is_fed_back_as_an_error_and_the_model_recovers():
    async with dummy_registry() as reg:
        llm = FakeLLM([
            LLMResponse(content="", tool_calls=[ToolCall("c1", "not_a_real_tool", {})]),
            LLMResponse(content="", tool_calls=[submit_call()]),
        ])
        result = await investigate("prepare an incident", llm, reg)
    assert result.stopped_reason == "draft_submitted"
    tool_msgs = [m for m in result.transcript if m["role"] == "tool"]
    assert any("Unknown tool" in m["content"] for m in tool_msgs)


@pytest.mark.asyncio
async def test_invalid_draft_is_rejected_with_a_specific_error_and_the_model_retries():
    bad_args = dict(VALID_DRAFT_ARGS)
    del bad_args["likely_cause"]  # required field missing
    async with dummy_registry() as reg:
        llm = FakeLLM([
            LLMResponse(content="", tool_calls=[submit_call("submit-bad", bad_args)]),
            LLMResponse(content="", tool_calls=[submit_call("submit-good")]),
        ])
        result = await investigate("prepare an incident", llm, reg)
    assert result.stopped_reason == "draft_submitted"
    tool_msgs = [m["content"] for m in result.transcript if m["role"] == "tool"]
    assert any("rejected" in c and "likely_cause" in c for c in tool_msgs)


@pytest.mark.asyncio
async def test_invalid_priority_band_enum_value_is_rejected_and_retried():
    bad_args = dict(VALID_DRAFT_ARGS, ticket_priority="P9")  # not a real TicketPriority value
    async with dummy_registry() as reg:
        llm = FakeLLM([
            LLMResponse(content="", tool_calls=[submit_call("submit-bad", bad_args)]),
            LLMResponse(content="", tool_calls=[submit_call("submit-good")]),
        ])
        result = await investigate("prepare an incident", llm, reg)
    assert result.stopped_reason == "draft_submitted" and result.turns_used == 2


@pytest.mark.asyncio
async def test_submit_args_double_wrapped_under_req_are_unwrapped_and_accepted():
    """A model that has just seen many req-wrapped real tools may wrap this
    pseudo-tool's args the same way - it must still work."""
    async with dummy_registry() as reg:
        llm = FakeLLM([LLMResponse(content="", tool_calls=[submit_call(args={"req": VALID_DRAFT_ARGS})])])
        result = await investigate("prepare an incident", llm, reg)
    assert result.stopped_reason == "draft_submitted" and result.draft.alarm_id == "ALM-000001"


@pytest.mark.asyncio
async def test_plain_prose_with_no_tool_call_is_nudged_back_not_accepted_as_the_draft():
    async with dummy_registry() as reg:
        llm = FakeLLM([
            LLMResponse(content="I think the answer is just this text."),
            LLMResponse(content="", tool_calls=[submit_call()]),
        ])
        result = await investigate("prepare an incident", llm, reg)
    assert result.stopped_reason == "draft_submitted"
    user_msgs = [m["content"] for m in result.transcript if m["role"] == "user"]
    assert any("Please continue investigating" in c for c in user_msgs)


# ---- bounded loop -----------------------------------------------------------------------
@pytest.mark.asyncio
async def test_loop_stops_at_max_turns_if_the_model_never_submits():
    async with dummy_registry() as reg:
        llm = FakeLLM([LLMResponse(content="", tool_calls=[ToolCall(f"c{i}", "dummy_tool", {"query": "x"})])
                      for i in range(3)])
        result = await investigate("prepare an incident", llm, reg, max_turns=3)
    assert result.stopped_reason == "max_turns" and result.draft is None and result.turns_used == 3
    assert len(result.tool_calls_made) == 3


# ---- citation/ticket grounding: the model must not fabricate evidence --------------------------
# Directly motivated by a live run: the model submitted a draft citing
# "KB-0001" and "KB-0002" - real, plausible-sounding doc IDs from this
# project's own corpus - despite never having called search_knowledge_base
# at all that run. The system prompt already says "do not fabricate
# evidence"; these tests prove the mechanical check that doesn't depend on
# the model obeying that on its own.
@pytest.mark.asyncio
async def test_citing_a_doc_id_never_actually_returned_is_rejected():
    know_srv = build_knowledge_server()
    async with connect_inprocess(know_srv) as session:
        reg = ToolRegistry()
        await reg.add(session)
        # never calls search_knowledge_base at all - straight to a citing submit
        llm = FakeLLM([
            LLMResponse(content="", tool_calls=[submit_call(args=dict(VALID_DRAFT_ARGS, citations=["KB-0001"]))]),
            LLMResponse(content="", tool_calls=[submit_call(args=VALID_DRAFT_ARGS)]),  # citations=[] this time
        ])
        result = await investigate("prepare an incident", llm, reg)
    assert result.stopped_reason == "draft_submitted" and result.draft.citations == []
    tool_msgs = [m["content"] for m in result.transcript if m["role"] == "tool"]
    assert any("not actually" in c and "KB-0001" in c for c in tool_msgs)


@pytest.mark.asyncio
async def test_citing_a_doc_id_actually_returned_by_search_is_accepted():
    know_srv = build_knowledge_server()
    async with connect_inprocess(know_srv) as session:
        reg = ToolRegistry()
        await reg.add(session)
        llm = FakeLLM([
            LLMResponse(content="", tool_calls=[ToolCall("c1", "search_knowledge_base", {
                "req": {"query": "high vibration compressor bearing coupling"}})]),
            LLMResponse(content="", tool_calls=[submit_call(args=dict(VALID_DRAFT_ARGS, citations=["KB-0001"]))]),
        ])
        result = await investigate("prepare an incident", llm, reg)
    assert result.stopped_reason == "draft_submitted" and result.draft.citations == ["KB-0001"]


@pytest.mark.asyncio
async def test_similar_ticket_id_never_actually_returned_is_rejected():
    transport = httpx.ASGITransport(app=ticket_app(TicketSettings(api_token=TOKEN)))
    client = SimulatorClient(base_url="http://sim", token=TOKEN, transport=transport)
    async with connect_inprocess(build_ticketing_server(client)) as session:
        reg = ToolRegistry()
        await reg.add(session)
        llm = FakeLLM([
            LLMResponse(content="", tool_calls=[submit_call(args=dict(
                VALID_DRAFT_ARGS, similar_ticket_ids=["INC0000001"]))]),  # fabricated, never looked up
            LLMResponse(content="", tool_calls=[submit_call(args=VALID_DRAFT_ARGS)]),
        ])
        result = await investigate("prepare an incident", llm, reg)
    assert result.stopped_reason == "draft_submitted" and result.draft.similar_ticket_ids == []
    tool_msgs = [m["content"] for m in result.transcript if m["role"] == "tool"]
    assert any("INC0000001" in c and "not actually" in c for c in tool_msgs)


@pytest.mark.asyncio
async def test_similar_ticket_id_actually_returned_by_find_similar_tickets_is_accepted():
    transport = httpx.ASGITransport(app=ticket_app(TicketSettings(api_token=TOKEN)))
    client = SimulatorClient(base_url="http://sim", token=TOKEN, transport=transport)
    async with connect_inprocess(build_ticketing_server(client)) as session:
        reg = ToolRegistry()
        await reg.add(session)
        found = await reg.call("find_similar_tickets", {"req": {"query": "compressor vibration bearing"}})
        real_id = found["candidates"][0]["ticket"]["ticket_id"]
        llm = FakeLLM([
            LLMResponse(content="", tool_calls=[ToolCall("c1", "find_similar_tickets",
                                                          {"req": {"query": "compressor vibration bearing"}})]),
            LLMResponse(content="", tool_calls=[submit_call(args=dict(
                VALID_DRAFT_ARGS, similar_ticket_ids=[real_id]))]),
        ])
        result = await investigate("prepare an incident", llm, reg)
    assert result.stopped_reason == "draft_submitted" and result.draft.similar_ticket_ids == [real_id]


@pytest.mark.asyncio
async def test_a_ticket_id_seen_via_list_tickets_also_counts_as_grounded():
    """The grounding check accepts any real prior sighting of the id - not
    only via find_similar_tickets - since list_tickets/get_ticket are
    equally legitimate ways to have actually seen a ticket."""
    transport = httpx.ASGITransport(app=ticket_app(TicketSettings(api_token=TOKEN)))
    client = SimulatorClient(base_url="http://sim", token=TOKEN, transport=transport)
    async with connect_inprocess(build_ticketing_server(client)) as session:
        reg = ToolRegistry()
        await reg.add(session)
        listing = await reg.call("list_tickets", {"req": {"page_size": 1}})
        real_id = listing["data"][0]["ticket_id"]
        llm = FakeLLM([
            LLMResponse(content="", tool_calls=[ToolCall("c1", "list_tickets", {"req": {"page_size": 1}})]),
            LLMResponse(content="", tool_calls=[submit_call(args=dict(
                VALID_DRAFT_ARGS, similar_ticket_ids=[real_id]))]),
        ])
        result = await investigate("prepare an incident", llm, reg)
    assert result.stopped_reason == "draft_submitted" and result.draft.similar_ticket_ids == [real_id]


# ---- LLM failure mid-investigation must not discard evidence already gathered ------------------
# Reproduces a real run: turns 1-4 made genuine progress (a real tool call
# succeeded), then the 5th call to the model timed out and crashed the whole
# investigation with a raw exception. This proves the fix.
@pytest.mark.asyncio
async def test_llm_timeout_after_real_progress_returns_gracefully_with_evidence_preserved():
    async with dummy_registry() as reg:
        llm = FakeLLM([
            LLMResponse(content="", tool_calls=[ToolCall("c1", "dummy_tool", {"query": "K-201"})]),
            LLMTimeoutError("simulated: local inference did not respond in time"),
        ])
        result = await investigate("prepare an incident", llm, reg)
    assert result.stopped_reason == "llm_error" and result.draft is None
    assert result.tool_calls_made == [("dummy_tool", {"query": "K-201"})]  # turn 1's real progress is kept
    assert result.turns_used == 1  # the failed turn itself is not counted as "used"


@pytest.mark.asyncio
async def test_llm_timeout_on_the_very_first_call_still_returns_cleanly():
    async with dummy_registry() as reg:
        llm = FakeLLM([LLMTimeoutError("simulated")])
        result = await investigate("prepare an incident", llm, reg)
    assert result.stopped_reason == "llm_error" and result.draft is None and result.tool_calls_made == []


# ---- repeat-tool circuit breaker ---------------------------------------------------------------
# Directly motivated by a real observed failure: a 3B local model called
# search_knowledge_base on all 8 turns of a live run and never touched any
# other tool. These tests reproduce that exact shape against FakeLLM.
@pytest.mark.asyncio
async def test_stuck_on_one_tool_stops_early_with_a_diagnosable_reason():
    async with dummy_registry() as reg:
        llm = FakeLLM([LLMResponse(content="", tool_calls=[ToolCall(f"c{i}", "dummy_tool", {"query": "x"})])
                      for i in range(8)])
        result = await investigate("prepare an incident", llm, reg, max_turns=8)
    assert result.stopped_reason == "stuck_repeating_tool"
    assert result.draft is None
    assert result.turns_used < 8  # stopped well before burning the whole turn budget
    assert len(llm.calls) == result.turns_used  # no wasted model calls after the cutoff


@pytest.mark.asyncio
async def test_a_warning_is_injected_before_the_hard_stop():
    async with dummy_registry() as reg:
        llm = FakeLLM([LLMResponse(content="", tool_calls=[ToolCall(f"c{i}", "dummy_tool", {"query": "x"})])
                      for i in range(8)])
        result = await investigate("prepare an incident", llm, reg, max_turns=8)
    user_msgs = [m["content"] for m in result.transcript if m["role"] == "user"]
    assert any("called dummy_tool" in c and "3 times in a row" in c for c in user_msgs)


@pytest.mark.asyncio
async def test_switching_tools_after_the_warning_avoids_the_hard_stop_and_can_still_succeed():
    """The breaker must not punish a model that reads the corrective nudge
    and actually changes behaviour - proves this isn't a blunt kill switch."""
    async with dummy_registry() as reg:
        llm = FakeLLM([
            LLMResponse(content="", tool_calls=[ToolCall("c1", "dummy_tool", {"query": "x"})]),
            LLMResponse(content="", tool_calls=[ToolCall("c2", "dummy_tool", {"query": "x"})]),
            LLMResponse(content="", tool_calls=[ToolCall("c3", "dummy_tool", {"query": "x"})]),  # hits the warning
            LLMResponse(content="", tool_calls=[submit_call()]),  # recovers immediately after
        ])
        result = await investigate("prepare an incident", llm, reg, max_turns=8)
    assert result.stopped_reason == "draft_submitted" and result.draft is not None
    assert result.turns_used == 4


@pytest.mark.asyncio
async def test_repeat_counter_resets_when_the_tool_changes():
    """Two calls to A, then B, then two more calls to A must NOT trip the
    breaker - it counts CONSECUTIVE repeats, not total occurrences."""
    srv = MCPServer("d3")

    @srv.tool(structured_output=True)
    async def tool_a(query: str) -> _EchoResult:
        """a"""
        return _EchoResult(echo=query)

    @srv.tool(structured_output=True)
    async def tool_b(query: str) -> _EchoResult:
        """b"""
        return _EchoResult(echo=query)

    async with connect_inprocess(srv) as session:
        reg = ToolRegistry()
        await reg.add(session)
        llm = FakeLLM([
            LLMResponse(content="", tool_calls=[ToolCall("c1", "tool_a", {"query": "x"})]),
            LLMResponse(content="", tool_calls=[ToolCall("c2", "tool_a", {"query": "x"})]),
            LLMResponse(content="", tool_calls=[ToolCall("c3", "tool_b", {"query": "x"})]),  # resets the streak
            LLMResponse(content="", tool_calls=[ToolCall("c4", "tool_a", {"query": "x"})]),
            LLMResponse(content="", tool_calls=[submit_call()]),
        ])
        result = await investigate("go", llm, reg, max_turns=8)
    assert result.stopped_reason == "draft_submitted"  # never hit the breaker
    warn_msgs = [m["content"] for m in result.transcript if m["role"] == "user" and "times in a row" in m.get("content", "")]
    assert warn_msgs == []


@pytest.mark.asyncio
async def test_max_turns_is_configurable_and_respected_exactly():
    srv = MCPServer("d2")

    @srv.tool(structured_output=True)
    async def t(x: str) -> _EchoResult:
        """t."""
        return _EchoResult(echo=x)

    async with connect_inprocess(srv) as session:
        reg = ToolRegistry()
        await reg.add(session)
        llm = FakeLLM([LLMResponse(content="", tool_calls=[ToolCall("c", "t", {"x": "1"})]) for _ in range(2)])
        result = await investigate("go", llm, reg, max_turns=2)
        assert result.turns_used == 2 and result.stopped_reason == "max_turns"


# ---- transcript / system prompt shape --------------------------------------------------------
@pytest.mark.asyncio
async def test_system_prompt_and_user_request_open_the_transcript():
    async with dummy_registry() as reg:
        llm = FakeLLM([LLMResponse(content="", tool_calls=[submit_call()])])
        result = await investigate("prepare an incident for K-201", llm, reg)
    assert result.transcript[0]["role"] == "system"
    assert result.transcript[1] == {"role": "user", "content": "prepare an incident for K-201"}


@pytest.mark.asyncio
async def test_tools_passed_to_the_llm_include_the_submit_draft_pseudo_tool():
    async with dummy_registry() as reg:
        llm = FakeLLM([LLMResponse(content="", tool_calls=[submit_call()])])
        await investigate("go", llm, reg)
    _, tools_seen = llm.calls[0]
    names = {t["function"]["name"] for t in tools_seen}
    assert "dummy_tool" in names and SUBMIT_DRAFT_TOOL_NAME in names


# ---- the real, integration-style investigation --------------------------------------------------
@pytest.mark.asyncio
async def test_realistic_multi_step_investigation_against_the_real_servers_produces_a_grounded_draft():
    """Mirrors the actual demo path: find the top-priority active EastRefinery
    alarm, gather evidence, submit a draft - against the REAL alarm,
    ticketing and knowledge servers and the REAL simulators (ASGI transport),
    not mocks. The scripted 'model' here plays a well-behaved model on
    purpose; test_orchestrator_agent.py's other tests already cover a
    misbehaving one."""
    alarm_transport = httpx.ASGITransport(app=alarm_app(AlarmSettings(api_token=TOKEN)))
    ticket_transport = httpx.ASGITransport(app=ticket_app(TicketSettings(api_token=TOKEN)))
    alarm_client = SimulatorClient(base_url="http://sim", token=TOKEN, transport=alarm_transport)
    ticket_client = SimulatorClient(base_url="http://sim", token=TOKEN, transport=ticket_transport)

    async with connect_inprocess(build_alarm_server(alarm_client)) as s1, \
              connect_inprocess(build_ticketing_server(ticket_client)) as s2, \
              connect_inprocess(build_knowledge_server()) as s3:
        reg = ToolRegistry()
        await reg.add(s1)
        await reg.add(s2)
        await reg.add(s3)

        # Discover the real top alarm first, the same way the "model" would via list_alarms.
        listing = await reg.call("list_alarms", {"req": {"site": "EastRefinery", "status": "active",
                                                          "sort_by": "severity"}})
        top = listing["data"][0]
        assert (top["asset_name"], top["alarm_name"]) == ("Recycle Gas Compressor K-201", "High Vibration")

        llm = FakeLLM([
            LLMResponse(content="", tool_calls=[ToolCall("c1", "list_alarms", {"req": {
                "site": "EastRefinery", "status": "active", "sort_by": "severity"}})]),
            LLMResponse(content="", tool_calls=[ToolCall("c2", "priority_score", {"alarm_id": top["alarm_id"]})]),
            LLMResponse(content="", tool_calls=[ToolCall("c3", "operator_recommendations", {"req": {
                "alarm_id": top["alarm_id"], "include_related": True, "include_historical_pattern": True}})]),
            LLMResponse(content="", tool_calls=[ToolCall("c4", "find_similar_tickets", {"req": {
                "asset_ids": [top["asset_id"]], "alarm_name": top["alarm_name"]}})]),
            LLMResponse(content="", tool_calls=[ToolCall("c5", "search_knowledge_base", {"req": {
                "query": "high vibration compressor bearing coupling"}})]),
            LLMResponse(content="", tool_calls=[submit_call(args={
                "alarm_id": top["alarm_id"], "alarm_name": top["alarm_name"], "asset_id": top["asset_id"],
                "asset_name": top["asset_name"], "priority_band": "critical",
                "likely_cause": "Likely bearing wear or rotor imbalance, consistent with KB-0001 and the "
                               "correlated compressor-train alarms.",
                "evidence": ["priority_score band=critical", "KB-0001 identifies bearing wear as a common cause"],
                "recommended_actions": ["Check lube oil pressure", "Schedule vibration analysis"],
                "similar_ticket_ids": [], "citations": ["KB-0001"], "ticket_priority": "P1",
                "ticket_short_description": "K-201 High Vibration - active critical alarm",
                "ticket_description": "Automated draft from copilot investigation.",
            })]),
        ])

        result = await investigate("Prepare an incident for the highest-priority active alarm in "
                                   "EastRefinery.", llm, reg)

        assert result.stopped_reason == "draft_submitted"
        assert [name for name, _ in result.tool_calls_made] == [
            "list_alarms", "priority_score", "operator_recommendations", "find_similar_tickets",
            "search_knowledge_base"]
        assert result.draft.alarm_id == top["alarm_id"] and result.draft.asset_id == top["asset_id"]
        assert result.draft.ticket_priority.value == "P1"
        assert "KB-0001" in result.draft.citations