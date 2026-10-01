"""Tests for create_ticket_from_draft: the ONE place a ticket is actually
created, deliberately outside the LLM loop. approved=False must never reach
the ticketing API at all - proven by call-count, not just by the exception
type, the same discipline used for the MCP-level gate in Phase 3."""

import httpx
import pytest

from orchestrator.actions import ApprovalRequiredError, create_ticket_from_draft
from orchestrator.draft import IncidentDraft
from orchestrator.mcp_client import ToolRegistry, connect_inprocess
from mcp_servers.ticketing_server import build_ticketing_server
from mcp_common.http_client import SimulatorClient
from ticketing_api.main import Settings, create_app

TOKEN = "test-token"


def make_draft(**overrides) -> IncidentDraft:
    base = dict(
        alarm_id="ALM-000001", alarm_name="High Vibration", asset_id="AST-0001",
        asset_name="Recycle Gas Compressor K-201",
        priority_band="critical", likely_cause="Bearing wear.", evidence=["Vibration trending up"],
        recommended_actions=["Check lube oil pressure"], similar_ticket_ids=[], citations=["KB-0001"],
        ticket_priority="P1", ticket_short_description="K-201 High Vibration - active critical alarm",
        ticket_description="Automated draft from copilot investigation.",
    )
    base.update(overrides)
    return IncidentDraft.model_validate(base)


async def registry():
    transport = httpx.ASGITransport(app=create_app(Settings(api_token=TOKEN)))
    client = SimulatorClient(base_url="http://sim", token=TOKEN, transport=transport)
    return build_ticketing_server(client)


@pytest.mark.asyncio
async def test_approved_false_raises_and_never_calls_the_api():
    srv = await registry()
    async with connect_inprocess(srv) as session:
        reg = ToolRegistry()
        await reg.add(session)
        before = (await reg.call("list_tickets", {"req": {"page_size": 1}}))["pagination"]["total"]
        with pytest.raises(ApprovalRequiredError):
            await create_ticket_from_draft(reg, make_draft(), approved=False)
        after = (await reg.call("list_tickets", {"req": {"page_size": 1}}))["pagination"]["total"]
        assert after == before


@pytest.mark.asyncio
async def test_approved_true_creates_a_ticket_with_fields_from_the_draft():
    srv = await registry()
    async with connect_inprocess(srv) as session:
        reg = ToolRegistry()
        await reg.add(session)
        draft = make_draft(alarm_id="ALM-999999", asset_id="AST-0001",
                           ticket_short_description="Custom short description",
                           ticket_description="Custom full description.", ticket_priority="P2")
        result = await create_ticket_from_draft(reg, draft, approved=True)
        t = result["ticket"]
        assert t["short_description"] == "Custom short description"
        assert t["description"] == "Custom full description."
        assert t["priority"] == "P2" and t["asset_id"] == "AST-0001" and t["alarm_id"] == "ALM-999999"
        assert t["state"] == "new" and "copilot-generated" in t["tags"]


@pytest.mark.asyncio
async def test_reported_by_defaults_to_copilot_and_is_overridable():
    srv = await registry()
    async with connect_inprocess(srv) as session:
        reg = ToolRegistry()
        await reg.add(session)
        # distinct alarm_name on each so the dedup logic (correctly) treats
        # them as two different tickets rather than folding the second into
        # the first - a mistake this exact test caught once already.
        default = await create_ticket_from_draft(reg, make_draft(alarm_id="A1", alarm_name="High Vibration"),
                                                  approved=True)
        assert default["ticket"]["reported_by"] == "copilot"
        custom = await create_ticket_from_draft(
            reg, make_draft(alarm_id="A2", alarm_name="High Discharge Temperature"),
            approved=True, reported_by="jane.operator")
        assert custom["ticket"]["reported_by"] == "jane.operator"
        assert default["ticket"]["ticket_id"] != custom["ticket"]["ticket_id"]


@pytest.mark.asyncio
async def test_duplicate_open_ticket_is_still_detected_when_creating_from_a_draft():
    """K-202's Low Lube Oil Pressure ticket is already open in the seed data
    (Phase 2) - approval from a draft must not bypass that dedup logic. This
    calls the real create_ticket_from_draft, not a hand-built request: it
    caught a real bug during development (IncidentDraft was missing
    alarm_name entirely, so this path could never dedupe) before this test
    existed to prove it."""
    srv = await registry()
    async with connect_inprocess(srv) as session:
        reg = ToolRegistry()
        await reg.add(session)
        existing = await reg.call("list_tickets", {"req": {"alarm_name": "Low Lube Oil Pressure", "page_size": 1}})
        asset_id = existing["data"][0]["asset_id"]
        draft = make_draft(alarm_id="ALM-NEW", alarm_name="Low Lube Oil Pressure", asset_id=asset_id,
                           ticket_short_description="Lube oil pressure low again")
        result = await create_ticket_from_draft(reg, draft, approved=True)
        assert result["duplicate_of"] == result["ticket"]["ticket_id"]


@pytest.mark.asyncio
async def test_a_genuinely_new_alarm_on_the_same_asset_is_not_falsely_deduped():
    """The flip side of the dedup test: a different alarm_name on an asset
    that already has SOME open ticket must still create a new one."""
    srv = await registry()
    async with connect_inprocess(srv) as session:
        reg = ToolRegistry()
        await reg.add(session)
        existing = await reg.call("list_tickets", {"req": {"alarm_name": "Low Lube Oil Pressure", "page_size": 1}})
        asset_id = existing["data"][0]["asset_id"]  # K-202, which has an open lube-oil ticket
        draft = make_draft(alarm_id="ALM-DIFFERENT", alarm_name="High Discharge Temperature", asset_id=asset_id)
        result = await create_ticket_from_draft(reg, draft, approved=True)
        assert result["duplicate_of"] is None


@pytest.mark.asyncio
async def test_unknown_asset_in_draft_is_a_clean_error_not_a_crash():
    srv = await registry()
    async with connect_inprocess(srv) as session:
        reg = ToolRegistry()
        await reg.add(session)
        draft = make_draft(asset_id="AST-9999")
        with pytest.raises(Exception, match="not_found"):
            await create_ticket_from_draft(reg, draft, approved=True)