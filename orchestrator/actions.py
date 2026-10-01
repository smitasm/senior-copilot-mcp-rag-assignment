"""The ONE place in this project that actually creates a ticket. Deliberately
NOT part of the LLM tool-calling loop: once a human has approved a specific
IncidentDraft (possibly after editing its fields), this plain deterministic
function builds the exact ticket request from that approved content and
calls create_ticket directly. The write path never depends on the LLM
choosing to call the tool correctly a second time - it never gets the chance
to call it at all.

This is layered on top of, not instead of, the MCP-level approval gate
(mcp_servers/ticketing_server.py): that gate still requires approved=true on
the wire. This module is what actually sets it, and only after its own
check passes - defense in depth, not a bypass.
"""

from __future__ import annotations

from orchestrator.draft import IncidentDraft
from orchestrator.mcp_client import ToolRegistry


class ApprovalRequiredError(Exception):
    """Raised instead of ever calling create_ticket when approved is not
    explicitly True."""


async def create_ticket_from_draft(tools: ToolRegistry, draft: IncidentDraft, *, approved: bool,
                                   reported_by: str = "copilot") -> dict:
    if not approved:
        raise ApprovalRequiredError("Ticket creation requires the user's explicit approval of this exact "
                                    "draft content; no ticket was created.")
    ticket = {
        "short_description": draft.ticket_short_description,
        "description": draft.ticket_description,
        "priority": draft.ticket_priority.value,
        "asset_id": draft.asset_id,
        "reported_by": reported_by,
        "alarm_id": draft.alarm_id,
        "alarm_name": draft.alarm_name,  # required for the ticketing API's duplicate-open-ticket detection
        "tags": ["copilot-generated"],
    }
    return await tools.call("create_ticket", {"req": {"approved": True, "ticket": ticket}})