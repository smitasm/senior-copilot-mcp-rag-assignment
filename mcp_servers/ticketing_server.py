"""MCP server wrapping the Ticketing API simulator: 4 tools.

`create_ticket` is the one WRITE operation in this whole project. The
approval gate is enforced HERE, in code, not left to the LLM's prompt
discipline: the tool requires `approved: true` and raises ToolError before
ever calling the ticketing API if it's missing. A 3B local model cannot be
trusted to always ask first purely from instructions; this makes "no ticket
without approval" true regardless of what the model does.
"""

from __future__ import annotations

import os

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import BaseModel, Field

from mcp_common.http_client import ApiError, SimulatorClient
from ticketing_api.models import (
    CreateTicketRequest, CreateTicketResponse, SimilarTicketsRequest, SimilarTicketsResponse, Ticket,
    TicketListQuery, TicketListResponse,
)


class TicketIdRequest(BaseModel):
    ticket_id: str = Field(description="Ticket id from list_tickets or find_similar_tickets, e.g. INC0000123.")


class CreateTicketToolRequest(BaseModel):
    approved: bool = Field(
        default=False,
        description="Must be explicitly set to true. Set it ONLY after the user has been shown the exact "
                    "ticket content and has clearly approved creating it. If false, or if the user has not "
                    "yet approved this specific content, do not call this tool with approved=true - ask "
                    "for approval first and wait for their answer.",
    )
    ticket: CreateTicketRequest


async def _call(coro):
    try:
        return await coro
    except ApiError as exc:
        raise ToolError(f"{exc.code}: {exc.message}") from exc


def build_ticketing_server(client: SimulatorClient) -> MCPServer:
    srv = MCPServer(
        "ticketing",
        instructions="Tools for the plant's ticketing system. list_tickets, get_ticket and "
                    "find_similar_tickets are read-only and safe to call freely. create_ticket WRITES data "
                    "and requires the user's explicit approval of the exact ticket content beforehand.",
    )

    @srv.tool(structured_output=True)
    async def list_tickets(req: TicketListQuery) -> TicketListResponse:
        """List/filter/page historical and open tickets."""
        data = await _call(client.get("/tickets", params=req.model_dump(mode="json", exclude_none=True)))
        return TicketListResponse.model_validate(data)

    @srv.tool(structured_output=True)
    async def get_ticket(req: TicketIdRequest) -> Ticket:
        """Get full detail for one ticket by its ticket_id."""
        return Ticket.model_validate(await _call(client.get(f"/tickets/{req.ticket_id}")))

    @srv.tool(structured_output=True)
    async def find_similar_tickets(req: SimilarTicketsRequest) -> SimilarTicketsResponse:
        """Find historical tickets similar to an alarm/asset/free-text
        description, ranked by relevance. Use this BEFORE creating a new
        ticket: it surfaces past resolutions as grounding evidence, and any
        OPEN ticket already covering the same issue (which create_ticket will
        also detect and reuse instead of duplicating)."""
        return SimilarTicketsResponse.model_validate(await _call(
            client.post("/tickets/similar", json_body=req.model_dump(mode="json", exclude_none=True))))

    @srv.tool(structured_output=True)
    async def create_ticket(req: CreateTicketToolRequest) -> CreateTicketResponse:
        """Create a new incident ticket. WRITE OPERATION: requires approved=true,
        set only after the user has explicitly approved this exact ticket
        content. If an open ticket already exists for the same asset+alarm,
        no new ticket is created; the existing one is returned instead
        (check the response's duplicate_of field)."""
        if not req.approved:
            raise ToolError("Not approved: creating a ticket requires the user's explicit approval of "
                            "this exact content first. Ask the user to confirm, then call this tool again "
                            "with approved=true.")
        body = req.ticket.model_dump(mode="json", exclude_none=True)
        return CreateTicketResponse.model_validate(await _call(client.post("/tickets", json_body=body)))

    return srv


if __name__ == "__main__":
    sim_client = SimulatorClient(
        base_url=os.getenv("TICKETING_API_URL", "http://localhost:8001"),
        token=os.getenv("TICKETING_API_TOKEN", "demo-token"),
        client_id="ticketing-mcp-server",
    )
    build_ticketing_server(sim_client).run(transport="streamable-http")