"""Domain models for the mock ticketing system (ServiceNow-Incident-style
fields: short_description, state, priority P1-P4, assignment_group).

Unlike the Alarm API, no Postman collection defines this contract - it's our
own design, kept close to a real ITSM tool so the mapping is easy to defend.
`ApiModel`/`Pagination` are imported rather than redefined, since they are
generic (extra="forbid" request base, page/page_size/total/total_pages).
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import Field, model_validator

from alarm_api.models import ApiModel, Pagination


class TicketState(str, Enum):
    new = "new"
    in_progress = "in_progress"
    on_hold = "on_hold"
    resolved = "resolved"
    closed = "closed"


class TicketPriority(str, Enum):
    p1_critical = "P1"
    p2_high = "P2"
    p3_moderate = "P3"
    p4_low = "P4"

    @property
    def rank(self) -> int:  # higher = more urgent, same convention as Severity.rank
        return {"P1": 3, "P2": 2, "P3": 1, "P4": 0}[self.value]


OPEN_STATES = (TicketState.new, TicketState.in_progress, TicketState.on_hold)
CLOSED_STATES = (TicketState.resolved, TicketState.closed)


class Ticket(ApiModel):
    ticket_id: str
    short_description: str
    description: str
    state: TicketState
    priority: TicketPriority
    category: str
    assignment_group: str
    assigned_to: str | None = None
    reported_by: str
    asset_id: str
    asset_name: str
    site: str
    unit: str
    alarm_id: str | None = None  # set when the ticket was raised from a specific alarm instance
    alarm_name: str | None = None  # set even for older tickets whose source alarm_id is not tracked
    opened_at: datetime
    updated_at: datetime
    resolved_at: datetime | None = None
    closed_at: datetime | None = None
    resolution_notes: str | None = None
    tags: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _state_matches_timestamps(self) -> "Ticket":
        has_res, has_closed = self.resolved_at is not None, self.closed_at is not None
        if self.state in CLOSED_STATES and not has_res:
            raise ValueError(f"state={self.state.value} requires resolved_at")
        if self.state == TicketState.closed and not has_closed:
            raise ValueError("state=closed requires closed_at")
        if self.state in OPEN_STATES and (has_res or has_closed):
            raise ValueError(f"state={self.state.value} must not have resolved_at/closed_at")
        if self.updated_at < self.opened_at:
            raise ValueError("updated_at cannot be before opened_at")
        if has_res and self.resolved_at < self.opened_at:
            raise ValueError("resolved_at cannot be before opened_at")
        if has_res and has_closed and self.closed_at < self.resolved_at:
            raise ValueError("closed_at cannot be before resolved_at")
        return self


class TicketListQuery(ApiModel):
    asset_id: str | None = None
    site: str | None = None
    unit: str | None = None
    state: TicketState | None = None
    priority: TicketPriority | None = None
    alarm_name: str | None = None
    opened_after: datetime | None = None
    opened_before: datetime | None = None
    page: int = Field(1, ge=1)
    page_size: int = Field(50, ge=1, le=200)
    sort_by: Literal["opened_at", "priority"] = "opened_at"
    sort_order: Literal["asc", "desc"] = "desc"


class TicketListResponse(ApiModel):
    data: list[Ticket]
    pagination: Pagination


class CreateTicketRequest(ApiModel):
    short_description: str = Field(min_length=1, max_length=160)
    description: str = Field(min_length=1, max_length=4000)
    priority: TicketPriority
    asset_id: str
    reported_by: str = Field(min_length=1)
    category: str | None = None  # defaults to the asset's maintenance_group if omitted
    assignment_group: str | None = None  # defaults to the asset's maintenance_group if omitted
    assigned_to: str | None = None
    alarm_id: str | None = None
    alarm_name: str | None = None
    tags: list[str] = Field(default_factory=list)


class CreateTicketResponse(ApiModel):
    ticket: Ticket
    duplicate_of: str | None = None  # set when an open ticket already covers this asset+alarm


class SimilarTicketsRequest(ApiModel):
    asset_ids: list[str] | None = None
    alarm_name: str | None = None
    query: str | None = None
    limit: int = Field(5, ge=1, le=20)
    min_score: float = Field(0.05, ge=0, le=1)

    @model_validator(mode="after")
    def _at_least_one_signal(self) -> "SimilarTicketsRequest":
        if not (self.asset_ids or self.alarm_name or self.query):
            raise ValueError("at least one of asset_ids, alarm_name or query is required")
        return self


class SimilarTicket(ApiModel):
    ticket: Ticket
    score: float = Field(ge=0, le=1)
    matched_terms: list[str]


class SimilarTicketsResponse(ApiModel):
    candidates: list[SimilarTicket]