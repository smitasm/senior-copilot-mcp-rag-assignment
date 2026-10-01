"""Business logic of the mock ticketing system (no HTTP in here).

Reuses `ServiceError`/`NotFoundError` from alarm_api.services - they carry
only (status, code, message, details), nothing alarm-specific, so importing
them here avoids redefining the same error taxonomy twice.

Asset metadata (name/site/unit/maintenance_group) is a STATIC snapshot taken
at startup from alarm_api's seed data, not a live HTTP call to the Alarm API.
The two simulators never call each other at request time: this mirrors how a
real CMDB feed syncs asset master data into a ticketing tool on a schedule,
rather than the ticketing tool depending on the alarm system being up.
"""

from __future__ import annotations

import math
import re
from typing import Callable

from alarm_api.models import AssetMetadata
from alarm_api.services import NotFoundError, ServiceError
from ticketing_api.models import (
    CLOSED_STATES, OPEN_STATES, CreateTicketRequest, CreateTicketResponse, Pagination, SimilarTicket,
    SimilarTicketsRequest, SimilarTicketsResponse, Ticket, TicketListQuery, TicketListResponse,
)

STOPWORDS = {"the", "a", "an", "is", "was", "were", "on", "in", "at", "to", "of", "and", "for",
            "with", "during", "this", "that", "alarm", "alarms", "ticket", "issue"}


def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOPWORDS}


class TicketService:
    def __init__(self, tickets: list[Ticket], now, assets: dict[str, AssetMetadata]) -> None:
        self._tickets: dict[str, Ticket] = {t.ticket_id: t for t in tickets}
        self._assets = assets
        self.now = now
        self._next_seq = len(tickets) + 1

    def _asset(self, asset_id: str) -> AssetMetadata:
        if asset_id not in self._assets:
            raise NotFoundError(f"Asset '{asset_id}' not found", [{"asset_id": asset_id}])
        return self._assets[asset_id]

    # ------------------------------------------------------------------
    def list_tickets(self, q: TicketListQuery) -> TicketListResponse:
        rows = [t for t in self._tickets.values()
                if (not q.asset_id or t.asset_id == q.asset_id) and (not q.site or t.site == q.site)
                and (not q.unit or t.unit == q.unit) and (not q.state or t.state == q.state)
                and (not q.priority or t.priority == q.priority)
                and (not q.alarm_name or (t.alarm_name or "").lower() == q.alarm_name.lower())
                and (not q.opened_after or t.opened_at >= q.opened_after)
                and (not q.opened_before or t.opened_at <= q.opened_before)]
        keyfn = ((lambda t: (t.priority.rank, t.opened_at)) if q.sort_by == "priority"
                else (lambda t: t.opened_at))
        rows.sort(key=keyfn, reverse=q.sort_order == "desc")
        total, begin = len(rows), (q.page - 1) * q.page_size
        return TicketListResponse(
            data=rows[begin:begin + q.page_size],
            pagination=Pagination(page=q.page, page_size=q.page_size, total=total,
                                  total_pages=math.ceil(total / q.page_size) if total else 0),
        )

    def get_ticket(self, ticket_id: str) -> Ticket:
        if ticket_id not in self._tickets:
            raise NotFoundError(f"Ticket '{ticket_id}' not found")
        return self._tickets[ticket_id]

    # ------------------------------------------------------------------
    def _open_duplicate(self, asset_id: str, alarm_name: str | None) -> Ticket | None:
        if not alarm_name:
            return None
        return next((t for t in self._tickets.values() if t.asset_id == asset_id and t.state in OPEN_STATES
                    and (t.alarm_name or "").lower() == alarm_name.lower()), None)

    def create_ticket(self, req: CreateTicketRequest) -> CreateTicketResponse:
        asset = self._asset(req.asset_id)
        if dup := self._open_duplicate(req.asset_id, req.alarm_name):
            return CreateTicketResponse(ticket=dup, duplicate_of=dup.ticket_id)
        ticket_id, self._next_seq = f"INC{self._next_seq:07d}", self._next_seq + 1
        ticket = Ticket(
            ticket_id=ticket_id, short_description=req.short_description, description=req.description,
            state="new", priority=req.priority, category=req.category or asset.maintenance_group or "General",
            assignment_group=req.assignment_group or asset.maintenance_group or "General",
            assigned_to=req.assigned_to, reported_by=req.reported_by, asset_id=asset.asset_id,
            asset_name=asset.name, site=asset.site, unit=asset.unit, alarm_id=req.alarm_id,
            alarm_name=req.alarm_name, opened_at=self.now, updated_at=self.now, tags=req.tags,
        )
        self._tickets[ticket_id] = ticket
        return CreateTicketResponse(ticket=ticket, duplicate_of=None)

    # ------------------------------------------------------------------
    def similar_tickets(self, req: SimilarTicketsRequest) -> SimilarTicketsResponse:
        qtext = " ".join(x for x in (req.query, req.alarm_name) if x)
        qtokens = _tokens(qtext)
        candidates: list[SimilarTicket] = []
        for t in self._tickets.values():
            ttext = " ".join([t.short_description, t.description, t.alarm_name or "", " ".join(t.tags)])
            overlap = qtokens & _tokens(ttext)
            # require at least 2 shared terms (or all of them, if the query itself
            # has fewer than 2) - one shared generic word like "high" or "low" is
            # noise, not evidence, and would otherwise pollute every result set.
            lexical = (len(overlap) / len(qtokens)) if qtokens and len(overlap) >= min(2, len(qtokens)) else 0.0
            bonus = (0.4 if req.alarm_name and (t.alarm_name or "").lower() == req.alarm_name.lower() else 0.0) + \
                   (0.2 if req.asset_ids and t.asset_id in req.asset_ids else 0.0)
            score = round(min(1.0, 0.5 * lexical + bonus), 3)
            if score >= req.min_score:
                candidates.append(SimilarTicket(ticket=t, score=score, matched_terms=sorted(overlap)))
        candidates.sort(key=lambda c: (-c.score, -c.ticket.opened_at.timestamp()))
        return SimilarTicketsResponse(candidates=candidates[:req.limit])