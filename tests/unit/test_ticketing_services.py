"""TicketService tests: filtering/sorting/pagination, duplicate-ticket
prevention, and the lexical similar-tickets search."""

import pytest

from alarm_api.seed import BFP101, BFP102, K201, K202, build_dataset
from alarm_api.services import NotFoundError, ServiceError
from ticketing_api import models as m
from ticketing_api.seed import build_ticket_dataset
from ticketing_api.services import TicketService


@pytest.fixture(scope="module")
def assets():
    return {a.asset_id: a for a in build_dataset().assets}


@pytest.fixture
def svc(assets):  # function-scoped: create_ticket mutates state
    return TicketService(build_ticket_dataset(), now=build_dataset().now, assets=assets)


def aid(assets, name):
    return next(a.asset_id for a in assets.values() if a.name == name)


# ---- listing ---------------------------------------------------------------
def test_filter_by_asset_and_state(svc, assets):
    q = m.TicketListQuery(asset_id=aid(assets, BFP101))
    res = svc.list_tickets(q)
    assert res.pagination.total >= 3 and all(t.asset_id == q.asset_id for t in res.data)


def test_sort_by_priority_puts_p1_first(svc):
    res = svc.list_tickets(m.TicketListQuery(sort_by="priority", sort_order="desc", page_size=200))
    assert res.data[0].priority == m.TicketPriority.p1_critical


def test_pagination_math(svc):
    res = svc.list_tickets(m.TicketListQuery(page_size=5))
    assert len(res.data) == 5 and res.pagination.total_pages == -(-res.pagination.total // 5)


def test_alarm_name_filter_is_case_insensitive(svc):
    res = svc.list_tickets(m.TicketListQuery(alarm_name="high vibration"))
    assert res.pagination.total >= 2 and all(t.alarm_name == "High Vibration" for t in res.data)


def test_get_ticket_not_found(svc):
    with pytest.raises(NotFoundError):
        svc.get_ticket("INC9999999")


# ---- create + duplicate prevention -----------------------------------------------
def test_create_ticket_fills_in_asset_context(svc, assets):
    req = m.CreateTicketRequest(short_description="Test alarm", description="Test.",
                                priority=m.TicketPriority.p3_moderate, asset_id=aid(assets, K201),
                                reported_by="unit-test", alarm_name="Some New Alarm")
    res = svc.create_ticket(req)
    assert res.duplicate_of is None
    t = res.ticket
    assert (t.state, t.asset_name, t.site, t.unit) == (m.TicketState.new, K201, "EastRefinery", "Unit 2")
    assert t.category == assets[aid(assets, K201)].maintenance_group
    assert svc.get_ticket(t.ticket_id) == t  # visible on the very next read


def test_create_ticket_for_k202_low_lube_oil_returns_the_existing_open_ticket(svc, assets):
    before = svc.list_tickets(m.TicketListQuery(page_size=200)).pagination.total
    req = m.CreateTicketRequest(short_description="Lube oil pressure alarm", description="New occurrence.",
                                priority=m.TicketPriority.p2_high, asset_id=aid(assets, K202),
                                reported_by="unit-test", alarm_name="Low Lube Oil Pressure")
    res = svc.create_ticket(req)
    assert res.duplicate_of is not None and res.ticket.ticket_id == res.duplicate_of
    assert res.ticket.state in m.OPEN_STATES
    after = svc.list_tickets(m.TicketListQuery(page_size=200)).pagination.total
    assert after == before  # no new ticket was created


def test_create_ticket_unknown_asset_is_404(svc):
    req = m.CreateTicketRequest(short_description="x", description="y", priority=m.TicketPriority.p4_low,
                                asset_id="AST-9999", reported_by="unit-test")
    with pytest.raises(NotFoundError):
        svc.create_ticket(req)


def test_create_ticket_without_alarm_name_never_dedupes(svc, assets):
    """A ticket with no alarm_name (e.g. generic PM work) must never be
    silently merged into an unrelated open ticket on the same asset."""
    req = m.CreateTicketRequest(short_description="Unrelated PM", description="y",
                                priority=m.TicketPriority.p4_low, asset_id=aid(assets, K202),
                                reported_by="unit-test")
    res = svc.create_ticket(req)
    assert res.duplicate_of is None


# ---- similar tickets ----------------------------------------------------------------
def test_similar_tickets_requires_at_least_one_signal():
    with pytest.raises(ValueError):
        m.SimilarTicketsRequest()


def test_similar_ranks_the_open_duplicate_first_for_k202(svc, assets):
    res = svc.similar_tickets(m.SimilarTicketsRequest(asset_ids=[aid(assets, K202)],
                                                       alarm_name="Low Lube Oil Pressure"))
    assert res.candidates and res.candidates[0].ticket.state in m.OPEN_STATES
    assert res.candidates[0].score > res.candidates[-1].score if len(res.candidates) > 1 else True


def test_similar_by_free_text_finds_k201_history(svc):
    res = svc.similar_tickets(m.SimilarTicketsRequest(query="compressor vibration bearing rotor"))
    names = {c.ticket.asset_name for c in res.candidates}
    assert K201 in names
    assert all(c.matched_terms for c in res.candidates)


def test_similar_min_score_filters_out_weak_matches(svc, assets):
    loose = svc.similar_tickets(m.SimilarTicketsRequest(asset_ids=[aid(assets, BFP102)], min_score=0.0))
    strict = svc.similar_tickets(m.SimilarTicketsRequest(asset_ids=[aid(assets, BFP102)], min_score=0.9))
    assert len(strict.candidates) <= len(loose.candidates)


def test_similar_rejects_single_generic_word_overlap_as_noise(svc, assets):
    """A ticket on an unrelated motor that merely shares the word 'high' with
    the query must not outrank or crowd out the real match."""
    res = svc.similar_tickets(m.SimilarTicketsRequest(alarm_name="High Vibration", min_score=0.0, limit=20))
    top = res.candidates[0]
    assert top.ticket.asset_name == K201 and top.score == 0.9  # lexical 0.5 + exact alarm_name bonus 0.4
    noise = [c for c in res.candidates if c.ticket.alarm_name == "High Winding Temperature"]
    assert all(c.score == 0.0 for c in noise)


def test_similar_results_are_score_ordered(svc):
    res = svc.similar_tickets(m.SimilarTicketsRequest(query="pressure alarm", min_score=0.0, limit=20))
    scores = [c.score for c in res.candidates]
    assert scores == sorted(scores, reverse=True)