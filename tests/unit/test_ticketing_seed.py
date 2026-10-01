"""The ticket dataset must be internally consistent and contain the
storylines the ticketing-service tests and the demo depend on."""

from datetime import timezone

import pytest

from alarm_api.seed import BFP101, BFP102, K201, K202, build_dataset
from ticketing_api.models import OPEN_STATES, TicketState
from ticketing_api.seed import SEED_NOW, build_ticket_dataset

UTC = timezone.utc


@pytest.fixture(scope="module")
def tickets():
    return build_ticket_dataset()


@pytest.fixture(scope="module")
def asset_ids():
    return {a.name: a.asset_id for a in build_dataset().assets}


def test_same_seed_is_identical():
    a, b = build_ticket_dataset(7), build_ticket_dataset(7)
    assert [t.model_dump() for t in a] == [t.model_dump() for t in b]


def test_different_seed_differs():
    a, b = build_ticket_dataset(1), build_ticket_dataset(2)
    assert [t.opened_at for t in a] != [t.opened_at for t in b]


def test_ticket_ids_unique_and_sorted_by_open_date(tickets):
    ids = [t.ticket_id for t in tickets]
    assert len(ids) == len(set(ids))
    opened = [t.opened_at for t in tickets]
    assert opened == sorted(opened)


def test_asset_fields_match_the_alarm_api_asset_catalogue(tickets, asset_ids):
    by_id = {v: k for k, v in asset_ids.items()}
    for t in tickets:
        assert t.asset_id in by_id and t.asset_name == by_id[t.asset_id]


def test_nothing_opens_after_now(tickets):
    assert all(t.opened_at <= SEED_NOW for t in tickets)


# ---- storylines the copilot demo depends on --------------------------------------
def test_k201_high_vibration_has_resolved_history(tickets, asset_ids):
    hits = [t for t in tickets if t.asset_id == asset_ids[K201] and t.alarm_name == "High Vibration"]
    assert len(hits) >= 2
    assert all(t.state in (TicketState.resolved, TicketState.closed) and t.resolution_notes for t in hits)


def test_k202_low_lube_oil_has_exactly_one_still_open_ticket(tickets, asset_ids):
    hits = [t for t in tickets if t.asset_id == asset_ids[K202] and t.alarm_name == "Low Lube Oil Pressure"]
    assert len(hits) == 1 and hits[0].state in OPEN_STATES
    assert hits[0].resolved_at is None and hits[0].closed_at is None


def test_bfp101_has_multiple_recurring_resolved_tickets(tickets, asset_ids):
    hits = [t for t in tickets if t.asset_id == asset_ids[BFP101]]
    assert len(hits) >= 3
    assert {t.alarm_name for t in hits} >= {"Seal Flush Low Flow", "Low Suction Pressure"}


def test_bfp102_has_a_very_recent_bearing_ticket(tickets, asset_ids):
    hits = [t for t in tickets if t.asset_id == asset_ids[BFP102] and t.alarm_name == "High Bearing Temperature"]
    assert len(hits) == 1
    assert (SEED_NOW - hits[0].opened_at).days <= 3


def test_p401_nuisance_ticket_matches_the_alarm_api_rationalization_advice(tickets):
    hit = next(t for t in tickets if t.alarm_name == "Low Discharge Flow")
    assert "deadband" in hit.resolution_notes.lower()


def test_some_tickets_have_no_alarm_link(tickets):
    assert sum(t.alarm_name is None for t in tickets) >= 2


def test_only_one_ticket_is_open(tickets):
    assert sum(t.state in OPEN_STATES for t in tickets) == 1