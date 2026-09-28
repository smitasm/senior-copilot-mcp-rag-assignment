"""The seed dataset must satisfy every precondition the Postman collections rely
on, plus the storylines the demo depends on. If someone edits seed.py and
breaks a chain, these tests fail before Newman does."""

from datetime import datetime, timedelta, timezone

import pytest

from alarm_api.models import AlarmStatus, Severity
from alarm_api.seed import BFP101, K201, K202, SEED_NOW, build_dataset

UTC = timezone.utc
WIN_START = datetime(2026, 5, 1, tzinfo=UTC)  # Postman window
WIN_END = datetime(2026, 7, 1, tzinfo=UTC)


@pytest.fixture(scope="module")
def ds():
    return build_dataset()


def in_window(a):
    return WIN_START <= a.start_time < WIN_END


def max_in_rolling_window(times, minutes):
    times, best, lo = sorted(times), 0, 0
    for hi, t in enumerate(times):
        while t - times[lo] > timedelta(minutes=minutes):
            lo += 1
        best = max(best, hi - lo + 1)
    return best


# ---- determinism ----------------------------------------------------------
def test_same_seed_is_identical():
    a, b = build_dataset(7), build_dataset(7)
    assert [x.model_dump() for x in a.alarms] == [x.model_dump() for x in b.alarms]


def test_different_seed_differs():
    assert [x.start_time for x in build_dataset(1).alarms] != [x.start_time for x in build_dataset(2).alarms]


# ---- structural integrity -------------------------------------------------
def test_ids_unique_and_sorted(ds):
    ids = [a.alarm_id for a in ds.alarms]
    assert len(ids) == len(set(ids))
    starts = [a.start_time for a in ds.alarms]
    assert starts == sorted(starts)


def test_alarms_reference_valid_assets_and_denormalised_fields_match(ds):
    by_id = {a.asset_id: a for a in ds.assets}
    for al in ds.alarms:
        asset = by_id[al.asset_id]
        assert (al.asset_name, al.site, al.unit) == (asset.name, asset.site, asset.unit)


def test_related_asset_ids_resolve(ds):
    ids = {a.asset_id for a in ds.assets}
    for a in ds.assets:
        assert set(a.related_asset_ids) <= ids


def test_each_unit_belongs_to_one_site(ds):
    pairs = {(a.unit, a.site) for a in ds.assets}
    assert len({u for u, _ in pairs}) == len(pairs)


def test_status_invariants(ds):
    for a in ds.alarms:
        assert a.start_time <= SEED_NOW
        if a.acknowledged_at:
            assert a.acknowledged_at >= a.start_time
        if a.status == AlarmStatus.cleared:
            assert a.end_time and a.end_time > a.start_time and a.end_time <= SEED_NOW
        elif a.status == AlarmStatus.active:
            assert a.end_time is None and a.acknowledged_at is None
        else:
            assert a.end_time is None and a.acknowledged_at is not None


# ---- Postman preconditions --------------------------------------------------
def test_asset_search_targets_exist(ds):
    names = {a.name for a in ds.assets}
    assert {"Boiler Feed Pump 101", "Boiler Feed Pump 102"} <= names


def test_at_least_three_compressors_and_first_three_are_the_east_cluster(ds):
    comps = [a for a in ds.assets if "compressor" in a.name.lower()]
    assert len(comps) >= 3
    assert all(c.site == "EastRefinery" and c.unit == "Unit 2" for c in comps[:3])


def test_unit5_has_three_or_more_motors(ds):
    motors = [a for a in ds.assets if a.unit == "Unit 5" and "motor" in a.name.lower()]
    assert len(motors) >= 3


def test_every_site_has_alarms_in_postman_window(ds):
    for site in ("EastRefinery", "NorthPlant", "SouthPlant"):
        assert any(a.site == site and in_window(a) for a in ds.alarms)


def test_east_refinery_has_active_alarms_with_clear_top_priority(ds):
    active = [a for a in ds.alarms if a.site == "EastRefinery" and a.status == AlarmStatus.active]
    assert len(active) >= 3
    top = max(active, key=lambda a: a.severity.rank)
    assert (top.asset_name, top.alarm_name, top.severity) == (K201, "High Vibration", Severity.critical)


def test_bfp102_has_an_active_alarm(ds):
    assert any(a.asset_name == "Boiler Feed Pump 102" and a.status == AlarmStatus.active for a in ds.alarms)


def test_unit5_has_safety_and_device_alarms_in_window(ds):
    kinds = {a.alarm_type.value for a in ds.alarms if a.unit == "Unit 5" and in_window(a)}
    assert {"safety", "device"} <= kinds


def test_unit2_flood_exists_inside_postman_window(ds):
    times = [a.start_time for a in ds.alarms if a.unit == "Unit 2" and in_window(a)]
    assert max_in_rolling_window(times, 10) >= 10


def test_unit3_has_critical_alarms_in_window(ds):
    assert sum(a.unit == "Unit 3" and a.severity == Severity.critical and in_window(a) for a in ds.alarms) >= 3


# ---- demo storylines ------------------------------------------------------------
def test_bfp101_recurring_high_severity_over_last_90_days(ds):
    cutoff = SEED_NOW - timedelta(days=90)
    hits = [a for a in ds.alarms if a.asset_name == BFP101 and a.start_time >= cutoff
            and a.severity in (Severity.high, Severity.critical)]
    assert len(hits) >= 30
    assert {a.alarm_name for a in hits} >= {"Low Suction Pressure", "High Bearing Temperature"}


def test_compressor_cluster_cooccurs_within_15_minutes(ds):
    k201 = [a.start_time for a in ds.alarms if a.asset_name == K201]
    k202 = [a.start_time for a in ds.alarms if a.asset_name == K202]
    pairs = sum(any(0 <= (t2 - t1).total_seconds() <= 900 for t2 in k202) for t1 in k201)
    assert pairs >= 10


def test_stale_alarms_exist_in_northplant_unit1(ds):
    stale = [a for a in ds.alarms if a.unit == "Unit 1" and in_window(a) and a.end_time
             and (a.end_time - a.start_time) > timedelta(minutes=180)]
    assert len(stale) >= 6


def test_unit4_chattering_repeats_far_beyond_threshold_8(ds):
    chatter = [a for a in ds.alarms if a.unit == "Unit 4" and a.alarm_name == "Low Discharge Flow" and in_window(a)]
    short = [a for a in chatter if (a.end_time - a.start_time) <= timedelta(minutes=5)]
    assert len(short) >= 8  # chattering = many very short repeats
    assert len(short) / len(chatter) >= 0.8  # and they dominate this alarm


def test_timeline_covers_last_90_days_from_now(ds):
    assert ds.alarms[0].start_time < WIN_START
    assert ds.alarms[-1].start_time >= SEED_NOW - timedelta(hours=1)