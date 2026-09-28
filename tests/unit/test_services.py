"""Service-layer tests. Several replay the Postman chaining flows step by step
(CHAIN-01/02/03/05/09/10) so we know the simulator can answer them."""

from datetime import datetime, timezone
from typing import get_args

import pytest

from alarm_api import models as m
from alarm_api.seed import BFP101, K201, K202, K203, build_dataset
from alarm_api.services import AlarmService, NotFoundError, ServiceError

UTC = timezone.utc
START, END = datetime(2026, 5, 1, tzinfo=UTC), datetime(2026, 7, 1, tzinfo=UTC)
WINDOW = m.TimeRange(start_time=START, end_time=END)


@pytest.fixture(scope="module")
def svc():
    return AlarmService(build_dataset())


def asset_id(svc, name):
    return svc.search_assets(name, limit=1).results[0].asset_id


# ---- assets --------------------------------------------------------------------
def test_search_exact_name_ranks_first(svc):
    res = svc.search_assets("Boiler Feed Pump 101", limit=5)
    assert res.results[0].name == BFP101
    assert res.total >= 2  # BFP102 matches partially, ranked lower


def test_search_compressor_returns_three_east_compressors_first(svc):
    res = svc.search_assets("compressor", limit=5).results
    assert len(res) >= 3 and all(r.unit == "Unit 2" for r in res[:3])


def test_search_motor_in_unit5(svc):
    res = svc.search_assets("motor", unit="Unit 5", limit=5)
    assert len(res.results) >= 3 and all(r.unit == "Unit 5" for r in res.results)


def test_search_no_match_is_empty_not_error(svc):
    assert svc.search_assets("zzzz-nothing").results == []


def test_search_blank_query_rejected(svc):
    with pytest.raises(ServiceError):
        svc.search_assets("   ")


def test_asset_metadata_and_not_found(svc):
    meta = svc.get_asset_metadata(asset_id(svc, BFP101))
    assert meta.criticality == "A" and len(meta.related_asset_ids) == 2
    with pytest.raises(NotFoundError):
        svc.get_asset_metadata("AST-9999")


# ---- alarm listing (pagination, sorting) ----------------------------------------
def test_pagination_math(svc):
    r = svc.list_alarms(m.AlarmListQuery(page_size=200))
    assert r.pagination.total_pages == -(-r.pagination.total // 200)
    assert len(r.data) == 200
    beyond = svc.list_alarms(m.AlarmListQuery(page=999, page_size=200))
    assert beyond.data == [] and beyond.pagination.total == r.pagination.total


def test_default_sort_is_recency_not_priority(svc):
    """CHAIN-09 step 1 takes data[0]. Recency-first means it is NOT the most
    severe alarm - the copilot must rank by priority, not trust row order."""
    q = m.AlarmListQuery(site="EastRefinery", status=m.AlarmStatus.active)
    first = svc.list_alarms(q).data[0]
    assert first.alarm_name != "High Vibration"


def test_sort_by_severity_puts_critical_first(svc):
    q = m.AlarmListQuery(site="EastRefinery", status=m.AlarmStatus.active, sort_by="severity")
    top = svc.list_alarms(q).data[0]
    assert (top.asset_name, top.severity) == (K201, m.Severity.critical)


def test_time_filter_is_inclusive_on_both_ends(svc):
    a = svc.list_alarms(m.AlarmListQuery(asset_id=asset_id(svc, BFP101), page_size=5)).data[0]
    q = m.AlarmListQuery(asset_id=a.asset_id, start_time=a.start_time, end_time=a.start_time)
    assert [x.alarm_id for x in svc.list_alarms(q).data] == [a.alarm_id]


def test_get_alarm_not_found(svc):
    with pytest.raises(NotFoundError):
        svc.get_alarm("ALM-NOPE")


# ---- summary / trends ---------------------------------------------------------------
def test_chain01_summary_for_bfp101(svc):
    req = m.SummaryRequest(asset_ids=[asset_id(svc, BFP101)], time_range=WINDOW,
                           severity=[m.Severity.high, m.Severity.critical], group_by=["alarm_name"],
                           kpis=["alarm_count", "recurring_rate", "avg_ack_delay"])
    res = svc.summary(req)
    names = {g.key["alarm_name"]: g for g in res.groups}
    assert names["Low Suction Pressure"].metrics["alarm_count"] >= 5
    assert "Seal Flush Low Flow" not in names  # medium severity filtered out
    assert res.totals["alarm_count"] == sum(g.metrics["alarm_count"] for g in res.groups)
    assert res.totals["recurring_rate"] > 50 and res.totals["avg_ack_delay"] > 0


def test_summary_group_by_asset_and_severity_for_unit(svc):
    req = m.SummaryRequest(unit="Unit 2", time_range=WINDOW, group_by=["asset_id", "asset_name", "severity"])
    res = svc.summary(req)
    assert res.groups and set(res.groups[0].key) == {"asset_id", "asset_name", "severity"}


def test_summary_without_group_by_returns_totals_only(svc):
    res = svc.summary(m.SummaryRequest(unit="Unit 3", time_range=WINDOW, kpis=["alarm_count", "critical_count"]))
    assert res.groups == [] and res.totals["critical_count"] >= 3


def test_trends_daily_buckets_are_continuous_and_sum_to_total(svc):
    aid = asset_id(svc, BFP101)
    res = svc.trends(m.TrendRequest(asset_ids=[aid], time_range=WINDOW, metrics=["alarm_count"]))
    assert len(res.series) == 61
    total = svc.summary(m.SummaryRequest(asset_ids=[aid], time_range=WINDOW)).totals["alarm_count"]
    assert sum(p.values["alarm_count"] for p in res.series) == total
    assert any(p.values["alarm_count"] == 0 for p in res.series)  # empty days present


def test_trends_rejects_absurd_bucket_counts(svc):
    wide = m.TimeRange(start_time=datetime(2026, 4, 1, tzinfo=UTC), end_time=datetime(2026, 9, 28, tzinfo=UTC))
    with pytest.raises(ServiceError):
        svc.trends(m.TrendRequest(unit="Unit 2", time_range=wide, bucket=m.Bucket.hourly))


# ---- correlation ---------------------------------------------------------------------------
def test_chain03_compressor_correlation(svc):
    ids = [a.asset_id for a in svc.search_assets("compressor", limit=3).results]
    res = svc.correlation(m.CorrelationRequest(asset_ids=ids, time_range=WINDOW, min_support=1))
    top = next(p for p in res.pairs if p.alarm_name_a == "High Discharge Temperature"
               and p.alarm_name_b == "Low Lube Oil Pressure")
    assert top.support >= 4 and abs(top.avg_lag_seconds - 360) < 120
    assert set(res.correlated_asset_ids) >= {ids[0], ids[1]}


def test_correlation_within_one_asset_finds_bfp101_chain(svc):
    """The 'likely contributing factors' story: seal flush -> low suction -> bearing temp."""
    res = svc.correlation(m.CorrelationRequest(asset_ids=[asset_id(svc, BFP101)], time_range=WINDOW))
    chain = {(p.alarm_name_a, p.alarm_name_b) for p in res.pairs if p.support >= 5}
    assert ("Seal Flush Low Flow", "Low Suction Pressure") in chain


def test_correlation_severity_threshold_excludes_lower_severity(svc):
    req = m.CorrelationRequest(asset_ids=[asset_id(svc, BFP101)], time_range=WINDOW,
                               severity_threshold=m.Severity.high)
    assert all("Seal Flush Low Flow" not in (p.alarm_name_a, p.alarm_name_b) for p in svc.correlation(req).pairs)


def test_correlation_unknown_asset_is_404(svc):
    with pytest.raises(NotFoundError):
        svc.correlation(m.CorrelationRequest(asset_ids=["AST-9999"], time_range=WINDOW))


# ---- flood ---------------------------------------------------------------------------------------
def test_chain02_flood_windows_feed_alarm_listing(svc):
    res = svc.flood(m.FloodRequest(unit="Unit 2", time_range=WINDOW))
    assert len(res.flood_windows) >= 2 and res.flood_index > 0
    w = res.flood_windows[0]
    assert (w.start.month, w.start.day) == (5, 14) and w.alarm_count >= 10
    listed = svc.list_alarms(m.AlarmListQuery(unit="Unit 2", start_time=w.start, end_time=w.end, page_size=200))
    assert listed.pagination.total == w.alarm_count  # window retrieves exactly the flood


def test_flood_none_where_no_flood(svc):
    assert svc.flood(m.FloodRequest(unit="Unit 5", time_range=WINDOW)).flood_windows == []


# ---- rationalization ----------------------------------------------------------------------------
def _cands(res):
    return {(c.alarm_name, c.reason) for c in res.candidates}


def test_rationalization_finds_chattering_stale_and_recurring(svc):
    chatter = svc.rationalization(m.RationalizationRequest(unit="Unit 4", time_range=WINDOW, recurrence_threshold=8))
    assert ("Low Discharge Flow", "chattering") in _cands(chatter)
    stale = svc.rationalization(m.RationalizationRequest(site="NorthPlant", unit="Unit 1", time_range=WINDOW,
                                                         recurrence_threshold=6))
    assert ("High Winding Temperature", "stale") in _cands(stale)
    rec = svc.rationalization(m.RationalizationRequest(asset_ids=[asset_id(svc, BFP101)], time_range=WINDOW))
    assert ("Low Suction Pressure", "recurring") in _cands(rec)


# ---- priority + recommendations (CHAIN-09 core) --------------------------------------------
def test_chain09_k201_is_highest_priority_active_east_alarm(svc):
    active = svc.list_alarms(m.AlarmListQuery(site="EastRefinery", status=m.AlarmStatus.active)).data
    scores = {a.alarm_id: svc.priority_score(a.alarm_id) for a in active}
    best = max(active, key=lambda a: scores[a.alarm_id].score)
    assert (best.asset_name, best.alarm_name) == (K201, "High Vibration")
    top = scores[best.alarm_id]
    assert top.band == "critical"
    assert abs(sum(f.contribution for f in top.factors) - top.score) < 0.11


def test_priority_score_is_bounded_and_explained(svc):
    for a in svc.list_alarms(m.AlarmListQuery(page_size=200)).data[:50]:
        r = svc.priority_score(a.alarm_id)
        assert 0 <= r.score <= 100 and len(r.factors) == 5


def test_recommendations_full_context_for_k201(svc):
    alarm = svc.list_alarms(m.AlarmListQuery(site="EastRefinery", status=m.AlarmStatus.active,
                                             sort_by="severity")).data[0]
    r = svc.recommendations(m.RecommendationRequest(alarm_id=alarm.alarm_id, include_related=True,
                                                    include_asset_context=True, include_historical_pattern=True))
    assert r.likely_cause and [a.rank for a in r.actions] == [1, 2, 3]
    assert {a.asset_name for a in r.related_alarms} >= {K202, K203}
    assert r.asset_context.name == K201
    pre = {p["alarm_name"] for p in r.historical_pattern["common_precursors"]}
    assert "High Discharge Temperature" in pre and r.historical_pattern["occurrences_total"] >= 5


def test_recommendations_flags_off_returns_lean_response(svc):
    aid = svc.list_alarms(m.AlarmListQuery(page_size=1)).data[0].alarm_id
    r = svc.recommendations(m.RecommendationRequest(alarm_id=aid))
    assert r.related_alarms == [] and r.asset_context is None and r.historical_pattern is None


def test_every_seeded_alarm_name_has_domain_knowledge(svc):
    from alarm_api.services import KNOWLEDGE
    assert {a.alarm_name for a in svc.ds.alarms} <= set(KNOWLEDGE)


# ---- calculations (CHAIN-04/07/10) ------------------------------------------------------------
@pytest.mark.parametrize("ctype", list(m.CalculationType))
def test_generate_then_execute_each_calculation(svc, ctype):
    filters = m.CalculationFilters(unit="Unit 3", start_time=START, end_time=END)
    gen = svc.generate_calculation(m.GenerateCalculationRequest(calculation_type=ctype, filters=filters))
    assert gen.code and gen.calculation_id.startswith("CALC-")
    res = svc.execute_calculation(m.ExecuteCalculationRequest(calculation_id=gen.calculation_id, filters=filters))
    assert res.value >= 0 and res.unit_of_measure


def test_calculation_ids_are_deterministic(svc):
    req = m.GenerateCalculationRequest(calculation_type=m.CalculationType.nuisance_alarm_score)
    assert svc.generate_calculation(req).calculation_id == svc.generate_calculation(req).calculation_id


def test_execute_unknown_calculation_is_404(svc):
    with pytest.raises(NotFoundError):
        svc.execute_calculation(m.ExecuteCalculationRequest(calculation_id="CALC-NOPE"))


def test_nuisance_score_is_high_for_chattering_unit4(svc):
    f = m.CalculationFilters(unit="Unit 4", start_time=START, end_time=END)
    gen = svc.generate_calculation(m.GenerateCalculationRequest(
        calculation_type=m.CalculationType.nuisance_alarm_score, filters=f))
    assert svc.execute_calculation(m.ExecuteCalculationRequest(calculation_id=gen.calculation_id, filters=f)).value > 40


# ---- KPI definitions consistency ------------------------------------------------------------------
def test_every_summary_kpi_is_defined_and_computable(svc):
    defined = {k.name for k in svc.kpi_definitions().kpis}
    assert set(get_args(m.SummaryKpi)) <= defined
    res = svc.summary(m.SummaryRequest(unit="Unit 2", time_range=WINDOW, kpis=list(get_args(m.SummaryKpi))))
    assert set(res.totals) == set(get_args(m.SummaryKpi))