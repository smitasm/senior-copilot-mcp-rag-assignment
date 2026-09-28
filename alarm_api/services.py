"""Business logic of the Alarm Management API simulator (no HTTP in here).

`AlarmService` answers every endpoint from an in-memory `Dataset`. Keeping it
free of FastAPI makes it trivially unit-testable; main.py is a thin HTTP shell.

Conventions
-----------
* Analytics time ranges are half-open: start <= alarm.start_time < end.
  GET /alarms filters are inclusive on both ends, so the [start, end] of a
  flood window returned by flood-analysis retrieves exactly that flood.
* Rates are percentages (0-100). Delays are seconds. Durations are minutes.
* "Recurring" = the same (asset, alarm_name) fired >= RECURRENCE_MIN times.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from typing import Callable

from alarm_api.models import (
    Alarm, AlarmListQuery, AlarmListResponse, AlarmStatus, AssetMetadata, AssetSearchResponse,
    AssetSummary, Bucket, CalculationFilters, CalculationType, CorrelatedPair, CorrelationRequest,
    CorrelationResponse, ExecuteCalculationRequest, ExecuteCalculationResponse, FloodRequest,
    FloodResponse, FloodWindow, GenerateCalculationRequest, GenerateCalculationResponse,
    KpiDefinition, KpiDefinitionsResponse, Pagination, PriorityFactor, PriorityScoreResponse,
    RationalizationCandidate, RationalizationRequest, RationalizationResponse, RecommendationRequest,
    RecommendationResponse, RecommendedAction, Severity, SummaryGroup, SummaryRequest,
    SummaryResponse, TimeRange, TrendPoint, TrendRequest, TrendResponse,
)
from alarm_api.seed import DATA_START, Dataset

UTC = timezone.utc
RECURRENCE_MIN = 5          # occurrences that make an alarm "recurring"
CHATTER_MAX_MINUTES = 5     # cleared this fast = chattering
STALE_MIN_OCCURRENCES = 3   # stale occurrences needed to flag a stale candidate
MAX_TREND_BUCKETS = 2000


# --------------------------------------------------------------------------
# Errors (main.py maps these to HTTP responses)
# --------------------------------------------------------------------------
class ServiceError(Exception):
    status, code = 400, "invalid_request"

    def __init__(self, message: str, details: list[dict] | None = None) -> None:
        super().__init__(message)
        self.message, self.details = message, details


class NotFoundError(ServiceError):
    status, code = 404, "not_found"


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------
def _aware(dt: datetime) -> datetime:
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _key(a: Alarm) -> tuple[str, str]:
    return (a.asset_id, a.alarm_name)


class AlarmService:
    def __init__(self, ds: Dataset) -> None:
        self.ds, self.now = ds, ds.now
        self._assets = {a.asset_id: a for a in ds.assets}
        self._alarms = {a.alarm_id: a for a in ds.alarms}
        self._calcs: dict[str, tuple[CalculationType, CalculationFilters]] = {}

    # ---- derived alarm properties --------------------------------------
    def _dur_min(self, a: Alarm) -> float:
        return ((a.end_time or self.now) - a.start_time).total_seconds() / 60

    def _is_chatter(self, a: Alarm) -> bool:
        return a.end_time is not None and self._dur_min(a) <= CHATTER_MAX_MINUTES

    def _is_stale(self, a: Alarm, threshold: float = 180) -> bool:
        return self._dur_min(a) > threshold

    @staticmethod
    def _ack_s(a: Alarm) -> float | None:
        return None if a.acknowledged_at is None else (a.acknowledged_at - a.start_time).total_seconds()

    # ---- selection ---------------------------------------------------------
    def _require_assets(self, ids: list[str]) -> None:
        missing = [i for i in ids if i not in self._assets]
        if missing:
            raise NotFoundError(f"Unknown asset id(s): {', '.join(missing)}", [{"asset_ids": missing}])

    def _select(self, *, asset_ids=None, site=None, unit=None, tr: TimeRange | None = None,
                severities=None, types=None) -> list[Alarm]:
        if asset_ids:
            self._require_assets(asset_ids)
        ids = set(asset_ids) if asset_ids else None
        lo, hi = (_aware(tr.start_time), _aware(tr.end_time)) if tr else (None, None)
        out = []
        for a in self.ds.alarms:
            if ids is not None and a.asset_id not in ids:
                continue
            if (site and a.site != site) or (unit and a.unit != unit):
                continue
            if tr and not (lo <= a.start_time < hi):
                continue
            if severities and a.severity not in severities:
                continue
            if types and a.alarm_type not in types:
                continue
            out.append(a)
        return out

    # ==================================================================
    # Assets
    # ==================================================================
    def search_assets(self, query: str, limit: int = 10, site: str | None = None,
                      unit: str | None = None) -> AssetSearchResponse:
        tokens = _tokens(query)
        if not tokens:
            raise ServiceError("query must contain at least one letter or digit")
        q = " ".join(query.lower().split())
        scored: list[tuple[float, AssetMetadata]] = []
        for a in self.ds.assets:
            if (site and a.site != site) or (unit and a.unit != unit):
                continue
            name = a.name.lower()
            if q in (name, a.asset_id.lower()):
                score = 100.0
            elif q in name:
                score = 80.0
            else:
                hay = set(_tokens(f"{a.name} {a.asset_type}"))
                hits = sum(t in hay for t in tokens)
                if not hits:
                    continue
                score = 20 + 40 * hits / len(tokens)
            scored.append((score, a))
        scored.sort(key=lambda x: (-x[0], x[1].asset_id))
        results = [AssetSummary(asset_id=a.asset_id, name=a.name, asset_type=a.asset_type,
                                site=a.site, unit=a.unit) for _, a in scored[:limit]]
        return AssetSearchResponse(results=results, total=len(scored))

    def get_asset_metadata(self, asset_id: str) -> AssetMetadata:
        if asset_id not in self._assets:
            raise NotFoundError(f"Asset '{asset_id}' not found")
        return self._assets[asset_id]

    # ==================================================================
    # Alarms
    # ==================================================================
    def list_alarms(self, q: AlarmListQuery) -> AlarmListResponse:
        lo = _aware(q.start_time) if q.start_time else None
        hi = _aware(q.end_time) if q.end_time else None
        rows = [a for a in self.ds.alarms
                if (not q.asset_id or a.asset_id == q.asset_id) and (not q.site or a.site == q.site)
                and (not q.unit or a.unit == q.unit) and (not q.status or a.status == q.status)
                and (not q.severity or a.severity == q.severity)
                and (not lo or a.start_time >= lo) and (not hi or a.start_time <= hi)]
        keyfn: dict[str, Callable[[Alarm], tuple]] = {
            "start_time": lambda a: (a.start_time, a.alarm_id),
            "severity": lambda a: (a.severity.rank, a.start_time, a.alarm_id),
            "alarm_name": lambda a: (a.alarm_name, a.start_time, a.alarm_id),
        }
        rows.sort(key=keyfn[q.sort_by], reverse=q.sort_order == "desc")
        total, begin = len(rows), (q.page - 1) * q.page_size
        return AlarmListResponse(
            data=rows[begin:begin + q.page_size],
            pagination=Pagination(page=q.page, page_size=q.page_size, total=total,
                                  total_pages=math.ceil(total / q.page_size)),
        )

    def get_alarm(self, alarm_id: str) -> Alarm:
        if alarm_id not in self._alarms:
            raise NotFoundError(f"Alarm '{alarm_id}' not found")
        return self._alarms[alarm_id]

    # ==================================================================
    # Summary / trends
    # ==================================================================
    def _metric(self, alarms: list[Alarm], name: str, recurring: set) -> float:
        n = len(alarms)
        if name == "alarm_count":
            return float(n)
        if name == "critical_count":
            return float(sum(a.severity == Severity.critical for a in alarms))
        if not n and name != "avg_ack_delay":
            return 0.0
        if name == "recurring_rate":
            return round(100 * sum(_key(a) in recurring for a in alarms) / n, 2)
        if name == "avg_ack_delay":
            delays = [d for a in alarms if (d := self._ack_s(a)) is not None]
            return round(sum(delays) / len(delays), 1) if delays else 0.0
        if name == "suppression_candidate_rate":
            return round(100 * sum(self._is_chatter(a) or self._is_stale(a) for a in alarms) / n, 2)
        raise ServiceError(f"Unknown KPI '{name}'")

    def summary(self, req: SummaryRequest) -> SummaryResponse:
        alarms = self._select(asset_ids=req.asset_ids, site=req.site, unit=req.unit, tr=req.time_range,
                              severities=set(req.severity or []), types=set(req.alarm_types or []))
        counts = Counter(_key(a) for a in alarms)
        recurring = {k for k, c in counts.items() if c >= RECURRENCE_MIN}
        groups: dict[tuple, list[Alarm]] = defaultdict(list)
        for a in alarms:
            keyvals = {"alarm_name": a.alarm_name, "asset_id": a.asset_id,
                       "asset_name": a.asset_name, "severity": a.severity.value}
            groups[tuple((g, keyvals[g]) for g in req.group_by)].append(a)
        out = [SummaryGroup(key=dict(k), metrics={m: self._metric(v, m, recurring) for m in req.kpis})
               for k, v in groups.items() if req.group_by]
        out.sort(key=lambda g: (-g.metrics.get("alarm_count", 0), sorted(g.key.items())))
        totals = {m: self._metric(alarms, m, recurring) for m in req.kpis}
        return SummaryResponse(time_range=req.time_range, groups=out, totals=totals)

    @staticmethod
    def _floor(dt: datetime, bucket: Bucket) -> datetime:
        if bucket == Bucket.hourly:
            return dt.replace(minute=0, second=0, microsecond=0)
        day = dt.replace(hour=0, minute=0, second=0, microsecond=0)
        return day - timedelta(days=day.weekday()) if bucket == Bucket.weekly else day

    def trends(self, req: TrendRequest) -> TrendResponse:
        step = {Bucket.hourly: timedelta(hours=1), Bucket.daily: timedelta(days=1),
                Bucket.weekly: timedelta(days=7)}[req.bucket]
        lo, hi = _aware(req.time_range.start_time), _aware(req.time_range.end_time)
        first = self._floor(lo, req.bucket)
        n_buckets = math.ceil((hi - first) / step)
        if n_buckets > MAX_TREND_BUCKETS:
            raise ServiceError(f"{n_buckets} buckets requested (max {MAX_TREND_BUCKETS}); use a coarser bucket "
                               "or a shorter time_range")
        alarms = self._select(asset_ids=req.asset_ids, site=req.site, unit=req.unit, tr=req.time_range)
        by_bucket: dict[datetime, list[Alarm]] = defaultdict(list)
        for a in alarms:
            by_bucket[self._floor(a.start_time, req.bucket)].append(a)
        series = [TrendPoint(bucket_start=(t := first + i * step),
                             values={m: self._metric(by_bucket.get(t, []), m, set()) for m in req.metrics})
                  for i in range(n_buckets)]
        return TrendResponse(bucket=req.bucket, series=series)

    # ==================================================================
    # Correlation / flood / rationalization
    # ==================================================================
    def correlation(self, req: CorrelationRequest) -> CorrelationResponse:
        keep = {s for s in Severity if s.rank >= req.severity_threshold.rank}
        events = sorted(self._select(asset_ids=req.asset_ids, tr=req.time_range, severities=keep),
                        key=lambda a: a.start_time)
        occurrences = Counter(_key(a) for a in events)
        window = timedelta(minutes=req.lag_window_minutes)
        support: Counter = Counter()
        lags: dict[tuple, list[float]] = defaultdict(list)
        for i, first in enumerate(events):
            seen: set = set()
            for second in events[i + 1:]:
                lag = second.start_time - first.start_time
                if lag > window:
                    break
                pair = (_key(first), _key(second))
                if lag <= timedelta(0) or pair[1] == pair[0] or pair[1] in seen:
                    continue
                seen.add(pair[1])
                support[pair] += 1
                lags[pair].append(lag.total_seconds())
        pairs = [CorrelatedPair(asset_id_a=a[0], alarm_name_a=a[1], asset_id_b=b[0], alarm_name_b=b[1],
                                support=s, confidence=round(min(1.0, s / occurrences[a]), 3),
                                avg_lag_seconds=round(sum(lags[(a, b)]) / s, 1))
                 for (a, b), s in support.items() if s >= req.min_support]
        pairs.sort(key=lambda p: (-p.support, -p.confidence, p.asset_id_a, p.alarm_name_a))
        cross = sorted({x for p in pairs if p.asset_id_a != p.asset_id_b for x in (p.asset_id_a, p.asset_id_b)})
        return CorrelationResponse(pairs=pairs, correlated_asset_ids=cross)

    def _flood_windows(self, alarms: list[Alarm], threshold: int, minutes: int) -> list[FloodWindow]:
        alarms = sorted(alarms, key=lambda a: a.start_time)
        span, lo, hits = timedelta(minutes=minutes), 0, []
        for hi, a in enumerate(alarms):
            while a.start_time - alarms[lo].start_time > span:
                lo += 1
            if hi - lo + 1 >= threshold:
                hits.append([alarms[lo].start_time, a.start_time])
        merged: list[list[datetime]] = []
        for s, e in hits:
            if merged and s <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], e)
            else:
                merged.append([s, e])
        windows = []
        for s, e in merged:
            inside = [a for a in alarms if s <= a.start_time <= e]
            top = [k for k, _ in Counter(a.asset_id for a in inside).most_common(3)]
            windows.append(FloodWindow(start=s, end=e, alarm_count=len(inside), top_assets=top))
        return windows

    def flood(self, req: FloodRequest) -> FloodResponse:
        alarms = self._select(site=req.site, unit=req.unit, tr=req.time_range)
        windows = self._flood_windows(alarms, req.threshold_count, req.rolling_window_minutes)
        in_flood = sum(w.alarm_count for w in windows)
        return FloodResponse(flood_windows=windows,
                             flood_index=round(100 * in_flood / len(alarms), 2) if alarms else 0.0)

    def rationalization(self, req: RationalizationRequest) -> RationalizationResponse:
        alarms = self._select(asset_ids=req.asset_ids, site=req.site, unit=req.unit, tr=req.time_range)
        grouped: dict[tuple, list[Alarm]] = defaultdict(list)
        for a in alarms:
            grouped[_key(a)].append(a)
        advice = {
            "chattering": "Add an on-delay or deadband, or filter the signal; review the alarm setpoint.",
            "stale": "Alarm stays active for hours: confirm it is actionable, else re-classify or remove.",
            "recurring": "Investigate the root cause and review priority; consider a repeat-alarm suppression rule.",
        }
        out = []
        for (aid, name), items in grouped.items():
            short = [a for a in items if self._is_chatter(a)]
            stale = [a for a in items if self._is_stale(a, req.stale_minutes_threshold)]
            if len(short) >= req.recurrence_threshold:
                reason, n = "chattering", len(short)
            elif len(stale) >= STALE_MIN_OCCURRENCES:
                reason, n = "stale", len(stale)
            elif len(items) >= req.recurrence_threshold:
                reason, n = "recurring", len(items)
            else:
                continue
            out.append(RationalizationCandidate(asset_id=aid, alarm_name=name, reason=reason,
                                                occurrences=n, suggestion=advice[reason]))
        out.sort(key=lambda c: (-c.occurrences, c.asset_id, c.alarm_name))
        return RationalizationResponse(candidates=out)

    # ==================================================================
    # Priority score / recommendations
    # ==================================================================
    def priority_score(self, alarm_id: str) -> PriorityScoreResponse:
        a = self.get_alarm(alarm_id)
        asset = self._assets[a.asset_id]
        sev = {Severity.low: 8, Severity.medium: 20, Severity.high: 35, Severity.critical: 50}[a.severity]
        crit = {"A": 20, "B": 12, "C": 5}[asset.criticality]
        prior = sum(1 for h in self.ds.alarms if _key(h) == _key(a)
                    and a.start_time - timedelta(days=30) <= h.start_time < a.start_time)
        factors = [
            PriorityFactor(name="severity", contribution=float(sev)),
            PriorityFactor(name="asset_criticality", contribution=float(crit)),
            PriorityFactor(name="recurrence_30d", contribution=round(min(15.0, prior * 1.5), 1)),
            PriorityFactor(name="duration", contribution=round(min(10.0, self._dur_min(a) / 60 * 2), 1)),
            PriorityFactor(name="unacknowledged", contribution=5.0 if a.status == AlarmStatus.active else 0.0),
        ]
        score = round(min(100.0, sum(f.contribution for f in factors)), 1)
        band = "critical" if score >= 75 else "high" if score >= 55 else "medium" if score >= 35 else "low"
        return PriorityScoreResponse(alarm_id=alarm_id, score=score, band=band, factors=factors)

    def recommendations(self, req: RecommendationRequest) -> RecommendationResponse:
        a = self.get_alarm(req.alarm_id)
        asset = self._assets[a.asset_id]
        cause, steps = KNOWLEDGE.get(a.alarm_name, (None, [
            "Verify the instrument reading locally.", "Check recent changes and operating conditions.",
            "Escalate to the area engineer if the condition persists."]))
        actions = [RecommendedAction(rank=i, action=s, rationale=f"Standard response for '{a.alarm_name}'.",
                                     confidence=round(0.85 - 0.15 * (i - 1), 2))
                   for i, s in enumerate(steps, start=1)]
        resp = RecommendationResponse(alarm_id=a.alarm_id, likely_cause=cause, actions=actions)
        scope = {asset.asset_id, *asset.related_asset_ids}
        if req.include_related:
            near = [x for x in self.ds.alarms if x.asset_id in scope and x.alarm_id != a.alarm_id
                    and abs((x.start_time - a.start_time).total_seconds()) <= 3600]
            resp.related_alarms = sorted(near, key=lambda x: x.start_time)[:10]
        if req.include_asset_context:
            resp.asset_context = asset
        if req.include_historical_pattern:
            resp.historical_pattern = self._pattern(a, scope)
        return resp

    def _pattern(self, a: Alarm, scope: set[str]) -> dict:
        hist = [h for h in self.ds.alarms if _key(h) == _key(a) and h.start_time < a.start_time]
        recent = [h for h in hist if h.start_time >= self.now - timedelta(days=90)]
        delays = [d for h in hist if (d := self._ack_s(h)) is not None]
        pre: Counter = Counter()
        for h in hist:
            for x in self.ds.alarms:
                if x.asset_id in scope and h.start_time - timedelta(minutes=30) <= x.start_time < h.start_time:
                    pre[(x.asset_id, x.asset_name, x.alarm_name)] += 1
        return {
            "occurrences_total": len(hist), "occurrences_last_90d": len(recent),
            "first_seen": hist[0].start_time.isoformat() if hist else None,
            "last_seen": hist[-1].start_time.isoformat() if hist else None,
            "avg_duration_minutes": round(sum(self._dur_min(h) for h in hist) / len(hist), 1) if hist else None,
            "avg_ack_delay_seconds": round(sum(delays) / len(delays), 1) if delays else None,
            "common_precursors": [{"asset_id": k[0], "asset_name": k[1], "alarm_name": k[2], "count": c}
                                  for k, c in pre.most_common(3)],
        }

    # ==================================================================
    # Calculation code (generate is descriptive; execute is a WHITELIST -
    # generated code text is never eval'd/exec'd)
    # ==================================================================
    def generate_calculation(self, req: GenerateCalculationRequest) -> GenerateCalculationResponse:
        payload = json.dumps([req.calculation_type.value, req.filters.model_dump(mode="json")], sort_keys=True)
        cid = "CALC-" + hashlib.sha1(payload.encode()).hexdigest()[:10].upper()
        self._calcs[cid] = (req.calculation_type, req.filters)
        desc, code = CALC_TEXT[req.calculation_type]
        return GenerateCalculationResponse(calculation_id=cid, calculation_type=req.calculation_type,
                                           code=code, description=desc)

    def execute_calculation(self, req: ExecuteCalculationRequest) -> ExecuteCalculationResponse:
        if req.calculation_id not in self._calcs:
            raise NotFoundError(f"Calculation '{req.calculation_id}' not found; call /calculation-code/generate first")
        ctype, stored = self._calcs[req.calculation_id]
        f = req.filters if req.filters.model_dump(exclude_none=True) else stored
        tr = TimeRange(start_time=f.start_time or DATA_START, end_time=f.end_time or self.now)
        alarms = self._select(site=f.site, unit=f.unit, tr=tr)
        runners: dict[CalculationType, Callable[[list[Alarm], TimeRange, CalculationFilters], tuple]] = {
            CalculationType.alarm_flood_index: self._calc_flood,
            CalculationType.critical_alarm_density: self._calc_critical_density,
            CalculationType.operator_response_efficiency: self._calc_response,
            CalculationType.nuisance_alarm_score: self._calc_nuisance,
        }
        value, unit, details = runners[ctype](alarms, tr, f)
        return ExecuteCalculationResponse(calculation_id=req.calculation_id, value=value,
                                          unit_of_measure=unit, details=details)

    def _calc_flood(self, alarms, tr, f):
        wins = self._flood_windows(alarms, 10, 10)
        n = sum(w.alarm_count for w in wins)
        return (round(100 * n / len(alarms), 2) if alarms else 0.0, "percent",
                {"flood_windows": len(wins), "alarms_in_floods": n, "total_alarms": len(alarms)})

    def _calc_critical_density(self, alarms, tr, f):
        days = max((_aware(tr.end_time) - _aware(tr.start_time)).total_seconds() / 86400, 1e-9)
        crit = sum(a.severity == Severity.critical for a in alarms)
        return (round(crit / days, 3), "critical_alarms_per_day",
                {"critical_alarms": crit, "total_alarms": len(alarms), "days": round(days, 1)})

    def _calc_response(self, alarms, tr, f):
        delays = [d for a in alarms if (d := self._ack_s(a)) is not None]
        fast = sum(d <= 600 for d in delays)
        return (round(100 * fast / len(alarms), 2) if alarms else 0.0, "percent_acked_within_10_min",
                {"avg_ack_delay_seconds": round(sum(delays) / len(delays), 1) if delays else None,
                 "unacknowledged": len(alarms) - len(delays), "total_alarms": len(alarms)})

    def _calc_nuisance(self, alarms, tr, f):
        chat = sum(self._is_chatter(a) for a in alarms)
        stale = sum(self._is_stale(a) for a in alarms)
        return (round(100 * (chat + stale) / len(alarms), 2) if alarms else 0.0, "percent",
                {"chattering": chat, "stale": stale, "total_alarms": len(alarms)})

    def kpi_definitions(self) -> KpiDefinitionsResponse:
        return KpiDefinitionsResponse(kpis=KPI_DEFINITIONS)


# --------------------------------------------------------------------------
# Static reference data
# --------------------------------------------------------------------------
KPI_DEFINITIONS = [
    KpiDefinition(name="alarm_count", description="Number of alarms in scope.", formula="count(alarms)", unit="count"),
    KpiDefinition(name="critical_count", description="Number of critical-severity alarms.",
                  formula="count(alarms where severity = critical)", unit="count"),
    KpiDefinition(name="recurring_rate", description="Share of alarms whose (asset, alarm_name) fired at least "
                  f"{RECURRENCE_MIN} times in the window.", formula="100 * recurring_alarms / total_alarms", unit="percent"),
    KpiDefinition(name="avg_ack_delay", description="Mean time from alarm start to acknowledgement.",
                  formula="mean(acknowledged_at - start_time)", unit="seconds"),
    KpiDefinition(name="suppression_candidate_rate", description="Share of alarms that are chattering (cleared within "
                  f"{CHATTER_MAX_MINUTES} min) or stale (active > 180 min).",
                  formula="100 * (chattering + stale) / total_alarms", unit="percent"),
    KpiDefinition(name="flood_index", description="Share of alarms raised inside alarm-flood windows.",
                  formula="100 * alarms_in_floods / total_alarms", unit="percent"),
    KpiDefinition(name="alarm_flood_index", description="Calculation: flood_index with 10 alarms / 10 minutes.",
                  formula="100 * alarms_in_floods / total_alarms", unit="percent"),
    KpiDefinition(name="critical_alarm_density", description="Critical alarms per day in scope.",
                  formula="critical_alarms / days", unit="critical_alarms_per_day"),
    KpiDefinition(name="operator_response_efficiency", description="Share of alarms acknowledged within 10 minutes.",
                  formula="100 * acked_within_600s / total_alarms", unit="percent"),
    KpiDefinition(name="nuisance_alarm_score", description="Share of alarms that are chattering or stale.",
                  formula="100 * (chattering + stale) / total_alarms", unit="percent"),
]

CALC_TEXT: dict[CalculationType, tuple[str, str]] = {
    CalculationType.alarm_flood_index: (
        "Percentage of alarms raised during flood windows (>=10 alarms in 10 min).",
        "flood_alarms = sum(w.alarm_count for w in flood_windows(alarms, 10, 10))\nreturn 100 * flood_alarms / len(alarms)"),
    CalculationType.critical_alarm_density: (
        "Critical alarms per day for the selected unit/site.",
        "critical = [a for a in alarms if a.severity == 'critical']\nreturn len(critical) / days"),
    CalculationType.operator_response_efficiency: (
        "Percentage of alarms acknowledged within 10 minutes.",
        "fast = [a for a in alarms if a.ack_delay is not None and a.ack_delay <= 600]\nreturn 100 * len(fast) / len(alarms)"),
    CalculationType.nuisance_alarm_score: (
        "Percentage of alarms that are chattering (<=5 min) or stale (>180 min).",
        "nuisance = [a for a in alarms if a.duration <= 5 or a.duration > 180]\nreturn 100 * len(nuisance) / len(alarms)"),
}

# alarm_name -> (likely cause, [ranked operator actions])
KNOWLEDGE: dict[str, tuple[str, list[str]]] = {
    "High Vibration": ("Rotor imbalance, bearing wear or lube-oil degradation on the rotating equipment.",
                       ["Confirm the reading on the local vibration monitor and check trend direction.",
                        "Verify lube oil pressure and temperature; reduce load if vibration is rising.",
                        "Prepare for controlled shutdown per the vibration trip procedure and notify mechanical."]),
    "High Discharge Temperature": ("Insufficient cooling, fouled intercooler or excessive compression ratio.",
                                   ["Check cooling water flow and intercooler outlet temperature.",
                                    "Review recycle/anti-surge valve position and suction conditions.",
                                    "Reduce throughput if temperature keeps rising."]),
    "Low Lube Oil Pressure": ("Lube oil pump degradation, filter blockage or oil cooler issue.",
                              ["Check standby lube oil pump auto-start and header pressure locally.",
                               "Inspect lube oil filter differential pressure.",
                               "Prepare to trip the machine if pressure falls to the trip setpoint."]),
    "Suction Pressure Low": ("Upstream supply shortfall or a closed/throttled suction valve.",
                             ["Check upstream supply and suction valve positions.",
                              "Review the anti-surge recycle valve for excessive opening.",
                              "Coordinate with the adjacent unit operator on supply."]),
    "Anti-Surge Valve Open": ("Operation near the surge line or a sticking anti-surge controller.",
                              ["Check compressor operating point against the surge map.",
                               "Verify anti-surge controller output and valve stroke.",
                               "Adjust load to move away from the surge line."]),
    "Low Suction Pressure": ("Inadequate NPSH from a low suction level, blocked strainer or upstream valve restriction.",
                             ["Check suction vessel level and strainer differential pressure.",
                              "Confirm suction valve is fully open.",
                              "Start the standby pump if suction pressure keeps falling."]),
    "High Bearing Temperature": ("Lubrication loss, misalignment or seal-flush failure heating the bearing.",
                                 ["Check bearing lube supply and cooling water to the bearing housing.",
                                  "Verify the seal flush flow.",
                                  "Switch to the standby pump if the temperature approaches the trip limit."]),
    "Seal Flush Low Flow": ("Blocked seal flush line or strainer, or a closed flush valve.",
                            ["Check the seal flush line valves and strainer.",
                             "Inspect the flush flow indicator locally.",
                             "Schedule seal inspection if the low flow recurs."]),
    "Pump Trip": ("Protective interlock activation (low suction, high bearing temperature or motor fault).",
                  ["Identify the first-out trip cause before any restart attempt.",
                   "Start the standby pump to restore flow.",
                   "Raise a maintenance ticket for the trip investigation."]),
    "Low Discharge Flow": ("Downstream restriction, partially closed control valve or pump wear.",
                           ["Confirm the flow transmitter reading against the valve position.",
                            "Check for a downstream blockage.",
                            "If it chatters repeatedly, review the setpoint and alarm deadband."]),
    "High Winding Temperature": ("Overloading, blocked ventilation or a failed cooling fan.",
                                 ["Check motor load current against the nameplate.",
                                  "Inspect cooling airflow and fan operation.",
                                  "Reduce load or start the standby drive."]),
    "Overload Trip": ("Sustained overcurrent from a mechanical jam, process overload or supply imbalance.",
                      ["Inspect the driven equipment for mechanical binding.",
                       "Check supply voltage and phase balance.",
                       "Do not reset more than once before electrical checks are completed."]),
    "Motor Fail To Start": ("Interlock not permissive, breaker fault or a control circuit failure.",
                            ["Check start permissives and breaker status.",
                             "Inspect the control circuit and fuses.",
                             "Start the standby motor and raise an electrical fault ticket."]),
    "Low Drum Level": ("Feedwater supply loss or a level-control malfunction.",
                       ["Check feedwater pump status and flow immediately.",
                        "Verify level transmitters against the gauge glass.",
                        "Follow the boiler low-level trip procedure and notify the shift supervisor."]),
    "High Steam Pressure": ("Reduced steam demand or a pressure-control valve fault.",
                            ["Check the steam header pressure controller output.",
                             "Confirm relief valve readiness.",
                             "Reduce firing rate."]),
    "Flame Failure": ("Fuel supply interruption, dirty scanner or an unstable burner.",
                      ["Confirm the burner management system trip state.",
                       "Purge before any relight attempt per the burner procedure.",
                       "Inspect the flame scanner and fuel supply."]),
    "Feedwater Flow Low": ("Feedwater pump degradation or a restricted feed line.",
                           ["Check the running feedwater pump discharge pressure.",
                            "Start the standby feedwater pump.",
                            "Inspect the feedwater control valve."]),
    "High Outlet Temperature": ("Excess firing or reduced process flow through the equipment.",
                                ["Check the firing rate and process flow.",
                                 "Verify temperature transmitter accuracy.",
                                 "Reduce firing or increase flow."]),
    "Low Fuel Gas Pressure": ("Fuel supply header disturbance or a plugged fuel gas filter.",
                              ["Check the fuel gas header pressure and the filter differential pressure.",
                               "Confirm the pressure control valve position.",
                               "Prepare for burner trip if the pressure keeps falling."]),
    "Burner Flame Unstable": ("Air/fuel ratio imbalance or a fouled burner tip.",
                              ["Check the air/fuel ratio and the oxygen trim.",
                               "Inspect burner tips and the scanner signal.",
                               "Schedule burner cleaning."]),
    "Low Deaerator Level": ("Insufficient makeup water or excessive feedwater draw.",
                            ["Check makeup water valve operation.", "Compare level transmitters.",
                             "Reduce feedwater demand if possible."]),
    "High Deaerator Pressure": ("Excess steam supply or a vent restriction.",
                                ["Check steam supply to the deaerator.", "Verify the vent is open.",
                                 "Monitor until pressure returns to normal."]),
    "Fouling Indicator High": ("Gradual deposit build-up on heat-transfer surfaces.",
                               ["Trend the fouling factor against the last cleaning.",
                                "Plan a cleaning at the next opportunity.",
                                "No immediate operator action required."]),
}