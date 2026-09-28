"""Domain models for the Alarm Management API simulator.

Single source of truth for:
  * FastAPI request/response validation (simulator)
  * MCP tool input/output schemas (mcp server)
  * Test fixtures and contract tests

Only a handful of response fields are pinned by the Postman collections
(results[].asset_id, data[].alarm_id, flood_windows[].start/.end,
calculation_id). Everything else is our own design, kept minimal and explicit.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


# --------------------------------------------------------------------------
# Enums
# --------------------------------------------------------------------------
class Severity(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"
    critical = "critical"

    @property
    def rank(self) -> int:
        return ["low", "medium", "high", "critical"].index(self.value)


class AlarmStatus(str, Enum):
    active = "active"
    acknowledged = "acknowledged"
    cleared = "cleared"


class AlarmType(str, Enum):
    process = "process"
    safety = "safety"
    device = "device"
    diagnostic = "diagnostic"


class Bucket(str, Enum):
    hourly = "hourly"
    daily = "daily"
    weekly = "weekly"


class CalculationType(str, Enum):
    alarm_flood_index = "alarm_flood_index"
    critical_alarm_density = "critical_alarm_density"
    operator_response_efficiency = "operator_response_efficiency"
    nuisance_alarm_score = "nuisance_alarm_score"


# --------------------------------------------------------------------------
# Shared building blocks
# --------------------------------------------------------------------------
class ApiModel(BaseModel):
    """Base: reject unknown request fields so typos fail loudly."""

    model_config = ConfigDict(extra="forbid", use_enum_values=False)


class TimeRange(ApiModel):
    start_time: datetime
    end_time: datetime

    @model_validator(mode="after")
    def _ordered(self) -> "TimeRange":
        if self.end_time <= self.start_time:
            raise ValueError("end_time must be after start_time")
        return self


class Pagination(ApiModel):
    page: int
    page_size: int
    total: int
    total_pages: int


class ErrorBody(ApiModel):
    code: str  # e.g. "unauthorized", "not_found", "validation_error"
    message: str
    details: list[dict[str, Any]] | None = None


class ErrorResponse(ApiModel):
    error: ErrorBody
    trace_id: str | None = None


class TraceContext(ApiModel):
    """Headers the API must accept and echo. NOTE: header is `trace_id`
    (underscore) exactly as in the Postman collections."""

    trace_id: str | None = None
    x_client_id: str | None = None
    x_metadata_tag: str | None = None


# --------------------------------------------------------------------------
# Assets
# --------------------------------------------------------------------------
class AssetSummary(ApiModel):
    asset_id: str
    name: str
    asset_type: str  # compressor | pump | motor | boiler | valve ...
    site: str  # EastRefinery | NorthPlant | SouthPlant
    unit: str  # Unit 1 .. Unit 5


class AssetSearchResponse(ApiModel):
    results: list[AssetSummary]  # Postman reads results[0].asset_id
    total: int


class AssetMetadata(AssetSummary):
    manufacturer: str | None = None
    model: str | None = None
    commissioned_on: datetime | None = None
    criticality: Literal["A", "B", "C"] = "B"
    maintenance_group: str | None = None
    related_asset_ids: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------
# Alarms
# --------------------------------------------------------------------------
class Alarm(ApiModel):
    alarm_id: str
    asset_id: str
    asset_name: str
    site: str
    unit: str
    alarm_name: str
    alarm_type: AlarmType
    severity: Severity
    status: AlarmStatus
    description: str | None = None
    start_time: datetime
    acknowledged_at: datetime | None = None
    end_time: datetime | None = None  # None while still active


class AlarmListQuery(ApiModel):
    asset_id: str | None = None
    site: str | None = None
    unit: str | None = None
    status: AlarmStatus | None = None
    severity: Severity | None = None
    start_time: datetime | None = None
    end_time: datetime | None = None
    page: int = Field(1, ge=1)
    page_size: int = Field(50, ge=1, le=200)
    sort_by: Literal["start_time", "severity", "alarm_name"] = "start_time"
    sort_order: Literal["asc", "desc"] = "desc"


class AlarmListResponse(ApiModel):
    data: list[Alarm]  # Postman reads data[0].alarm_id
    pagination: Pagination


# --------------------------------------------------------------------------
# Scope shared by analytics endpoints (asset_ids OR site/unit)
# --------------------------------------------------------------------------
class ScopedRequest(ApiModel):
    asset_ids: list[str] | None = None
    site: str | None = None
    unit: str | None = None
    time_range: TimeRange


# --------------------------------------------------------------------------
# POST /alarms/summary
# --------------------------------------------------------------------------
GroupBy = Literal["alarm_name", "asset_id", "asset_name", "severity"]
SummaryKpi = Literal[
    "alarm_count",
    "critical_count",
    "recurring_rate",
    "avg_ack_delay",
    "suppression_candidate_rate",
]


class SummaryRequest(ScopedRequest):
    severity: list[Severity] | None = None
    alarm_types: list[AlarmType] | None = None
    group_by: list[GroupBy] = Field(default_factory=list)
    kpis: list[SummaryKpi] = Field(default_factory=lambda: ["alarm_count"])


class SummaryGroup(ApiModel):
    key: dict[str, str]  # {"alarm_name": "High Discharge Pressure"}
    metrics: dict[str, float]


class SummaryResponse(ApiModel):
    time_range: TimeRange
    groups: list[SummaryGroup]
    totals: dict[str, float]


# --------------------------------------------------------------------------
# POST /alarms/trends
# --------------------------------------------------------------------------
TrendMetric = Literal["alarm_count", "avg_ack_delay"]


class TrendRequest(ScopedRequest):
    bucket: Bucket = Bucket.daily
    metrics: list[TrendMetric] = Field(default_factory=lambda: ["alarm_count"])


class TrendPoint(ApiModel):
    bucket_start: datetime
    values: dict[str, float]


class TrendResponse(ApiModel):
    bucket: Bucket
    series: list[TrendPoint]


# --------------------------------------------------------------------------
# POST /alarms/correlation
# --------------------------------------------------------------------------
class CorrelationRequest(ApiModel):
    asset_ids: list[str] = Field(min_length=1)
    time_range: TimeRange
    correlation_method: Literal["cooccurrence"] = "cooccurrence"
    lag_window_minutes: int = Field(15, ge=1, le=240)
    severity_threshold: Severity = Severity.medium
    min_support: int = Field(1, ge=1)


class CorrelatedPair(ApiModel):
    asset_id_a: str
    alarm_name_a: str
    asset_id_b: str
    alarm_name_b: str
    support: int  # times the pair co-occurred within the lag window
    confidence: float = Field(ge=0, le=1)
    avg_lag_seconds: float


class CorrelationResponse(ApiModel):
    pairs: list[CorrelatedPair]
    correlated_asset_ids: list[str]


# --------------------------------------------------------------------------
# POST /alarms/flood-analysis
# --------------------------------------------------------------------------
class FloodRequest(ApiModel):
    unit: str | None = None
    site: str | None = None
    time_range: TimeRange
    threshold_count: int = Field(10, ge=1)
    rolling_window_minutes: int = Field(10, ge=1)


class FloodWindow(ApiModel):
    start: datetime  # Postman reads flood_windows[0].start / .end
    end: datetime
    alarm_count: int
    top_assets: list[str] = Field(default_factory=list)


class FloodResponse(ApiModel):
    flood_windows: list[FloodWindow]
    flood_index: float | None = None


# --------------------------------------------------------------------------
# POST /alarms/rationalization-candidates
# --------------------------------------------------------------------------
class RationalizationRequest(ApiModel):
    asset_ids: list[str] | None = None
    site: str | None = None
    unit: str | None = None
    time_range: TimeRange
    recurrence_threshold: int = Field(5, ge=1)
    stale_minutes_threshold: int = Field(180, ge=1)


class RationalizationCandidate(ApiModel):
    asset_id: str
    alarm_name: str
    reason: Literal["chattering", "stale", "recurring"]
    occurrences: int
    suggestion: str


class RationalizationResponse(ApiModel):
    candidates: list[RationalizationCandidate]


# --------------------------------------------------------------------------
# POST /alarms/priority-score
# --------------------------------------------------------------------------
class PriorityScoreRequest(ApiModel):
    alarm_id: str


class PriorityFactor(ApiModel):
    name: str  # severity | asset_criticality | recurrence | duration ...
    contribution: float


class PriorityScoreResponse(ApiModel):
    alarm_id: str
    score: float = Field(ge=0, le=100)
    band: Literal["low", "medium", "high", "critical"]
    factors: list[PriorityFactor]


# --------------------------------------------------------------------------
# POST /recommendations/operator-actions
# --------------------------------------------------------------------------
class RecommendationRequest(ApiModel):
    alarm_id: str
    include_related: bool = False
    include_asset_context: bool = False
    include_historical_pattern: bool = False


class RecommendedAction(ApiModel):
    rank: int
    action: str
    rationale: str
    confidence: float = Field(ge=0, le=1)


class RecommendationResponse(ApiModel):
    alarm_id: str
    likely_cause: str | None = None
    actions: list[RecommendedAction]
    related_alarms: list[Alarm] = Field(default_factory=list)
    asset_context: AssetMetadata | None = None
    historical_pattern: dict[str, Any] | None = None


# --------------------------------------------------------------------------
# POST /calculation-code/generate  and  /execute
# --------------------------------------------------------------------------
class CalculationFilters(ApiModel):
    site: str | None = None
    unit: str | None = None
    start_time: datetime | None = None
    end_time: datetime | None = None


class GenerateCalculationRequest(ApiModel):
    calculation_type: CalculationType
    filters: CalculationFilters = Field(default_factory=CalculationFilters)


class GenerateCalculationResponse(ApiModel):
    calculation_id: str  # Postman reads calculation_id
    calculation_type: CalculationType
    code: str  # descriptive, NEVER exec()'d - execute uses a whitelist
    description: str


class ExecuteCalculationRequest(ApiModel):
    calculation_id: str
    filters: CalculationFilters = Field(default_factory=CalculationFilters)


class ExecuteCalculationResponse(ApiModel):
    calculation_id: str
    value: float
    unit_of_measure: str
    details: dict[str, Any] = Field(default_factory=dict)


# --------------------------------------------------------------------------
# GET /analytics/kpi-definitions  and  GET /health
# --------------------------------------------------------------------------
class KpiDefinition(ApiModel):
    name: str
    description: str
    formula: str
    unit: str


class KpiDefinitionsResponse(ApiModel):
    kpis: list[KpiDefinition]


class HealthResponse(ApiModel):
    status: Literal["ok"] = "ok"
    version: str = "1.0.0"