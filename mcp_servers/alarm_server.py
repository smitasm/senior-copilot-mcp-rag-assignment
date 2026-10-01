"""MCP server wrapping the Alarm API simulator: 14 tools, one per endpoint.

Every tool reuses a model from alarm_api.models as its argument and/or return
type - the same models the simulator itself validates against (File 1) and
that AlarmService implements (File 4). That's the single source of truth
promise made back at the start of this project: nobody hand-writes a JSON
schema here, MCPServer generates it from the Pydantic model.

`build_alarm_server(client)` is a factory so tests can inject a SimulatorClient
whose transport is an in-process ASGI app (no real network, no real port).
The `__main__` block below is the only place that builds a real one.
"""

from __future__ import annotations

import os

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import BaseModel, Field

from alarm_api.models import (
    Alarm, AlarmListQuery, AlarmListResponse, AssetMetadata, AssetSearchResponse, CorrelationRequest,
    CorrelationResponse, ExecuteCalculationRequest, ExecuteCalculationResponse, FloodRequest, FloodResponse,
    GenerateCalculationRequest, GenerateCalculationResponse, KpiDefinitionsResponse, PriorityScoreResponse,
    RationalizationRequest, RationalizationResponse, RecommendationRequest, RecommendationResponse,
    SummaryRequest, SummaryResponse, TrendRequest, TrendResponse,
)
from mcp_common.http_client import ApiError, SimulatorClient


# --------------------------------------------------------------------------
# Small argument models for the endpoints alarm_api.models doesn't already
# cover (plain path/query params, not JSON bodies).
# --------------------------------------------------------------------------
class SearchAssetsRequest(BaseModel):
    query: str = Field(min_length=1, description="Free-text asset name or id, e.g. 'Boiler Feed Pump 101'.")
    limit: int = Field(10, ge=1, le=100)
    site: str | None = Field(None, description="e.g. EastRefinery, NorthPlant, SouthPlant")
    unit: str | None = Field(None, description="e.g. Unit 2")


class AssetIdRequest(BaseModel):
    asset_id: str = Field(description="Asset id from asset_search, e.g. AST-0001.")


class AlarmIdRequest(BaseModel):
    alarm_id: str = Field(description="Alarm id from list_alarms, e.g. ALM-000123.")


async def _call(coro):
    """Every tool routes its API call through here: an ApiError from the
    simulator (bad auth, unknown id, validation failure, exhausted retries...)
    becomes a ToolError, whose message the model actually gets to read -
    a bare exception here would be swallowed into a generic 'tool crashed'
    message (verified against the real MCP SDK before writing this)."""
    try:
        return await coro
    except ApiError as exc:
        raise ToolError(f"{exc.code}: {exc.message}") from exc


def build_alarm_server(client: SimulatorClient) -> MCPServer:
    srv = MCPServer(
        "alarm-management",
        instructions="Tools for querying alarms, assets, analytics and recommendations from the plant's "
                    "Alarm Management system. Read-only: no tool here creates or changes anything.",
    )

    @srv.tool(structured_output=True)
    async def search_assets(req: SearchAssetsRequest) -> AssetSearchResponse:
        """Find an asset by name or id. Use this first when the user names an
        asset (e.g. 'Boiler Feed Pump 101') to resolve it to an asset_id."""
        data = await _call(client.get("/assets/search", params=req.model_dump(mode="json", exclude_none=True)))
        return AssetSearchResponse.model_validate(data)

    @srv.tool(structured_output=True)
    async def get_asset_metadata(req: AssetIdRequest) -> AssetMetadata:
        """Get full metadata for one asset: manufacturer, criticality, maintenance group, related assets."""
        return AssetMetadata.model_validate(await _call(client.get(f"/assets/{req.asset_id}/metadata")))

    @srv.tool(structured_output=True)
    async def list_alarms(req: AlarmListQuery) -> AlarmListResponse:
        """List/filter/page alarms. NOTE: results are sorted by recency by
        default, NOT by priority - the first row is the newest alarm, not
        necessarily the most urgent one. Use priority_score to rank alarms."""
        data = await _call(client.get("/alarms", params=req.model_dump(mode="json", exclude_none=True)))
        return AlarmListResponse.model_validate(data)

    @srv.tool(structured_output=True)
    async def get_alarm(req: AlarmIdRequest) -> Alarm:
        """Get full detail for one alarm by its alarm_id."""
        return Alarm.model_validate(await _call(client.get(f"/alarms/{req.alarm_id}")))

    @srv.tool(structured_output=True)
    async def alarm_summary(req: SummaryRequest) -> SummaryResponse:
        """Aggregate alarm KPIs (count, critical_count, recurring_rate,
        avg_ack_delay, suppression_candidate_rate) over a time range, optionally
        grouped by alarm_name/asset/severity."""
        return SummaryResponse.model_validate(await _call(
            client.post("/alarms/summary", json_body=req.model_dump(mode="json", exclude_none=True))))

    @srv.tool(structured_output=True)
    async def alarm_trends(req: TrendRequest) -> TrendResponse:
        """Time-bucketed alarm counts/ack-delay over a time range (hourly/daily/weekly)."""
        return TrendResponse.model_validate(await _call(
            client.post("/alarms/trends", json_body=req.model_dump(mode="json", exclude_none=True))))

    @srv.tool(structured_output=True)
    async def correlate_alarms(req: CorrelationRequest) -> CorrelationResponse:
        """Find alarm pairs that tend to fire close together in time on the
        given assets - use this to find likely contributing/related alarms."""
        return CorrelationResponse.model_validate(await _call(
            client.post("/alarms/correlation", json_body=req.model_dump(mode="json", exclude_none=True))))

    @srv.tool(structured_output=True)
    async def flood_analysis(req: FloodRequest) -> FloodResponse:
        """Detect alarm-flood windows (many alarms in a short rolling window) for a unit or site."""
        return FloodResponse.model_validate(await _call(
            client.post("/alarms/flood-analysis", json_body=req.model_dump(mode="json", exclude_none=True))))

    @srv.tool(structured_output=True)
    async def rationalization_candidates(req: RationalizationRequest) -> RationalizationResponse:
        """Find alarms worth rationalizing: chattering (fires and clears
        repeatedly), stale (stays active for a long time), or recurring."""
        return RationalizationResponse.model_validate(await _call(
            client.post("/alarms/rationalization-candidates", json_body=req.model_dump(mode="json", exclude_none=True))))

    @srv.tool(structured_output=True)
    async def priority_score(req: AlarmIdRequest) -> PriorityScoreResponse:
        """Score one alarm's priority (0-100, banded low/medium/high/critical)
        with an explainable breakdown by factor. Use this to rank several
        active alarms instead of trusting list_alarms' default row order."""
        return PriorityScoreResponse.model_validate(await _call(
            client.post("/alarms/priority-score", json_body={"alarm_id": req.alarm_id})))

    @srv.tool(structured_output=True)
    async def operator_recommendations(req: RecommendationRequest) -> RecommendationResponse:
        """Get the likely cause and ranked recommended operator actions for
        one alarm. Set include_related/include_asset_context/
        include_historical_pattern=true to also get nearby related alarms,
        full asset context, and this alarm's own history (occurrence count,
        common precursor alarms) - the evidence an incident report should cite."""
        return RecommendationResponse.model_validate(await _call(
            client.post("/recommendations/operator-actions", json_body=req.model_dump(mode="json", exclude_none=True))))

    @srv.tool(structured_output=True)
    async def generate_calculation(req: GenerateCalculationRequest) -> GenerateCalculationResponse:
        """Define a named plant KPI calculation (e.g. alarm_flood_index,
        nuisance_alarm_score) for a scope/time filter. Returns a calculation_id
        to pass to execute_calculation - call this first, then that."""
        return GenerateCalculationResponse.model_validate(await _call(
            client.post("/calculation-code/generate", json_body=req.model_dump(mode="json", exclude_none=True))))

    @srv.tool(structured_output=True)
    async def execute_calculation(req: ExecuteCalculationRequest) -> ExecuteCalculationResponse:
        """Run a calculation_id from generate_calculation and get its numeric result."""
        return ExecuteCalculationResponse.model_validate(await _call(
            client.post("/calculation-code/execute", json_body=req.model_dump(mode="json", exclude_none=True))))

    @srv.tool(structured_output=True)
    async def kpi_definitions() -> KpiDefinitionsResponse:
        """List every KPI name this system knows, with its formula and unit."""
        return KpiDefinitionsResponse.model_validate(await _call(client.get("/analytics/kpi-definitions")))

    return srv


if __name__ == "__main__":
    sim_client = SimulatorClient(
        base_url=os.getenv("ALARM_API_URL", "http://localhost:8000"),
        token=os.getenv("ALARM_API_TOKEN", "demo-token"),
        client_id="alarm-mcp-server",
    )
    build_alarm_server(sim_client).run(transport="streamable-http")