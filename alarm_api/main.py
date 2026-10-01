"""HTTP layer of the Alarm Management API simulator.

Thin shell over `AlarmService`: routing, bearer auth, trace-header propagation,
uniform error bodies, structured access logs and (opt-in) fault injection.

Run:    uvicorn alarm_api.main:app --port 8000
Config: ALARM_API_TOKEN (default demo-token, dev only), ALARM_API_SEED,
        ALARM_API_ENABLE_FAULTS (true|false, default false)

Fault injection (only when ALARM_API_ENABLE_FAULTS=true) exists so the MCP
server's retry/timeout logic can be tested against a real HTTP service:
    x-simulate-fault: 503        -> always answer 503
    x-simulate-fault: 503:2      -> answer 503 for the first 2 calls with the
                                    same trace_id + path, then behave normally
    x-simulate-delay-ms: 3000    -> sleep before answering (capped at 10 s)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import secrets
import time
import uuid
from dataclasses import dataclass
from typing import Annotated

from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from alarm_api import models as m
from alarm_api.seed import build_dataset
from alarm_api.services import AlarmService, ServiceError

DEFAULT_DEV_TOKEN = "demo-token"  # matches the Postman collections; NEVER use outside local dev

access_log = logging.getLogger("alarm_api.access")
if not access_log.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(message)s"))
    access_log.addHandler(_h)
    access_log.setLevel(logging.INFO)
    access_log.propagate = False


@dataclass(frozen=True)
class Settings:
    api_token: str = DEFAULT_DEV_TOKEN
    seed: int = 42
    enable_faults: bool = False

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            api_token=os.getenv("ALARM_API_TOKEN", DEFAULT_DEV_TOKEN),
            seed=int(os.getenv("ALARM_API_SEED", "42")),
            enable_faults=os.getenv("ALARM_API_ENABLE_FAULTS", "false").lower() in ("1", "true", "yes"),
        )


# --------------------------------------------------------------------------
# Errors: every non-2xx response has the same envelope (m.ErrorResponse)
# --------------------------------------------------------------------------
def _trace(request: Request) -> str | None:
    return getattr(request.state, "trace_id", None)


def _error(status: int, code: str, message: str, trace_id: str | None,
           details: list[dict] | None = None, headers: dict | None = None) -> JSONResponse:
    body = m.ErrorResponse(error=m.ErrorBody(code=code, message=message, details=details), trace_id=trace_id)
    return JSONResponse(status_code=status, content=body.model_dump(mode="json", exclude_none=True),
                        headers=headers)


# --------------------------------------------------------------------------
# Dependencies
# --------------------------------------------------------------------------
def require_auth(request: Request, authorization: Annotated[str | None, Header()] = None) -> None:
    scheme, _, token = (authorization or "").partition(" ")
    challenge = {"WWW-Authenticate": "Bearer"}
    if scheme.lower() != "bearer" or not token:
        raise HTTPException(401, "Missing or malformed Authorization header; expected 'Bearer <token>'",
                            headers=challenge)
    expected = request.app.state.settings.api_token
    if not secrets.compare_digest(token.encode(), expected.encode()):  # constant-time compare
        raise HTTPException(401, "Invalid access token", headers=challenge)


def get_service(request: Request) -> AlarmService:
    return request.app.state.service


Svc = Annotated[AlarmService, Depends(get_service)]
router = APIRouter(dependencies=[Depends(require_auth)])


# --------------------------------------------------------------------------
# Routes (paths and verbs exactly as in the Postman collections)
# --------------------------------------------------------------------------
@router.get("/assets/search", response_model=m.AssetSearchResponse, tags=["assets"])
def search_assets(svc: Svc, query: Annotated[str, Query(min_length=1, max_length=200)],
                  limit: Annotated[int, Query(ge=1, le=100)] = 10,
                  site: str | None = None, unit: str | None = None):
    return svc.search_assets(query, limit, site, unit)


@router.get("/assets/{asset_id}/metadata", response_model=m.AssetMetadata, tags=["assets"])
def asset_metadata(asset_id: str, svc: Svc):
    return svc.get_asset_metadata(asset_id)


@router.get("/alarms", response_model=m.AlarmListResponse, tags=["alarms"])
def list_alarms(q: Annotated[m.AlarmListQuery, Query()], svc: Svc):
    return svc.list_alarms(q)


@router.get("/alarms/{alarm_id}", response_model=m.Alarm, tags=["alarms"])
def get_alarm(alarm_id: str, svc: Svc):
    return svc.get_alarm(alarm_id)


@router.post("/alarms/summary", response_model=m.SummaryResponse, tags=["analytics"])
def alarm_summary(body: m.SummaryRequest, svc: Svc):
    return svc.summary(body)


@router.post("/alarms/trends", response_model=m.TrendResponse, tags=["analytics"])
def alarm_trends(body: m.TrendRequest, svc: Svc):
    return svc.trends(body)


@router.post("/alarms/correlation", response_model=m.CorrelationResponse, tags=["analytics"])
def alarm_correlation(body: m.CorrelationRequest, svc: Svc):
    return svc.correlation(body)


@router.post("/alarms/flood-analysis", response_model=m.FloodResponse, tags=["analytics"])
def flood_analysis(body: m.FloodRequest, svc: Svc):
    return svc.flood(body)


@router.post("/alarms/rationalization-candidates", response_model=m.RationalizationResponse, tags=["analytics"])
def rationalization(body: m.RationalizationRequest, svc: Svc):
    return svc.rationalization(body)


@router.post("/alarms/priority-score", response_model=m.PriorityScoreResponse, tags=["analytics"])
def priority_score(body: m.PriorityScoreRequest, svc: Svc):
    return svc.priority_score(body.alarm_id)


@router.post("/recommendations/operator-actions", response_model=m.RecommendationResponse, tags=["recommendations"])
def operator_actions(body: m.RecommendationRequest, svc: Svc):
    return svc.recommendations(body)


@router.post("/calculation-code/generate", response_model=m.GenerateCalculationResponse, tags=["calculations"])
def generate_calculation(body: m.GenerateCalculationRequest, svc: Svc):
    return svc.generate_calculation(body)


@router.post("/calculation-code/execute", response_model=m.ExecuteCalculationResponse, tags=["calculations"])
def execute_calculation(body: m.ExecuteCalculationRequest, svc: Svc):
    return svc.execute_calculation(body)


@router.get("/analytics/kpi-definitions", response_model=m.KpiDefinitionsResponse, tags=["analytics"])
def kpi_definitions(svc: Svc):
    return svc.kpi_definitions()


# --------------------------------------------------------------------------
# App factory
# --------------------------------------------------------------------------
def create_app(settings: Settings | None = None, service: AlarmService | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    app = FastAPI(title="Alarm Management API Simulator", version="1.0.0",
                  description="Simulated source system for the Incident and Ticket Enrichment Copilot.")
    app.state.settings = settings
    app.state.service = service or AlarmService(build_dataset(settings.seed))
    fault_counts: dict[tuple[str, str], int] = {}

    if settings.api_token == DEFAULT_DEV_TOKEN:
        access_log.warning(json.dumps({"event": "startup", "warning": "using default dev token"}))

    async def maybe_fault(request: Request, trace_id: str) -> JSONResponse | None:
        delay = request.headers.get("x-simulate-delay-ms", "")
        if delay.isdigit():
            await asyncio.sleep(min(int(delay), 10_000) / 1000)
        status_s, _, times_s = request.headers.get("x-simulate-fault", "").partition(":")
        if not (status_s.isdigit() and 400 <= int(status_s) <= 599):
            return None
        status, key = int(status_s), (trace_id, request.url.path)
        if times_s.isdigit() and fault_counts.get(key, 0) >= int(times_s):
            return None  # transient fault already served N times -> succeed
        if len(fault_counts) > 10_000:
            fault_counts.clear()
        fault_counts[key] = fault_counts.get(key, 0) + 1
        headers = {"Retry-After": "1"} if status in (429, 503) else None
        return _error(status, "simulated_fault", f"Simulated {status} response", trace_id, headers=headers)

    @app.middleware("http")
    async def trace_and_log(request: Request, call_next):
        started = time.perf_counter()
        trace_id = request.headers.get("trace_id") or str(uuid.uuid4())  # NB: underscore header
        request.state.trace_id = trace_id
        echo = {"trace_id": trace_id}
        for name in ("x-client-id", "x-metadata-tag"):
            if value := request.headers.get(name):
                echo[name] = value
        try:
            fault = None
            if settings.enable_faults and request.url.path != "/health":
                fault = await maybe_fault(request, trace_id)
            response = fault or await call_next(request)
        except Exception:  # last-resort guard: never leak a stack trace
            access_log.exception("unhandled error trace_id=%s", trace_id)
            response = _error(500, "internal_error", "Unexpected server error", trace_id)
        response.headers.update(echo)
        access_log.info(json.dumps({
            "event": "request", "trace_id": trace_id, "client_id": echo.get("x-client-id"),
            "metadata_tag": echo.get("x-metadata-tag"), "method": request.method, "path": request.url.path,
            "status": response.status_code, "duration_ms": round((time.perf_counter() - started) * 1000, 1),
        }))  # never logs headers/bodies, so tokens cannot leak
        return response

    @app.exception_handler(ServiceError)
    async def on_service_error(request: Request, exc: ServiceError):
        return _error(exc.status, exc.code, exc.message, _trace(request), exc.details)

    @app.exception_handler(RequestValidationError)
    async def on_validation_error(request: Request, exc: RequestValidationError):
        details = [{"loc": [str(p) for p in e["loc"]], "msg": e["msg"], "type": e["type"]} for e in exc.errors()]
        return _error(422, "validation_error", "Request validation failed", _trace(request), details)

    @app.exception_handler(StarletteHTTPException)
    async def on_http_error(request: Request, exc: StarletteHTTPException):
        code = {401: "unauthorized", 404: "not_found", 405: "method_not_allowed"}.get(exc.status_code, "http_error")
        return _error(exc.status_code, code, str(exc.detail), _trace(request), headers=exc.headers)

    @app.get("/health", response_model=m.HealthResponse, tags=["system"])
    def health():
        return m.HealthResponse()

    app.include_router(router)
    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("alarm_api.main:app", host="0.0.0.0", port=int(os.getenv("ALARM_API_PORT", "8000")))