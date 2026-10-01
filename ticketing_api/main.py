"""HTTP layer of the mock ticketing system.

Deliberately mirrors alarm_api/main.py's conventions (auth, trace_id
propagation, error envelope, fault injection) so the MCP client can treat
both simulators the same way. The two files are NOT unified behind a shared
helper yet - alarm_api/main.py already shipped and is under test, so this
duplicates ~100 lines rather than risk touching it. A shared `common/`
module is a reasonable hardening-phase cleanup, noted as known debt.

Run:    uvicorn ticketing_api.main:app --port 8001
Config: TICKETING_API_TOKEN (default demo-token, dev only), TICKETING_API_SEED,
        TICKETING_API_ENABLE_FAULTS (true|false)
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

from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from alarm_api.models import ErrorBody, ErrorResponse, HealthResponse
from alarm_api.seed import SEED_NOW, build_dataset
from alarm_api.services import ServiceError
from ticketing_api import models as m
from ticketing_api.seed import build_ticket_dataset
from ticketing_api.services import TicketService

DEFAULT_DEV_TOKEN = "demo-token"

access_log = logging.getLogger("ticketing_api.access")
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
            api_token=os.getenv("TICKETING_API_TOKEN", DEFAULT_DEV_TOKEN),
            seed=int(os.getenv("TICKETING_API_SEED", "42")),
            enable_faults=os.getenv("TICKETING_API_ENABLE_FAULTS", "false").lower() in ("1", "true", "yes"),
        )


def _trace(request: Request) -> str | None:
    return getattr(request.state, "trace_id", None)


def _error(status: int, code: str, message: str, trace_id: str | None,
          details: list[dict] | None = None, headers: dict | None = None) -> JSONResponse:
    body = ErrorResponse(error=ErrorBody(code=code, message=message, details=details), trace_id=trace_id)
    return JSONResponse(status_code=status, content=body.model_dump(mode="json", exclude_none=True),
                        headers=headers)


def require_auth(request: Request, authorization: Annotated[str | None, Header()] = None) -> None:
    scheme, _, token = (authorization or "").partition(" ")
    challenge = {"WWW-Authenticate": "Bearer"}
    if scheme.lower() != "bearer" or not token:
        raise HTTPException(401, "Missing or malformed Authorization header; expected 'Bearer <token>'",
                            headers=challenge)
    expected = request.app.state.settings.api_token
    if not secrets.compare_digest(token.encode(), expected.encode()):
        raise HTTPException(401, "Invalid access token", headers=challenge)


def get_service(request: Request) -> TicketService:
    return request.app.state.service


Svc = Annotated[TicketService, Depends(get_service)]
router = APIRouter(dependencies=[Depends(require_auth)])


@router.get("/tickets", response_model=m.TicketListResponse, tags=["tickets"])
def list_tickets(q: Annotated[m.TicketListQuery, Depends()], svc: Svc):
    return svc.list_tickets(q)


@router.get("/tickets/{ticket_id}", response_model=m.Ticket, tags=["tickets"])
def get_ticket(ticket_id: str, svc: Svc):
    return svc.get_ticket(ticket_id)


@router.post("/tickets", response_model=m.CreateTicketResponse, status_code=201, tags=["tickets"])
def create_ticket(body: m.CreateTicketRequest, svc: Svc):
    return svc.create_ticket(body)


@router.post("/tickets/similar", response_model=m.SimilarTicketsResponse, tags=["tickets"])
def similar_tickets(body: m.SimilarTicketsRequest, svc: Svc):
    return svc.similar_tickets(body)


def create_app(settings: Settings | None = None, service: TicketService | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    app = FastAPI(title="Ticketing System Simulator", version="1.0.0",
                 description="Mock ITSM ticketing system for the Incident and Ticket Enrichment Copilot.")
    app.state.settings = settings
    if service is None:
        assets = {a.asset_id: a for a in build_dataset(settings.seed).assets}
        service = TicketService(build_ticket_dataset(settings.seed), now=SEED_NOW, assets=assets)
    app.state.service = service
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
            return None
        if len(fault_counts) > 10_000:
            fault_counts.clear()
        fault_counts[key] = fault_counts.get(key, 0) + 1
        headers = {"Retry-After": "1"} if status in (429, 503) else None
        return _error(status, "simulated_fault", f"Simulated {status} response", trace_id, headers=headers)

    @app.middleware("http")
    async def trace_and_log(request: Request, call_next):
        started = time.perf_counter()
        trace_id = request.headers.get("trace_id") or str(uuid.uuid4())
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
        except Exception:
            access_log.exception("unhandled error trace_id=%s", trace_id)
            response = _error(500, "internal_error", "Unexpected server error", trace_id)
        response.headers.update(echo)
        access_log.info(json.dumps({
            "event": "request", "trace_id": trace_id, "client_id": echo.get("x-client-id"),
            "metadata_tag": echo.get("x-metadata-tag"), "method": request.method, "path": request.url.path,
            "status": response.status_code, "duration_ms": round((time.perf_counter() - started) * 1000, 1),
        }))
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

    @app.get("/health", response_model=HealthResponse, tags=["system"])
    def health():
        return HealthResponse()

    app.include_router(router)
    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("ticketing_api.main:app", host="0.0.0.0", port=int(os.getenv("TICKETING_API_PORT", "8001")))