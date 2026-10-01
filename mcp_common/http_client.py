"""Shared HTTP client the MCP servers use to call the Alarm API and Ticketing
API simulators over real HTTP (not in-process imports) - in Docker these are
separate containers, and testing the actual network boundary (auth headers,
retries, timeouts) matters more than testing a Python function call would.

Every logical call keeps ONE trace_id across all of its retry attempts, which
lets a test drive the simulators' own `x-simulate-fault: 503:N` fault
injection (built in alarm_api/main.py and ticketing_api/main.py) and assert
on retry behaviour against the real service, with no HTTP mocking library.
"""

from __future__ import annotations

import asyncio
import random
import uuid
from dataclasses import dataclass, field
from typing import Any

import httpx

RETRYABLE_STATUS = {429, 502, 503, 504}
DEFAULT_TIMEOUT_S = 10.0
DEFAULT_MAX_RETRIES = 3
DEFAULT_BACKOFF_BASE_S = 0.2
MAX_BACKOFF_S = 5.0


class ApiError(Exception):
    """A call to a simulator failed. `status`/`code`/`message`/`details` come
    straight from the simulator's ErrorResponse envelope when available;
    network-level failures (timeout, connection refused) get a synthetic
    status/code so callers can handle both the same way."""

    def __init__(self, status: int, code: str, message: str,
                details: list[dict] | None = None, trace_id: str | None = None) -> None:
        super().__init__(f"[{status} {code}] {message}")
        self.status, self.code, self.message = status, code, message
        self.details, self.trace_id = details, trace_id

    @property
    def is_retryable_family(self) -> bool:
        """Whether THIS status is normally retried - informational only; the
        client already retried what it could before raising."""
        return self.status in RETRYABLE_STATUS or self.status >= 500


@dataclass
class SimulatorClient:
    """One instance per target simulator (Alarm API, Ticketing API)."""

    base_url: str
    token: str
    client_id: str = "mcp-server"
    timeout_s: float = DEFAULT_TIMEOUT_S
    max_retries: int = DEFAULT_MAX_RETRIES
    backoff_base_s: float = DEFAULT_BACKOFF_BASE_S
    transport: httpx.AsyncBaseTransport | None = None  # override in tests (ASGITransport)
    _client: httpx.AsyncClient = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout_s,
                                         transport=self.transport)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "SimulatorClient":
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    # ----------------------------------------------------------------
    async def get(self, path: str, *, params: dict[str, Any] | None = None,
                  trace_id: str | None = None, metadata_tag: str | None = None) -> dict[str, Any]:
        return await self._request("GET", path, params=params, trace_id=trace_id, metadata_tag=metadata_tag)

    async def post(self, path: str, *, json_body: dict[str, Any] | None = None,
                   trace_id: str | None = None, metadata_tag: str | None = None) -> dict[str, Any]:
        return await self._request("POST", path, json_body=json_body, trace_id=trace_id, metadata_tag=metadata_tag)

    # ----------------------------------------------------------------
    async def _request(self, method: str, path: str, *, params: dict[str, Any] | None = None,
                       json_body: dict[str, Any] | None = None, trace_id: str | None = None,
                       metadata_tag: str | None = None) -> dict[str, Any]:
        trace_id = trace_id or str(uuid.uuid4())
        headers = {"Authorization": f"Bearer {self.token}", "trace_id": trace_id, "x-client-id": self.client_id}
        if metadata_tag:
            headers["x-metadata-tag"] = metadata_tag
        # drop None query params rather than sending "asset_id=None"
        clean_params = {k: v for k, v in (params or {}).items() if v is not None} if params else None

        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = await self._client.request(method, path, params=clean_params, json=json_body,
                                                   headers=headers)
            except (httpx.TimeoutException, httpx.ConnectError, httpx.ReadError) as exc:
                last_exc = exc
                if attempt < self.max_retries:
                    await asyncio.sleep(self._delay(attempt))
                    continue
                raise ApiError(504, "gateway_timeout", f"{type(exc).__name__} calling {method} {path}: {exc}",
                               trace_id=trace_id) from exc

            if resp.status_code < 400:
                return resp.json() if resp.content else {}

            body = self._safe_json(resp)
            err = (body or {}).get("error", {})
            code, message = err.get("code", "http_error"), err.get("message", resp.text[:500])
            if resp.status_code in RETRYABLE_STATUS and attempt < self.max_retries:
                await asyncio.sleep(self._retry_after(resp) or self._delay(attempt))
                continue
            raise ApiError(resp.status_code, code, message, err.get("details"),
                           body.get("trace_id", trace_id) if body else trace_id)

        raise ApiError(504, "gateway_timeout", f"Exhausted retries calling {method} {path}",
                       trace_id=trace_id) from last_exc

    def _delay(self, attempt: int) -> float:
        base = min(self.backoff_base_s * (2 ** attempt), MAX_BACKOFF_S)
        return base + random.uniform(0, base * 0.25)  # jitter, avoids retry storms

    @staticmethod
    def _retry_after(resp: httpx.Response) -> float | None:
        value = resp.headers.get("Retry-After")
        try:
            return float(value) if value is not None else None
        except ValueError:
            return None

    @staticmethod
    def _safe_json(resp: httpx.Response) -> dict | None:
        try:
            return resp.json()
        except ValueError:
            return None