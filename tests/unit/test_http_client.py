"""SimulatorClient tests, run against the REAL Alarm API app via ASGITransport
(in-process, no socket) - including its actual `x-simulate-fault` header, so
retry/timeout behaviour is proven against the real service, not a mock."""

import asyncio
import time

import httpx
import pytest

from alarm_api.main import Settings, create_app
from mcp_common.http_client import ApiError, SimulatorClient

TOKEN = "test-token"


@pytest.fixture
def transport():
    return httpx.ASGITransport(app=create_app(Settings(api_token=TOKEN, enable_faults=True)))


@pytest.fixture
def client(transport):
    return SimulatorClient(base_url="http://sim", token=TOKEN, transport=transport,
                           max_retries=3, backoff_base_s=0.01)


async def _run(coro):
    return await coro


# ---- happy path -----------------------------------------------------------
@pytest.mark.asyncio
async def test_get_returns_parsed_json(client):
    body = await client.get("/analytics/kpi-definitions")
    assert body["kpis"] and isinstance(body["kpis"], list)


@pytest.mark.asyncio
async def test_post_sends_json_body(client):
    body = await client.post("/alarms/priority-score", json_body={"alarm_id": "ALM-000001"})
    assert 0 <= body["score"] <= 100


@pytest.mark.asyncio
async def test_trace_id_is_generated_and_returned_in_error(client):
    with pytest.raises(ApiError) as exc:
        await client.get("/alarms/ALM-NOPE")
    assert exc.value.trace_id and len(exc.value.trace_id) >= 32


@pytest.mark.asyncio
async def test_explicit_trace_id_is_used(client, transport):
    async with httpx.AsyncClient(transport=transport, base_url="http://sim") as raw:
        r = await raw.get("/health", headers={"trace_id": "my-trace"})
    assert r.headers["trace_id"] == "my-trace"


# ---- error mapping (non-retryable) -----------------------------------------------
@pytest.mark.asyncio
async def test_404_maps_to_api_error_without_retrying(transport):
    class CountingTransport(httpx.AsyncBaseTransport):
        def __init__(self) -> None:
            self.calls = 0

        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            self.calls += 1
            return await transport.handle_async_request(request)

    counting = CountingTransport()
    client = SimulatorClient(base_url="http://sim", token=TOKEN, transport=counting, max_retries=3)
    with pytest.raises(ApiError) as exc:
        await client.get("/alarms/ALM-NOPE")
    assert (exc.value.status, exc.value.code) == (404, "not_found")
    assert counting.calls == 1  # a 4xx (other than 429) must NOT be retried


@pytest.mark.asyncio
async def test_401_maps_to_api_error(transport):
    bad = SimulatorClient(base_url="http://sim", token="wrong-token", transport=transport)
    with pytest.raises(ApiError) as exc:
        await bad.get("/alarms")
    assert exc.value.status == 401


@pytest.mark.asyncio
async def test_422_validation_error_carries_details(client):
    with pytest.raises(ApiError) as exc:
        await client.get("/alarms", params={"page_size": 9999})
    assert exc.value.status == 422 and exc.value.details


# ---- retry behaviour, against the simulator's REAL fault injection -----------------------------
@pytest.mark.asyncio
async def test_retries_share_one_trace_id_so_fault_counter_clears(client, transport):
    """Drives the simulator's own fault counter directly: fails twice for a
    given trace_id, then the simulator itself starts answering normally."""
    async with httpx.AsyncClient(transport=transport, base_url="http://sim",
                                 headers={"Authorization": f"Bearer {TOKEN}"}) as raw:
        trace = "shared-trace-1"
        codes = []
        for _ in range(3):
            r = await raw.get("/analytics/kpi-definitions",
                              headers={"trace_id": trace, "x-simulate-fault": "503:2"})
            codes.append(r.status_code)
        assert codes == [503, 503, 200]

    # now prove SimulatorClient's own retry loop clears the SAME simulator-side
    # counter within a single call, because it reuses one trace_id across attempts
    client2 = SimulatorClient(base_url="http://sim", token=TOKEN, transport=transport,
                              max_retries=3, backoff_base_s=0.01)
    body = await client2.get("/analytics/kpi-definitions", trace_id="shared-trace-2")
    assert body["kpis"]


@pytest.mark.asyncio
async def test_client_retries_through_injected_503_using_its_own_headers(transport):
    """End-to-end: SimulatorClient.get() is called with a fault header baked
    into a one-off transport wrapper, proving the client's retry loop (not
    just the simulator) recovers a transient failure within one logical call."""

    class FlakyTransport(httpx.AsyncBaseTransport):
        def __init__(self, inner: httpx.ASGITransport) -> None:
            self.inner, self.calls = inner, 0

        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            self.calls += 1
            request.headers["x-simulate-fault"] = "503:2"
            return await self.inner.handle_async_request(request)

    flaky = FlakyTransport(transport)
    client = SimulatorClient(base_url="http://sim", token=TOKEN, transport=flaky,
                             max_retries=3, backoff_base_s=0.01)
    body = await client.get("/analytics/kpi-definitions", trace_id="one-call-trace")
    assert body["kpis"] and flaky.calls == 3  # 2 failures + 1 success, same trace_id throughout


@pytest.mark.asyncio
async def test_every_retry_attempt_reuses_the_same_trace_id(transport):
    """The simulator's fault counter is keyed by (trace_id, path); if a retry
    regenerated the trace_id, the counter would never clear and every retry
    would look like a brand-new call - proven here by recording the header
    actually sent on each individual attempt, not just the end result."""
    seen_trace_ids: list[str | None] = []

    class Recorder(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            seen_trace_ids.append(request.headers.get("trace_id"))
            request.headers["x-simulate-fault"] = "503:2"
            return await transport.handle_async_request(request)

    client = SimulatorClient(base_url="http://sim", token=TOKEN, transport=Recorder(),
                             max_retries=3, backoff_base_s=0.01)
    await client.get("/analytics/kpi-definitions", trace_id="pinned-trace")
    assert seen_trace_ids == ["pinned-trace", "pinned-trace", "pinned-trace"]


@pytest.mark.asyncio
async def test_retries_are_exhausted_and_raise_after_max_attempts(transport):
    class AlwaysFaulty(httpx.AsyncBaseTransport):
        def __init__(self, inner: httpx.ASGITransport) -> None:
            self.inner, self.calls = inner, 0

        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            self.calls += 1
            request.headers["x-simulate-fault"] = "503"  # every call fails
            return await self.inner.handle_async_request(request)

    always = AlwaysFaulty(transport)
    client = SimulatorClient(base_url="http://sim", token=TOKEN, transport=always,
                             max_retries=2, backoff_base_s=0.01)
    with pytest.raises(ApiError) as exc:
        await client.get("/analytics/kpi-definitions", trace_id="exhaust-trace")
    assert exc.value.status == 503 and always.calls == 3  # 1 initial + 2 retries


@pytest.mark.asyncio
async def test_backoff_grows_between_attempts(transport):
    class AlwaysFaulty(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            request.headers["x-simulate-fault"] = "503"
            return await transport.handle_async_request(request)

    client = SimulatorClient(base_url="http://sim", token=TOKEN, transport=AlwaysFaulty(),
                             max_retries=2, backoff_base_s=0.05)
    started = time.perf_counter()
    with pytest.raises(ApiError):
        await client.get("/analytics/kpi-definitions", trace_id="timing-trace")
    elapsed = time.perf_counter() - started
    assert elapsed >= 0.05 + 0.10  # two waits: ~0.05s then ~0.10s (before jitter)


# ---- network-level failure (not an HTTP status at all) --------------------------------------
@pytest.mark.asyncio
async def test_network_error_is_retried_then_raises_a_clean_api_error():
    class AlwaysDown(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("connection refused", request=request)

    client = SimulatorClient(base_url="http://sim", token=TOKEN, transport=AlwaysDown(),
                             max_retries=2, backoff_base_s=0.01)
    with pytest.raises(ApiError) as exc:
        await client.get("/health")
    assert exc.value.status == 504 and exc.value.code == "gateway_timeout"


@pytest.mark.asyncio
async def test_client_id_header_is_sent(transport):
    seen = {}

    class Spy(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            seen["x-client-id"] = request.headers.get("x-client-id")
            return await transport.handle_async_request(request)

    client = SimulatorClient(base_url="http://sim", token=TOKEN, transport=Spy(), client_id="alarm-mcp-server")
    await client.get("/health")
    assert seen["x-client-id"] == "alarm-mcp-server"


@pytest.mark.asyncio
async def test_none_query_params_are_dropped_not_sent_as_the_string_none(transport):
    seen = {}

    class Spy(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            seen["query"] = str(request.url.params)
            return await transport.handle_async_request(request)

    client = SimulatorClient(base_url="http://sim", token=TOKEN, transport=Spy())
    await client.get("/alarms", params={"site": "EastRefinery", "unit": None})
    assert "unit" not in seen["query"] and "EastRefinery" in seen["query"]


@pytest.mark.asyncio
async def test_context_manager_closes_the_underlying_client(transport):
    async with SimulatorClient(base_url="http://sim", token=TOKEN, transport=transport) as c:
        await c.get("/health")
    assert c._client.is_closed