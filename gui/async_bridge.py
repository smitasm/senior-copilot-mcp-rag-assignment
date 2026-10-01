"""The actual orchestration logic behind the GUI, kept separate from
Streamlit rendering so it can be unit-tested directly (Streamlit apps
themselves are hard to test meaningfully; the logic behind them isn't).

Streamlit's execution model reruns the whole script top-to-bottom on every
interaction and has no native asyncio support, so app.py wraps each of these
functions in a single `asyncio.run(...)` per button click - fresh MCP
connections opened and cleanly closed within that one call, never held
across reruns. That matches the pattern already proven throughout this
project's own tests (connect, use, close, within one async scope) rather
than fighting Streamlit's rerun model with a persisted event loop.
"""

from __future__ import annotations

from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass

from mcp_common.http_client import SimulatorClient
from mcp_servers.alarm_server import build_alarm_server
from mcp_servers.knowledge_server import build_knowledge_server
from mcp_servers.ticketing_server import build_ticketing_server
from orchestrator.actions import ApprovalRequiredError, create_ticket_from_draft
from orchestrator.agent import InvestigationResult, investigate
from orchestrator.draft import IncidentDraft
from orchestrator.llm import LLMClient, OllamaLLM
from orchestrator.mcp_client import ToolCallError, ToolRegistry, connect_inprocess
from rag.embeddings import EmbeddingProvider, HashingEmbeddingProvider


@dataclass(frozen=True)
class ConnectionConfig:
    alarm_api_url: str = "http://localhost:8000"
    ticketing_api_url: str = "http://localhost:8001"
    api_token: str = "demo-token"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:3b"


@asynccontextmanager
async def build_registry(cfg: ConnectionConfig, *, include_alarm: bool = True, include_ticketing: bool = True,
                         include_knowledge: bool = True, embedder: EmbeddingProvider | None = None,
                         alarm_client: SimulatorClient | None = None,
                         ticket_client: SimulatorClient | None = None):
    """Connects whichever of the three MCP servers the caller needs, in
    process, and yields one aggregated ToolRegistry. `include_*` flags exist
    because approve_and_create only needs the ticketing tool - opening all
    three for a single create_ticket call would be wasted work every time a
    user clicks Approve.

    `alarm_client`/`ticket_client` are injectable (same dependency-injection
    pattern SimulatorClient itself already uses for its `transport` param) so
    tests can point this at real in-process simulators via ASGI transport,
    exactly like every other test in this project - rather than needing a
    real uvicorn process on a real port just to exercise this function.

    Uses AsyncExitStack rather than a hand-rolled conditional context-manager
    wrapper around connect_inprocess: an earlier version added its own extra
    @asynccontextmanager layer in between, and an exception propagating
    through that extra layer broke the MCP SDK's internal TaskGroup
    cancel-scope nesting ("unhandled errors in a TaskGroup") - confirmed by
    hitting exactly that error before switching to AsyncExitStack, which is
    the standard tool for conditionally entering a variable number of async
    context managers into one combined scope without adding that kind of
    extra nesting."""
    owns_alarm = alarm_client is None
    owns_ticket = ticket_client is None
    alarm_client = alarm_client or SimulatorClient(base_url=cfg.alarm_api_url, token=cfg.api_token, client_id="gui")
    ticket_client = ticket_client or SimulatorClient(base_url=cfg.ticketing_api_url, token=cfg.api_token,
                                                     client_id="gui")
    reg = ToolRegistry()
    try:
        async with AsyncExitStack() as stack:
            if include_alarm:
                s = await stack.enter_async_context(connect_inprocess(build_alarm_server(alarm_client)))
                await reg.add(s)
            if include_ticketing:
                s = await stack.enter_async_context(connect_inprocess(build_ticketing_server(ticket_client)))
                await reg.add(s)
            if include_knowledge:
                s = await stack.enter_async_context(
                    connect_inprocess(build_knowledge_server(embedder or HashingEmbeddingProvider())))
                await reg.add(s)
            yield reg
    finally:
        if owns_alarm:
            await alarm_client.aclose()
        if owns_ticket:
            await ticket_client.aclose()


async def run_investigation(user_request: str, cfg: ConnectionConfig, *, llm: LLMClient | None = None,
                            alarm_client: SimulatorClient | None = None,
                            ticket_client: SimulatorClient | None = None) -> InvestigationResult:
    """Runs one full investigation against the real Alarm, Ticketing and
    Knowledge MCP servers (or a caller-supplied `llm`, e.g. FakeLLM in
    tests).

    Any exception is caught INSIDE the `async with build_registry(...)`
    block and only re-raised after it has exited cleanly - confirmed
    directly that letting an exception propagate WHILE that context is still
    open breaks the MCP SDK's internal TaskGroup cancel-scope invariants
    ("unhandled errors in a TaskGroup"), regardless of how the context
    manager itself is structured. This is the general, robust pattern for
    every function in this file that might raise inside a build_registry
    block, not a one-off workaround."""
    error: Exception | None = None
    result: InvestigationResult | None = None
    async with build_registry(cfg, alarm_client=alarm_client, ticket_client=ticket_client) as reg:
        own_llm = llm is None
        client = llm or OllamaLLM(base_url=cfg.ollama_base_url, model=cfg.ollama_model)
        try:
            result = await investigate(user_request, client, reg)
        except Exception as exc:  # noqa: BLE001 - re-raised below, after the context closes cleanly
            error = exc
        finally:
            if own_llm:
                await client.aclose()
    if error is not None:
        raise error
    return result


async def approve_and_create(draft: IncidentDraft, cfg: ConnectionConfig, *, approved: bool,
                             reported_by: str = "copilot-gui-user",
                             ticket_client: SimulatorClient | None = None) -> dict:
    """The GUI's own call site for the one write operation in this project.
    Requires an explicit `approved` argument - same discipline as
    orchestrator/actions.py and the MCP-level gate underneath it. This
    function does not decide whether the user approved; app.py's own
    confirmation UI does, and must pass approved=True only after that.

    See run_investigation's docstring for why exceptions are caught inside
    the build_registry block and re-raised after it closes, rather than
    letting ApprovalRequiredError (an entirely expected, common case here)
    propagate through it directly."""
    error: Exception | None = None
    result: dict | None = None
    async with build_registry(cfg, include_alarm=False, include_knowledge=False,
                              ticket_client=ticket_client) as reg:
        try:
            result = await create_ticket_from_draft(reg, draft, approved=approved, reported_by=reported_by)
        except Exception as exc:  # noqa: BLE001
            error = exc
    if error is not None:
        raise error
    return result


async def fetch_ticket_details(ticket_ids: list[str], cfg: ConnectionConfig, *,
                               ticket_client: SimulatorClient | None = None) -> list[dict]:
    """Looks up full ticket content for display (the GUI shows more than
    just an ID for each similar ticket). Best-effort: a ticket that fails to
    load (e.g. a stale id) is skipped rather than failing the whole batch -
    this is a display convenience, not safety-critical grounding data."""
    if not ticket_ids:
        return []
    async with build_registry(cfg, include_alarm=False, include_knowledge=False,
                              ticket_client=ticket_client) as reg:
        out = []
        for tid in ticket_ids:
            try:
                out.append(await reg.call("get_ticket", {"req": {"ticket_id": tid}}))
            except ToolCallError:
                continue
        return out


HEALTH_CHECK_TIMEOUT_S = 3.0


async def check_connections(cfg: ConnectionConfig, *, alarm_client: SimulatorClient | None = None,
                            ticket_client: SimulatorClient | None = None) -> dict[str, bool]:
    """Health-checks the two REST simulators before the user tries to run a
    full investigation against a service that isn't up.

    Deliberately builds its own clients with timeout_s=3 and max_retries=0,
    NOT SimulatorClient's library defaults (10s timeout, 3 retries) - those
    are right for a real API call that might hit a transient blip, but wrong
    here: retrying a genuinely-down service 3 times with backoff defeats the
    purpose of a FAST connectivity check, and on at least one real run (a
    Windows machine, where a refused connection apparently doesn't fail as
    fast as it does on Linux) that amplification pushed a single health
    check past 15 seconds. A health check answering "is this up at all" has
    no business retrying; one quick attempt per service is the right
    behaviour regardless of platform."""
    results = {}
    pairs = (("alarm_api", cfg.alarm_api_url, alarm_client), ("ticketing_api", cfg.ticketing_api_url, ticket_client))
    for name, url, injected in pairs:
        owns = injected is None
        client = injected or SimulatorClient(base_url=url, token=cfg.api_token,
                                             timeout_s=HEALTH_CHECK_TIMEOUT_S, max_retries=0)
        try:
            await client.get("/health")
            results[name] = True
        except Exception:
            results[name] = False
        finally:
            if owns:
                await client.aclose()
    return results


__all__ = ["ApprovalRequiredError", "ConnectionConfig", "approve_and_create", "build_registry",
          "check_connections", "fetch_ticket_details", "run_investigation"]