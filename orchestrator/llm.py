"""LLM provider abstraction. Every test uses FakeLLM (deterministic, scripted,
no network); OllamaLLM is the real, swappable local-model backend used only
at runtime - no test imports it against a live server.

OllamaLLM's request/response shape follows Ollama's documented tool-calling
flow, confirmed directly against a live qwen2.5:3b server before writing this
(a tool call's `arguments` field arrives already parsed as a dict, not a JSON
string to decode - that's an Ollama-specific detail worth getting right).

Retry/timeout handling exists because a real run surfaced it directly: a
local CPU model can take tens of seconds per turn once the conversation has
grown, and a bare timeout with no retry crashed an entire multi-turn
investigation - discarding every tool result already gathered - on one slow
call. LLMTimeoutError is the clean, typed exception a caller (agent.py, and
eventually the GUI) can catch, instead of a raw httpx exception surfacing
three async-context-manager layers deep.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class LLMResponse:
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)


class LLMClient(Protocol):
    async def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> LLMResponse: ...


class LLMTimeoutError(Exception):
    """The LLM did not respond in time, even after retries. Local CPU
    inference can genuinely take longer as conversation context grows - this
    is not necessarily a bug in the caller, but it needs a clean, catchable
    exception rather than a raw httpx error."""


class OllamaLLM:
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "qwen2.5:3b",
                timeout_s: float = 90.0, keep_alive: str = "10m", max_retries: int = 1,
                retry_backoff_s: float = 2.0, num_ctx: int = 16384,
                transport: httpx.AsyncBaseTransport | None = None) -> None:
        # num_ctx matters a lot here: measured directly, this project's system
        # prompt plus its ~20 tool schemas alone total roughly 6,900 tokens -
        # already 168% of Ollama's 4096-token DEFAULT context window, before a
        # single turn of conversation happens. Without overriding it, Ollama
        # silently truncates every request, which both degrades tool
        # selection (truncated instructions/tools) and makes every call much
        # slower (it reprocesses a large, truncated context from scratch each
        # time rather than cleanly extending a small one) - confirmed by
        # measuring the real payload size and by `ollama ps` showing the
        # default 4096 context in a live run.
        self.model, self.keep_alive, self.num_ctx = model, keep_alive, num_ctx
        self.timeout_s, self.max_retries, self.retry_backoff_s = timeout_s, max_retries, retry_backoff_s
        self._client = httpx.AsyncClient(base_url=base_url, timeout=timeout_s, transport=transport)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "OllamaLLM":
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    async def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> LLMResponse:
        body: dict[str, Any] = {"model": self.model, "stream": False, "messages": messages,
                                "keep_alive": self.keep_alive, "options": {"num_ctx": self.num_ctx}}
        if tools:
            body["tools"] = tools

        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = await self._client.post("/api/chat", json=body)
                resp.raise_for_status()
                msg = resp.json()["message"]
                calls = [ToolCall(id=tc.get("id") or f"call_{i}", name=tc["function"]["name"],
                                  arguments=tc["function"].get("arguments") or {})
                         for i, tc in enumerate(msg.get("tool_calls") or [])]
                return LLMResponse(content=msg.get("content") or "", tool_calls=calls)
            except (httpx.TimeoutException, httpx.ConnectError) as exc:
                last_exc = exc
                if attempt < self.max_retries:
                    await asyncio.sleep(self.retry_backoff_s)
                    continue
                raise LLMTimeoutError(
                    f"Ollama did not respond after {self.max_retries + 1} attempt(s), "
                    f"{self.timeout_s}s timeout each. Local inference can slow down substantially as the "
                    "conversation grows; check `ollama ps` to confirm the model is actually staying loaded "
                    "between calls, free up system memory, or raise timeout_s."
                ) from exc
        raise LLMTimeoutError("Exhausted retries calling Ollama") from last_exc  # unreachable in practice


class FakeLLM:
    """Deterministic scripted LLM: returns one LLMResponse per call, in the
    order given. A scripted item may also be an Exception instance (e.g.
    LLMTimeoutError) - it is raised instead of returned, for testing how a
    caller handles a mid-investigation LLM failure. Raises AssertionError if
    asked for more calls than were scripted, so a test whose agent loop runs
    longer than expected fails loudly instead of hanging or silently
    repeating the last response."""

    def __init__(self, script: list[LLMResponse | Exception]) -> None:
        self._script: list[LLMResponse | Exception] = list(script)
        self.calls: list[tuple[list[dict[str, Any]], list[dict[str, Any]]]] = []

    async def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> LLMResponse:
        self.calls.append((messages, tools))
        if not self._script:
            raise AssertionError(f"FakeLLM script exhausted after {len(self.calls)} call(s)")
        item = self._script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item