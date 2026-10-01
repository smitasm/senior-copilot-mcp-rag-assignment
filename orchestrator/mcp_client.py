"""Connects the orchestrator to this project's MCP servers and exposes a flat
tool registry the LLM can call by name.

MCP servers run IN-PROCESS with the orchestrator via the SDK's in-memory
transport - real protocol semantics (schemas, discovery, error format), no
network hop - since they're always deployed together. The two REST
simulators those tools ultimately call (Alarm API, Ticketing API) remain
genuine separate HTTP services, since those model real external systems.
Swapping an MCP server to run remotely, if ever needed, only touches
`connect_inprocess` below; nothing else in this file or in agent.py changes.
"""

from __future__ import annotations

import asyncio
import contextlib
from contextlib import asynccontextmanager
from typing import Any

from mcp.client.session import ClientSession
from mcp.server.mcpserver import MCPServer
from mcp.shared.memory import create_client_server_memory_streams


class ToolCallError(Exception):
    """A tool call failed. `str(exc)` is the exact text the MCP layer
    reported - already a clean message thanks to ToolError in the servers
    themselves (Phase 3), safe to feed straight back to the LLM."""


@asynccontextmanager
async def connect_inprocess(server: MCPServer):
    """Open one long-lived ClientSession to `server` over in-memory streams,
    for the caller's use across an arbitrary number of tool calls (unlike
    tests/mcp_protocol/helpers.py, which opens a fresh session per call -
    fine for isolated protocol tests, wasteful for a real conversation)."""
    async with create_client_server_memory_streams() as (client_streams, server_streams):
        c_read, c_write = client_streams
        s_read, s_write = server_streams
        task = asyncio.create_task(
            server._lowlevel_server.run(s_read, s_write, server._lowlevel_server.create_initialization_options())
        )
        try:
            async with ClientSession(c_read, c_write) as session:
                await session.initialize()
                yield session
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task


class ToolRegistry:
    """Aggregates tools from one or more initialized ClientSessions into a
    single flat, LLM-facing tool list, and dispatches calls by name.

    Every real tool in this project takes one wrapped parameter (`req`) -
    that's how MCPServer schemas a single-Pydantic-model function argument
    (see Phase 3). A small local model, having seen ~19 tool schemas, can
    plausibly pass FLAT arguments instead of the `req` wrapper for one of
    them; `call()` auto-wraps when a tool's schema expects exactly `req` and
    the caller didn't supply it, rather than failing a call over a shape the
    model almost got right.
    """

    def __init__(self) -> None:
        self._dispatch: dict[str, ClientSession] = {}
        self._specs: list[dict[str, Any]] = []
        self._param_names: dict[str, set[str]] = {}

    async def add(self, session: ClientSession) -> None:
        result = await session.list_tools()
        for t in result.tools:
            if t.name in self._dispatch:
                raise ValueError(f"duplicate tool name across MCP servers: {t.name!r}")
            self._dispatch[t.name] = session
            self._param_names[t.name] = set((t.input_schema or {}).get("properties", {}))
            self._specs.append({"type": "function", "function": {
                "name": t.name, "description": t.description or "", "parameters": t.input_schema}})

    def llm_tools(self) -> list[dict[str, Any]]:
        return list(self._specs)

    @property
    def tool_names(self) -> list[str]:
        return list(self._dispatch)

    async def call(self, name: str, arguments: dict[str, Any]) -> Any:
        session = self._dispatch.get(name)
        if session is None:
            raise ToolCallError(f"Unknown tool '{name}'. Available tools: {', '.join(self.tool_names)}")
        arguments = dict(arguments or {})
        if self._param_names.get(name) == {"req"} and "req" not in arguments:
            arguments = {"req": arguments}  # normalize: model passed flat args for a req-wrapped tool
        result = await session.call_tool(name, arguments)
        text = result.content[0].text if result.content else ""
        if result.is_error:
            raise ToolCallError(text)
        return result.structured_content if result.structured_content is not None else text