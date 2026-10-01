"""Test harness for driving an MCPServer through a REAL ClientSession, over
in-memory streams - not calling server.call_tool() directly, which bypasses
protocol-level behaviour (error wrapping, schema validation) that a real MCP
client actually experiences. This is a plain helper module (no test_ prefix),
so pytest never tries to collect it as a test file itself.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any

from mcp.client.session import ClientSession
from mcp.server.mcpserver import MCPServer
from mcp.shared.memory import create_client_server_memory_streams
from mcp.types import CallToolResult, Tool


async def call_tool(server: MCPServer, name: str, args: dict[str, Any]) -> CallToolResult:
    """Call one tool through the real MCP wire protocol and return its result."""
    async with create_client_server_memory_streams() as (client_streams, server_streams):
        c_read, c_write = client_streams
        s_read, s_write = server_streams
        server_task = asyncio.create_task(
            server._lowlevel_server.run(s_read, s_write, server._lowlevel_server.create_initialization_options())
        )
        try:
            async with ClientSession(c_read, c_write) as session:
                await session.initialize()
                return await session.call_tool(name, args)
        finally:
            server_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await server_task


async def list_tools(server: MCPServer) -> list[Tool]:
    async with create_client_server_memory_streams() as (client_streams, server_streams):
        c_read, c_write = client_streams
        s_read, s_write = server_streams
        server_task = asyncio.create_task(
            server._lowlevel_server.run(s_read, s_write, server._lowlevel_server.create_initialization_options())
        )
        try:
            async with ClientSession(c_read, c_write) as session:
                await session.initialize()
                result = await session.list_tools()
                return result.tools
        finally:
            server_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await server_task