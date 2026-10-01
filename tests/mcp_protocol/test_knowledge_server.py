"""Protocol-level tests for the knowledge-base MCP server, mirroring the
alarm/ticketing server test style. Uses HashingEmbeddingProvider (default),
never OllamaEmbeddingProvider - no network dependency in tests."""

import pytest

from mcp_servers.knowledge_server import build_knowledge_server
from tests.mcp_protocol.helpers import call_tool, list_tools


@pytest.fixture(scope="module")
def server():
    return build_knowledge_server()


@pytest.mark.asyncio
async def test_one_tool_is_registered_with_a_description(server):
    tools = await list_tools(server)
    assert len(tools) == 1 and tools[0].name == "search_knowledge_base" and tools[0].description


@pytest.mark.asyncio
async def test_search_returns_formatted_text_and_structured_results(server):
    res = await call_tool(server, "search_knowledge_base",
                          {"req": {"query": "high vibration compressor bearing"}})
    assert not res.is_error
    body = res.structured_content
    assert body["results"] and body["results"][0]["doc_id"] == "KB-0001"
    assert "KB-0001" in body["formatted"]


@pytest.mark.asyncio
async def test_injection_flag_surfaces_through_the_tool(server):
    res = await call_tool(server, "search_knowledge_base",
                          {"req": {"query": "vendor remote access AI assistant automated instructions",
                                  "top_k": 10}})
    results = res.structured_content["results"]
    assert any(r["flagged"] for r in results)
    assert "CAUTION" in res.structured_content["formatted"]


@pytest.mark.asyncio
async def test_empty_query_is_rejected_by_schema(server):
    res = await call_tool(server, "search_knowledge_base", {"req": {"query": ""}})
    assert res.is_error


@pytest.mark.asyncio
async def test_top_k_is_respected(server):
    res = await call_tool(server, "search_knowledge_base", {"req": {"query": "alarm", "top_k": 2}})
    assert len(res.structured_content["results"]) <= 2