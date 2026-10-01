"""LLM abstraction tests. OllamaLLM is tested only against a mock transport
that mimics Ollama's documented /api/chat response shape - never against a
live server, so these tests are fast and deterministic in CI."""

import httpx
import pytest

from orchestrator.llm import FakeLLM, LLMResponse, LLMTimeoutError, OllamaLLM, ToolCall


# ---- FakeLLM ----------------------------------------------------------------
@pytest.mark.asyncio
async def test_fakellm_returns_scripted_responses_in_order():
    r1, r2 = LLMResponse(content="first"), LLMResponse(content="second")
    llm = FakeLLM([r1, r2])
    assert await llm.chat([], []) is r1
    assert await llm.chat([], []) is r2


@pytest.mark.asyncio
async def test_fakellm_raises_when_script_exhausted():
    llm = FakeLLM([LLMResponse(content="only one")])
    await llm.chat([], [])
    with pytest.raises(AssertionError, match="exhausted"):
        await llm.chat([], [])


@pytest.mark.asyncio
async def test_fakellm_records_every_call_for_assertions():
    llm = FakeLLM([LLMResponse(content="ok")])
    messages = [{"role": "user", "content": "hi"}]
    tools = [{"type": "function", "function": {"name": "x"}}]
    await llm.chat(messages, tools)
    assert llm.calls == [(messages, tools)]


def test_llm_response_has_tool_calls_property():
    assert not LLMResponse(content="x").has_tool_calls
    assert LLMResponse(content="", tool_calls=[ToolCall("1", "t", {})]).has_tool_calls


# ---- OllamaLLM, against a mock transport that mimics real Ollama responses ----------------
class MockOllamaTransport(httpx.AsyncBaseTransport):
    def __init__(self, response_json: dict) -> None:
        self.response_json, self.last_request_json = response_json, None

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        import json as _json
        self.last_request_json = _json.loads(request.content)
        return httpx.Response(200, json=self.response_json)


@pytest.mark.asyncio
async def test_ollama_parses_a_tool_call_response_exactly_like_the_real_server():
    # this exact shape was confirmed against a live qwen2.5:3b server
    transport = MockOllamaTransport({"message": {"role": "assistant", "content": "",
                                                 "tool_calls": [{"id": "call_abc", "function": {
                                                     "name": "asset_search", "arguments": {"query": "Boiler Feed Pump 101"}}}]}})
    async with OllamaLLM(model="qwen2.5:3b", transport=transport) as llm:
        response = await llm.chat([{"role": "user", "content": "find the pump"}], tools=[])
    assert response.tool_calls == [ToolCall(id="call_abc", name="asset_search",
                                            arguments={"query": "Boiler Feed Pump 101"})]


@pytest.mark.asyncio
async def test_ollama_parses_a_plain_text_response_with_no_tool_calls():
    transport = MockOllamaTransport({"message": {"role": "assistant", "content": "Here is my answer."}})
    async with OllamaLLM(transport=transport) as llm:
        response = await llm.chat([{"role": "user", "content": "hi"}], tools=[])
    assert response.content == "Here is my answer." and not response.has_tool_calls


@pytest.mark.asyncio
async def test_ollama_request_body_includes_model_messages_and_keep_alive():
    transport = MockOllamaTransport({"message": {"role": "assistant", "content": "ok"}})
    async with OllamaLLM(model="qwen2.5:3b", keep_alive="5m", transport=transport) as llm:
        await llm.chat([{"role": "user", "content": "hi"}], tools=[])
    body = transport.last_request_json
    assert body["model"] == "qwen2.5:3b" and body["stream"] is False and body["keep_alive"] == "5m"
    assert body["messages"] == [{"role": "user", "content": "hi"}]
    assert "tools" not in body  # omitted entirely when there are none


@pytest.mark.asyncio
async def test_ollama_request_overrides_num_ctx_beyond_the_4096_default():
    """The 4096 default is not enough for this project's own tool schemas
    (measured: ~6,900 tokens for the system prompt + ~20 tools alone) -
    every request must explicitly ask for a bigger context window."""
    transport = MockOllamaTransport({"message": {"role": "assistant", "content": "ok"}})
    async with OllamaLLM(transport=transport) as llm:  # default num_ctx
        await llm.chat([{"role": "user", "content": "hi"}], tools=[])
    assert transport.last_request_json["options"]["num_ctx"] > 4096


@pytest.mark.asyncio
async def test_num_ctx_is_configurable():
    transport = MockOllamaTransport({"message": {"role": "assistant", "content": "ok"}})
    async with OllamaLLM(num_ctx=32768, transport=transport) as llm:
        await llm.chat([{"role": "user", "content": "hi"}], tools=[])
    assert transport.last_request_json["options"]["num_ctx"] == 32768


@pytest.mark.asyncio
async def test_ollama_request_body_includes_tools_when_provided():
    transport = MockOllamaTransport({"message": {"role": "assistant", "content": "ok"}})
    tools = [{"type": "function", "function": {"name": "search_assets"}}]
    async with OllamaLLM(transport=transport) as llm:
        await llm.chat([{"role": "user", "content": "hi"}], tools=tools)
    assert transport.last_request_json["tools"] == tools


@pytest.mark.asyncio
async def test_ollama_generates_a_call_id_when_the_server_omits_one():
    transport = MockOllamaTransport({"message": {"role": "assistant", "content": "",
                                                 "tool_calls": [{"function": {"name": "x", "arguments": {}}}]}})
    async with OllamaLLM(transport=transport) as llm:
        response = await llm.chat([], tools=[])
    assert response.tool_calls[0].id  # non-empty, even though the mock server didn't supply one


@pytest.mark.asyncio
async def test_ollama_raises_on_http_error():
    class Failing(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, json={"error": "model not found"})

    async with OllamaLLM(transport=Failing()) as llm:
        with pytest.raises(httpx.HTTPStatusError):
            await llm.chat([], tools=[])


# ---- retry/timeout resilience -------------------------------------------------------
# Directly motivated by a real run: four successful calls (~30s each as
# context grew), then a fifth that timed out and crashed the whole
# investigation with a raw httpx exception three context-managers deep.
class _TimesOutNTimesThenSucceeds(httpx.AsyncBaseTransport):
    def __init__(self, fail_times: int, response_json: dict) -> None:
        self.fail_times, self.response_json, self.calls = fail_times, response_json, 0

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.calls += 1
        if self.calls <= self.fail_times:
            raise httpx.ReadTimeout("simulated slow local inference", request=request)
        return httpx.Response(200, json=self.response_json)


@pytest.mark.asyncio
async def test_ollama_retries_a_timeout_then_succeeds():
    transport = _TimesOutNTimesThenSucceeds(1, {"message": {"role": "assistant", "content": "ok"}})
    async with OllamaLLM(max_retries=1, retry_backoff_s=0.01, transport=transport) as llm:
        response = await llm.chat([{"role": "user", "content": "hi"}], tools=[])
    assert response.content == "ok" and transport.calls == 2


@pytest.mark.asyncio
async def test_ollama_raises_llmtimeouterror_not_a_raw_httpx_exception_after_exhausting_retries():
    transport = _TimesOutNTimesThenSucceeds(99, {"message": {"role": "assistant", "content": "unreachable"}})
    async with OllamaLLM(max_retries=1, retry_backoff_s=0.01, transport=transport) as llm:
        with pytest.raises(LLMTimeoutError):
            await llm.chat([{"role": "user", "content": "hi"}], tools=[])
    assert transport.calls == 2  # 1 initial + 1 retry, not more


@pytest.mark.asyncio
async def test_ollama_default_is_zero_retries_beyond_the_one_configured():
    """max_retries=0 (a caller who wants to fail fast) makes exactly one attempt."""
    transport = _TimesOutNTimesThenSucceeds(1, {"message": {"role": "assistant", "content": "ok"}})
    async with OllamaLLM(max_retries=0, retry_backoff_s=0.01, transport=transport) as llm:
        with pytest.raises(LLMTimeoutError):
            await llm.chat([{"role": "user", "content": "hi"}], tools=[])
    assert transport.calls == 1


@pytest.mark.asyncio
async def test_connect_error_is_also_retried_the_same_as_a_timeout():
    class DownThenUp(httpx.AsyncBaseTransport):
        def __init__(self) -> None:
            self.calls = 0

        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            self.calls += 1
            if self.calls == 1:
                raise httpx.ConnectError("connection refused", request=request)
            return httpx.Response(200, json={"message": {"role": "assistant", "content": "ok"}})

    transport = DownThenUp()
    async with OllamaLLM(max_retries=1, retry_backoff_s=0.01, transport=transport) as llm:
        response = await llm.chat([], tools=[])
    assert response.content == "ok"


# ---- FakeLLM can simulate a mid-script failure ------------------------------------------------
@pytest.mark.asyncio
async def test_fakellm_can_raise_a_scripted_exception_instead_of_returning():
    llm = FakeLLM([LLMResponse(content="first"), LLMTimeoutError("simulated"), LLMResponse(content="third")])
    assert (await llm.chat([], [])).content == "first"
    with pytest.raises(LLMTimeoutError, match="simulated"):
        await llm.chat([], [])
    assert (await llm.chat([], [])).content == "third"  # script continues normally after the exception