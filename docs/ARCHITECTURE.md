# Architecture

This document goes deeper than the README: every design decision here was
made in response to something specific - either a requirement in the
assignment or a failure actually observed while testing against the real
local model - not from a generic template. Where that's the case, the
observed failure is named explicitly.

## Design principles

1. **The Alarm API must be reached through MCP, not directly.** This is
   stated outright in the assignment, and it shapes everything else: the
   orchestrator never imports or calls `alarm_api`/`ticketing_api` code
   directly, only ever through an MCP `ClientSession`'s `call_tool`.
2. **MCP and RAG must participate in the same workflow, not run side by
   side.** `search_knowledge_base` is itself an MCP tool
   (`mcp_servers/knowledge_server.py`), discovered and called through the
   exact same `ClientSession` machinery as the alarm/ticketing tools - RAG
   isn't a separate subsystem the agent occasionally consults, it's one
   more tool in the same registry.
3. **Nothing is written without explicit human approval.** The only write
   operation anywhere in the system is `create_ticket`, and it refuses to
   run without `approved=True` - see "The approval gate" below.
4. **Every claim in the final draft must be traceable.** Not just "cite your
   sources" as a prompt instruction (which a 3B model will not reliably
   follow under pressure) but a hard check that runs after the model
   produces its draft and rejects/flags anything it can't back up with an
   actual tool call from this run.

## The 19 MCP tools

### `alarm_server.py` (14 tools) -> Alarm API simulator

| Tool | What it does |
|---|---|
| `search_assets` | Find an asset by name or id. The natural first call when the user names an asset/unit in free text. |
| `get_asset_metadata` | Full metadata for one asset: manufacturer, criticality, maintenance group, related assets. |
| `list_alarms` | List/filter/page alarms, sorted by recency by default. |
| `get_alarm` | Full detail for one alarm by its alarm_id. |
| `alarm_summary` | Aggregate alarm KPIs (count, critical_count, recurring_rate, ...) over a scope. |
| `alarm_trends` | Time-bucketed alarm counts / ack-delay over a range (hourly/daily/weekly). |
| `correlate_alarms` | Find alarm pairs that tend to fire close together in time on the same asset/unit. |
| `flood_analysis` | Detect alarm-flood windows (many alarms in a short rolling window) for a unit/site. |
| `rationalization_candidates` | Find alarms worth rationalizing - chattering (fires/clears rapidly), standing, etc. |
| `priority_score` | Score one alarm's priority 0-100, banded low/medium/high/critical. |
| `operator_recommendations` | Likely cause + ranked recommended operator actions for one alarm. |
| `generate_calculation` | Define a named plant KPI calculation (e.g. alarm_flood_index). |
| `execute_calculation` | Run a `calculation_id` from `generate_calculation` and get its numeric result. |
| `kpi_definitions` | List every KPI this system knows, with its formula and unit. |

### `ticketing_server.py` (4 tools) -> Ticketing API mock

| Tool | What it does |
|---|---|
| `list_tickets` | List/filter/page historical and open tickets. |
| `get_ticket` | Full detail for one ticket by its ticket_id. |
| `find_similar_tickets` | Find historical tickets similar to an alarm/asset/free-text description - the duplicate-detection tool. |
| `create_ticket` | **Write operation.** Create a new incident ticket. Requires `approved=true`; raises `ApprovalRequiredError` before any HTTP call otherwise. |

### `knowledge_server.py` (1 tool) -> RAG over the knowledge base

| Tool | What it does |
|---|---|
| `search_knowledge_base` | Search the plant's knowledge base (troubleshooting guides, safety procedures, maintenance history) and return cited, chunked passages. |

## The approval gate

`create_ticket`'s check lives in the MCP tool itself
(`mcp_servers/ticketing_server.py`), not only in the GUI, deliberately:
a gate that exists solely in the Streamlit layer is bypassed by anything
else that can reach the MCP server - a different client, a future
integration, a bug in the GUI's own session-state handling. Putting it in
the tool means the *only* way `create_ticket` ever reaches the Ticketing
API is if the caller explicitly passed `approved=True`, and the GUI is the
only caller in this system that ever sets that flag, and only after the
on-screen checkbox is ticked and confirmed. This is tested directly in
`tests/unit/test_orchestrator_actions.py` and
`tests/mcp_protocol/test_ticketing_server.py`.

## The agent loop (`orchestrator/agent.py`)

A bounded loop, max 8 turns, with three mechanisms added specifically
because of failures observed running the real model (qwen2.5:3b, CPU
inference) rather than designed in abstract:

- **Repeat-tool circuit breaker.** An early run called
  `search_knowledge_base` eight times in a row with materially the same
  arguments, burning the entire turn budget without making progress. The
  loop now tracks identical-enough consecutive calls, injects a warning
  into the model's context at 3 repeats, and forcibly stops the loop at 5.
- **Grounding check.** A separate run produced a draft citing `KB-0001` and
  `KB-0002` without ever having called `search_knowledge_base` that turn -
  plausible-sounding, fabricated citations. After the model submits its
  draft (via the `submit_incident_draft` pseudo-tool - see below), the
  agent diffs every citation and ticket ID the draft claims against what
  the tool calls actually recorded in this run's trace, and flags anything
  that doesn't match. This is a check on evidence provenance, not on the
  model's reasoning - it confirms a cited source was really retrieved, not
  that the model's conclusion from it is correct.
- **`submit_incident_draft` pseudo-tool.** Rather than asking the model to
  produce well-formed structured output as free text and hoping it
  complies, the final answer is forced through one more "tool call" with a
  strict schema (`orchestrator/draft.py`'s `IncidentDraft`), reusing the
  same typed-arguments machinery the model already uses for every other
  tool rather than introducing a second, less reliable output path.

The loop also has a hard LLM-call timeout (90s) with one retry; if both
attempts fail, the run stops cleanly with `stopped_reason="llm_error"` and
the GUI renders whatever evidence was gathered before the failure rather
than losing it - see the README's Known Limitations for why this happens
and how often.

## RAG design (`rag/`)

- **12 knowledge-base documents**, split by markdown section into 51
  chunks (`rag/chunking.py`) - sized for passage-level citation rather than
  whole-document retrieval, so a cited source points at a specific
  paragraph, not "somewhere in this 2,000-word guide."
- **Hybrid retrieval** (`rag/retrieval.py`): BM25 lexical scoring combined
  with semantic embedding similarity (Ollama `nomic-embed-text`), rather
  than either alone - BM25 alone misses paraphrases, embeddings alone can
  over-match on topical similarity for passages that aren't actually
  relevant.
- **Deliberate prompt-injection test case.** `KB-0099` contains an
  "Automated Assistant Instructions" section with injected text attempting
  to redirect the model's behavior. It is *flagged*, not deleted - the
  retrieval pipeline still returns it like any other chunk, but
  `format_for_prompt` wraps every retrieved chunk as inert, clearly-bounded
  data in the prompt rather than as instructions, so the injection is
  present in the corpus (proving the pipeline doesn't just curate it away)
  but inert when it reaches the model. See `tests/unit/test_rag_retrieval.py`.

## Testing strategy

- **No test depends on Ollama being reachable.** Every orchestrator/agent
  test uses a deterministic `FakeLLM`
  (`tests/unit/test_orchestrator_agent.py`) that returns scripted tool
  calls and a scripted final draft - this is what makes the 333-test suite
  run in ~25s locally and in CI with no external services at all.
- **MCP protocol tests are not mocked at the tool-call layer.** They run a
  real `mcp.client.session.ClientSession` over `mcp.shared.memory`'s
  in-memory transport against the real server objects
  (`tests/mcp_protocol/`), so they exercise actual tool discovery, schema
  validation, and `ToolError` propagation - the same code path production
  traffic uses, just without a network hop.
- **Integration tests use FastAPI's `TestClient`** directly against the
  simulators (`tests/integration/`), covering the same HTTP contract the
  Postman collections in `postman/` check manually.
- **GUI tests use Streamlit's `AppTest`**
  (`tests/unit/test_gui_app.py`), which drives the real `gui/app.py`
  through its actual Streamlit execution model (not a hand-rolled mock of
  it), with the `FakeLLM` behind `async_bridge.py` so the whole pipeline -
  input box to rendered draft - is exercised end-to-end without Ollama.

## A few implementation details worth knowing (not just "what" but "why it was tricky")

- **MCP SDK v2.2** uses `MCPServer`, not the `FastMCP` name from v1 - and
  `tool.input_schema`/`result.is_error`, not the camelCase
  `inputSchema`/`isError` from the TypeScript SDK's naming convention,
  which this project's early drafts briefly (and incorrectly) assumed.
- **anyio `TaskGroup` cancel-scope invariant**: any exception that
  propagates *out* of `connect_inprocess()`'s or `build_registry()`'s
  `async with` block (rather than being caught inside it and re-raised
  after it closes) corrupts anyio's cancel scope for the rest of the
  process. `gui/async_bridge.py` catches every exception path inside these
  context managers and only re-raises once they've closed cleanly.
- **`num_ctx=16384`** is sent explicitly on every Ollama request
  (`orchestrator/llm.py`), not just set once via `/set parameter` in an
  interactive `ollama run` session - the default of 4096 overflows partway
  through a real investigation once 19 tool schemas plus several turns of
  accumulated history are in context, and relying solely on a
  manually-saved model variant is fragile if someone pulls the base model
  fresh.
