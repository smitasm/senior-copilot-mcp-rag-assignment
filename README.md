# Incident and Ticket Enrichment Copilot

An agent that investigates a process-plant alarm, cites the knowledge-base
articles it actually consulted, drafts a structured incident report, and
only writes anything to the ticketing system after a human explicitly
approves it. Nothing is written automatically, ever.

Built for the ABB final-round assignment. Three simulated/mocked backends
(an Alarm API, a ServiceNow-style Ticketing API, and a RAG knowledge base),
reached exclusively through MCP tools, orchestrated by a local LLM
(Ollama / qwen2.5:3b) behind a Streamlit GUI.

```
"Prepare an incident for the highest-priority active alarm in EastRefinery."
   -> list_alarms -> priority_score -> operator_recommendations
   -> search_knowledge_base -> [cited draft, shown for review]
   -> (you click Approve) -> create_ticket (only now, only if approved)
```

## Quick start

### Docker (recommended - this is what `docker compose up --build` runs)

```bash
docker compose up --build
```

This starts three containers from one shared image (see [`Dockerfile`](Dockerfile)):
`alarm-api` (:8000), `ticketing-api` (:8001), and `gui` (:8501), with the GUI
waiting on the other two's health checks before it starts. Open
**http://localhost:8501**.

Ollama is **not** containerized - it runs on your host (this is how it was
built and tested throughout: native Ollama has GPU access a container
usually doesn't). The `gui` container reaches it at
`host.docker.internal:11434`, which Docker Desktop (Windows/Mac) provides
automatically; `docker-compose.yml` adds the `extra_hosts` entry needed for
that hostname to also resolve on a native Linux Docker host.

**Before you start the stack**, make sure Ollama is running with the model
pulled and its context window raised above the 4096 default (19 MCP tool
schemas plus a growing conversation overflow it otherwise - see
[Known limitations](#known-limitations)):

```bash
ollama pull qwen2.5:3b
ollama pull nomic-embed-text
ollama run qwen2.5:3b
/set parameter num_ctx 16384
/save qwen2.5:3b
```

Run the test suite inside a fresh container (no local Python needed):

```bash
docker compose run --rm gui python -m pytest tests -q
```

### Local (no Docker)

```bash
python -m venv .venv && source .venv/bin/activate   # .venv\Scripts\activate on Windows
pip install -r requirements.txt

# three terminals:
python -m uvicorn alarm_api.main:app --port 8000
python -m uvicorn ticketing_api.main:app --port 8001
python -m streamlit run gui/app.py
```

See [`.env.example`](.env.example) for every configurable value (all have
working defaults - copying it is optional, not required).

## What you're looking at

The GUI has one input box ("What would you like investigated?") and shows,
after you click Investigate:

- **An editable incident draft** - alarm summary, likely cause, recommended
  action, severity - with every factual claim traceable to a specific MCP
  tool call or knowledge-base citation.
- **An MCP trace panel** - every tool the agent actually called, with its
  arguments, in order. This is the receipt that proves the agent used MCP
  rather than hallucinating the alarm data.
- **A similar/duplicate tickets panel** - so you don't file a second ticket
  for something already open (see K-202 in the seed data for a deliberate
  example).
- **An approval checkbox**, gating the only write action in the whole
  system (`create_ticket`). Nothing is written to the ticketing system until
  you tick it and confirm.

## Architecture

```
 Streamlit GUI (gui/app.py)
   |  (async_bridge.py - bridges Streamlit's sync model to the async agent loop)
   v
 Orchestrator agent (orchestrator/agent.py)
   |  - bounded loop, max 8 turns
   |  - repeat-tool circuit breaker (warns at 3, stops at 5 identical calls)
   |  - grounding check: every citation/ticket ID in the final draft must
   |    trace back to a tool the agent actually called this run
   |  - forces structured output via a submit_incident_draft pseudo-tool,
   |    rather than trusting the model to format free text correctly
   v
 In-process MCP clients, one ClientSession per server (orchestrator/mcp_client.py)
   |
   +--> alarm_server.py (14 tools)  --> Alarm API simulator (FastAPI, :8000)
   +--> ticketing_server.py (4 tools, create_ticket gated on approved=True)
   |                                --> Ticketing API mock (FastAPI, :8001)
   +--> knowledge_server.py (1 tool: search_knowledge_base)
                                     --> rag/ (hybrid BM25 + semantic retrieval
                                              over 12 KB docs, 51 chunks)
```

**Why MCP servers run in-process rather than as separate containers/processes:**
the assignment requires the agent to reach the Alarm API *through* MCP, not
directly - an in-process `ClientSession` over an in-memory transport
(`mcp.shared.memory`) gives that guarantee with zero network hops to debug
and no extra containers to orchestrate, while still being the real MCP
protocol (tool discovery, typed schemas, `ToolError` propagation) rather
than a shortcut that merely looks like it.

**Why the approval gate lives in the MCP tool itself
(`mcp_servers/ticketing_server.py`), not just in the GUI:** a GUI-only gate
is trivially bypassed by anything that talks to the MCP server directly
(another client, a future integration, a bug in the GUI's own state
handling). `create_ticket` raises `ApprovalRequiredError` before any HTTP
call is made unless `approved=True` is passed explicitly - the GUI is the
only caller that ever sets it, and only after the checkbox is ticked.

**Why a repeat-tool circuit breaker and a grounding check:** both came from
watching the real 3B model misbehave during development, not from
speculation. The circuit breaker exists because an early qwen2.5:3b run
called `search_knowledge_base` eight times in a row with no new arguments,
burning the turn budget. The grounding check exists because a separate run
fabricated citations to `KB-0001`/`KB-0002` without ever calling
`search_knowledge_base` that turn - the check catches exactly that by
diffing the draft's claimed sources against the tool calls actually
recorded in the trace.

For the full design rationale behind every one of these decisions, the
complete 19-tool MCP catalog, and the RAG retrieval design, see
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Project structure

```
alarm_api/        Alarm API simulator - models, seed data, services, FastAPI app
ticketing_api/     Ticketing API mock - same shape, ServiceNow-style ITSM model
mcp_common/        Shared HTTP client (retry/timeout) used by both MCP servers
mcp_servers/        alarm_server.py (14 tools), ticketing_server.py (4 tools),
                    knowledge_server.py (1 RAG tool)
rag/               Document loading, chunking, embeddings, hybrid retrieval
orchestrator/       llm.py, mcp_client.py, draft.py, agent.py, actions.py
gui/               async_bridge.py, app.py (Streamlit)
postman/           Contract tests + scenario/chaining collections for the
                    Alarm API simulator (run via Newman; see docs/ARCHITECTURE.md)
scripts/           check_newman_report.py
tests/             unit/, integration/, mcp_protocol/ - 333 tests, no
                    external dependencies, no Ollama required
docs/              ARCHITECTURE.md, DEMO_SCRIPT.md
```

## Testing

```bash
python -m pytest tests -q          # 333 tests, ~25s, nothing external required
python -m pytest tests --cov       # with coverage
```

Every test that touches the "LLM" uses a deterministic `FakeLLM`
(see `tests/unit/test_orchestrator_agent.py`) - the suite never calls
Ollama, so it's exactly as fast and reliable in CI as it is locally. MCP
protocol tests run through a real `ClientSession` over in-memory streams
(not mocked tool calls), so they exercise the actual wire-level contract.

GitHub Actions ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)) runs
this suite plus a Docker build check on every push and PR.

## Known limitations

- **Local-model inference time is variable.** On CPU-only inference,
  individual turns have taken anywhere from a few seconds to over 90s
  depending on system load and how long the conversation has grown. The
  agent has a 90s timeout with one retry per turn; if both attempts time
  out, the run stops cleanly with `stopped_reason="llm_error"` and the GUI
  shows exactly what evidence was gathered before the failure (it does not
  crash or lose the trace - this is a demonstrated, intentional failure
  mode, not an unhandled one).
- **Grounding is a heuristic, not a proof.** The grounding check confirms
  that every cited KB ID or ticket ID in the final draft matches something
  the agent's tool calls actually returned this run - it cannot verify the
  model's *reasoning* about that evidence is correct, only that the
  evidence it's citing is real and was actually retrieved.
- **Newman/Postman contract tests are not wired into CI** - they're
  documented as a manual verification step (`postman/`) rather than
  automated, to keep CI fast and dependency-light; the same HTTP contract
  is separately covered by `tests/integration/` via FastAPI's TestClient.
- **Single-user, single-session.** There's no auth beyond the shared dev
  token, no multi-tenant isolation, and no persistence of past
  investigations beyond the in-session audit trail shown in the GUI.
- **GUI tests use Streamlit's `AppTest`**, which exercises the app's logic
  and rendering but not real browser interaction (file uploads, JS-driven
  widgets, etc. - none of which this app uses, but worth naming as a scope
  boundary).

## License

Built as a take-home assignment; not intended for production use as-is.
