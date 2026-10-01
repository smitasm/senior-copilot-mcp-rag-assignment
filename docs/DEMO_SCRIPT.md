# Demo video script (target: 8-10 minutes, hard cap per the assignment)

Written as a shot list, not a transcript to read word-for-word - say it in
your own words, but hit every numbered beat in order. Each beat names
*why* it's there: it maps to something specific the evaluators check (see
the bracketed note), so skipping one isn't just "less thorough," it's a
gap in what's actually being graded.

Record the K-201 High Vibration scenario - it's the one the seed data was
built around (critical, active, unacknowledged - see README). If the local
model times out mid-recording the way it did during development, **that's
a fine thing to leave in**: narrate what happened ("the model didn't
respond in time, and here's the evidence it preserved instead of
crashing") rather than cutting and re-recording for a clean take. A real
failure handled gracefully is more convincing than a demo that implies this
never happens.

## 1. Cold open (0:00-0:30)

One sentence: what this is and the one rule that matters most. *"This is
an incident copilot that investigates a plant alarm through MCP tools,
cites its sources, and never writes a ticket without my explicit
approval."*

## 2. `docker compose up --build` (0:30-1:30)

Run it live from a clean `git clone` if you can manage the timing; if a
full build is too slow to show in real time, cut to it already running and
say so. Point out the three services coming up
(`alarm-api`, `ticketing-api`, `gui`) and that the GUI waits on the other
two's health checks before starting - then open `http://localhost:8501`.
**[Checks: "`docker compose up --build` must work."]**

## 3. Orient on the GUI (1:30-2:00)

Sidebar: connection URLs, Ollama model. Main panel: the single input box.
Nothing clever here - just enough that the next section makes sense.

## 4. Run the investigation, narrating the MCP trace live (2:00-4:30)

Type: *"Prepare an incident for the highest-priority active alarm in
EastRefinery."* Click Investigate.

As the MCP trace panel fills in, **say the tool names out loud as they
appear** and explicitly name what they prove:

> "`list_alarms` - that's a real MCP tool call against the Alarm API
> simulator, not the orchestrator reading the database directly. `priority_score`,
> `operator_recommendations` - same thing, same client session."

**[Checks: "Copilot MUST use MCP for Alarm API, not direct calls" - this
is the one beat in the whole video that most directly proves it, so don't
rush past it.]**

## 5. Point at the RAG citation in the draft (4:30-5:30)

When `search_knowledge_base` appears in the trace and the draft shows a
`KB-XXXX` citation, pause on it:

> "This citation isn't a separate RAG system bolted on the side -
> `search_knowledge_base` is tool #19 in the exact same MCP registry as the
> alarm tools you just saw. Same client session, same trace panel."

**[Checks: "MCP and RAG must participate in SAME workflow" and "Source
citations must be visible in GUI" - both land in this one beat.]**

## 6. Similar/duplicate tickets panel (5:30-6:15)

Show the panel that checks for an existing open ticket before anything
gets created. If investigating K-202 specifically, this is where its
pre-existing Low Lube Oil Pressure ticket shows up as a deliberate
duplicate-detection example; for K-201, show the panel confirming nothing
is open for it yet, which is equally worth a sentence ("nothing open, so
creating one is the right call").

## 7. The approval gate (6:15-7:30)

Scroll to the approval checkbox. **Leave it unchecked and show that
nothing is written yet** - the draft is sitting there fully formed, but no
ticket exists. Then check it, confirm, and show the ticket actually appear
(either the GUI's own confirmation or a quick look at the ticketing API's
`/tickets` listing).

**[Checks: "Write operations require explicit confirmation" - the
unchecked-state pause before you tick it is what actually demonstrates the
gate, not just the after state.]**

## 8. Tests and CI, briefly (7:30-8:15)

`python -m pytest tests -q` (or `docker compose run --rm gui python -m
pytest tests -q`) - let the "333 passed" line sit on screen for a couple
seconds. Flash the `.github/workflows/ci.yml` file or the Actions tab.
Mention in passing: no secrets committed, `.env.example` only - everything
in the repo runs with the default dev token.

## 9. Close (8:15-9:30)

Thirty seconds on the architecture diagram from `README.md` /
`docs/ARCHITECTURE.md` - orchestrator, three MCP servers, the approval
gate living in the tool itself rather than only the GUI. One honest
sentence on the known limitation that matters most: local CPU inference
time varies, and the system is built to degrade cleanly rather than crash
when it runs long - if you saw that happen during recording, this is where
you point back to it.

Stop talking. End the recording.
