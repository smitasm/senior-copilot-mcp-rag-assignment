"""The investigation loop: an LLM-driven, tool-calling agent bounded to
MAX_TURNS, ending only when the model submits a schema-valid IncidentDraft
(or runs out of turns). This is deliberately the ONLY thing the agent does -
it never creates a ticket itself; see orchestrator/actions.py for the
separate, non-LLM-mediated write path that runs after a human approves.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from orchestrator.draft import SUBMIT_DRAFT_TOOL_NAME, IncidentDraft, submit_draft_tool_spec
from orchestrator.llm import LLMClient, LLMTimeoutError, ToolCall
from orchestrator.mcp_client import ToolCallError, ToolRegistry

MAX_TURNS = 8

# A local model that fixates on one easy-schema tool (observed directly: a 3B
# model called search_knowledge_base on all 8 turns and never touched
# list_alarms) must not be allowed to silently burn the whole turn budget.
# REPEAT_WARN_TURNS injects an explicit correction; REPEAT_STOP_TURNS ends the
# run early with a distinct, diagnosable stopped_reason instead of a generic
# "max_turns" that hides what actually happened.
REPEAT_WARN_TURNS = 3
REPEAT_STOP_TURNS = 5

SYSTEM_PROMPT = (
    "You are an incident-preparation assistant for a plant's Alarm Management system. You have tools to "
    "query alarms, assets, analytics, historical tickets, and a knowledge base.\n\n"
    "For a request to prepare an incident for a specific alarm, or for 'the highest-priority active alarm' "
    "in some scope, follow this procedure:\n"
    "1. Call list_alarms to find candidate active alarms in that scope (e.g. site and status=active), "
    "sorted by severity if possible.\n"
    "2. If there is more than one candidate, call priority_score on the top 1-3 candidates by severity to "
    "find the single highest-priority alarm.\n"
    "3. Call operator_recommendations for that alarm, with include_related, include_asset_context and "
    "include_historical_pattern all set to true, to get its likely cause and supporting evidence.\n"
    "4. Call find_similar_tickets for that alarm's asset and alarm_name.\n"
    "5. Call search_knowledge_base ONCE, using the alarm_name and key symptoms as the query, for "
    "supporting reference material.\n"
    "6. Call submit_incident_draft exactly once, as your final step, with a complete, evidence-grounded "
    "draft.\n\n"
    "Do not call the same tool more than twice in a row - if you are unsure what to do next, call "
    "list_alarms. Do not fabricate evidence; every claim in your draft should trace back to a tool result "
    "you actually retrieved. In particular, citations must only list doc_id values that "
    "search_knowledge_base actually returned to you in THIS conversation, and similar_ticket_ids must only "
    "list ticket_id values that find_similar_tickets, list_tickets or get_ticket actually returned to you - "
    "never invent an id, even a plausible-looking one, that you have not actually seen in a tool result. "
    "You do not have a tool to create a ticket directly - drafting is your only job; a human will review "
    "and approve before anything is created."
)


@dataclass
class InvestigationResult:
    draft: IncidentDraft | None
    transcript: list[dict[str, Any]]  # full message history, for a GUI "MCP trace" panel
    tool_calls_made: list[tuple[str, dict[str, Any]]]  # (tool_name, arguments), in order, for grounding audits
    turns_used: int
    stopped_reason: str  # "draft_submitted" | "max_turns" | "stuck_repeating_tool" | "llm_error"


def _unwrap_if_double_wrapped(arguments: dict[str, Any]) -> dict[str, Any]:
    """submit_incident_draft is NOT an MCP tool (its schema is IncidentDraft's
    own fields, flat) - but the model has just seen many real tools that DO
    take a single wrapped `req` argument, and may copy that pattern here too.
    If the only key is `req` and it's a dict, use that instead."""
    if set(arguments) == {"req"} and isinstance(arguments["req"], dict):
        return arguments["req"]
    return arguments


def _record_grounding(name: str, result: Any, seen_doc_ids: set[str], seen_ticket_ids: set[str]) -> None:
    """Track which doc_ids / ticket_ids were ACTUALLY returned by a real tool
    call, so a submitted draft's citations/similar_ticket_ids can be checked
    against reality rather than trusted at face value. Directly motivated by
    a live run: the model submitted citations to KB-0001/KB-0002 - real,
    plausible-sounding doc IDs - without ever having called
    search_knowledge_base at all. The system prompt already says 'do not
    fabricate evidence'; a weak local model cannot be relied on to obey that
    on its own, the same lesson as the approval gate and the repeat-tool
    breaker."""
    if not isinstance(result, dict):
        return
    if name == "search_knowledge_base":
        seen_doc_ids.update(r.get("doc_id") for r in result.get("results", []) if r.get("doc_id"))
    elif name == "find_similar_tickets":
        seen_ticket_ids.update(c.get("ticket", {}).get("ticket_id") for c in result.get("candidates", [])
                               if c.get("ticket", {}).get("ticket_id"))
    elif name == "list_tickets":
        seen_ticket_ids.update(t.get("ticket_id") for t in result.get("data", []) if t.get("ticket_id"))
    elif name == "get_ticket" and result.get("ticket_id"):
        seen_ticket_ids.add(result["ticket_id"])
    elif name == "create_ticket" and result.get("ticket", {}).get("ticket_id"):
        seen_ticket_ids.add(result["ticket"]["ticket_id"])


def _ungrounded_references(draft: IncidentDraft, seen_doc_ids: set[str], seen_ticket_ids: set[str]
                          ) -> tuple[set[str], set[str]]:
    return set(draft.citations) - seen_doc_ids, set(draft.similar_ticket_ids) - seen_ticket_ids


def _grounding_error_text(bad_citations: set[str], bad_tickets: set[str]) -> str:
    parts = [f"{SUBMIT_DRAFT_TOOL_NAME} rejected: this draft references evidence that was not actually "
            "retrieved during this investigation."]
    if bad_citations:
        parts.append(f"citations {sorted(bad_citations)} do not match any doc_id actually returned by "
                     "search_knowledge_base. Call search_knowledge_base and cite only what it actually "
                     "returns, or remove these citations.")
    if bad_tickets:
        parts.append(f"similar_ticket_ids {sorted(bad_tickets)} do not match any ticket_id actually "
                     "returned by find_similar_tickets, list_tickets or get_ticket. Look them up for real, "
                     "or remove these ids.")
    parts.append("Call submit_incident_draft again with corrected values.")
    return " ".join(parts)


async def investigate(user_request: str, llm: LLMClient, tools: ToolRegistry,
                      max_turns: int = MAX_TURNS) -> InvestigationResult:
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_request},
    ]
    tool_specs = tools.llm_tools() + [submit_draft_tool_spec()]
    tool_calls_made: list[tuple[str, dict[str, Any]]] = []
    last_tool_name: str | None = None
    repeat_count = 0
    seen_doc_ids: set[str] = set()
    seen_ticket_ids: set[str] = set()

    for turn in range(1, max_turns + 1):
        try:
            response = await llm.chat(messages, tool_specs)
        except LLMTimeoutError:
            # A transient LLM failure must not discard the tool results and
            # transcript already gathered - a real timed-out run motivated
            # this: turns 1-4 had already made a genuine tool call before
            # turn 5's call to the model failed.
            return InvestigationResult(draft=None, transcript=messages, tool_calls_made=tool_calls_made,
                                       turns_used=turn - 1, stopped_reason="llm_error")

        if not response.has_tool_calls:
            # Plain prose with no tool call: never accept it as the draft.
            # Nudge the model back toward using its tools / submitting.
            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": f"Please continue investigating using the available "
                             f"tools, and call {SUBMIT_DRAFT_TOOL_NAME} with your final draft when ready. "
                             "Do not answer in plain text."})
            continue

        messages.append({"role": "assistant", "content": response.content, "tool_calls": [
            {"id": c.id, "function": {"name": c.name, "arguments": c.arguments}} for c in response.tool_calls
        ]})

        for call in response.tool_calls:
            if call.name == SUBMIT_DRAFT_TOOL_NAME:
                draft = _try_submit(call)
                if draft is None:
                    messages.append({"role": "tool", "content": _draft_validation_error_text(call)})
                    continue
                bad_citations, bad_tickets = _ungrounded_references(draft, seen_doc_ids, seen_ticket_ids)
                if bad_citations or bad_tickets:
                    messages.append({"role": "tool", "content": _grounding_error_text(bad_citations, bad_tickets)})
                    continue
                return InvestigationResult(draft=draft, transcript=messages,
                                           tool_calls_made=tool_calls_made, turns_used=turn,
                                           stopped_reason="draft_submitted")

            tool_calls_made.append((call.name, call.arguments))
            try:
                result = await tools.call(call.name, call.arguments)
                _record_grounding(call.name, result, seen_doc_ids, seen_ticket_ids)
                content = result if isinstance(result, str) else json.dumps(result, default=str)
            except ToolCallError as exc:
                content = f"ERROR: {exc}"
            messages.append({"role": "tool", "content": content})

            repeat_count = repeat_count + 1 if call.name == last_tool_name else 1
            last_tool_name = call.name
            if repeat_count >= REPEAT_STOP_TURNS:
                return InvestigationResult(draft=None, transcript=messages, tool_calls_made=tool_calls_made,
                                           turns_used=turn, stopped_reason="stuck_repeating_tool")
            if repeat_count == REPEAT_WARN_TURNS:
                messages.append({"role": "user", "content": (
                    f"You have called {call.name} {repeat_count} times in a row. Do not call it again. "
                    "If you now have enough information, call submit_incident_draft. Otherwise call a "
                    "DIFFERENT tool - if you have not yet called list_alarms, call that.")})

    return InvestigationResult(draft=None, transcript=messages, tool_calls_made=tool_calls_made,
                               turns_used=max_turns, stopped_reason="max_turns")


def _try_submit(call: ToolCall) -> IncidentDraft | None:
    try:
        return IncidentDraft.model_validate(_unwrap_if_double_wrapped(call.arguments))
    except ValidationError:
        return None


def _draft_validation_error_text(call: ToolCall) -> str:
    try:
        IncidentDraft.model_validate(_unwrap_if_double_wrapped(call.arguments))
    except ValidationError as exc:
        return f"{SUBMIT_DRAFT_TOOL_NAME} rejected - fix these fields and call it again: {exc}"
    return f"{SUBMIT_DRAFT_TOOL_NAME} rejected for an unknown reason - please try again."