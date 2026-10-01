"""Streamlit GUI for the Incident and Ticket Enrichment Copilot.

Deliberately thin: almost everything here is rendering. The actual
orchestration logic (connecting MCP servers, running the investigation,
creating a ticket, health-checking the simulators) lives in
gui/async_bridge.py and is unit-tested there - Streamlit apps themselves are
hard to test meaningfully, so the split keeps the testable logic testable.

Run:    streamlit run gui/app.py
Needs:  the Alarm API and Ticketing API running (see sidebar for URLs) and
        Ollama running locally with the configured model pulled.
"""

from __future__ import annotations

import asyncio
import os
from datetime import datetime, timezone

import streamlit as st

from gui.async_bridge import (
    ApprovalRequiredError, ConnectionConfig, approve_and_create, check_connections, fetch_ticket_details,
    run_investigation,
)
from orchestrator.draft import IncidentDraft
from rag.documents import DOCUMENTS

DOC_TITLES = {d.doc_id: d.title for d in DOCUMENTS}

STOPPED_REASON_MESSAGES = {
    "max_turns": "The investigation used all of its turns without submitting a final draft. The tool "
                "calls below show what it found along the way - you can try again, or narrow the request.",
    "stuck_repeating_tool": "The model got stuck calling the same tool repeatedly without making "
                            "progress, so the investigation was stopped early rather than wasting the "
                            "rest of its turn budget. Try again, or try a smaller/faster model.",
    "llm_error": "The local model did not respond in time (even after a retry). Evidence gathered before "
                "the failure is preserved below. Check that Ollama is running and responsive, then try "
                "again.",
}


def _run_async(coro):
    return asyncio.run(coro)


def _cfg_from_sidebar() -> ConnectionConfig:
    st.sidebar.header("Connections")
    alarm_url = st.sidebar.text_input("Alarm API URL", os.getenv("ALARM_API_URL", "http://localhost:8000"))
    ticket_url = st.sidebar.text_input("Ticketing API URL", os.getenv("TICKETING_API_URL", "http://localhost:8001"))
    token = st.sidebar.text_input("API token", os.getenv("ALARM_API_TOKEN", "demo-token"), type="password")
    st.sidebar.header("Local model")
    ollama_url = st.sidebar.text_input("Ollama URL", os.getenv("OLLAMA_URL", "http://localhost:11434"))
    ollama_model = st.sidebar.text_input("Model", os.getenv("OLLAMA_MODEL", "qwen2.5:3b"))
    cfg = ConnectionConfig(alarm_api_url=alarm_url, ticketing_api_url=ticket_url, api_token=token,
                           ollama_base_url=ollama_url, ollama_model=ollama_model)

    if st.sidebar.button("Check connections"):
        with st.sidebar:
            with st.spinner("Checking..."):
                health = _run_async(check_connections(cfg))
            for name, ok in health.items():
                (st.success if ok else st.error)(f"{name}: {'reachable' if ok else 'NOT reachable'}")
    return cfg


def _init_state() -> None:
    st.session_state.setdefault("result", None)
    st.session_state.setdefault("audit_log", [])
    st.session_state.setdefault("loaded_draft_alarm_id", None)


def _seed_edit_fields(draft: IncidentDraft) -> None:
    """Widget values are seeded from a NEW draft only once (keyed on
    alarm_id) - not on every Streamlit rerun, or the user's own edits would
    be wiped out every time they type a character."""
    if st.session_state.loaded_draft_alarm_id == draft.alarm_id:
        return
    st.session_state.loaded_draft_alarm_id = draft.alarm_id
    st.session_state.edit_likely_cause = draft.likely_cause
    st.session_state.edit_evidence = "\n".join(draft.evidence)
    st.session_state.edit_actions = "\n".join(draft.recommended_actions)
    st.session_state.edit_priority = draft.ticket_priority.value
    st.session_state.edit_short_desc = draft.ticket_short_description
    st.session_state.edit_description = draft.ticket_description
    st.session_state.approved_checkbox = False


def _draft_from_edit_fields(draft: IncidentDraft) -> IncidentDraft:
    """Rebuilds an IncidentDraft from the (possibly user-edited) widget
    values, through the SAME Pydantic validation the model's own submission
    was checked against - a bad edit (e.g. an empty required field, or a
    short_description over 160 chars) is caught here the same way it would
    have been caught during the investigation itself."""
    return IncidentDraft(
        alarm_id=draft.alarm_id, alarm_name=draft.alarm_name, asset_id=draft.asset_id,
        asset_name=draft.asset_name, priority_band=draft.priority_band,
        likely_cause=st.session_state.edit_likely_cause,
        evidence=[line.strip() for line in st.session_state.edit_evidence.splitlines() if line.strip()],
        recommended_actions=[line.strip() for line in st.session_state.edit_actions.splitlines() if line.strip()],
        similar_ticket_ids=draft.similar_ticket_ids, citations=draft.citations,
        ticket_priority=st.session_state.edit_priority, ticket_short_description=st.session_state.edit_short_desc,
        ticket_description=st.session_state.edit_description,
    )


def _render_mcp_trace(result) -> None:
    with st.expander(f"MCP trace - {len(result.tool_calls_made)} tool call(s)", expanded=False):
        if not result.tool_calls_made:
            st.caption("No tool calls were made.")
        for i, (name, args) in enumerate(result.tool_calls_made, start=1):
            st.markdown(f"**{i}. `{name}`**")
            st.json(args, expanded=False)


def _render_citations(citations: list[str]) -> None:
    if not citations:
        st.caption("No knowledge-base articles were cited.")
        return
    for doc_id in citations:
        st.markdown(f"- **{doc_id}**: {DOC_TITLES.get(doc_id, '(unknown document)')}")


def _render_similar_tickets(ticket_ids: list[str], cfg: ConnectionConfig) -> None:
    if not ticket_ids:
        st.caption("No similar historical tickets were found.")
        return
    if st.button("Load similar ticket details", key="load_tickets_btn"):
        with st.spinner("Fetching ticket details..."):
            st.session_state.similar_ticket_details = _run_async(fetch_ticket_details(ticket_ids, cfg))
    for t in st.session_state.get("similar_ticket_details", []):
        with st.expander(f"{t['ticket_id']} - {t['short_description']} ({t['state']})"):
            st.write(f"**Asset:** {t['asset_name']}  |  **Priority:** {t['priority']}  |  "
                    f"**Opened:** {t['opened_at']}")
            if t.get("resolution_notes"):
                st.write(f"**Resolution:** {t['resolution_notes']}")
    if not st.session_state.get("similar_ticket_details") and not st.session_state.get("load_tickets_btn"):
        st.caption(f"{len(ticket_ids)} similar ticket(s) found: {', '.join(ticket_ids)} "
                  "(click above to load full details)")


def _render_draft(result, cfg: ConnectionConfig) -> None:
    draft = result.draft
    _seed_edit_fields(draft)

    st.subheader("Incident draft")
    c1, c2, c3 = st.columns(3)
    c1.metric("Alarm", draft.alarm_name)
    c2.metric("Asset", draft.asset_name)
    c3.metric("Priority band", draft.priority_band.upper())
    st.caption(f"alarm_id: `{draft.alarm_id}`  |  asset_id: `{draft.asset_id}`")

    st.text_area("Likely cause", key="edit_likely_cause", height=80)
    st.text_area("Evidence (one per line)", key="edit_evidence", height=100)
    st.text_area("Recommended actions (one per line)", key="edit_actions", height=100)

    st.subheader("Citations")
    _render_citations(draft.citations)

    st.subheader("Similar tickets")
    _render_similar_tickets(draft.similar_ticket_ids, cfg)

    st.subheader("Ticket to create")
    st.selectbox("Priority", ["P1", "P2", "P3", "P4"], key="edit_priority")
    st.text_input("Short description", key="edit_short_desc", max_chars=160)
    st.text_area("Description", key="edit_description", height=120)

    _render_mcp_trace(result)

    st.divider()
    approved = st.checkbox("I have reviewed this draft and approve creating this ticket exactly as shown "
                           "above.", key="approved_checkbox")
    if st.button("Approve & Create Ticket", type="primary", disabled=not approved):
        try:
            final_draft = _draft_from_edit_fields(draft)
        except Exception as exc:
            st.error(f"Cannot create ticket - the edited content is invalid: {exc}")
            return
        with st.spinner("Creating ticket..."):
            try:
                created = _run_async(approve_and_create(final_draft, cfg, approved=True))
            except ApprovalRequiredError as exc:
                st.error(str(exc))
                return
            except Exception as exc:
                st.error(f"Failed to create the ticket: {exc}")
                return
        ticket = created["ticket"]
        if created["duplicate_of"]:
            st.warning(f"An open ticket already covers this asset and alarm: **{ticket['ticket_id']}** "
                      f"({ticket['state']}). No new ticket was created.")
        else:
            st.success(f"Ticket **{ticket['ticket_id']}** created.")
        st.session_state.audit_log.append({
            "time": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            "alarm": draft.alarm_name, "asset": draft.asset_name,
            "ticket_id": ticket["ticket_id"], "duplicate": bool(created["duplicate_of"]),
        })


def _render_no_draft(result) -> None:
    st.warning(STOPPED_REASON_MESSAGES.get(result.stopped_reason, f"Stopped: {result.stopped_reason}"))
    _render_mcp_trace(result)


def _render_audit_trail() -> None:
    """Renders as a hand-built markdown table rather than st.table(), which
    pulls in the full pandas+pyarrow dependency chain just to display a
    handful of text rows - real overhead in the Docker image, and its
    cold-start cost caused a 15s+ AppTest timeout on a fresh Windows install
    (freshly-installed pandas 3.0.6 in particular). Nothing here needs a
    DataFrame: no sorting, filtering, or numeric columns, just text."""
    if not st.session_state.audit_log:
        return
    st.divider()
    st.subheader("Audit trail (this session)")
    header = "| Time | Alarm | Asset | Ticket | Duplicate |\n|---|---|---|---|---|"
    rows = "\n".join(
        f"| {e['time']} | {e['alarm']} | {e['asset']} | {e['ticket_id']} | {'yes' if e['duplicate'] else 'no'} |"
        for e in st.session_state.audit_log
    )
    st.markdown(f"{header}\n{rows}")


def main() -> None:
    st.set_page_config(page_title="Incident and Ticket Enrichment Copilot", layout="wide")
    _init_state()
    st.title("Incident and Ticket Enrichment Copilot")
    st.caption("Investigates an alarm using MCP tools over the Alarm API, Ticketing API and a knowledge "
              "base, then drafts an incident for your review. Nothing is written until you approve it.")

    cfg = _cfg_from_sidebar()

    request = st.text_area("What would you like investigated?",
                           value="Prepare an incident for the highest-priority active alarm in EastRefinery.",
                           height=70)
    if st.button("Investigate", type="primary"):
        st.session_state.loaded_draft_alarm_id = None  # force re-seeding of edit fields for the new draft
        st.session_state.pop("similar_ticket_details", None)
        with st.spinner("Investigating - this typically takes 1-3 minutes on a local model..."):
            try:
                st.session_state.result = _run_async(run_investigation(request, cfg))
            except Exception as exc:
                st.session_state.result = None
                st.error(f"Investigation failed unexpectedly: {exc}")

    result = st.session_state.result
    if result is not None:
        if result.stopped_reason == "draft_submitted":
            _render_draft(result, cfg)
        else:
            _render_no_draft(result)

    _render_audit_trail()


if __name__ == "__main__":
    main()