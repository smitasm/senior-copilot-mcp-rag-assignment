"""GUI smoke tests via Streamlit's AppTest framework.

Deliberately scoped: these prove the app script loads cleanly, renders the
right thing for each investigation outcome, and gates the write action on
explicit confirmation - the properties that matter for a GUI's correctness
and that AppTest can genuinely verify. They do NOT click "Investigate" or
"Approve & Create Ticket" for real, since app.py's main() calls
run_investigation/approve_and_create with no way to inject a FakeLLM or a
test transport from outside - those functions are already thoroughly tested
directly in test_gui_async_bridge.py, against real in-process simulators.
Re-proving that same logic through a real button click would need live
Ollama and live network servers running during the test, for no real
additional confidence over what's already proven. Manually seeding
st.session_state.result (constructed from the same real IncidentDraft /
InvestigationResult classes app.py itself uses) is the AppTest-recommended
way to test rendering logic in isolation from the async work that produces
that state.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from orchestrator.agent import InvestigationResult
from orchestrator.draft import IncidentDraft

# AppTest.from_file resolves a relative path against the file that CALLS it,
# not the CWD - so a bare "gui/app.py" broke as soon as this ran from
# tests/unit/ instead of the repo root where it was first tried
# interactively. An absolute path computed from this test file's own
# location is correct regardless of where pytest is invoked from.
APP_PATH = str(Path(__file__).resolve().parents[2] / "gui" / "app.py")


def make_draft(**overrides) -> IncidentDraft:
    base = dict(
        alarm_id="ALM-000001", alarm_name="High Vibration", asset_id="AST-0001",
        asset_name="Recycle Gas Compressor K-201", priority_band="critical",
        likely_cause="Bearing wear.", evidence=["Vibration trending up"],
        recommended_actions=["Check lube oil pressure"], similar_ticket_ids=[], citations=[],
        ticket_priority="P1", ticket_short_description="K-201 High Vibration - active critical alarm",
        ticket_description="Automated draft.",
    )
    base.update(overrides)
    return IncidentDraft.model_validate(base)


def make_result(**overrides) -> InvestigationResult:
    base = dict(draft=make_draft(), transcript=[{"role": "system", "content": "x"}],
               tool_calls_made=[("list_alarms", {"site": "EastRefinery"})], turns_used=3,
               stopped_reason="draft_submitted")
    base.update(overrides)
    return InvestigationResult(**base)


def run_with_result(result: InvestigationResult | None) -> AppTest:
    at = AppTest.from_file(APP_PATH)
    at.session_state["result"] = result
    at.run(timeout=15)
    return at


# ---- app loads cleanly -------------------------------------------------------
def test_app_loads_without_exception_and_no_result_yet():
    at = run_with_result(None)
    assert not at.exception
    assert at.title[0].value == "Incident and Ticket Enrichment Copilot"
    assert any(b.label == "Investigate" for b in at.button)


def test_sidebar_has_all_connection_fields():
    at = run_with_result(None)
    labels = {ti.label for ti in at.sidebar.text_input}
    assert labels == {"Alarm API URL", "Ticketing API URL", "API token", "Ollama URL", "Model"}


def test_no_draft_sections_render_before_any_investigation_runs():
    at = run_with_result(None)
    assert at.subheader == []


# ---- check-connections must not crash when backends are unreachable -----------------------
def test_check_connections_handles_unreachable_backends_without_crashing():
    """Points the sidebar URLs at a port nothing could plausibly be
    listening on, rather than relying on localhost:8000/8001 genuinely
    being down - on a developer's own machine (exactly where this project
    lives), those are very likely to actually be running in another
    terminal, which made an earlier version of this test fail not because
    anything was broken, but because its own assumption about ambient
    machine state was wrong."""
    at = run_with_result(None)
    for ti in at.sidebar.text_input:
        if ti.label in ("Alarm API URL", "Ticketing API URL"):
            ti.set_value("http://127.0.0.1:1")  # port 1 is reserved; nothing binds to it
    at.run(timeout=15)

    check_btn = next(b for b in at.sidebar.button if b.label == "Check connections")
    check_btn.click().run(timeout=15)
    assert not at.exception
    errors = [e.value for e in at.sidebar.error]
    assert len(errors) == 2 and all("NOT reachable" in e for e in errors)


# ---- successful draft renders correctly ------------------------------------------------
def test_draft_form_renders_all_real_content():
    at = run_with_result(make_result())
    assert not at.exception
    subheaders = {h.value for h in at.subheader}
    assert subheaders == {"Incident draft", "Citations", "Similar tickets", "Ticket to create"}
    metrics = {m.label: m.value for m in at.metric}
    assert metrics == {"Alarm": "High Vibration", "Asset": "Recycle Gas Compressor K-201",
                       "Priority band": "CRITICAL"}
    likely_cause = next(ta for ta in at.text_area if ta.label == "Likely cause")
    assert likely_cause.value == "Bearing wear."


def test_citation_doc_id_resolves_to_its_real_title():
    at = run_with_result(make_result(draft=make_draft(citations=["KB-0001"])))
    assert any("KB-0001" in m.value and "Troubleshooting High Vibration" in m.value for m in at.markdown)


def test_unknown_citation_id_does_not_crash_the_render():
    at = run_with_result(make_result(draft=make_draft(citations=["KB-9999"])))
    assert not at.exception
    assert any("KB-9999" in m.value and "unknown document" in m.value for m in at.markdown)


def test_similar_ticket_ids_are_listed_before_details_are_loaded():
    at = run_with_result(make_result(draft=make_draft(similar_ticket_ids=["INC0000001"])))
    assert any("INC0000001" in c.value for c in at.caption)


# ---- approval is gated on the confirmation checkbox --------------------------------------
def test_approve_button_is_disabled_until_the_checkbox_is_checked():
    at = run_with_result(make_result())
    approve = next(b for b in at.button if b.label == "Approve & Create Ticket")
    assert approve.disabled is True

    at.checkbox[0].check().run(timeout=15)
    approve = next(b for b in at.button if b.label == "Approve & Create Ticket")
    assert approve.disabled is False


# ---- every non-submitted stopped_reason renders its own explanation, not a draft ---------------
# Expected text is hardcoded here rather than imported from gui.app's own
# STOPPED_REASON_MESSAGES - importing it would compare the rendered message
# against the very dict that produced it, which stays trivially "equal" even
# if all three messages were collapsed into one generic string (confirmed:
# that exact mutation passed every test here until this was fixed to check
# independently-stated expected text instead).
EXPECTED_FAILURE_MESSAGES = {
    "max_turns": "without submitting a final draft",
    "stuck_repeating_tool": "got stuck calling the same tool repeatedly",
    "llm_error": "did not respond in time",
}


@pytest.mark.parametrize("reason", ["max_turns", "stuck_repeating_tool", "llm_error"])
def test_each_failure_mode_renders_its_own_message_and_no_draft_form(reason):
    at = run_with_result(make_result(draft=None, stopped_reason=reason))
    assert not at.exception
    assert at.subheader == []  # no draft-form sections
    assert len(at.warning) == 1 and EXPECTED_FAILURE_MESSAGES[reason] in at.warning[0].value


def test_the_three_failure_messages_are_genuinely_different_from_each_other():
    """Directly guards against the exact mutation that slipped through
    before this test existed: all three STOPPED_REASON_MESSAGES collapsed
    into one identical generic string."""
    from gui.app import STOPPED_REASON_MESSAGES
    values = list(STOPPED_REASON_MESSAGES.values())
    assert len(set(values)) == len(values) == 3


def test_mcp_trace_is_shown_even_when_no_draft_was_produced():
    at = run_with_result(make_result(draft=None, stopped_reason="max_turns",
                                     tool_calls_made=[("list_alarms", {"site": "EastRefinery"})]))
    assert any("1 tool call" in e.label for e in at.expander)


# ---- edited fields survive a rerun (not wiped by Streamlit's rerun cycle) ----------------------
def test_editing_likely_cause_persists_across_a_rerun():
    at = run_with_result(make_result())
    likely_cause = next(ta for ta in at.text_area if ta.label == "Likely cause")
    likely_cause.set_value("Edited cause text.").run(timeout=15)
    likely_cause_after = next(ta for ta in at.text_area if ta.label == "Likely cause")
    assert likely_cause_after.value == "Edited cause text."


def test_a_new_draft_with_a_different_alarm_id_resets_the_edit_fields():
    """_seed_edit_fields only re-seeds on a NEW alarm_id - proves it actually
    does re-seed when the alarm genuinely changes, the other half of the
    property test_editing_likely_cause_persists_across_a_rerun checks."""
    at = run_with_result(make_result())
    likely_cause = next(ta for ta in at.text_area if ta.label == "Likely cause")
    likely_cause.set_value("Edited cause text.").run(timeout=15)

    new_result = make_result(draft=make_draft(alarm_id="ALM-999999", likely_cause="Different real cause."))
    at.session_state["result"] = new_result
    at.session_state["loaded_draft_alarm_id"] = None  # mirrors what the Investigate button does
    at.run(timeout=15)
    likely_cause_after = next(ta for ta in at.text_area if ta.label == "Likely cause")
    assert likely_cause_after.value == "Different real cause."


# ---- audit trail ------------------------------------------------------------------------------
def test_audit_trail_is_hidden_when_empty():
    at = run_with_result(None)
    assert "Audit trail (this session)" not in {h.value for h in at.subheader}


def test_audit_trail_renders_a_logged_entry():
    """Checks the hand-built markdown table (not st.table/pandas - see
    _render_audit_trail's docstring for why) contains the logged entry's
    real values."""
    at = AppTest.from_file(APP_PATH)
    at.session_state["audit_log"] = [{"time": "2026-09-30 12:00:00 UTC", "alarm": "High Vibration",
                                      "asset": "K-201", "ticket_id": "INC0000019", "duplicate": False}]
    at.run(timeout=15)
    assert "Audit trail (this session)" in {h.value for h in at.subheader}
    table_md = next(m.value for m in at.markdown if "INC0000019" in m.value)
    assert "High Vibration" in table_md and "K-201" in table_md and "| no |" in table_md


def test_audit_trail_marks_a_duplicate_entry_distinctly():
    at = AppTest.from_file(APP_PATH)
    at.session_state["audit_log"] = [{"time": "2026-09-30 12:00:00 UTC", "alarm": "Low Lube Oil Pressure",
                                      "asset": "K-202", "ticket_id": "INC0000018", "duplicate": True}]
    at.run(timeout=15)
    table_md = next(m.value for m in at.markdown if "INC0000018" in m.value)
    assert "| yes |" in table_md