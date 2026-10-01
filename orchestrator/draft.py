"""The agent's final output. Rather than parsing free-form LLM prose (fragile,
especially on a small local model), the agent FORCES the model to submit its
conclusion as a call to a pseudo-tool, `submit_incident_draft`, whose
arguments are validated against IncidentDraft - the same structured-output
pattern used throughout this project (models.py in Phase 1, every MCP tool in
Phase 3), now applied to the model's final answer instead of just its
intermediate tool arguments. A bad submission doesn't crash the run; agent.py
feeds the validation error back and lets the model retry.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from ticketing_api.models import TicketPriority

SUBMIT_DRAFT_TOOL_NAME = "submit_incident_draft"


class IncidentDraft(BaseModel):
    alarm_id: str = Field(description="The alarm_id this incident is about.")
    alarm_name: str = Field(description="The alarm_name this incident is about, e.g. 'High Vibration' - "
                            "required so the ticketing system can detect and reuse an existing open ticket "
                            "for the same asset and alarm instead of creating a duplicate.")
    asset_id: str
    asset_name: str
    priority_band: str = Field(description="low | medium | high | critical, from priority_score.")
    likely_cause: str = Field(min_length=1, description="One or two sentences on the probable root cause, "
                              "grounded in the evidence actually gathered - not invented.")
    evidence: list[str] = Field(min_length=1, description="Bullet-point facts supporting likely_cause, each "
                                "traceable to a specific tool result, e.g. 'K-202 Low Lube Oil Pressure fired "
                                "6 minutes before this alarm' or 'KB-0001: bearing wear shows a gradual "
                                "vibration increase, matching this alarm's trend'.")
    recommended_actions: list[str] = Field(min_length=1)
    similar_ticket_ids: list[str] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list, description="doc_id values from search_knowledge_base "
                                 "whose content informed likely_cause or recommended_actions.")
    ticket_priority: TicketPriority
    ticket_short_description: str = Field(min_length=1, max_length=160)
    ticket_description: str = Field(min_length=1)


def submit_draft_tool_spec() -> dict:
    return {"type": "function", "function": {
        "name": SUBMIT_DRAFT_TOOL_NAME,
        "description": "Submit your final incident draft once you have gathered enough evidence: the "
                       "alarm's priority score, its likely cause and recommended actions via "
                       "operator_recommendations, and relevant similar tickets and/or knowledge-base "
                       "articles. Call this exactly once, as your final step. You do not have a tool to "
                       "create a ticket - this draft will be reviewed by a human before anything is "
                       "created.",
        "parameters": IncidentDraft.model_json_schema(),
    }}