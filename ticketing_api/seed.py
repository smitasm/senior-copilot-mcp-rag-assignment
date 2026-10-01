"""Deterministic historical ticket dataset for the mock ticketing system.

Reuses the SAME assets and alarm_name vocabulary as alarm_api.seed, so a
ticket search by alarm_name or asset_id lines up with real alarm data. `now`
defaults to alarm_api.seed.SEED_NOW, so a ticket the copilot creates "today"
sits on the same timeline as the active alarms.

Storylines (deliberately, for the demo):
  * K-201 "High Vibration" has 2 resolved tickets in its history -> the
    grounding evidence for the copilot's "likely cause" on the active alarm.
  * K-202 "Low Lube Oil Pressure" has ONE STILL-OPEN ticket -> tests / demos
    the "don't create a duplicate, an open ticket already covers this" path.
  * BFP101 has several resolved tickets (seal, bearing) -> recurring-issue
    evidence lining up with alarm_api's BFP101 storyline.
  * P-401 has a resolved ticket whose note matches the rationalization
    suggestion ("adjusted alarm deadband") -> ties two subsystems together.
  * A few tickets carry no alarm_id/alarm_name (generic PM work) so search
    filters are exercised against noise, not just signal.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

from alarm_api.seed import BFP101, BFP102, K201, K202, K203, SEED_NOW, build_dataset
from ticketing_api.models import Ticket, TicketPriority, TicketState

UTC = timezone.utc
DEFAULT_SEED = 42
TICKET_START = datetime(2026, 4, 5, tzinfo=UTC)

REPORTERS = ["R. Kumar (Board Operator)", "S. Iyer (Board Operator)", "A. Mehta (Field Operator)",
            "P. Rao (Shift Supervisor)", "N. Das (Field Operator)"]
TECHNICIANS = ["J. Fernandes (Rotating Equip.)", "K. Bose (Instrumentation)", "M. Pillai (Electrical)",
              "T. Nair (Mechanical)", "V. Menon (Boiler Systems)"]


@dataclass
class _Raw:
    asset: str
    short: str
    desc: str
    priority: TicketPriority
    opened_offset_days: float
    duration_hours: float  # to resolution; None-like via still_open flag
    still_open: bool = False
    alarm_name: str | None = None
    resolution: str | None = None
    tags: list[str] = field(default_factory=list)


def _spec(raw: list[_Raw]) -> list[_Raw]:
    return raw


RAW: list[_Raw] = [
    # ---- K-201 High Vibration: resolved history (grounds the active-alarm demo) ----
    _Raw(K201, "Compressor K-201 high vibration on startup", "Vibration alarm during startup ramp-up.",
        TicketPriority.p2_high, 18, 30, alarm_name="High Vibration",
        resolution="Coupling found misaligned; realigned and rebalanced rotor. Vibration returned to baseline.",
        tags=["vibration", "coupling", "rotor-balance"]),
    _Raw(K201, "K-201 vibration trending high", "Vibration crept up over several shifts, approaching alarm limit.",
        TicketPriority.p2_high, 61, 20, alarm_name="High Vibration",
        resolution="Bearing showed early wear on inspection; replaced drive-end bearing.",
        tags=["vibration", "bearing"]),
    # ---- K-202 Low Lube Oil Pressure: ONE STILL-OPEN ticket ----
    _Raw(K202, "K-202 lube oil pressure trending low", "Lube oil header pressure has been drifting down for 2 shifts.",
        TicketPriority.p2_high, 0.03, 0, still_open=True, alarm_name="Low Lube Oil Pressure",
        tags=["lube-oil", "compressor"]),
    # ---- BFP102: very recent, ties to the OTHER active alarm on the demo (High Bearing Temperature) ----
    _Raw(BFP102, "BFP-102 bearing temperature trending up", "Bearing temperature trending up over the last shift.",
        TicketPriority.p2_high, 2, 20, alarm_name="High Bearing Temperature",
        resolution="Cooling water valve to the bearing housing was partially closed; valve opened, temperature "
                   "trending down. Recommend monitoring, may recur if the valve drifts closed again.",
        tags=["bearing", "cooling-water"]),
    # ---- K-203: one resolved, unrelated in time, for correlation-cluster realism ----
    _Raw(K203, "K-203 suction pressure low during startup", "Suction pressure alarm during compressor startup sequence.",
        TicketPriority.p3_moderate, 40, 6, alarm_name="Suction Pressure Low",
        resolution="Upstream valve was left throttled after maintenance; fully opened, pressure normalized.",
        tags=["suction", "valve"]),
    # ---- BFP101: recurring seal/bearing history ----
    _Raw(BFP101, "BFP-101 seal flush low flow, recurring", "Seal flush flow alarm recurring roughly every few days.",
        TicketPriority.p3_moderate, 12, 4, alarm_name="Seal Flush Low Flow",
        resolution="Flush line strainer was partially blocked; cleaned strainer, flow restored.",
        tags=["seal", "recurring"]),
    _Raw(BFP101, "BFP-101 bearing temperature high", "Bearing temperature alarm, pump still running.",
        TicketPriority.p2_high, 35, 8, alarm_name="High Bearing Temperature",
        resolution="Seal flush had been degraded for some time, starving the bearing of cooling. Flush line "
                   "cleaned and bearing temperature monitored back to normal; recommend addressing seal flush "
                   "chattering to prevent recurrence.",
        tags=["bearing", "seal"]),
    _Raw(BFP101, "BFP-101 low suction pressure, repeat occurrence", "Suction pressure alarm, third time this month.",
        TicketPriority.p2_high, 70, 5, alarm_name="Low Suction Pressure",
        resolution="Suction strainer differential pressure was high; strainer cleaned. Recommend evaluating "
                   "strainer mesh size, this is a recurring issue.",
        tags=["suction", "recurring", "strainer"]),
    _Raw(BFP101, "BFP-101 seal replacement", "Following repeated seal flush alarms, mechanical seal inspected.",
        TicketPriority.p3_moderate, 95, 12, alarm_name="Seal Flush Low Flow",
        resolution="Mechanical seal showed wear; replaced seal assembly. Flush flow stable since.",
        tags=["seal", "recurring"]),
    # ---- Boiler B-301: critical safety history ----
    _Raw("Steam Boiler B-301", "B-301 low drum level trip", "Boiler tripped on low-low drum level.",
        TicketPriority.p1_critical, 25, 3, alarm_name="Low Drum Level",
        resolution="Feedwater control valve positioner had failed; replaced positioner, re-commissioned level "
                   "control loop, boiler restarted per procedure.",
        tags=["boiler", "trip", "safety"]),
    _Raw("Steam Boiler B-301", "B-301 flame failure during load change", "Flame scanner lost signal during a rapid load reduction.",
        TicketPriority.p1_critical, 88, 2, alarm_name="Flame Failure",
        resolution="Flame scanner lens was fouled; cleaned lens and verified signal strength. Purged and relit "
                   "per burner management procedure.",
        tags=["boiler", "burner", "safety"]),
    # ---- Unit 4 P-401: chattering resolved with a deadband fix ----
    _Raw("Reformer Feed Pump P-401", "P-401 discharge flow alarm chattering", "Low discharge flow alarm chattering "
        "repeatedly, dozens of times per day, no real process impact observed.",
        TicketPriority.p4_low, 55, 48, alarm_name="Low Discharge Flow",
        resolution="Confirmed with process engineering that flow is within normal operating band; alarm setpoint "
                   "had no deadband. Added a deadband and 10s on-delay to the alarm configuration; chattering "
                   "stopped.",
        tags=["nuisance", "deadband", "rationalization"]),
    # ---- Unit 1 (North): stale-alarm ticket, closed without root-cause fix ----
    _Raw("Desalter Motor M-101", "M-101 winding temperature alarm active for extended period",
        "Winding temperature alarm has been active for several hours without operator action.",
        TicketPriority.p3_moderate, 30, 26, alarm_name="High Winding Temperature",
        resolution="Reading confirmed within safe operating range on local gauge; alarm setpoint under review by "
                   "reliability engineering, no immediate action taken.",
        tags=["stale", "motor"]),
    # ---- Motor trips on Unit 5 ----
    _Raw("Lube Oil Pump Motor M-502", "M-502 overload trip", "Motor tripped on thermal overload protection.",
        TicketPriority.p2_high, 45, 5, alarm_name="Overload Trip",
        resolution="Coupling to K-201 lube oil pump was binding; freed coupling, verified free rotation, motor "
                   "restarted successfully.",
        tags=["motor", "trip"]),
    _Raw("Cooling Tower Fan Motor M-503", "M-503 fan motor winding temperature high",
        "Winding temperature trending up on the cooling tower fan motor.",
        TicketPriority.p3_moderate, 20, 6, alarm_name="High Winding Temperature",
        resolution="Fan bearing grease had hardened, increasing load; regreased bearing, temperature normalized.",
        tags=["motor", "cooling-tower"]),
    # ---- generic tickets: no alarm link at all (search-noise realism) ----
    _Raw("Crude Charge Pump P-101", "Quarterly preventive maintenance - P-101", "Routine PM per maintenance schedule.",
        TicketPriority.p4_low, 15, 4, resolution="PM completed, no abnormalities found.", tags=["pm", "routine"]),
    _Raw("Deaerator DA-301", "Annual inspection - DA-301", "Scheduled annual internal inspection.",
        TicketPriority.p4_low, 100, 6, resolution="Inspection completed, vessel in good condition.",
        tags=["pm", "inspection"]),
    _Raw("Atmospheric Heater H-101", "H-101 refractory inspection follow-up", "Follow-up on prior refractory hot spot observation.",
        TicketPriority.p3_moderate, 130, 5, resolution="Hot spot within acceptable limits, continue monitoring.",
        tags=["pm", "refractory"]),
]


def build_ticket_dataset(seed: int = DEFAULT_SEED, now: datetime = SEED_NOW) -> list[Ticket]:
    """Deterministic: same seed -> identical tickets. Reuses alarm_api's asset
    catalogue so asset_id/site/unit/category line up exactly."""
    rng = random.Random(seed)
    assets = {a.name: a for a in build_dataset(seed, now).assets}
    tickets: list[Ticket] = []
    for n, r in enumerate(sorted(RAW, key=lambda x: x.opened_offset_days, reverse=True), start=1):
        asset = assets[r.asset]
        opened = now - timedelta(days=r.opened_offset_days, hours=rng.uniform(0, 4))
        if opened > now:
            opened = now - timedelta(hours=rng.uniform(1, 4))
        if r.still_open:
            state = rng.choice([TicketState.new, TicketState.in_progress])
            resolved = closed = None
            updated = opened + timedelta(hours=rng.uniform(0, 2))
        else:
            resolved = opened + timedelta(hours=r.duration_hours)
            if resolved > now:
                resolved = now - timedelta(hours=rng.uniform(0, 1))
            closed_delay = rng.uniform(4, 72)
            closed = resolved + timedelta(hours=closed_delay)
            if closed > now:
                state, closed, updated = TicketState.resolved, None, resolved
            else:
                state, updated = TicketState.closed, closed
        tickets.append(Ticket(
            ticket_id=f"INC{n:07d}", short_description=r.short, description=r.desc, state=state,
            priority=r.priority, category=asset.maintenance_group, assignment_group=asset.maintenance_group,
            assigned_to=None if r.still_open and state == TicketState.new else rng.choice(TECHNICIANS),
            reported_by=rng.choice(REPORTERS), asset_id=asset.asset_id, asset_name=asset.name,
            site=asset.site, unit=asset.unit, alarm_name=r.alarm_name, opened_at=opened, updated_at=updated,
            resolved_at=resolved, closed_at=closed, resolution_notes=r.resolution, tags=r.tags,
        ))
    return sorted(tickets, key=lambda t: t.opened_at)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Generate and inspect the ticket seed dataset.")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    tickets = build_ticket_dataset(args.seed)
    states = {}
    for t in tickets:
        states[t.state.value] = states.get(t.state.value, 0) + 1
    print(f"tickets={len(tickets)} states={states} "
          f"first={tickets[0].opened_at:%Y-%m-%d} last={tickets[-1].opened_at:%Y-%m-%d}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps([t.model_dump(mode="json") for t in tickets], indent=2))
        print(f"wrote {args.out}")