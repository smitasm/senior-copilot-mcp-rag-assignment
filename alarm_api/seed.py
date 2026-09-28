"""Deterministic synthetic dataset for the Alarm API simulator.

`build_dataset()` is a pure function: same seed -> identical assets and alarms,
so tests, demos and Docker runs all see the same world.

Design goals
------------
1. Satisfy every Postman precondition (assets to search, active EastRefinery
   alarms, a Unit 2 flood, compressors to correlate, Unit 5 motors, ...).
2. Cover BOTH time windows we need: Postman's 2026-05-01..2026-07-01 and
   "last 90 days" relative to SEED_NOW (2026-09-28).
3. Contain deliberate "storylines" (recurring pump trouble, a compressor
   cluster, floods, chattering, stale alarms) so analytics endpoints return
   meaningful results and the demo has a real story to tell.
4. Alarm NAMES are stable join keys: the ticket history and the RAG document
   corpus refer to the same names ("High Vibration", "Low Suction Pressure").
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from alarm_api.models import Alarm, AlarmStatus, AlarmType, AssetMetadata, Severity

UTC = timezone.utc
SEED_NOW = datetime(2026, 9, 28, 12, 0, tzinfo=UTC)  # the simulator's "now"
DATA_START = datetime(2026, 4, 1, tzinfo=UTC)
DEFAULT_SEED = 42

P, D, S, G = AlarmType.process, AlarmType.device, AlarmType.safety, AlarmType.diagnostic
LOW, MED, HIGH, CRIT = Severity.low, Severity.medium, Severity.high, Severity.critical


# --------------------------------------------------------------------------
# Assets  (list order defines asset ids: AST-0001, AST-0002, ...)
# Each unit belongs to exactly one site, so "Unit 2" is unambiguous.
#   EastRefinery: Unit 2, 3, 5   NorthPlant: Unit 1   SouthPlant: Unit 4
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class AssetSpec:
    name: str
    asset_type: str
    site: str
    unit: str
    manufacturer: str
    model: str
    criticality: str
    group: str
    related: tuple[str, ...] = ()


K201 = "Recycle Gas Compressor K-201"
K202 = "Recycle Gas Compressor K-202"
K203 = "Make-up Gas Compressor K-203"
BFP101 = "Boiler Feed Pump 101"
BFP102 = "Boiler Feed Pump 102"
B301 = "Steam Boiler B-301"
M101 = "Desalter Motor M-101"
H101 = "Atmospheric Heater H-101"
P401 = "Reformer Feed Pump P-401"

E, N, SO = "EastRefinery", "NorthPlant", "SouthPlant"
ASSET_SPECS: list[AssetSpec] = [
    AssetSpec(K201, "compressor", E, "Unit 2", "Siemens Energy", "STC-SV", "A", "Rotating Equipment East", (K202, K203, "Lube Oil Pump Motor M-502")),
    AssetSpec(K202, "compressor", E, "Unit 2", "Siemens Energy", "STC-SV", "A", "Rotating Equipment East", (K201, K203)),
    AssetSpec(K203, "compressor", E, "Unit 2", "Atlas Copco", "ZH-4500", "A", "Rotating Equipment East", (K201, K202)),
    AssetSpec("Hydrotreater Feed Pump P-201", "pump", E, "Unit 2", "Flowserve", "HPX", "B", "Rotating Equipment East"),
    AssetSpec(BFP101, "pump", E, "Unit 3", "KSB", "HGM-S", "A", "Utilities East", (BFP102, B301)),
    AssetSpec(BFP102, "pump", E, "Unit 3", "KSB", "HGM-S", "A", "Utilities East", (BFP101, B301)),
    AssetSpec(B301, "boiler", E, "Unit 3", "Babcock & Wilcox", "FM-120", "A", "Utilities East", (BFP101, BFP102)),
    AssetSpec("Deaerator DA-301", "deaerator", E, "Unit 3", "Cochrane", "DA-30", "B", "Utilities East"),
    AssetSpec("Cooling Water Pump Motor M-501", "motor", E, "Unit 5", "ABB", "M3BP-315", "B", "Electrical East"),
    AssetSpec("Lube Oil Pump Motor M-502", "motor", E, "Unit 5", "ABB", "M3BP-160", "B", "Electrical East", (K201,)),
    AssetSpec("Cooling Tower Fan Motor M-503", "motor", E, "Unit 5", "ABB", "M3BP-250", "C", "Electrical East"),
    AssetSpec("Main Drive Motor M-504", "motor", E, "Unit 5", "ABB", "AMA-400", "B", "Electrical East"),
    AssetSpec("Crude Charge Pump P-101", "pump", N, "Unit 1", "Sulzer", "OH2", "B", "Mechanical North"),
    AssetSpec(H101, "heater", N, "Unit 1", "Fired Heaters Inc", "FH-88", "B", "Mechanical North"),
    AssetSpec("Overhead Gas Compressor K-101", "compressor", N, "Unit 1", "Ingersoll Rand", "CentacII", "B", "Mechanical North"),
    AssetSpec(M101, "motor", N, "Unit 1", "Siemens", "1LE1", "C", "Electrical North"),
    AssetSpec(P401, "pump", SO, "Unit 4", "Grundfos", "NK-150", "B", "Mechanical South"),
    AssetSpec("Hydrogen Recycle Motor M-401", "motor", SO, "Unit 4", "WEG", "W22", "B", "Electrical South"),
    AssetSpec("Product Cooler E-401", "heat_exchanger", SO, "Unit 4", "Alfa Laval", "AlfaNova", "C", "Mechanical South"),
    AssetSpec("Naphtha Splitter Reboiler H-401", "heater", SO, "Unit 4", "Fired Heaters Inc", "FH-52", "B", "Mechanical South"),
]

# --------------------------------------------------------------------------
# Alarm templates per asset type: (name, type, severity, weight, description)
# --------------------------------------------------------------------------
TEMPLATES: dict[str, list[tuple[str, AlarmType, Severity, float, str]]] = {
    "compressor": [
        ("High Discharge Temperature", P, HIGH, 3, "Discharge temperature above high limit."),
        ("Low Lube Oil Pressure", D, HIGH, 2, "Lube oil header pressure below low limit."),
        ("High Vibration", D, CRIT, 1, "Shaft vibration above trip-advisory limit."),
        ("Suction Pressure Low", P, MED, 3, "Suction pressure below normal operating range."),
        ("Anti-Surge Valve Open", P, MED, 2, "Anti-surge recycle valve opened beyond threshold."),
    ],
    "pump": [
        ("Low Suction Pressure", P, HIGH, 3, "Pump suction pressure below NPSH margin limit."),
        ("High Bearing Temperature", D, HIGH, 2, "Bearing metal temperature above high limit."),
        ("Seal Flush Low Flow", D, MED, 2, "Mechanical seal flush flow below minimum."),
        ("Pump Trip", S, CRIT, 1, "Pump tripped on protective interlock."),
        ("Low Discharge Flow", P, LOW, 2, "Discharge flow below minimum flow setpoint."),
    ],
    "motor": [
        ("High Winding Temperature", D, MED, 3, "Stator winding temperature above alarm limit."),
        ("Overload Trip", S, HIGH, 1, "Motor tripped on thermal overload."),
        ("Motor Fail To Start", D, MED, 1, "Start command issued but no run feedback."),
        ("High Vibration", D, HIGH, 1, "Motor bearing vibration above alarm limit."),
    ],
    "boiler": [
        ("Low Drum Level", S, CRIT, 0.5, "Steam drum level below low-low limit."),
        ("High Steam Pressure", P, MED, 3, "Steam header pressure above high limit."),
        ("Flame Failure", S, CRIT, 0.3, "Burner flame scanner lost flame signal."),
        ("Feedwater Flow Low", P, HIGH, 2, "Boiler feedwater flow below minimum."),
    ],
    "heater": [
        ("High Outlet Temperature", P, HIGH, 3, "Heater outlet temperature above high limit."),
        ("Low Fuel Gas Pressure", P, MED, 3, "Fuel gas pressure below low limit."),
        ("Burner Flame Unstable", D, MED, 2, "Flame scanner reports unstable combustion."),
    ],
    "deaerator": [
        ("Low Deaerator Level", P, MED, 2, "Deaerator storage level below low limit."),
        ("High Deaerator Pressure", P, LOW, 2, "Deaerator pressure above normal range."),
    ],
    "heat_exchanger": [
        ("High Outlet Temperature", P, MED, 2, "Cooler outlet temperature above limit."),
        ("Fouling Indicator High", G, LOW, 2, "Calculated fouling factor above threshold."),
    ],
}
RATE_PER_DAY = {"compressor": 0.30, "pump": 0.30, "motor": 0.25, "boiler": 0.30,
                "heater": 0.25, "heat_exchanger": 0.15, "deaerator": 0.15}
_T = {t: {x[0]: x for x in v} for t, v in TEMPLATES.items()}  # lookup by name


@dataclass
class Dataset:
    assets: list[AssetMetadata]
    alarms: list[Alarm]
    now: datetime

    def asset_by_name(self, name: str) -> AssetMetadata:
        return next(a for a in self.assets if a.name == name)


@dataclass
class _Raw:
    asset: str
    name: str
    atype: AlarmType
    severity: Severity
    start: datetime
    end: datetime | None
    ack: datetime | None
    description: str


class _Builder:
    def __init__(self, now: datetime) -> None:
        self.now = now
        self.raw: list[_Raw] = []

    def add(self, asset: str, name: str, atype: AlarmType, sev: Severity, start: datetime,
            duration_min: float | None, ack_delay_min: float | None, description: str | None = None) -> None:
        """duration_min=None -> still in alarm at `now` (active / acknowledged)."""
        if description is None:
            spec_type = next(s.asset_type for s in ASSET_SPECS if s.name == asset)
            description = next((t[4] for t in TEMPLATES[spec_type] if t[0] == name), name)
        end = None if duration_min is None else start + timedelta(minutes=duration_min)
        if end is not None and end > self.now:
            return  # never let history leak past "now"
        ack = None if ack_delay_min is None else start + timedelta(minutes=ack_delay_min)
        if ack is not None and end is not None:
            ack = min(ack, end)
        self.raw.append(_Raw(asset, name, atype, sev, start, end, ack, description))


def _lower(sev: Severity, rng: random.Random) -> Severity:
    if sev.rank > 0 and rng.random() < 0.12:
        return list(Severity)[sev.rank - 1]
    return sev


def _background(b: _Builder, rng: random.Random) -> None:
    end = b.now - timedelta(hours=6)
    span_min = (end - DATA_START).total_seconds() / 60
    span_days = span_min / 1440
    for spec in ASSET_SPECS:
        tmpls = TEMPLATES[spec.asset_type]
        n = round(RATE_PER_DAY[spec.asset_type] * span_days * (0.8 + 0.4 * rng.random()))
        for _ in range(n):
            name, atype, sev, _w, desc = rng.choices(tmpls, weights=[t[3] for t in tmpls])[0]
            start = DATA_START + timedelta(minutes=rng.random() * span_min)
            duration = min(240.0, max(2.0, rng.lognormvariate(3.3, 0.8)))
            r = rng.random()
            ack = None if r < 0.10 else (rng.uniform(20, 60) if r < 0.20 else rng.uniform(1, 15))
            b.add(spec.name, name, atype, _lower(sev, rng), start, duration, ack, desc)


def _storyline_bfp101(b: _Builder, rng: random.Random) -> None:
    """Recurring: seal flush low -> low suction pressure -> high bearing temp, every ~4 days."""
    t, i = datetime(2026, 4, 8, 9, 10, tzinfo=UTC), 0
    while t < datetime(2026, 9, 24, tzinfo=UTC):
        s = t + timedelta(hours=rng.randint(-6, 6), minutes=rng.randint(0, 59))
        m = timedelta(minutes=1)
        b.add(BFP101, "Seal Flush Low Flow", D, MED, s, 25, rng.randint(3, 12))
        b.add(BFP101, "Low Suction Pressure", P, HIGH, s + 9 * m, 40, rng.choice([4, 6, 9, 25, 35]))
        b.add(BFP101, "High Bearing Temperature", D, CRIT if i % 3 == 0 else HIGH, s + 21 * m, 55, rng.randint(5, 30))
        t += timedelta(days=4)
        i += 1


def _storyline_compressors(b: _Builder, rng: random.Random) -> None:
    """Co-occurring cluster on K-201/K-202/K-203 every ~9 days (correlation target)."""
    t, i = datetime(2026, 4, 12, 4, 30, tzinfo=UTC), 0
    while t < datetime(2026, 9, 20, tzinfo=UTC):
        s = t + timedelta(minutes=rng.randint(0, 90))
        m = timedelta(minutes=1)
        b.add(K201, "High Discharge Temperature", P, HIGH, s, 45, rng.randint(2, 10))
        b.add(K202, "Low Lube Oil Pressure", D, HIGH, s + 6 * m, 35, rng.randint(3, 12))
        b.add(K203, "Suction Pressure Low", P, MED, s + 11 * m, 30, rng.randint(4, 15))
        if i % 2 == 0:
            b.add(K201, "High Vibration", D, CRIT, s + 14 * m, 60, rng.randint(2, 8))
        t += timedelta(days=9)
        i += 1


def _storyline_floods(b: _Builder, rng: random.Random) -> None:
    """Three Unit 2 alarm floods (26 alarms inside 9 minutes)."""
    unit2 = [a for a in ASSET_SPECS if a.unit == "Unit 2"]
    for start in (datetime(2026, 5, 14, 3, 20, tzinfo=UTC), datetime(2026, 6, 9, 15, 5, tzinfo=UTC),
                  datetime(2026, 8, 21, 11, 30, tzinfo=UTC)):
        for _ in range(26):
            spec = rng.choice(unit2)
            tm = rng.choices(TEMPLATES[spec.asset_type], weights=[t[3] for t in TEMPLATES[spec.asset_type]])[0]
            b.add(spec.name, tm[0], tm[1], _lower(tm[2], rng), start + timedelta(seconds=rng.uniform(0, 540)),
                  rng.randint(8, 60), rng.choice([None, 3, 8]), tm[4])


def _storyline_chatter_and_stale(b: _Builder, rng: random.Random) -> None:
    # SouthPlant Unit 4: P-401 chattering bursts (nuisance alarm)
    t = datetime(2026, 4, 3, 2, 0, tzinfo=UTC)
    while t < datetime(2026, 9, 25, tzinfo=UTC):
        cur = t
        for _ in range(rng.randint(6, 10)):
            b.add(P401, "Low Discharge Flow", P, LOW, cur, rng.randint(1, 3), None)
            cur += timedelta(minutes=rng.randint(1, 4))
        t += timedelta(days=5)
    # NorthPlant Unit 1: stale alarms that sit for many hours
    for k in range(10):
        s = datetime(2026, 5, 2, 6, tzinfo=UTC) + timedelta(days=6 * k, hours=(k * 3) % 24)
        b.add(M101, "High Winding Temperature", D, MED, s, rng.uniform(360, 1200), rng.uniform(200, 400))
    for k in range(8):
        s = datetime(2026, 5, 3, 8, tzinfo=UTC) + timedelta(days=8 * k, hours=(k * 5) % 24)
        b.add(H101, "Low Fuel Gas Pressure", P, MED, s, rng.uniform(240, 600), rng.uniform(150, 300))


def _storyline_unit3_unit5(b: _Builder, rng: random.Random) -> None:
    # B-301: critical safety alarms (critical-alarm-density KPI for Unit 3)
    for k in range(12):
        s = datetime(2026, 4, 20, 5, tzinfo=UTC) + timedelta(days=13 * k + rng.random() * 3, hours=rng.randint(0, 12))
        name = "Flame Failure" if k % 3 == 2 else "Low Drum Level"
        b.add(B301, name, S, CRIT, s, rng.randint(10, 45), rng.randint(1, 5))
    # Unit 5 motors: guaranteed safety + device alarms inside the Postman window
    for d, asset, name, atype, sev in [
        ((5, 7), "Lube Oil Pump Motor M-502", "Overload Trip", S, HIGH),
        ((5, 22), "Cooling Water Pump Motor M-501", "Motor Fail To Start", D, MED),
        ((6, 6), "Lube Oil Pump Motor M-502", "Overload Trip", S, HIGH),
        ((6, 19), "Main Drive Motor M-504", "Motor Fail To Start", D, MED),
    ]:
        b.add(asset, name, atype, sev, datetime(2026, d[0], d[1], 10, 15, tzinfo=UTC), 30, 6)


def _active_now(b: _Builder) -> None:
    """Alarms still in alarm at SEED_NOW. K-201 High Vibration is the demo hero:
    critical, unacknowledged, on a criticality-A asset, correlated with K-202/K-203."""
    m = timedelta(minutes=1)
    n = b.now
    b.add(K201, "High Vibration", D, CRIT, n - 42 * m, None, None)
    b.add(K202, "Low Lube Oil Pressure", D, HIGH, n - 36 * m, None, None)
    b.add(K203, "Suction Pressure Low", P, MED, n - 31 * m, None, 5)  # acknowledged
    b.add(BFP102, "High Bearing Temperature", D, HIGH, n - 180 * m, None, None)
    b.add(B301, "High Steam Pressure", P, MED, n - 95 * m, None, 10)  # acknowledged
    b.add("Cooling Tower Fan Motor M-503", "High Winding Temperature", D, MED, n - 120 * m, None, None)
    b.add("Crude Charge Pump P-101", "Low Suction Pressure", P, LOW, n - 300 * m, None, None)
    b.add("Naphtha Splitter Reboiler H-401", "Burner Flame Unstable", D, MED, n - 50 * m, None, None)


def build_dataset(seed: int = DEFAULT_SEED, now: datetime = SEED_NOW) -> Dataset:
    rng = random.Random(seed)
    b = _Builder(now)
    _background(b, rng)
    _storyline_bfp101(b, rng)
    _storyline_compressors(b, rng)
    _storyline_floods(b, rng)
    _storyline_chatter_and_stale(b, rng)
    _storyline_unit3_unit5(b, rng)
    _active_now(b)

    ids = {s.name: f"AST-{i:04d}" for i, s in enumerate(ASSET_SPECS, start=1)}
    assets = [
        AssetMetadata(
            asset_id=ids[s.name], name=s.name, asset_type=s.asset_type, site=s.site, unit=s.unit,
            manufacturer=s.manufacturer, model=s.model,
            commissioned_on=datetime(2012 + i % 9, 1 + i % 12, 15, tzinfo=UTC),
            criticality=s.criticality, maintenance_group=s.group,
            related_asset_ids=[ids[r] for r in s.related],
        )
        for i, s in enumerate(ASSET_SPECS)
    ]
    spec_by_name = {s.name: s for s in ASSET_SPECS}
    alarms: list[Alarm] = []
    for n, r in enumerate(sorted(b.raw, key=lambda x: (x.start, x.asset, x.name)), start=1):
        sp = spec_by_name[r.asset]
        status = (AlarmStatus.cleared if r.end else
                  AlarmStatus.acknowledged if r.ack else AlarmStatus.active)
        alarms.append(Alarm(
            alarm_id=f"ALM-{n:06d}", asset_id=ids[r.asset], asset_name=r.asset, site=sp.site, unit=sp.unit,
            alarm_name=r.name, alarm_type=r.atype, severity=r.severity, status=status,
            description=r.description, start_time=r.start, acknowledged_at=r.ack, end_time=r.end,
        ))
    return Dataset(assets=assets, alarms=alarms, now=now)


def _summary(ds: Dataset) -> str:
    c_sev = Counter(a.severity.value for a in ds.alarms)
    c_stat = Counter(a.status.value for a in ds.alarms)
    return (f"assets={len(ds.assets)} alarms={len(ds.alarms)} severity={dict(c_sev)} "
            f"status={dict(c_stat)} first={ds.alarms[0].start_time:%Y-%m-%d} last={ds.alarms[-1].start_time:%Y-%m-%d}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Generate and inspect the seed dataset.")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--out", type=Path, help="optional path to dump the dataset as JSON")
    args = ap.parse_args()
    data = build_dataset(args.seed)
    print(_summary(data))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({
            "assets": [a.model_dump(mode="json") for a in data.assets],
            "alarms": [a.model_dump(mode="json") for a in data.alarms],
        }, indent=2))
        print(f"wrote {args.out}")